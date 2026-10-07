"""
things to note between maxrl and grpo
compared with GRPO the only algorithmic change is the advantage:
GRPO :  A_i = (r_i - r_hat) / (std + eps)
MaxRL:  A_i = (r_i - r_hat) / (r_hat + eps)          r_hat = mean reward of the prompt's N rollouts
with a binary reward, no KL term, and one on-policy gradient step per rollout batch.
"""
import math
import os

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from config import RLHFConfig
from dataset import MazeDataset_RL, _tokenizer
from model import NanoQwen
from rewards import MazeEnv, parse_moves, reward_fn


def compute_advantages(rewards, group_size, eps=1e-6):
    """
    MaxRL advantage (paper Alg. 1 line 9; official compute_maxrl_outcome_advantage):
        A_i = (r_i - r_hat) / (r_hat + eps)
    rewards: (B*G,) binary rewards. Groups with no correct rollout get A_i = 0.
    """
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    return ((grouped - mean) / (mean + eps)).view(-1)


def policy_loss(new_logprobs, old_logprobs, advantages, mask, clip_coef, total_tokens):
    """
    PPO-clipped surrogate, aggregated as a token-mean over the whole rollout batch
    (verl's default loss_agg_mode). With ppo_epochs == 1 the ratio is 1 and this is
    exactly the REINFORCE gradient with MaxRL advantages.
    """
    ratio = torch.exp(new_logprobs - old_logprobs)
    adv = advantages.unsqueeze(1)
    surrogate = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * adv)
    return -(surrogate * mask).sum() / total_tokens


def pass_at_k(n, c, k):
    """Unbiased pass@k estimator (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


class MaxRLTrainer:
    def __init__(self):
        self.config = cfg = RLHFConfig()
        self.device = cfg.device

        self.model = NanoQwen(cfg)
        self.model.load_state_dict(torch.load(cfg.checkpoint_path, map_location="cpu"))
        self.model.to(self.device)

        self.tokenizer = _tokenizer()
        self.done_id = self.tokenizer.token_to_id("DONE")
        self.pad_id = self.tokenizer.token_to_id("<pad>")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps)))

        train_ds = MazeDataset_RL(tokenizer=self.tokenizer, json_file_path="data/train.json")
        self.train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        self._data_iter = iter(self.train_loader)

        val_ds = MazeDataset_RL(tokenizer=self.tokenizer, json_file_path="data/val.json")
        self.val_prompts = [val_ds[i][0] for i in range(min(cfg.eval_prompts, len(val_ds)))]

        os.makedirs(cfg.save_dir, exist_ok=True)

    # data
    def _next_batch(self):
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.train_loader)
            return next(self._data_iter)

    def encode_prompts(self, prompts):
        ids = [e.ids for e in self.tokenizer.encode_batch(prompts)]
        L = len(ids[0])
        assert all(len(x) == L for x in ids), "prompts must have equal length (fixed grid size)"
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    # rollout
    @torch.no_grad()
    def rollout(self, prompt_ids, n):
        """
        Sample n completions per prompt at temperature 1.
        Returns full sequences (prompt + completion), a completion mask that covers
        tokens up to and including the first DONE, and the sampling log-probs.
        """
        self.model.eval()
        cfg = self.config
        x_all = prompt_ids.repeat_interleave(n, dim=0)
        prompt_len = x_all.size(1)
        max_new = min(cfg.max_new_tokens, cfg.max_seq_len - prompt_len)

        gens, logps = [], []
        for s in range(0, x_all.size(0), cfg.rollout_micro_batch):
            x = x_all[s:s + cfg.rollout_micro_batch]
            done = torch.zeros(x.size(0), dtype=torch.bool, device=self.device)
            step_logps = []
            for _ in range(max_new):
                logits, _ = self.model(x, return_last_only=True)
                logp = logits.log_softmax(-1)
                next_ids = torch.multinomial(logp.exp(), num_samples=1)
                step_logps.append(logp.gather(-1, next_ids).squeeze(1))
                x = torch.cat([x, next_ids], dim=1)
                done |= next_ids.squeeze(1) == self.done_id
                if done.all():
                    break
            gen = x[:, prompt_len:]
            lp = torch.stack(step_logps, dim=1)
            pad = max_new - gen.size(1)
            gens.append(F.pad(gen, (0, pad), value=self.pad_id))
            logps.append(F.pad(lp, (0, pad), value=0.0))

        gen = torch.cat(gens, dim=0)
        old_logprobs = torch.cat(logps, dim=0)

        is_done = gen == self.done_id
        mask = ((is_done.cumsum(dim=1) - is_done.long()) == 0).long()   # 1 up to & incl. first DONE

        T = int(mask.sum(dim=1).max().item())                            # trim to longest real completion
        gen, old_logprobs, mask = gen[:, :T], old_logprobs[:, :T], mask[:, :T]
        return torch.cat([x_all, gen], dim=1), mask, old_logprobs

    # rewards
    def compute_rewards(self, prompts, completions, n):
        envs = [MazeEnv(p.split()) for p in prompts]
        rewards, n_format_err = [], 0
        for i in range(completions.size(0)):
            toks = [self.tokenizer.id_to_token(t) for t in completions[i].tolist()]
            moves = parse_moves(toks)
            if moves is None:
                n_format_err += 1
                outcome = None
            else:
                outcome = envs[i // n].simulate(moves)
            rewards.append(reward_fn(outcome))
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        return rewards, {"format_err_frac": n_format_err / len(rewards)}

    # train step
    def train_step(self, step):
        cfg = self.config
        prompts, _ = self._next_batch()
        prompt_ids = self.encode_prompts(prompts)

        tokens, mask, old_logprobs = self.rollout(prompt_ids, cfg.G)
        T = mask.size(1)
        rewards, rstats = self.compute_rewards(prompts, tokens[:, -T:], cfg.G)
        advantages = compute_advantages(rewards, cfg.G, cfg.adv_eps)

        self.model.train()
        total_tokens = mask.sum().clamp(min=1).float()
        N = tokens.size(0)
        loss_total = 0.0
        entropy_sum, kl_sum, diag_tokens = 0.0, 0.0, 0.0
        lr = self.optimizer.param_groups[0]["lr"]
        for _ in range(cfg.ppo_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            for s in range(0, N, cfg.train_micro_batch):
                sl = slice(s, s + cfg.train_micro_batch)
                if not advantages[sl].any():          # zero advantage -> zero gradient, skip the forward
                    continue
                logits, _ = self.model(tokens[sl])
                comp_logits = logits[:, -T - 1:-1]     # positions that predict the T completion tokens
                labels = tokens[sl][:, -T:]
                logp_all = comp_logits.log_softmax(-1)
                new_logprobs = logp_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                loss = policy_loss(new_logprobs, old_logprobs[sl], advantages[sl], mask[sl],
                                   cfg.clip_coef, total_tokens)
                loss.backward()
                loss_total += loss.item()
                with torch.no_grad():                  # diagnostics on the tokens that carry gradient
                    m = mask[sl].float()
                    entropy_sum += (-(logp_all.exp() * logp_all).sum(-1) * m).sum().item()
                    kl_sum += ((old_logprobs[sl] - new_logprobs) * m).sum().item()
                    diag_tokens += m.sum().item()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
        self.scheduler.step()

        grouped = rewards.view(-1, cfg.G)
        group_pass = grouped.mean(dim=1)
        log = {
            "train/loss": loss_total / cfg.ppo_epochs,
            "train/grad_norm": float(grad_norm),
            "train/pass_rate": rewards.mean().item(),
            "train/frac_groups_all_fail": (group_pass == 0).float().mean().item(),
            "train/frac_groups_all_pass": (group_pass == 1).float().mean().item(),
            "train/adv_abs_mean": advantages.abs().mean().item(),
            "train/completion_len": mask.sum(dim=1).float().mean().item(),
            "train/format_err_frac": rstats["format_err_frac"],
            "train/lr": lr,
            # entropy / approx_kl are 0 on steps where every group was all-fail (no forward pass ran)
            "train/entropy": entropy_sum / max(diag_tokens, 1.0),
            "train/approx_kl": kl_sum / max(diag_tokens, 1.0),   # ~0 when on-policy (ppo_epochs == 1)
        }
        print(f"step {step:5d} | loss {log['train/loss']:+.4f} | pass {log['train/pass_rate']:.3f} | "
              f"all_fail {log['train/frac_groups_all_fail']:.2f} all_pass {log['train/frac_groups_all_pass']:.2f} | "
              f"len {log['train/completion_len']:.1f} | fmt_err {log['train/format_err_frac']:.3f} | "
              f"lr {lr:.1e} | gnorm {log['train/grad_norm']:.3f} | ent {log['train/entropy']:.3f} | "
              f"kl {log['train/approx_kl']:+.4f}")
        return log

    # eval step
    @torch.no_grad()
    def evaluate(self):
        cfg = self.config
        n = cfg.eval_n
        prompts_per_chunk = max(1, cfg.rollout_micro_batch // n)
        correct = []
        for s in range(0, len(self.val_prompts), prompts_per_chunk):
            prompts = self.val_prompts[s:s + prompts_per_chunk]
            tokens, mask, _ = self.rollout(self.encode_prompts(prompts), n)
            rewards, _ = self.compute_rewards(prompts, tokens[:, -mask.size(1):], n)
            correct.extend(rewards.view(-1, n).sum(dim=1).long().tolist())
        metrics = {}
        for k in cfg.eval_pass_k:
            if k <= n:
                metrics[f"val/pass@{k}"] = sum(pass_at_k(n, c, k) for c in correct) / len(correct)
        print("eval | " + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
        return metrics

    # train loop
    def train(self):
        cfg = self.config
        wandb.log(self.evaluate(), step=0)
        for step in range(1, cfg.rl_steps + 1):
            log = self.train_step(step)
            if step % cfg.eval_every == 0:
                log.update(self.evaluate())
            wandb.log(log, step=step)
            if step % cfg.save_every == 0:
                torch.save(self.model.state_dict(), os.path.join(cfg.save_dir, f"step_{step}.pt"))


if __name__ == "__main__":
    cfg = RLHFConfig()
    wandb.init(project="maze-maxrl", config=cfg.__dict__)
    MaxRLTrainer().train()
