import torch
from model import NanoQwen
from config  import RLHFConfig
from dataset import _tokenizer
from rewards import MazeEnv, reward_fn, DELTAS
from torch.utils.data import DataLoader
import wandb
from dataset import MazeDataset_RL

wandb.init(project="maze-maxrl")

def compute_advantages(rewards, group_size):
    
    #rewards:    (B*G,) scalar reward per completion
    #group_size: G, number of completions per prompt
    #returns:    (B*G,) normalized advantages
    
    B = rewards.shape[0] // group_size
    grouped = rewards.view(B, group_size)                       # (B, G)
    mean = grouped.mean(dim=1, keepdim=True)                    # (B, 1)
    std  = grouped.std(dim=1, keepdim=True)                     # (B, 1)
    advantages = (grouped - mean) / (std + 1e-8)               
    return advantages.view(-1)


def calculate_clipped_objective(
    logprobs,
    mb_logprobs,
    mb_advantages,
    clip_coef,
    mask,
    n_correct,
):
    """
    logprobs: new actions
    mb_logprobs: old actions
    mask:       (N, T) 1 for real completion tokens, 0 for pad
    n_correct:  number of correct completions (the normalizer)
    """

    diff_ = logprobs - mb_logprobs
    ratio = torch.exp(diff_)

    non_clipped = ratio * mb_advantages
    clipped = torch.clip(ratio, 1-clip_coef, 1+clip_coef)*mb_advantages
    surrogate = torch.minimum(non_clipped, clipped)                 # (N, T) per token

    mask = mask.to(surrogate.dtype)
    # average over each completion's own tokens (the 1/|o_i| term), then
    # normalize by the number of CORRECT completions instead of all completions
    per_seq = (surrogate * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)   # (N,)
    return per_seq.sum() / n_correct

def calc_kl_penalty(logits, ref_logits, kl_coef):

    ref_logprobs = ref_logits.log_softmax(-1)
    logprobs = logits.log_softmax(-1)

    probs = logprobs.exp()

    kl_div = (probs *(logprobs - ref_logprobs)).sum(-1)
    return kl_coef * kl_div.mean()



class RLHFTrainer:
    def __init__(self):

        self.config = RLHFConfig()

        self.model = NanoQwen(self.config)
        self.ref_model = NanoQwen(self.config)

        
        self.ref_model.requires_grad_(False)

        self.device = self.config.device
        self.model.load_state_dict(torch.load(self.config.checkpoint_path, map_location=self.device))
        self.ref_model.load_state_dict(torch.load(self.config.checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.ref_model.to(self.device)

        self.tokenizer= _tokenizer()


        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.group_size = self.config.G
        self.clip_coef = self.config.clip_coef
        self.kl_coef = self.config.kl_coef
        self.grad_clip = self.config.grad_clip
        self.lr = self.config.lr

    @torch.no_grad()
    def rollout(self, prompts):
        self.model.eval()
        encs = self.tokenizer.encode_batch(prompts)
        enc_ids = [e.ids for e in encs]

        prompts_tensor = torch.tensor(enc_ids, dtype=torch.long, device=self.device)

        group_size = self.config.G

        x = prompts_tensor.repeat_interleave(group_size, dim=0)
        prompt_len = x.size(1)

        eos_id = self.tokenizer.token_to_id("DONE")

        done = torch.zeros(x.size(0), dtype=torch.bool, device=self.device)

        max_steps = self.config.max_seq_len - prompt_len

        step_logprobs = []
        for _ in range(max_steps):
            logits, _ = self.model(x, return_last_only=True)
            logp = logits.log_softmax(-1)
            next_ids = torch.multinomial(logp.exp(), num_samples=1)

            step_logprobs.append(logp.gather(-1, next_ids).squeeze(1)) # B*G

            x = torch.cat([x, next_ids], dim=1)
            done |= next_ids.squeeze(1) == eos_id

            if done.all(): break

        generated_ids = x[:, prompt_len:]
        old_logprobs = torch.stack(step_logprobs, dim=1)

        is_eos = (generated_ids == eos_id)
        eos_before = is_eos.cumsum(dim=1) - is_eos.long()
        completion_mask = (eos_before==0).long()

        return x, completion_mask, old_logprobs.detach()
        

    def compute_rewards(self, prompts, completions, optimal_lens):

        G = self.config.G
        envs = [MazeEnv(p.split()) for p in prompts]
        rewards = []

        for i in range(completions.size(0)):
            prompt_idx = i // G  # which prompt this completion belongs to
            optimal_len = optimal_lens[prompt_idx]
            env = envs[prompt_idx]
            
            moves = []
            for token_id in completions[i]:
                token = self.tokenizer.id_to_token(token_id.item())
                if token not in DELTAS:     # DONE, <eos>, <pad>, or any non-move -> end of path
                    break
                moves.append(token)
            outcome = env.simulate(moves)
            reward = reward_fn(outcome, optimal_lens[prompt_idx].item())
            rewards.append(reward)

            if i == 0:   # DEBUG: inspect one completion
                print(f"[debug] moves ({len(moves)}): {moves[:20]}")
                print(f"[debug] outcome: {outcome}")

        n_solved = sum(1 for r in rewards if r > 0)   # DEBUG
        print(f"[debug] solved {n_solved}/{len(rewards)}")

        return torch.tensor(rewards, dtype=torch.float, device=self.device)

    # def gather_logprobs(self, logits, action_tokens):        
    #     """
    #     the model returns raw logits, this function is meant to help turn them to logprobs
    #     """
    #     logprobs = logits.log_softmax(dim=-1)
    #     return logprobs.gather(-1, action_tokens.unsqueeze(-1)).squeeze(-1)
        
    def compute_objective(self, tokens, completion_mask, old_logprobs, advantages, rewards):

        T = completion_mask.size(1)

        full_seq_logits, _ = self.model(tokens)
        self.ref_model.eval()
        with torch.no_grad():
            full_ref_logits, _ = self.ref_model(tokens)

        comp_logits = full_seq_logits[:,:-1][:,-T:]
        comp_ref_logits = full_ref_logits[:, :-1][:, -T:]
        labels = tokens[:, -T:]

        new_logprobs = comp_logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        adv = advantages.unsqueeze(1).expand(-1, T) #(B*G,) -> (B*G, T)

        n_correct = (rewards > 0).float().sum().clamp(min=1)   # number of correct completions

        surrogate = calculate_clipped_objective(
            new_logprobs, old_logprobs, adv, self.clip_coef, completion_mask, n_correct)

        kl = calc_kl_penalty(comp_logits, comp_ref_logits, self.kl_coef)

        return -surrogate  + kl, kl
        

    def train_one_batch(self):
        rl_dataset = MazeDataset_RL(tokenizer=self.tokenizer, json_file_path="data/train.json")
        rl_dataloader = DataLoader(rl_dataset, batch_size=self.config.batch_size, shuffle=True)


        self.model.train()
        
        batch = next(iter(rl_dataloader))
        prompts, optimal_lens = batch

        tokens, completion_mask, old_logprobs = self.rollout(prompts)
        rewards = self.compute_rewards(prompts, tokens[:, -completion_mask.size(1):], optimal_lens)
        advantages = compute_advantages(rewards, self.config.G).detach()

        losses=0
        kls=0
        for _ in range(2):
            
            self.model.train()

            loss, kl = self.compute_objective(tokens, completion_mask, old_logprobs, advantages, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            losses += loss.item()
            kls += kl.item()

        avg_loss = losses / 2
        avg_kl = kls / 2
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()
        n_solved = (rewards > 0).sum().item()
        reward_max = rewards.max().item()

        print(f"Loss: {avg_loss:.4f}, Reward Mean: {reward_mean:.4f}, Reward Std: {reward_std:.4f}, "
              f"KL: {avg_kl:.4f}, solved: {n_solved}/{rewards.numel()}, r_max: {reward_max:.3f}")
        
        wandb.log({
            "loss": avg_loss,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "kl": avg_kl,
        })

    def train(self):
        for epoch in range(self.config.epochs):
            self.train_one_batch()
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f"checkpoint_epoch_{epoch}.pt")


            




if __name__ == "__main__":
    trainer = RLHFTrainer()
    trainer.train()
