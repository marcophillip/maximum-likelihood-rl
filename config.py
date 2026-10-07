from dataclasses import dataclass
import torch


@dataclass
class NanoQwenConfig:
    vocab_size: int = 32
    max_seq_len: int = 512
    hidden_dim: int = 256
    intermediate_dim: int = 1024
    num_heads: int = 4
    num_kv_groups: int = 2
    num_layers: int = 4
    theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: str = "checkpoints/nanoqwen_1500.pt"   # SFT checkpoint (paper: 1500 SFT steps)

    batch_size: int = 64
    epochs: int = 100


@dataclass
class RLHFConfig(NanoQwenConfig):

    G: int = 32                     # rollouts per prompt (paper sweeps 4..128)
    batch_size: int = 64            # prompts per RL step (paper: 256)
    lr: float = 1e-5                # 1e-4 with a fresh Adam collapsed the policy in one step
    warmup_steps: int = 50          # linear lr warmup; Adam's first steps are sign(g) * lr
    rl_steps: int = 3000            # paper: ~9000
    max_new_tokens: int = 64        # paths are ~30 tokens; caps runaway no-DONE completions
    ppo_epochs: int = 1             # one gradient step per rollout batch -> on-policy
    clip_coef: float = 0.2          # verl default; a no-op when ppo_epochs == 1
    grad_clip: float = 1.0
    adv_eps: float = 1e-6           # epsilon in (r - r_hat) / (r_hat + eps)

    # memory knobs: B*G sequences are processed in chunks of this size
    rollout_micro_batch: int = 512
    train_micro_batch: int = 256

    # held-out evaluation: pass@k with temperature-1 sampling (paper's metric)
    eval_every: int = 100
    eval_prompts: int = 400         # val.json has 400 mazes
    eval_n: int = 8                 # samples per val prompt
    eval_pass_k: tuple = (1, 4, 8)

    save_every: int = 250
    save_dir: str = "checkpoints/maxrl"
