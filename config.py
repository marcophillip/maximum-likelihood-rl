from dataclasses import dataclass

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
    device="cuda"
    checkpoint_path="checkpoints/nanoqwen.pt"


    batch_size=64
    epochs=100



@dataclass
class RLHFConfig(NanoQwenConfig):
    G: int =5  # number of completions per prompt during training (grpo)
    clip_coef: float = 0.2
    kl_coef: float = 0.1
    grad_clip: float = 1.0
    lr: float = 1e-6
