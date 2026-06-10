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
