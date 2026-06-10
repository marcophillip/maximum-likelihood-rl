import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.hidden_dim // config.num_heads

        assert self.head_dim % 2 == 0, "RoPE dimension must be even"

        self.dim = self.head_dim

        inv_freq = 1.0 / (
            config.theta ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(config.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, seq_len: int):
        """
        Returns cos/sin shaped (B, 1, S, dim) for broadcast over heads.
        If position_ids (B, S) is provided, gather those positions;
        otherwise fall back to contiguous 0..seq_len-1.
        """

        cos = self.cos_cached[:seq_len]       # (S, dim) -> broadcasts over (B, H, S, dim)
        sin = self.sin_cached[:seq_len]       # (S, dim)

        return cos, sin

    def apply_rope(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        # x: (B, H, S, dim)  cos/sin: (B, 1, S, dim)
        return (x * cos) + (self._rotate_half(x) * sin)
        


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_groups = config.num_kv_groups
        self.head_dim = config.hidden_dim // config.num_heads
        self.queries_per_kv = config.num_heads // config.num_kv_groups

        self.rope = RotaryPositionEmbedding(config)

        self.q_proj  = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.k_proj  = nn.Linear(config.hidden_dim, config.num_kv_groups * self.head_dim, bias=False)
        self.v_proj  = nn.Linear(config.hidden_dim, config.num_kv_groups * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
    ):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)


        q = self.q_norm(q)  # normalize q and k before RoPE
        k = self.k_norm(k)

        cos, sin = self.rope(seq_len)
        q = self.rope.apply_rope(q, cos, sin)
        k = self.rope.apply_rope(k, cos, sin)

        if self.queries_per_kv > 1:
            k = k.unsqueeze(2).expand(batch_size, self.num_kv_groups, self.queries_per_kv, seq_len, self.head_dim)
            k = k.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(batch_size, self.num_kv_groups, self.queries_per_kv, seq_len, self.head_dim)
            v = v.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True, 
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(output)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.l2 = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.l2(self.activation(self.l1(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = GroupedQueryAttention(config)
        self.mlp = MLP(config)
        self.norm1 = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.attn(x1)

        x2 = self.norm2(x)
        x = x + self.mlp(x2)

        return x




class NanoQwen(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.hidden_dim),
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps),
        ))

        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
        
    def forward(self, input_ids, targets=None):
        x = self.transformer.wte(input_ids)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :]).squeeze(1)
            loss = None

        return logits, loss
    
    @classmethod
    def generate(model, tokenizer, maze_tokens: list[str]):
        model.eval()

        input_ids = tokenizer.encode(maze_tokens)

        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        x = x.to(next(model.parameters()).device)

        generated = []

        with torch.no_grad():
            for _ in range(NanoQwenConfig.max_sequence_len):
                logits, _ = model(x)



        
if __name__ == "__main__":
    config = NanoQwenConfig()
    model = NanoQwen(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    logits, loss = model(input_ids, targets)
    print(f"Logits shape: {logits.shape}, Loss: {loss.item()}")

