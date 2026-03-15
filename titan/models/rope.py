import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChunkedRotaryEmbedding(nn.Module):
    """
    Implements Chunked Rotary Position Embeddings (RoPE) for long-context extension.
    Dynamically scales the base frequency based on sequence length, similar to YaRN
    or NTK-aware scaling, to extrapolate beyond the trained context window.
    """
    def __init__(self, dim, max_position_embeddings=8192, base=10000.0, scaling_factor=1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor

        # Inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build initial cos/sin cache
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        # Apply scaling factor for longer contexts (dynamic scaling)
        if self.scaling_factor > 1.0:
            t = t / self.scaling_factor
            
        freqs = torch.outer(t, self.inv_freq.to(device))
        # Different from paper, but follows implementation: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            # Dynamically scale base if we exceed cache
            # Simple linear scaling fallback if context exceeds expectations
            self.scaling_factor = seq_len / self.max_position_embeddings
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Applies rotary position embeddings to queries and keys.
    q, k: [bs, num_heads, seq_len, head_dim]
    """
    if position_ids is not None:
        # Use specific position IDs (e.g., when resuming from a recurrent memory state)
        cos = cos[position_ids].unsqueeze(1) # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)
    else:
        # Trivial sequence aligned
        cos = cos.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
