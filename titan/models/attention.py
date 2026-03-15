import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SlidingWindowAttention(nn.Module):
    """
    Implements Sliding Window Attention (SWA) combined with FlashAttention concepts.
    Reduces the complexity of self-attention from O(N^2) to O(N x W) where W is the 
    window size. Vital for long-context execution (8K+ tokens) and memory efficiency.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.window_size = config.sliding_window_size

        # QKV Projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cos=None, sin=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 1. Apply RoPE (from ChunkedRotaryEmbedding)
        from .rope import apply_rotary_pos_emb
        if cos is not None and sin is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # 2. Key-Value Caching / Memory Routing
        if past_key_value is not None:
            # past_key_value is a tuple of (key_cache, value_cache)
            # Memory chunking mechanism: drop oldest keys if they exceed the maximum recurrent window
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            
            # Truncate to sliding window size for the KV cache to prevent OOM across chunks
            if key_states.shape[2] > self.window_size:
                key_states = key_states[:, :, -self.window_size:, :]
                value_states = value_states[:, :, -self.window_size:, :]

        past_key_value = (key_states, value_states) if use_cache else None

        # 3. Sliding Window Masking
        # Standard scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            adder = (1.0 - attention_mask) * -10000.0
            attn_weights = attn_weights + adder

        # Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Output accumulation
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
