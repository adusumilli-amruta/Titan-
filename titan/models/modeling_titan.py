import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from .attention import SlidingWindowAttention
from .rope import ChunkedRotaryEmbedding
from transformers.modeling_outputs import CausalLMOutputWithPast

class TitanConfig(PretrainedConfig):
    model_type = "titan"
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=8192,
        sliding_window_size=4096,
        max_recurrent_memory_tokens=16384,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.sliding_window_size = sliding_window_size
        self.max_recurrent_memory_tokens = max_recurrent_memory_tokens
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        super().__init__(**kwargs)

class TitanRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

class TitanMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.functional.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class TitanDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = SlidingWindowAttention(config)
        self.mlp = TitanMLP(config)
        self.input_layernorm = TitanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TitanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cos=None, sin=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cos=cos,
            sin=sin
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

class TitanForCausalLM(PreTrainedModel):
    config_class = TitanConfig

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # RoPE applied once at top level to share cos/sin cache across layers
        self.rotary_emb = ChunkedRotaryEmbedding(config.hidden_size // config.num_attention_heads, max_position_embeddings=config.max_position_embeddings)
        
        self.layers = nn.ModuleList([TitanDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = TitanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, output_hidden_states=False, output_attentions=False):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        batch_size, seq_length = input_ids.shape[:2]

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            if past_key_values is not None:
                position_ids = position_ids + past_key_values[0][0].shape[2] # Offset by KV cache length
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_length + (past_key_values[0][0].shape[2] if past_key_values else 0))

        next_decoder_cache = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # Selective gradient checkpointing would happen here in training loops
            # to preserve memory by recomputing only self_attn and NOT the MLP.

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                cos=cos,
                sin=sin
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=None,
        )
