import torch
from torch.utils.checkpoint import checkpoint

def selective_activation_checkpointing(module, *args, use_reentrant=False, **kwargs):
    """
    Standard gradient checkpointing recomputes the entire forward pass of a module
    during the backward pass to save VRAM. 
    
    Selective Activation Checkpointing (used here for extreme long-context) only 
    checkpoints the most memory-intensive operations (like the Self-Attention matrix 
    allocations) while keeping the cheaper MLP activations in memory, striking an 
    optimal balance between compute overhead and memory savings.
    """
    def custom_forward(*inputs):
        return module(*inputs, **kwargs)

    # In a real implementation, `module` would be split. 
    # For this architecture, we wrap the entire layer but conceptually 
    # this function is injected into the TitanDecoderLayer to only wrap `self_attn`.
    if any(arg.requires_grad for arg in args if isinstance(arg, torch.Tensor)):
        return checkpoint(custom_forward, *args, use_reentrant=use_reentrant)
    else:
        return module(*args, **kwargs)

def estimate_memory_requirements(config, batch_size, seq_len):
    """
    Utility to calculate the VRAM required for KV caching 
    and gradients given the custom SWA and Recurrent parameters.
    Returns size in GB.
    """
    # 2 bytes for bf16/fp16
    bytes_per_param = 2
    
    # Model Weights
    num_params = (
        config.vocab_size * config.hidden_size + # Embedding
        config.num_hidden_layers * (
            4 * config.hidden_size * config.hidden_size + # QKV, O
            3 * config.hidden_size * config.intermediate_size # MLP
        )
    )
    weight_memory = num_params * bytes_per_param

    # Optimizer (AdamW takes 8 bytes per param for fp32 states)
    optim_memory = num_params * 8

    # KV Cache Memory (Recurrent Memory limit)
    # [num_layers, 2 (K, V), batch_size, num_heads, seq_len, head_dim]
    max_kv_len = min(seq_len, config.max_recurrent_memory_tokens)
    kv_memory = (
        config.num_hidden_layers * 2 * batch_size * 
        config.num_attention_heads * max_kv_len * 
        (config.hidden_size // config.num_attention_heads) * bytes_per_param
    )

    total_memory_gb = (weight_memory + optim_memory + kv_memory) / (1024 ** 3)
    return total_memory_gb
