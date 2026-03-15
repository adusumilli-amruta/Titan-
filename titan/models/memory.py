import torch
import torch.nn as nn

class RecurrentMemoryState(nn.Module):
    """
    Manages the Key-Value (KV) cache for chunks of sequence during long-context execution.
    Like Transformer-XL or MemGPT, this intercepts the 'past_key_value' from the transformer
    blocks and persists them across forward passes for the same document/interaction.
    """
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.max_memory_tokens = config.max_recurrent_memory_tokens

        # Instead of storing in parameters, these are transient states managed during training
        # or inference loops. We store them in a dictionary keyed by the batch ID or generation session.
        self.state_cache = {}

    def init_state(self, batch_id, device, dtype):
        """Initializes an empty cache state for a new interaction/document."""
        self.state_cache[batch_id] = [None for _ in range(self.num_layers)]

    def get_state(self, batch_id):
        """Retrieves the KV cache for the current layer chunking."""
        if batch_id not in self.state_cache:
            return None
        return self.state_cache.get(batch_id)

    def update_state(self, batch_id, layer_idx, key_cache, value_cache):
        """
        Updates the memory buffer.
        key_cache, value_cache: [bsz, num_heads, seq_len, head_dim]
        """
        if batch_id not in self.state_cache:
            self.init_state(batch_id, key_cache.device, key_cache.dtype)

        # Truncate if we exceed the physical memory limit of our recurrence window
        if key_cache.shape[2] > self.max_memory_tokens:
            key_cache = key_cache[:, :, -self.max_memory_tokens:, :]
            value_cache = value_cache[:, :, -self.max_memory_tokens:, :]

        # Detach to stop exploding gradients across completely massive BPTT (Backpropagation Through Time)
        # unless explicitly doing truncated BPTT.
        self.state_cache[batch_id][layer_idx] = (key_cache.detach(), value_cache.detach())

    def clear_state(self, batch_id):
        """Frees memory for a completed document."""
        if batch_id in self.state_cache:
            del self.state_cache[batch_id]
