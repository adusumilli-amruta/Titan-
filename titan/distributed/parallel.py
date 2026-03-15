import torch

try:
    from accelerate import Accelerator
    _ACCELERATE_AVAILABLE = True
except ImportError:
    Accelerator = None
    _ACCELERATE_AVAILABLE = False

class MicroBatchParallelHandler:
    """
    Manages physical VRAM constraints by splitting a desired global batch size 
    into smaller micro-batches that fit on the GPU, accumulating their gradients, 
    and stepping the optimizer once the global batch size is reached.
    """
    def __init__(self, accelerator: Accelerator, global_batch_size: int, micro_batch_size: int):
        self.accelerator = accelerator
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        
        # Calculate how many steps needed before optimizer.step()
        self.gradient_accumulation_steps = global_batch_size // (micro_batch_size * accelerator.num_processes)
        
        if self.gradient_accumulation_steps < 1:
            self.gradient_accumulation_steps = 1
            
        # Optional check: is it perfectly divisible?
        assert global_batch_size % (micro_batch_size * accelerator.num_processes) == 0, \
            "Global batch size must be divisible by (micro_batch_size * num_gpus)"

    def should_step(self, step: int) -> bool:
        """Returns True if it's time to synchronize gradients and update weights."""
        return (step + 1) % self.gradient_accumulation_steps == 0

    def backward_step(self, loss: torch.Tensor, model, optimizer, lr_scheduler, step: int):
        """
        Handles the backward pass, scaling the loss appropriately so the
        accumulated gradients equal the expected global batch gradient.
        """
        loss = loss / self.gradient_accumulation_steps
        self.accelerator.backward(loss)

        if self.should_step(step):
            # Clip gradients to prevent exploding logits (very common in PPO)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
