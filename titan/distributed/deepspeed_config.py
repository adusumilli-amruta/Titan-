import json
import os

def create_zero3_config(
        batch_size_per_gpu, 
        gradient_accumulation_steps, 
        offload_optimizer=True, 
        offload_param=True,
        save_path="ds_zero3_config.json"
    ):
    """
    Generates a DeepSpeed ZeRO-3 configuration JSON that enables training 
    massive models by offloading optimizer states and parameters to CPU RAM 
    or NVMe when not actively used in the forward/backward pass.
    """
    config = {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": "auto"
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": "auto",
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": batch_size_per_gpu,
        "wall_clock_breakdown": False
    }

    if offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
    if offload_param:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }

    with open(save_path, "w") as f:
        json.dump(config, f, indent=4)
    
    return save_path

def get_deepspeed_env_vars():
    """Returns local rank and world size from standard PyTorch distributed launch."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, world_size
