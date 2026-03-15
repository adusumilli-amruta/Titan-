import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

try:
    import deepspeed
    _DEEPSPEED_AVAILABLE = True
except ImportError:
    _DEEPSPEED_AVAILABLE = False

from titan.models.modeling_titan import TitanForCausalLM, TitanConfig
from titan.distributed.deepspeed_config import get_deepspeed_env_vars
from titan.distributed.parallel import MicroBatchParallelHandler

def pretrain_loop(
    dataset,
    config_dict,
    epochs=1,
    global_batch_size=1024,
    micro_batch_size=8,
    learning_rate=3e-4,
    ds_config_path="ds_zero3_config.json"
):
    """
    Standard Distributed Causal Language Modeling Pre-training loop.
    Optimized for massive throughput using DeepSpeed ZeRO-3 and Microbatching.
    """
    local_rank, world_size = get_deepspeed_env_vars()
    
    # 1. Initialize Model Geometry
    config = TitanConfig(**config_dict)
    model = TitanForCausalLM(config)
    
    # 2. Setup Data
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=True)

    # 3. Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    total_steps = len(dataloader) * epochs // (global_batch_size // (micro_batch_size * world_size))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=total_steps)

    # 4. DeepSpeed Initialization
    model_engine, optimizer, dataloader, scheduler = deepspeed.initialize(
        args=None,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        lr_scheduler=scheduler,
        training_data=dataset,
        config=ds_config_path
    )

    # Note: DeepSpeed internally handles the gradient accumulation steps defined in the JSON.
    # The MicroBatchParallelHandler provided in `titan/distributed` is an Accelerator-native
    # alternative if the user chooses not to use DeepSpeed.

    # 5. Training Loop
    model_engine.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            # Causal LM: input = batch, label = batch shifted by 1
            input_ids = batch["input_ids"].to(model_engine.local_rank)
            labels = batch["labels"].to(model_engine.local_rank)

            outputs = model_engine(input_ids=input_ids)
            logits = outputs.logits
            
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))

            # DeepSpeed handles scaling, backward, and step natively 
            # if accumulation boundaries are met.
            model_engine.backward(loss)
            model_engine.step()

            if step % 100 == 0 and local_rank == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

    if local_rank == 0:
        model_engine.save_checkpoint("checkpoints/pretrain_final")
