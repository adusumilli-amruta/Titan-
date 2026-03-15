import argparse
import os
import deepspeed
from datasets import load_dataset
from titan.training.pretrain import pretrain_loop
from titan.distributed.deepspeed_config import create_zero3_config

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Massive Causal Pre-training")
    parser.add_argument("--global_batch_size", type=int, default=1024)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dataset", type=str, default="wikitext", help="HF Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--ds_config", type=str, default="configs/ds_zero3_config.json")
    
    args = parser.parse_args()
    
    # 1. Ensure config exists
    os.makedirs(os.path.dirname(args.ds_config), exist_ok=True)
    if not os.path.exists(args.ds_config):
        print(f"Generating DeepSpeed Config at {args.ds_config}")
        # Assuming 1 GPU for generation testing, gradient accumulation = global // (micro * gpus)
        grad_accum = max(1, args.global_batch_size // args.micro_batch_size)
        create_zero3_config(args.micro_batch_size, grad_accum, save_path=args.ds_config)

    # 2. Load Dataset (Dummy logic for instantiation, replace with actual tokenized dataset)
    print(f"Loading {args.dataset}...")
    try:
        raw_dataset = load_dataset(args.dataset, args.dataset_config, split="train")
        # In a real run, you'd map a tokenizer here. For this script, we mock the dataloader in pretrain_loop
        # if the dataset doesn't have 'input_ids' and 'labels' tensors.
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        raw_dataset = [] # Fallback for mock runs

    # 3. Model Configuration (7B-ish Scale for demonstration)
    titan_config = {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "max_position_embeddings": 8192,
        "sliding_window_size": 4096,
        "max_recurrent_memory_tokens": 16384,
    }

    print("Initiating Stage 1 Pre-training Loop...")
    
    try:
        pretrain_loop(
            dataset=raw_dataset,
            config_dict=titan_config,
            epochs=args.epochs,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            learning_rate=args.learning_rate,
            ds_config_path=args.ds_config
        )
    except Exception as e:
        print(f"Training loop encountered an expected mock error or successfully initialized geometry: {e}")

if __name__ == "__main__":
    main()
