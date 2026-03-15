import torch
from torch.utils.data import Dataset
import json
import os
from typing import Optional, List, Dict

class PreferenceDataset(Dataset):
    """
    Dataset for RLHF reward model training using pairwise preference data.
    
    Each sample contains a (chosen, rejected) pair where:
    - chosen: the human-preferred response
    - rejected: the human-rejected response
    
    Compatible with standard preference formats:
    - Anthropic HH-RLHF
    - UltraFeedback
    - Custom JSONL with {"prompt", "chosen", "rejected"} fields
    
    The Bradley-Terry reward model learns to assign higher scalar scores
    to chosen responses vs rejected ones.
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048, 
                 split: str = "train"):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []
        self._load_data(data_path, split)
    
    def _load_data(self, data_path: str, split: str):
        """Loads preference pairs from JSONL or HuggingFace datasets."""
        if os.path.isfile(data_path) and data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    self.pairs.append({
                        "prompt": item["prompt"],
                        "chosen": item["chosen"],
                        "rejected": item["rejected"],
                    })
        else:
            # Try loading from HuggingFace
            try:
                from datasets import load_dataset
                ds = load_dataset(data_path, split=split)
                for item in ds:
                    self.pairs.append({
                        "prompt": item.get("prompt", item.get("instruction", "")),
                        "chosen": item.get("chosen", item.get("chosen_response", "")),
                        "rejected": item.get("rejected", item.get("rejected_response", "")),
                    })
            except Exception as e:
                print(f"Warning: Could not load dataset from {data_path}: {e}")

    def _tokenize_pair(self, prompt: str, response: str):
        """Tokenizes a prompt+response and returns input_ids and attention_mask."""
        tok = self.tokenizer._load_tokenizer()
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        encoded = tok(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        chosen = self._tokenize_pair(pair["prompt"], pair["chosen"])
        rejected = self._tokenize_pair(pair["prompt"], pair["rejected"])
        return {
            "chosen": chosen,
            "rejected": rejected,
        }


class PreferenceCollator:
    """
    Custom collator for batching preference pairs.
    Separates chosen and rejected into their own sub-batches
    for efficient parallel forward passes through the reward model.
    """
    
    def __call__(self, batch: List[Dict]) -> Dict:
        chosen_input_ids = torch.stack([item["chosen"]["input_ids"] for item in batch])
        chosen_masks = torch.stack([item["chosen"]["attention_mask"] for item in batch])
        rejected_input_ids = torch.stack([item["rejected"]["input_ids"] for item in batch])
        rejected_masks = torch.stack([item["rejected"]["attention_mask"] for item in batch])
        
        return {
            "chosen": {
                "input_ids": chosen_input_ids,
                "attention_mask": chosen_masks,
            },
            "rejected": {
                "input_ids": rejected_input_ids,
                "attention_mask": rejected_masks,
            },
        }


class ExecutionFeedbackDataset(Dataset):
    """
    Compiler-Driven RLAIF (Reinforcement Learning from AI Feedback) dataset.
    
    Instead of human preferences, this dataset uses code execution results
    as the reward signal. The model generates code, which is compiled and
    tested. Successful executions become "chosen" and failures become "rejected".
    
    This is the core innovation of Project Titan: using a Python compiler
    or unit test suite as the "reward model" for self-play improvement.
    
    Each sample format:
    {
        "problem": "Write a function that returns the nth Fibonacci number",
        "test_cases": ["assert fib(0) == 0", "assert fib(5) == 5"],
        "chosen_solution": "def fib(n): ...",     # Passes all tests
        "rejected_solution": "def fib(n): ...",   # Fails tests
        "chosen_execution_log": "All 2 tests passed",
        "rejected_execution_log": "AssertionError at test 2"
    }
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        if os.path.isfile(data_path):
            with open(data_path, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line.strip()))
    
    def _format_code_prompt(self, problem: str, solution: str, execution_log: str):
        """Formats a code generation trajectory for reward learning."""
        text = (
            f"<|im_start|>system\nYou are a code generation assistant. "
            f"Write correct, efficient Python code.<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n<|think|>Let me solve this step by step.<|/think|>\n"
            f"```python\n{solution}\n```\n"
            f"Execution Result: {execution_log}<|im_end|>"
        )
        tok = self.tokenizer._load_tokenizer()
        encoded = tok(text, max_length=self.max_length, truncation=True,
                      padding="max_length", return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        chosen = self._format_code_prompt(
            sample["problem"], sample["chosen_solution"], 
            sample.get("chosen_execution_log", "All tests passed")
        )
        rejected = self._format_code_prompt(
            sample["problem"], sample["rejected_solution"],
            sample.get("rejected_execution_log", "Execution failed")
        )
        return {"chosen": chosen, "rejected": rejected}
