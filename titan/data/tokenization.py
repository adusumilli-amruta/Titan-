import os
import json
from typing import Optional

class TitanTokenizer:
    """
    Lightweight tokenizer wrapper that standardizes BPE/SentencePiece 
    interfaces across all three training stages.
    
    Handles:
    - Prompt template injection for SFT and RLHF stages
    - Dynamic sequence packing for pre-training efficiency
    - Special token management (<|im_start|>, <|im_end|>, <|pad|>, <|tool_call|>)
    
    In production, this wraps a HuggingFace AutoTokenizer with custom 
    special tokens for tool-use and chain-of-thought formatting.
    """
    
    # Special token definitions
    SPECIAL_TOKENS = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "tool_call_token": "<|tool_call|>",
        "tool_result_token": "<|tool_result|>",
        "think_token": "<|think|>",
        "end_think_token": "<|/think|>",
    }

    # Prompt templates for different training stages
    PRETRAIN_TEMPLATE = "{text}"
    
    SFT_TEMPLATE = (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n{assistant}<|im_end|>"
    )
    
    TOOL_USE_TEMPLATE = (
        "<|im_start|>system\nYou have access to the following tools:\n{tool_schema}\n"
        "When calling a tool, emit a JSON payload wrapped in <|tool_call|>...<|im_end|> tags.<|im_end|>\n"
        "<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<|think|>{reasoning}<|/think|>\n"
        "<|tool_call|>{tool_call}<|im_end|>\n"
        "<|tool_result|>{tool_result}<|im_end|>\n"
        "<|im_start|>assistant\n{final_answer}<|im_end|>"
    )

    def __init__(self, tokenizer_name_or_path: str = "meta-llama/Llama-2-7b-hf", max_length: int = 8192):
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name_or_path
        self._tokenizer = None  # Lazy load

    def _load_tokenizer(self):
        """Lazily loads the HuggingFace tokenizer to avoid import overhead."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name, 
                    trust_remote_code=True
                )
                # Register special tokens
                special_tokens = list(self.SPECIAL_TOKENS.values())
                self._tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self.SPECIAL_TOKENS["pad_token"]
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer '{self.tokenizer_name}': {e}")
        return self._tokenizer

    @property
    def vocab_size(self):
        return self._load_tokenizer().vocab_size

    def encode_pretrain(self, text: str, return_tensors: str = "pt"):
        """Tokenizes raw text for causal pre-training (no prompt template)."""
        tok = self._load_tokenizer()
        return tok(
            text, 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors=return_tensors
        )

    def encode_sft(self, system: str, user: str, assistant: str, return_tensors: str = "pt"):
        """Tokenizes a (system, user, assistant) conversation turn for SFT."""
        formatted = self.SFT_TEMPLATE.format(system=system, user=user, assistant=assistant)
        tok = self._load_tokenizer()
        encoded = tok(
            formatted,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=return_tensors
        )
        
        # Create labels: mask the prompt tokens (system + user) with -100
        # so the model only learns to predict the assistant response
        labels = encoded["input_ids"].clone()
        prompt_end = formatted.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
        prompt_tokens = tok(formatted[:prompt_end], return_tensors=return_tensors)["input_ids"]
        labels[:, :prompt_tokens.shape[1]] = -100
        encoded["labels"] = labels
        
        return encoded

    def encode_tool_use(self, tool_schema: str, user: str, reasoning: str, 
                        tool_call: str, tool_result: str, final_answer: str,
                        return_tensors: str = "pt"):
        """Tokenizes a full tool-use Chain-of-Thought trajectory for RLHF training."""
        formatted = self.TOOL_USE_TEMPLATE.format(
            tool_schema=tool_schema, user=user, reasoning=reasoning,
            tool_call=tool_call, tool_result=tool_result, final_answer=final_answer
        )
        tok = self._load_tokenizer()
        return tok(
            formatted,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=return_tensors
        )

    def decode(self, token_ids, skip_special_tokens: bool = True):
        """Decodes token IDs back to text."""
        tok = self._load_tokenizer()
        return tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def save_config(self, save_dir: str):
        """Persists the tokenizer configuration and special token maps."""
        os.makedirs(save_dir, exist_ok=True)
        config = {
            "tokenizer_name": self.tokenizer_name,
            "max_length": self.max_length,
            "special_tokens": self.SPECIAL_TOKENS,
        }
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)
