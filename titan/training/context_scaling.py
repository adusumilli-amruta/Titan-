import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from titan.models.modeling_titan import TitanForCausalLM, TitanConfig
from titan.models.memory import RecurrentMemoryState
from titan.distributed.memory_utils import selective_activation_checkpointing

class ContextScalingTrainer:
    """
    Stage 2 Trainer: Mid-Training Long-Context Scaling.
    
    This trainer takes a pre-trained checkpoint and extends its effective
    context window by:
    
    1. Scaling the RoPE base frequency (NTK-aware / YaRN scaling)
    2. Expanding the sliding window size
    3. Training on long documents using the Recurrent Memory mechanism
    
    The recurrent memory allows processing arbitrarily long documents 
    through fixed-size (8K) windows by caching and passing KV states 
    between chunks, similar to Transformer-XL.
    
    Key insight: Instead of training on 32K tokens at once (OOM),
    process 4 chunks of 8K tokens sequentially, passing hidden states
    between chunks via the RecurrentMemoryState.
    """
    
    def __init__(self, model, config, memory_state, optimizer, scheduler, 
                 chunk_size=8192, device="cuda"):
        self.model = model
        self.config = config
        self.memory_state = memory_state
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.chunk_size = chunk_size
        self.device = device
        
    @classmethod
    def from_pretrained(cls, checkpoint_path, new_config_overrides=None, 
                        learning_rate=2e-5, device="cuda"):
        """
        Loads a pre-trained checkpoint and modifies the architecture 
        for long-context scaling.
        """
        # Load base config and model
        config = TitanConfig.from_pretrained(checkpoint_path)
        
        # Apply scaling overrides
        if new_config_overrides:
            for key, value in new_config_overrides.items():
                setattr(config, key, value)
        
        model = TitanForCausalLM.from_pretrained(checkpoint_path, config=config)
        
        # Modify the RoPE embeddings for extended context
        new_scaling = config.max_position_embeddings / 8192  # Original max
        model.rotary_emb.scaling_factor = new_scaling
        model.rotary_emb._set_cos_sin_cache(
            seq_len=config.max_position_embeddings,
            device=model.device,
            dtype=torch.float32
        )
        
        # Initialize recurrent memory
        memory_state = RecurrentMemoryState(config)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, 
                                                      num_training_steps=10000)
        
        return cls(model, config, memory_state, optimizer, scheduler, 
                   chunk_size=config.sliding_window_size, device=device)

    def train_on_chunked_document(self, document_chunks, doc_id):
        """
        Processes a single long document split into sequential chunks.
        
        The Recurrent Memory mechanism caches KV states from each chunk
        and passes them to the next chunk, creating an effective context
        window that spans the entire document while only using chunk_size 
        memory at any point.
        
        Args:
            document_chunks: List of dicts with 'input_ids', 'labels', etc.
            doc_id: Unique identifier for this document's memory state
        """
        self.memory_state.init_state(doc_id, self.device, torch.float32)
        total_loss = 0.0
        
        for chunk_idx, chunk in enumerate(document_chunks):
            input_ids = chunk["input_ids"].unsqueeze(0).to(self.device)
            labels = chunk["labels"].unsqueeze(0).to(self.device)
            attention_mask = chunk["attention_mask"].unsqueeze(0).to(self.device)
            
            # Retrieve cached KV states from previous chunk
            past_key_values = self.memory_state.get_state(doc_id)
            
            # Convert list of layer states to tuple format expected by model
            if past_key_values is not None and past_key_values[0] is not None:
                past_key_values = tuple(past_key_values)
            else:
                past_key_values = None
            
            # Forward pass with memory
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,   # Must cache for memory handoff
            )
            
            logits = outputs.logits
            past_key_values_out = outputs.past_key_values
            
            # Compute causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
            # Backward only on this chunk (truncated BPTT)
            loss.backward()
            
            # Update memory state for next chunk
            if past_key_values_out is not None:
                for layer_idx, (k, v) in enumerate(past_key_values_out):
                    self.memory_state.update_state(doc_id, layer_idx, k, v)
            
            total_loss += loss.item()
            
            # Step optimizer every chunk (or accumulate across chunks)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        # Clean up memory for this document
        self.memory_state.clear_state(doc_id)
        
        avg_loss = total_loss / len(document_chunks)
        return avg_loss

    def train_epoch(self, dataloader, epoch=0):
        """
        Runs one epoch of context scaling training.
        Documents are processed chunk-by-chunk with recurrent memory handoffs.
        """
        self.model.train()
        epoch_loss = 0.0
        num_docs = 0
        
        # Group chunks by doc_id for sequential processing
        current_doc_id = None
        current_doc_chunks = []
        
        for batch in dataloader:
            doc_id = batch.get("doc_id", 0)
            
            # If we hit a new document, process the previous one
            if isinstance(doc_id, torch.Tensor):
                doc_id = doc_id.item()
            
            if doc_id != current_doc_id and current_doc_chunks:
                loss = self.train_on_chunked_document(current_doc_chunks, current_doc_id)
                epoch_loss += loss
                num_docs += 1
                
                if num_docs % 10 == 0:
                    print(f"  Epoch {epoch} | Doc {num_docs} | Avg Loss: {epoch_loss/num_docs:.4f}")
                
                current_doc_chunks = []
            
            current_doc_id = doc_id
            current_doc_chunks.append(batch)
        
        # Process the last document
        if current_doc_chunks:
            loss = self.train_on_chunked_document(current_doc_chunks, current_doc_id)
            epoch_loss += loss
            num_docs += 1
        
        avg_epoch_loss = epoch_loss / max(num_docs, 1)
        print(f"Epoch {epoch} Complete | Avg Loss: {avg_epoch_loss:.4f} | Documents: {num_docs}")
        return avg_epoch_loss
