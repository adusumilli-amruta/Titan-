import torch
from torch.utils.data import IterableDataset, Dataset
import os
import json
import random

class StreamingTextDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for massive pre-training corpora.
    
    Instead of loading the entire dataset into RAM, this reads from sharded 
    JSONL files on disk (or S3) and yields tokenized chunks on-the-fly.
    Crucial for multi-terabyte datasets like The Pile, RedPajama, or SlimPajama.
    
    Supports:
    - Sharded file reading with worker distribution
    - Sequence packing (concatenate short documents to fill context windows)
    - Dynamic padding/truncation to max_length
    """
    
    def __init__(self, data_dir: str, tokenizer, max_length: int = 4096, 
                 shuffle_shards: bool = True, seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        
        # Discover all shard files
        self.shard_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith(('.jsonl', '.json', '.txt'))
        ]) if os.path.isdir(data_dir) else [data_dir]

    def _get_worker_shards(self):
        """Distributes shards across DataLoader workers for parallel I/O."""
        worker_info = torch.utils.data.get_worker_info()
        shards = list(self.shard_files)
        
        if self.shuffle_shards:
            rng = random.Random(self.seed)
            rng.shuffle(shards)
        
        if worker_info is not None:
            # Split shards across workers
            per_worker = len(shards) // worker_info.num_workers
            worker_id = worker_info.id
            shards = shards[worker_id * per_worker : (worker_id + 1) * per_worker]
        
        return shards

    def _read_documents(self, filepath):
        """Reads documents from a single shard file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield data.get("text", data.get("content", ""))
                except json.JSONDecodeError:
                    # Plain text file fallback
                    yield line

    def _pack_sequences(self, documents):
        """
        Sequence packing: concatenates short documents together to fill 
        the full context window, maximizing GPU utilization.
        
        Instead of padding a 200-token document to 4096 tokens (wasting 95% compute),
        we pack multiple documents into a single training example.
        """
        buffer_ids = []
        
        for doc in documents:
            tok = self.tokenizer._load_tokenizer()
            doc_ids = tok.encode(doc, add_special_tokens=False)
            buffer_ids.extend(doc_ids)
            buffer_ids.append(tok.eos_token_id or 2)  # Document boundary
            
            # Once we have enough tokens, yield packed sequences
            while len(buffer_ids) >= self.max_length:
                chunk = buffer_ids[:self.max_length]
                buffer_ids = buffer_ids[self.max_length:]
                
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                }

    def __iter__(self):
        shards = self._get_worker_shards()
        
        def doc_generator():
            for shard in shards:
                yield from self._read_documents(shard)
        
        yield from self._pack_sequences(doc_generator())


class ChunkedDocumentDataset(Dataset):
    """
    For mid-training long-context scaling (Stage 2).
    
    Takes very long documents (books, codebases, research papers) and splits them
    into overlapping chunks that the Recurrent Memory mechanism can process
    sequentially. Each chunk carries metadata about its position in the document
    to support the recurrent KV-cache handoff.
    
    chunk[0] → process → cache KV states
    chunk[1] → process with cached KV → update cache  
    chunk[2] → process with cached KV → update cache
    ...
    
    This simulates processing an arbitrarily long document through an 8K window.
    """
    
    def __init__(self, documents, tokenizer, chunk_size: int = 8192, 
                 overlap: int = 512, max_chunks_per_doc: int = 16):
        super().__init__()
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_chunks_per_doc = max_chunks_per_doc
        
        # Pre-process all documents into chunk metadata
        self.chunks = []
        self._build_chunks(documents)

    def _build_chunks(self, documents):
        """Splits documents into overlapping windows with position metadata."""
        for doc_idx, doc in enumerate(documents):
            tok = self.tokenizer._load_tokenizer()
            token_ids = tok.encode(doc, add_special_tokens=False)
            
            if len(token_ids) < self.chunk_size:
                # Short document: pad to chunk_size
                padded = token_ids + [tok.pad_token_id or 0] * (self.chunk_size - len(token_ids))
                self.chunks.append({
                    "input_ids": padded[:self.chunk_size],
                    "doc_id": doc_idx,
                    "chunk_idx": 0,
                    "total_chunks": 1,
                    "is_continuation": False,
                })
                continue

            # Long document: create overlapping chunks
            stride = self.chunk_size - self.overlap
            num_chunks = min(
                (len(token_ids) - self.overlap) // stride + 1,
                self.max_chunks_per_doc
            )
            
            for chunk_idx in range(num_chunks):
                start = chunk_idx * stride
                end = start + self.chunk_size
                chunk_ids = token_ids[start:end]
                
                # Pad last chunk if needed
                if len(chunk_ids) < self.chunk_size:
                    chunk_ids = chunk_ids + [tok.pad_token_id or 0] * (self.chunk_size - len(chunk_ids))
                
                self.chunks.append({
                    "input_ids": chunk_ids[:self.chunk_size],
                    "doc_id": doc_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": num_chunks,
                    "is_continuation": chunk_idx > 0,
                })

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return {
            "input_ids": torch.tensor(chunk["input_ids"], dtype=torch.long),
            "labels": torch.tensor(chunk["input_ids"], dtype=torch.long),
            "attention_mask": torch.ones(self.chunk_size, dtype=torch.long),
            "doc_id": chunk["doc_id"],
            "chunk_idx": chunk["chunk_idx"],
            "total_chunks": chunk["total_chunks"],
            "is_continuation": chunk["is_continuation"],
        }
