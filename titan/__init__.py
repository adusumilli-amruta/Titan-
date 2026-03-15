"""
Titan: End-to-End Distributed Framework for Transformer
Pre-training, Long-Context Scaling, and Compiler-Driven RLHF Alignment.

Modules:
    models      - Custom Decoder-Only Transformer with SWA, RoPE, and Recurrent Memory
    distributed - DeepSpeed ZeRO-3, activation checkpointing, microbatching
    training    - Pre-training, Reward Modeling, PPO alignment loops
    eval        - Automated reasoning benchmarks and report generation
    data        - Tokenization, streaming datasets, preference data utilities
    serving     - FastAPI REST API, inference engine, production middleware
    cloud       - Azure Blob Storage, Azure ML tracking, Key Vault config loader
    monitoring  - SQLite experiment DB, Prometheus metrics, health dashboard
"""

__version__ = "0.1.0"
__author__ = "AI Researcher"

