from setuptools import setup, find_packages

setup(
    name="titan",
    version="0.1.0",
    description="End-to-End Distributed Framework for Transformer Pretraining, Long-Context, and RLHF",
    author="AI Researcher",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "deepspeed>=0.10.0",
        "accelerate>=0.20.0",
        "matplotlib>=3.7.1",
        "datasets",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "serving": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.5.0",
            "httpx>=0.25.0",
        ],
        "cloud": [
            "azure-storage-blob>=12.19.0",
            "azure-identity>=1.15.0",
            "azure-keyvault-secrets>=4.7.0",
        ],
        "monitoring": [
            "psutil>=5.9.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "flake8",
            "black",
            "isort",
            "mypy",
        ],
        "all": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.5.0",
            "httpx>=0.25.0",
            "azure-storage-blob>=12.19.0",
            "azure-identity>=1.15.0",
            "azure-keyvault-secrets>=4.7.0",
            "psutil>=5.9.0",
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
