# ──────────────────────────────────────────────────────────────
# Titan Inference Server — Production Dockerfile
# Multi-stage build for optimized image size
# ──────────────────────────────────────────────────────────────

# Stage 1: Builder (install dependencies)
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements.txt \
    fastapi uvicorn[standard] pydantic

COPY . .
RUN pip install --no-cache-dir --prefix=/install -e .

# Stage 2: Runtime (minimal image)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /app /app
WORKDIR /app

# Environment variables
ENV TITAN_MODEL_PATH=/models/checkpoint
ENV TITAN_DEVICE=cuda
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose API port
EXPOSE 8000

# Launch with uvicorn (production workers)
CMD ["python3", "-m", "uvicorn", "titan.serving.api_server:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
