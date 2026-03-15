import time
import uuid
import logging
from typing import Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from titan.serving.inference_engine import InferenceEngine, GenerationRequest

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Request / Response Schemas (OpenAI-compatible API format)
# ──────────────────────────────────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    """OpenAI-compatible /v1/completions request schema."""
    model: str = "titan-7b"
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    stream: bool = False
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stop: Optional[list] = None


class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible /v1/chat/completions request schema."""
    model: str = "titan-7b"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: dict


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: dict


# ──────────────────────────────────────────────────────────────────────────────
# Application Factory
# ──────────────────────────────────────────────────────────────────────────────

# Global engine instance (initialized at startup)
engine: Optional[InferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook: loads the model on startup, cleans up on shutdown."""
    global engine
    import os
    model_path = os.environ.get("TITAN_MODEL_PATH", "checkpoints/pretrain/final")
    device = os.environ.get("TITAN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    engine = InferenceEngine(model_path=model_path, device=device)
    engine.load_model()
    logger.info("Titan API Server ready.")
    yield
    logger.info("Shutting down Titan API Server.")


def create_app() -> FastAPI:
    """Factory function for the FastAPI application."""
    app = FastAPI(
        title="Titan Inference API",
        description=(
            "Production-grade REST API for the Titan LLM. "
            "Provides OpenAI-compatible endpoints for text completion "
            "and chat, with streaming support."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health & Status Endpoints ─────────────────────────────────────────

    @app.get("/health", tags=["System"])
    async def health_check():
        """Returns server health status and readiness for inference."""
        return {
            "status": "healthy" if engine and engine.is_loaded else "loading",
            "timestamp": int(time.time()),
        }

    @app.get("/v1/models", tags=["System"])
    async def list_models():
        """Lists available models (OpenAI-compatible)."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "titan-7b",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "titan",
                }
            ],
        }

    @app.get("/v1/stats", tags=["Monitoring"])
    async def get_stats():
        """Returns runtime statistics: GPU memory, request counts, throughput."""
        if not engine:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        return engine.get_stats()

    # ── Completion Endpoint ───────────────────────────────────────────────

    @app.post("/v1/completions", response_model=CompletionResponse, tags=["Inference"])
    async def create_completion(req: CompletionRequest):
        """
        Generate text completions for a given prompt.
        Compatible with OpenAI's /v1/completions API format.
        """
        if not engine or not engine.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        request_id = f"cmpl-{uuid.uuid4().hex[:12]}"

        # Tokenize prompt
        if engine.tokenizer:
            input_ids = engine.tokenizer.encode(req.prompt, return_tensors="pt").squeeze(0)
        else:
            raise HTTPException(status_code=500, detail="Tokenizer not available")

        gen_request = GenerationRequest(
            request_id=request_id,
            input_ids=input_ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            do_sample=req.temperature > 0,
            repetition_penalty=req.repetition_penalty,
            stream=req.stream,
        )

        # Streaming response
        if req.stream:
            async def event_stream():
                async for chunk in engine.generate_stream(gen_request):
                    import json
                    data = {
                        "id": request_id,
                        "object": "text_completion",
                        "choices": [{"index": 0, "text": chunk["token"], "finish_reason": chunk["finish_reason"]}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # Standard response
        response = engine.generate(gen_request)

        return CompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=req.model,
            choices=[CompletionChoice(text=response.generated_text, finish_reason=response.finish_reason)],
            usage={
                "prompt_tokens": input_ids.shape[0],
                "completion_tokens": response.generated_tokens,
                "total_tokens": input_ids.shape[0] + response.generated_tokens,
            },
        )

    # ── Chat Completion Endpoint ──────────────────────────────────────────

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["Inference"])
    async def create_chat_completion(req: ChatCompletionRequest):
        """
        Generate chat completions from conversation messages.
        Compatible with OpenAI's /v1/chat/completions API format.
        """
        if not engine or not engine.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Format conversation into prompt
        prompt_parts = []
        for msg in req.messages:
            prompt_parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        full_prompt = "\n".join(prompt_parts)

        if engine.tokenizer:
            input_ids = engine.tokenizer.encode(full_prompt, return_tensors="pt").squeeze(0)
        else:
            raise HTTPException(status_code=500, detail="Tokenizer not available")

        gen_request = GenerationRequest(
            request_id=request_id,
            input_ids=input_ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.temperature > 0,
        )

        response = engine.generate(gen_request)

        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=response.generated_text),
                    finish_reason=response.finish_reason,
                )
            ],
            usage={
                "prompt_tokens": input_ids.shape[0],
                "completion_tokens": response.generated_tokens,
                "total_tokens": input_ids.shape[0] + response.generated_tokens,
            },
        )

    return app


# Entry point: uvicorn titan.serving.api_server:app
app = create_app()
