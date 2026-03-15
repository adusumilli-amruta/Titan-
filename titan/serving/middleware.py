import time
import logging
import uuid
from functools import wraps
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with timing, status codes, and request metadata.
    Vital for production monitoring, debugging, and audit trails.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.time()

        # Attach request ID for tracing
        request.state.request_id = request_id

        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"| client={request.client.host if request.client else 'unknown'}"
        )

        try:
            response = await call_next(request)
            elapsed_ms = (time.time() - start) * 1000

            logger.info(
                f"[{request_id}] {request.method} {request.url.path} "
                f"→ {response.status_code} | {elapsed_ms:.1f}ms"
            )

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
            return response

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"→ ERROR | {elapsed_ms:.1f}ms | {str(e)}"
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token-bucket rate limiter per client IP.
    Prevents abuse and ensures fair resource allocation across tenants.
    """

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.rpm = requests_per_minute
        self._buckets: dict = defaultdict(lambda: {"tokens": requests_per_minute, "last_refill": time.time()})

    def _refill(self, bucket: dict):
        now = time.time()
        elapsed = now - bucket["last_refill"]
        refill_amount = elapsed * (self.rpm / 60.0)
        bucket["tokens"] = min(self.rpm, bucket["tokens"] + refill_amount)
        bucket["last_refill"] = now

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        bucket = self._buckets[client_ip]
        self._refill(bucket)

        if bucket["tokens"] < 1:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return Response(
                content='{"error": "Rate limit exceeded. Try again later."}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

        bucket["tokens"] -= 1
        return await call_next(request)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Simple API key authentication middleware.
    In production, this integrates with Azure Active Directory (AAD)
    or Azure Key Vault for secrets management.
    """

    def __init__(self, app, api_keys: list = None, exempt_paths: list = None):
        super().__init__(app)
        self.api_keys = set(api_keys or [])
        self.exempt_paths = set(exempt_paths or ["/health", "/docs", "/openapi.json", "/redoc"])
        self.enabled = len(self.api_keys) > 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        if request.url.path in self.exempt_paths:
            return await call_next(request)

        api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
        if api_key not in self.api_keys:
            return Response(
                content='{"error": "Invalid or missing API key."}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global exception handler that catches unhandled errors and returns
    structured JSON error responses instead of HTML stack traces.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except HTTPException:
            raise  # Let FastAPI handle known HTTP exceptions
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during inference")
            return Response(
                content='{"error": "GPU out of memory. Try a shorter prompt or lower max_tokens."}',
                status_code=507,
                media_type="application/json",
            )
        except Exception as e:
            logger.exception(f"Unhandled server error: {e}")
            return Response(
                content=f'{{"error": "Internal server error: {str(e)}"}}',
                status_code=500,
                media_type="application/json",
            )


# Missing import at at the top level for torch in ErrorHandlingMiddleware
import torch
