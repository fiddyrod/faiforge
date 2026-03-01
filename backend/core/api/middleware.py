"""
API Middleware

Provides:
- API key authentication (Bearer token)
- Per-key rate limiting (sliding window, in-memory)
"""

import os
import time
from collections import defaultdict, deque
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Bearer token authentication middleware.

    Configure by setting FAIFORGE_API_KEYS env var (comma-separated keys).
    If the env var is not set (e.g. development mode), all requests pass through.

    Exempted paths: /health, /docs, /openapi.json, /metrics, /redoc
    """

    EXEMPT_PREFIXES = ("/health", "/docs", "/openapi.json", "/metrics", "/redoc")

    def __init__(self, app, api_keys: Optional[list[str]] = None):
        super().__init__(app)
        # Build a set for O(1) lookups; empty set = auth disabled
        raw = api_keys or _load_keys_from_env()
        self._keys: set[str] = set(k for k in raw if k)
        self._enabled = bool(self._keys)

    async def dispatch(self, request: Request, call_next):
        if not self._enabled:
            return await call_next(request)

        # Skip auth for exempt paths
        for prefix in self.EXEMPT_PREFIXES:
            if request.url.path.startswith(prefix):
                return await call_next(request)

        # Extract Bearer token
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Missing or invalid Authorization header. Expected: Bearer <key>"},
            )

        token = auth[len("Bearer "):]
        if token not in self._keys:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API key"},
            )

        # Attach key to request state for rate limiter
        request.state.api_key = token
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-key sliding window rate limiter (in-memory).

    Defaults: 60 requests / 60 seconds.
    Configurable via FAIFORGE_RATE_LIMIT_REQUESTS and FAIFORGE_RATE_LIMIT_WINDOW env vars.

    Returns 429 with Retry-After header when the limit is exceeded.
    If APIKeyMiddleware is not enabled (auth disabled), rate limiting is skipped.
    """

    EXEMPT_PREFIXES = ("/health", "/docs", "/openapi.json", "/metrics", "/redoc")

    def __init__(self, app, requests_per_window: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self._limit = int(os.getenv("FAIFORGE_RATE_LIMIT_REQUESTS", str(requests_per_window)))
        self._window = int(os.getenv("FAIFORGE_RATE_LIMIT_WINDOW", str(window_seconds)))
        # key -> deque of timestamps (monotonic)
        self._windows: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        # Skip if no api_key attached (auth disabled or exempt path)
        api_key: Optional[str] = getattr(request.state, "api_key", None)
        if api_key is None:
            return await call_next(request)

        for prefix in self.EXEMPT_PREFIXES:
            if request.url.path.startswith(prefix):
                return await call_next(request)

        now = time.monotonic()
        window = self._windows[api_key]

        # Evict timestamps older than the window
        cutoff = now - self._window
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self._limit:
            retry_after = int(self._window - (now - window[0])) + 1
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={"error": f"Rate limit exceeded. Retry after {retry_after}s."},
            )

        window.append(now)
        return await call_next(request)


def _load_keys_from_env() -> list[str]:
    """Load API keys from FAIFORGE_API_KEYS env var (comma-separated)."""
    raw = os.getenv("FAIFORGE_API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]
