"""
Prometheus metrics for FAIForge.

Metric families:
  http_requests_total            - HTTP request count [method, path, status_code]
  http_request_duration_seconds  - HTTP latency histogram [method, path]
  llm_requests_total             - LLM inference count [model, status]
  llm_request_duration_seconds   - LLM inference latency [model]
  llm_tokens_total               - Token usage [model, token_type]
  llm_cost_usd_total             - Accumulated cost [model]
  cache_requests_total           - Cache lookup count [result]
  rag_requests_total             - RAG operation count [operation, status]
  rag_documents_ingested_total   - Total documents ingested
  rag_chunks_created_total       - Total chunks created
  models_loaded                  - Gauge: currently loaded models
"""
import time
from typing import Callable

from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CONTENT_TYPE_LATEST,
    generate_latest,
)


# ---------------------------------------------------------------------------
# HTTP metrics
# ---------------------------------------------------------------------------

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)


# ---------------------------------------------------------------------------
# LLM inference metrics
# ---------------------------------------------------------------------------

llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM inference requests",
    ["model", "status"],
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM inference latency in seconds",
    ["model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens processed",
    ["model", "token_type"],  # token_type: prompt / completion
)

llm_cost_usd_total = Counter(
    "llm_cost_usd_total",
    "Total LLM cost in USD",
    ["model"],
)


# ---------------------------------------------------------------------------
# Semantic cache metrics
# ---------------------------------------------------------------------------

cache_requests_total = Counter(
    "cache_requests_total",
    "Total semantic cache lookups",
    ["result"],  # result: hit / miss
)


# ---------------------------------------------------------------------------
# RAG pipeline metrics
# ---------------------------------------------------------------------------

rag_requests_total = Counter(
    "rag_requests_total",
    "Total RAG pipeline operations",
    ["operation", "status"],  # operation: ingest / query, status: success / error
)

rag_documents_ingested_total = Counter(
    "rag_documents_ingested_total",
    "Total documents ingested into the RAG pipeline",
)

rag_chunks_created_total = Counter(
    "rag_chunks_created_total",
    "Total text chunks created during RAG ingestion",
)


# ---------------------------------------------------------------------------
# System / registry metrics
# ---------------------------------------------------------------------------

models_loaded = Gauge(
    "models_loaded",
    "Number of models currently loaded in the registry",
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# Paths excluded from HTTP metrics (noisy, low-signal)
_EXCLUDED_PATHS = frozenset({"/metrics", "/health", "/favicon.ico"})


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Track per-request HTTP metrics (count + latency) for Prometheus."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in _EXCLUDED_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.perf_counter() - start
            http_requests_total.labels(
                method=request.method,
                path=request.url.path,
                status_code=str(status_code),
            ).inc()
            http_request_duration_seconds.labels(
                method=request.method,
                path=request.url.path,
            ).observe(duration)


# ---------------------------------------------------------------------------
# /metrics response helper
# ---------------------------------------------------------------------------

def metrics_response() -> Response:
    """Return a Prometheus text-format metrics response."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
