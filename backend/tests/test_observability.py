"""Tests for observability: JSON logging, request IDs, log_with_context, middleware."""
import json
import logging
import pytest
from unittest.mock import MagicMock, AsyncMock

from core.observability import (
    setup_logging,
    get_logger,
    log_with_context,
    set_request_id,
    get_request_id,
    JSONFormatter,
)


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------

def make_record(msg="Test message", level=logging.INFO, extra_fields=None, exc_info=None):
    record = logging.LogRecord(
        name="faiforge",
        level=level,
        pathname="",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )
    if extra_fields:
        record.extra_fields = extra_fields
    return record


class TestJSONFormatter:
    def test_output_is_valid_json(self):
        formatter = JSONFormatter()
        output = formatter.format(make_record())
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_message_field_present(self):
        formatter = JSONFormatter()
        data = json.loads(formatter.format(make_record("Hello world")))
        assert data["message"] == "Hello world"

    def test_level_field_present(self):
        formatter = JSONFormatter()
        data = json.loads(formatter.format(make_record(level=logging.WARNING)))
        assert data["level"] == "WARNING"

    def test_timestamp_field_present(self):
        formatter = JSONFormatter()
        data = json.loads(formatter.format(make_record()))
        assert "timestamp" in data
        assert data["timestamp"].endswith("Z")

    def test_logger_name_field_present(self):
        formatter = JSONFormatter()
        data = json.loads(formatter.format(make_record()))
        assert data["logger"] == "faiforge"

    def test_extra_fields_merged_into_output(self):
        formatter = JSONFormatter()
        record = make_record(extra_fields={"event": "my_event", "latency_ms": 42})
        data = json.loads(formatter.format(record))
        assert data["event"] == "my_event"
        assert data["latency_ms"] == 42

    def test_no_request_id_when_not_set(self):
        set_request_id(None)
        formatter = JSONFormatter()
        data = json.loads(formatter.format(make_record()))
        assert "request_id" not in data

    def test_request_id_included_when_set(self):
        set_request_id("req-abc-123")
        try:
            formatter = JSONFormatter()
            data = json.loads(formatter.format(make_record()))
            assert data["request_id"] == "req-abc-123"
        finally:
            set_request_id(None)

    def test_exception_info_included_when_present(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc = sys.exc_info()
        record = make_record(level=logging.ERROR, exc_info=exc)
        data = json.loads(formatter.format(record))
        assert "exception" in data
        assert "ValueError" in data["exception"]


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_returns_logger_instance(self):
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)

    def test_logger_name_is_faiforge(self):
        logger = setup_logging()
        assert logger.name == "faiforge"

    def test_sets_info_level(self):
        logger = setup_logging(log_level="INFO")
        assert logger.level == logging.INFO

    def test_sets_debug_level(self):
        logger = setup_logging(log_level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_sets_warning_level(self):
        logger = setup_logging(log_level="WARNING")
        assert logger.level == logging.WARNING

    def test_json_formatter_attached_for_json_format(self):
        logger = setup_logging(log_format="json")
        assert any(isinstance(h.formatter, JSONFormatter) for h in logger.handlers)

    def test_no_json_formatter_for_text_format(self):
        logger = setup_logging(log_format="text")
        assert not any(isinstance(h.formatter, JSONFormatter) for h in logger.handlers)

    def test_does_not_propagate(self):
        logger = setup_logging()
        assert logger.propagate is False


# ---------------------------------------------------------------------------
# Request ID context var
# ---------------------------------------------------------------------------

class TestRequestId:
    def test_set_and_get_round_trip(self):
        set_request_id("test-id-456")
        assert get_request_id() == "test-id-456"

    def test_set_to_none(self):
        set_request_id(None)
        assert get_request_id() is None

    def test_overwrite_existing_id(self):
        set_request_id("first")
        set_request_id("second")
        assert get_request_id() == "second"


# ---------------------------------------------------------------------------
# log_with_context
# ---------------------------------------------------------------------------

class TestLogWithContext:
    def test_extra_fields_attached_to_record(self):
        logger = get_logger()
        captured = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured.append(record)

        handler = CapturingHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            log_with_context(
                logger, "info", "test message",
                event="test_event", model="gpt-4o-mini", latency_ms=99,
            )
            assert len(captured) == 1
            record = captured[0]
            assert record.extra_fields["event"] == "test_event"
            assert record.extra_fields["model"] == "gpt-4o-mini"
            assert record.extra_fields["latency_ms"] == 99
        finally:
            logger.removeHandler(handler)

    def test_warning_level_used_correctly(self):
        logger = get_logger()
        captured = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured.append(record)

        handler = CapturingHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            log_with_context(logger, "warning", "warning message")
            assert captured[0].levelno == logging.WARNING
        finally:
            logger.removeHandler(handler)

    def test_error_level_used_correctly(self):
        logger = get_logger()
        captured = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured.append(record)

        handler = CapturingHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            log_with_context(logger, "error", "something failed", error="details")
            assert captured[0].levelno == logging.ERROR
        finally:
            logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# RequestLoggingMiddleware (unit-level, not full integration)
# ---------------------------------------------------------------------------

class TestRequestLoggingMiddleware:
    @pytest.mark.asyncio
    async def test_adds_x_request_id_header(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from core.observability import RequestLoggingMiddleware

        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/ping")
        def ping():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/ping")
        assert "x-request-id" in response.headers
        # UUID format: 8-4-4-4-12
        req_id = response.headers["x-request-id"]
        assert len(req_id) == 36

    @pytest.mark.asyncio
    async def test_each_request_gets_unique_id(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from core.observability import RequestLoggingMiddleware

        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/ping")
        def ping():
            return {"status": "ok"}

        client = TestClient(app)
        id1 = client.get("/ping").headers["x-request-id"]
        id2 = client.get("/ping").headers["x-request-id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

def _get_sample(sample_name: str, **labels) -> float:
    """Read a sample value from the default Prometheus registry.

    prometheus_client strips '_total' from Counter metric.name internally,
    so we search by sample.name (which keeps the full name as-is).
    """
    from prometheus_client import REGISTRY
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == sample_name:
                if all(sample.labels.get(k) == v for k, v in labels.items()):
                    return sample.value
    return 0.0


class TestPrometheusMetrics:
    """Smoke-tests: metrics objects exist and can be incremented."""

    def test_llm_requests_counter_increments(self):
        from core.observability.metrics import llm_requests_total
        before = _get_sample("llm_requests_total", model="test-model", status="success")
        llm_requests_total.labels(model="test-model", status="success").inc()
        after = _get_sample("llm_requests_total", model="test-model", status="success")
        assert after == before + 1

    def test_cache_hit_counter_increments(self):
        from core.observability.metrics import cache_requests_total
        before = _get_sample("cache_requests_total", result="hit")
        cache_requests_total.labels(result="hit").inc()
        after = _get_sample("cache_requests_total", result="hit")
        assert after == before + 1

    def test_cache_miss_counter_increments(self):
        from core.observability.metrics import cache_requests_total
        before = _get_sample("cache_requests_total", result="miss")
        cache_requests_total.labels(result="miss").inc()
        after = _get_sample("cache_requests_total", result="miss")
        assert after == before + 1

    def test_rag_ingest_counter_increments(self):
        from core.observability.metrics import rag_requests_total
        before = _get_sample("rag_requests_total", operation="ingest", status="success")
        rag_requests_total.labels(operation="ingest", status="success").inc()
        after = _get_sample("rag_requests_total", operation="ingest", status="success")
        assert after == before + 1

    def test_rag_query_counter_increments(self):
        from core.observability.metrics import rag_requests_total
        before = _get_sample("rag_requests_total", operation="query", status="success")
        rag_requests_total.labels(operation="query", status="success").inc()
        after = _get_sample("rag_requests_total", operation="query", status="success")
        assert after == before + 1

    def test_models_loaded_gauge_set(self):
        from core.observability.metrics import models_loaded
        models_loaded.set(4)
        assert _get_sample("models_loaded") == 4.0

    def test_llm_tokens_counter_increments(self):
        from core.observability.metrics import llm_tokens_total
        before = _get_sample("llm_tokens_total", model="gpt-4o-mini", token_type="prompt")
        llm_tokens_total.labels(model="gpt-4o-mini", token_type="prompt").inc(100)
        after = _get_sample("llm_tokens_total", model="gpt-4o-mini", token_type="prompt")
        assert after == before + 100

    def test_llm_cost_counter_increments(self):
        from core.observability.metrics import llm_cost_usd_total
        before = _get_sample("llm_cost_usd_total", model="gpt-4o-mini")
        llm_cost_usd_total.labels(model="gpt-4o-mini").inc(0.001)
        after = _get_sample("llm_cost_usd_total", model="gpt-4o-mini")
        assert abs(after - (before + 0.001)) < 1e-9

    def test_rag_documents_counter_increments(self):
        from core.observability.metrics import rag_documents_ingested_total
        before = _get_sample("rag_documents_ingested_total")
        rag_documents_ingested_total.inc(5)
        after = _get_sample("rag_documents_ingested_total")
        assert after == before + 5

    def test_rag_chunks_counter_increments(self):
        from core.observability.metrics import rag_chunks_created_total
        before = _get_sample("rag_chunks_created_total")
        rag_chunks_created_total.inc(12)
        after = _get_sample("rag_chunks_created_total")
        assert after == before + 12


class TestPrometheusMiddleware:
    """Integration: middleware tracks HTTP metrics and excludes health/metrics paths."""

    def _make_app(self):
        from fastapi import FastAPI
        from core.observability import PrometheusMiddleware

        app = FastAPI()
        app.add_middleware(PrometheusMiddleware)

        @app.get("/ping")
        def ping():
            return {"status": "ok"}

        @app.get("/health")
        def health():
            return {"status": "healthy"}

        return app

    def test_increments_http_requests_total(self):
        from fastapi.testclient import TestClient
        client = TestClient(self._make_app())
        before = _get_sample("http_requests_total", method="GET", path="/ping", status_code="200")
        client.get("/ping")
        after = _get_sample("http_requests_total", method="GET", path="/ping", status_code="200")
        assert after == before + 1

    def test_records_duration_histogram(self):
        from fastapi.testclient import TestClient
        client = TestClient(self._make_app())
        before_count = _get_sample("http_request_duration_seconds_count", method="GET", path="/ping")
        client.get("/ping")
        after_count = _get_sample("http_request_duration_seconds_count", method="GET", path="/ping")
        assert after_count == before_count + 1

    def test_health_path_excluded_from_metrics(self):
        from fastapi.testclient import TestClient
        client = TestClient(self._make_app())
        before = _get_sample("http_requests_total", method="GET", path="/health", status_code="200")
        client.get("/health")
        after = _get_sample("http_requests_total", method="GET", path="/health", status_code="200")
        assert after == before  # no increment

    def test_multiple_requests_accumulate(self):
        from fastapi.testclient import TestClient
        client = TestClient(self._make_app())
        before = _get_sample("http_requests_total", method="GET", path="/ping", status_code="200")
        client.get("/ping")
        client.get("/ping")
        client.get("/ping")
        after = _get_sample("http_requests_total", method="GET", path="/ping", status_code="200")
        assert after == before + 3


class TestMetricsEndpoint:
    """/metrics endpoint returns valid Prometheus text."""

    def _make_app(self):
        from fastapi import FastAPI
        from fastapi.responses import Response
        from core.observability.metrics import metrics_response

        app = FastAPI()

        @app.get("/metrics")
        def prometheus_metrics():
            return metrics_response()

        return app

    def test_returns_200(self):
        from fastapi.testclient import TestClient
        client = TestClient(self._make_app())
        assert client.get("/metrics").status_code == 200

    def test_content_type_is_prometheus(self):
        from fastapi.testclient import TestClient
        client = TestClient(self._make_app())
        ct = client.get("/metrics").headers["content-type"]
        assert "text/plain" in ct

    def test_body_contains_known_metric(self):
        from fastapi.testclient import TestClient
        client = TestClient(self._make_app())
        body = client.get("/metrics").text
        assert "http_requests_total" in body
        assert "llm_requests_total" in body
        assert "cache_requests_total" in body
        assert "models_loaded" in body
