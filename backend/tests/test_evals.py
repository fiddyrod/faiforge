"""Tests for the P2 evals module: LLMJudge, ABRouter, InMemoryEvalStore, RAGEvalPipeline."""
import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from core.evals import (
    LLMJudge,
    JudgeResult,
    InMemoryEvalStore,
    ABRouter,
    Experiment,
    Variant,
    RAGEvalPipeline,
    EvalInput,
    EvalResult,
)
from core.evals.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
)


# =============================================================================
# Fixtures
# =============================================================================

def _mock_adapter(content: str = '{"score": 8.5, "reasoning": "Good answer."}'):
    adapter = Mock()
    adapter.complete = AsyncMock(return_value=Mock(
        content=content,
        model="gpt-4o-mini",
        input_tokens=50,
        output_tokens=30,
        cost_usd=0.0001,
        latency_ms=120.0,
    ))
    return adapter


@pytest.fixture
def eval_store():
    return InMemoryEvalStore()


@pytest.fixture
def mock_judge():
    adapter = _mock_adapter()
    return LLMJudge(adapter=adapter)


@pytest.fixture
def ab_router(eval_store):
    return ABRouter(store=eval_store)


@pytest.fixture
def sample_experiment():
    return Experiment(
        id="test-exp",
        variants=[
            Variant(id="variant-a", system_prompt="You are formal.", weight=0.5),
            Variant(id="variant-b", system_prompt="You are casual.", weight=0.5),
        ],
        routing="round_robin",
    )


# =============================================================================
# InMemoryEvalStore
# =============================================================================

class TestInMemoryEvalStore:
    def test_append_and_retrieve_feedback(self, eval_store):
        eval_store.append_feedback({"message_id": "msg-1", "rating": "thumbs_up"})
        eval_store.append_feedback({"message_id": "msg-2", "rating": "thumbs_down"})

        feedbacks = eval_store.get_feedbacks()
        assert len(feedbacks) == 2
        assert feedbacks[0]["message_id"] == "msg-1"

    def test_feedback_auto_assigns_id_and_timestamp(self, eval_store):
        eval_store.append_feedback({"message_id": "msg-1"})
        fb = eval_store.get_feedbacks()[0]
        assert "id" in fb
        assert "created_at" in fb

    def test_feedback_pagination(self, eval_store):
        for i in range(10):
            eval_store.append_feedback({"message_id": f"msg-{i}"})
        page = eval_store.get_feedbacks(limit=3, offset=5)
        assert len(page) == 3
        assert page[0]["message_id"] == "msg-5"

    def test_total_feedbacks(self, eval_store):
        for i in range(5):
            eval_store.append_feedback({"message_id": f"msg-{i}"})
        assert eval_store.total_feedbacks() == 5

    def test_ab_results(self, eval_store):
        eval_store.append_ab_result({"experiment_id": "exp-1", "variant_id": "a", "latency_ms": 100})
        eval_store.append_ab_result({"experiment_id": "exp-1", "variant_id": "b", "latency_ms": 200})
        eval_store.append_ab_result({"experiment_id": "exp-2", "variant_id": "x", "latency_ms": 150})

        exp1_results = eval_store.get_ab_results("exp-1")
        assert len(exp1_results) == 2
        assert eval_store.get_ab_results("exp-2") == [{"experiment_id": "exp-2", "variant_id": "x", "latency_ms": 150, **{k: v for k, v in eval_store.get_ab_results("exp-2")[0].items() if k == "created_at"}}] or len(eval_store.get_ab_results("exp-2")) == 1


# =============================================================================
# LLMJudge
# =============================================================================

class TestLLMJudge:
    @pytest.mark.asyncio
    async def test_judge_parses_json_response(self, mock_judge):
        result = await mock_judge.judge(
            question="What is Python?",
            response="Python is a programming language.",
        )
        assert isinstance(result, JudgeResult)
        assert result.score == 8.5
        assert result.reasoning == "Good answer."
        assert result.model == "gpt-4o-mini"
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_judge_clamps_score_to_0_10(self):
        adapter = _mock_adapter('{"score": 15.0, "reasoning": "Too high."}')
        judge = LLMJudge(adapter=adapter)
        result = await judge.judge("q", "a")
        assert result.score == 10.0

    @pytest.mark.asyncio
    async def test_judge_fallback_on_non_json(self):
        adapter = _mock_adapter("Score: 7 out of 10. Decent answer.")
        judge = LLMJudge(adapter=adapter)
        result = await judge.judge("q", "a")
        assert result.score == 7.0

    @pytest.mark.asyncio
    async def test_judge_includes_context_in_prompt(self):
        adapter = _mock_adapter('{"score": 9.0, "reasoning": "Uses context well."}')
        judge = LLMJudge(adapter=adapter)
        await judge.judge("q", "a", context="some context")
        call_args = adapter.complete.call_args
        messages = call_args[1]["messages"] if call_args[1] else call_args[0][0]
        user_msg = next(m for m in messages if m.role == "user")
        assert "some context" in user_msg.content


# =============================================================================
# ABRouter
# =============================================================================

class TestABRouter:
    def test_register_experiment(self, ab_router, sample_experiment):
        ab_router.register_experiment(sample_experiment)
        experiments = ab_router.list_experiments()
        assert len(experiments) == 1
        assert experiments[0]["id"] == "test-exp"

    def test_round_robin_routing(self, ab_router, sample_experiment):
        ab_router.register_experiment(sample_experiment)
        selections = [ab_router.select_variant("test-exp").id for _ in range(4)]
        assert selections == ["variant-a", "variant-b", "variant-a", "variant-b"]

    def test_random_routing_returns_valid_variant(self, eval_store):
        router = ABRouter(store=eval_store)
        exp = Experiment(
            id="rand-exp",
            variants=[
                Variant(id="a", system_prompt="A"),
                Variant(id="b", system_prompt="B"),
            ],
            routing="random",
        )
        router.register_experiment(exp)
        for _ in range(20):
            v = router.select_variant("rand-exp")
            assert v.id in ("a", "b")

    def test_select_variant_unknown_experiment(self, ab_router):
        with pytest.raises(KeyError):
            ab_router.select_variant("nonexistent")

    def test_register_experiment_no_variants_raises(self, ab_router):
        with pytest.raises(ValueError):
            ab_router.register_experiment(Experiment(id="empty", variants=[]))

    def test_get_stats_aggregates_correctly(self, ab_router, sample_experiment):
        ab_router.register_experiment(sample_experiment)
        ab_router.record_result("test-exp", "variant-a", {"latency_ms": 100, "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.001})
        ab_router.record_result("test-exp", "variant-a", {"latency_ms": 200, "input_tokens": 20, "output_tokens": 10, "cost_usd": 0.002})
        ab_router.record_result("test-exp", "variant-b", {"latency_ms": 150, "input_tokens": 15, "output_tokens": 8, "cost_usd": 0.0015})

        stats = ab_router.get_stats("test-exp")
        assert stats["total_requests"] == 3
        variant_a_stats = next(v for v in stats["variants"] if v["id"] == "variant-a")
        assert variant_a_stats["requests"] == 2
        assert variant_a_stats["avg_latency_ms"] == 150.0

    def test_get_stats_unknown_experiment(self, ab_router):
        with pytest.raises(KeyError):
            ab_router.get_stats("nonexistent")


# =============================================================================
# RAGEvalPipeline
# =============================================================================

class TestRAGEvalPipeline:
    def _make_pipeline(self, adapter):
        judge = LLMJudge(adapter=adapter)
        return RAGEvalPipeline(metrics=[
            FaithfulnessMetric(judge=judge),
            AnswerRelevancyMetric(judge=judge),
        ])

    @pytest.mark.asyncio
    async def test_pipeline_runs_all_metrics(self):
        adapter = _mock_adapter('{"score": 0.8, "reasoning": "Good."}')
        pipeline = self._make_pipeline(adapter)
        inp = EvalInput(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["AI stands for artificial intelligence."],
        )
        results = await pipeline.run(inp)
        assert "faithfulness" in results
        assert "answer_relevancy" in results
        assert all(0.0 <= r.score <= 1.0 for r in results.values())

    @pytest.mark.asyncio
    async def test_pipeline_respects_metric_names_filter(self):
        adapter = _mock_adapter('{"score": 0.9, "reasoning": "Great."}')
        pipeline = self._make_pipeline(adapter)
        inp = EvalInput(question="q", answer="a", contexts=["c"])
        results = await pipeline.run(inp, metric_names=["faithfulness"])
        assert list(results.keys()) == ["faithfulness"]

    @pytest.mark.asyncio
    async def test_pipeline_handles_metric_failure_gracefully(self):
        adapter = Mock()
        adapter.complete = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        pipeline = self._make_pipeline(adapter)
        inp = EvalInput(question="q", answer="a", contexts=["c"])
        results = await pipeline.run(inp)
        # All metrics should return an error EvalResult, not raise
        for r in results.values():
            assert r.score == 0.0
            assert "failed" in r.reasoning.lower()

    @pytest.mark.asyncio
    async def test_context_recall_requires_ground_truth(self):
        adapter = _mock_adapter('{"score": 0.7, "reasoning": "ok"}')
        judge = LLMJudge(adapter=adapter)
        metric = ContextRecallMetric(judge=judge)
        inp = EvalInput(question="q", answer="a", contexts=["c"])  # no ground_truth
        result = await metric.compute(inp)
        assert result.score == 0.0
        assert "ground_truth" in result.reasoning

    @pytest.mark.asyncio
    async def test_context_recall_with_ground_truth(self):
        adapter = _mock_adapter('{"score": 0.85, "reasoning": "Context covers ground truth."}')
        judge = LLMJudge(adapter=adapter)
        metric = ContextRecallMetric(judge=judge)
        inp = EvalInput(question="q", answer="a", contexts=["c"], ground_truth="gt")
        result = await metric.compute(inp)
        assert result.score == 0.85


# =============================================================================
# API Integration Tests
# =============================================================================

class TestEvalsAPI:
    """Integration tests for /v1/evals/* endpoints via FastAPI TestClient."""

    @pytest.fixture
    def client(self, mock_config):
        from fastapi.testclient import TestClient
        from core.api.server import create_app
        from core.inference.registry import ModelRegistry

        registry = ModelRegistry()
        adapter = Mock()
        adapter.complete = AsyncMock(return_value=Mock(
            content='{"score": 7.5, "reasoning": "Good answer."}',
            model="gpt-4o-mini",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.00001,
            latency_ms=100.0,
            tool_calls=None,
            finish_reason="stop",
        ))
        registry.register("gpt-4o-mini", adapter)

        with patch("core.api.server.load_registry", return_value=registry):
            app = create_app(mock_config, openai_api_key="",
                             routing_config_path="/nonexistent/routing.yaml")
            with TestClient(app) as c:
                yield c

    def test_judge_endpoint(self, client):
        resp = client.post("/v1/evals/judge", json={
            "question": "What is Python?",
            "response": "Python is a programming language.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert "reasoning" in data

    def test_feedback_endpoint_stores_record(self, client):
        resp = client.post("/v1/evals/feedback", json={
            "message_id": "msg-1",
            "question": "What is AI?",
            "response": "AI is great.",
            "rating": "thumbs_up",
            "run_judge": False,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        list_resp = client.get("/v1/evals/feedback")
        assert list_resp.status_code == 200
        assert list_resp.json()["total"] == 1

    def test_rag_eval_endpoint(self, client):
        resp = client.post("/v1/evals/rag", json={
            "question": "What is AI?",
            "answer": "AI is artificial intelligence.",
            "contexts": ["AI stands for artificial intelligence."],
            "metrics": ["faithfulness", "answer_relevancy"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert data["metrics_run"] == ["faithfulness", "answer_relevancy"]

    def test_ab_experiment_create_and_stats(self, client):
        create_resp = client.post("/v1/evals/ab/experiments", json={
            "id": "test-exp",
            "variants": [
                {"id": "a", "system_prompt": "Be formal.", "weight": 0.5},
                {"id": "b", "system_prompt": "Be casual.", "weight": 0.5},
            ],
            "routing": "round_robin",
        })
        assert create_resp.status_code == 200
        assert create_resp.json()["status"] == "created"

        stats_resp = client.get("/v1/evals/ab/experiments/test-exp/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert stats["experiment_id"] == "test-exp"
        assert len(stats["variants"]) == 2

    def test_ab_stats_unknown_experiment(self, client):
        resp = client.get("/v1/evals/ab/experiments/nonexistent/stats")
        assert resp.status_code == 404

    def test_list_experiments(self, client):
        client.post("/v1/evals/ab/experiments", json={
            "id": "exp-1",
            "variants": [
                {"id": "a", "system_prompt": "A"},
                {"id": "b", "system_prompt": "B"},
            ],
        })
        resp = client.get("/v1/evals/ab/experiments")
        assert resp.status_code == 200
        assert len(resp.json()["experiments"]) == 1

    def test_chat_completion_with_ab_experiment(self, client):
        # Register experiment first
        client.post("/v1/evals/ab/experiments", json={
            "id": "chat-exp",
            "variants": [
                {"id": "formal", "system_prompt": "Be formal."},
                {"id": "casual", "system_prompt": "Be casual."},
            ],
            "routing": "round_robin",
        })

        resp = client.post(
            "/v1/chat/completions?experiment_id=chat-exp",
            json={"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4o-mini"}
        )
        assert resp.status_code == 200

    def test_chat_completion_unknown_ab_experiment(self, client):
        resp = client.post(
            "/v1/chat/completions?experiment_id=nonexistent",
            json={"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4o-mini"}
        )
        assert resp.status_code == 404
