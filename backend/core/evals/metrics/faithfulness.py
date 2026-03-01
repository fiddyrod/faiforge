import asyncio
import json

from .base import EvalInput, EvalResult, EvalMetric

try:
    from ragas.metrics import faithfulness as _ragas_faithfulness
    from ragas import evaluate as _ragas_evaluate
    from datasets import Dataset as _Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    _ragas_faithfulness = None
    _ragas_evaluate = None
    _Dataset = None
    RAGAS_AVAILABLE = False

_JUDGE_PROMPT = (
    "You are evaluating the faithfulness of an AI answer. "
    "Faithfulness measures whether every claim in the answer is supported by the provided context. "
    "An unfaithful answer introduces facts not present in the context.\n\n"
    "Context:\n{contexts}\n\nAnswer: {answer}\n\n"
    "Rate faithfulness 0.0 (completely hallucinated) to 1.0 (fully grounded). "
    'Return ONLY valid JSON: {{"score": <0.0-1.0>, "reasoning": "<1-2 sentences>"}}'
)


class FaithfulnessMetric(EvalMetric):
    """Faithfulness: are all answer claims supported by the retrieved context?

    Uses Ragas when available, falls back to LLM-as-judge.
    """

    name = "faithfulness"

    def __init__(self, judge=None):
        """
        Args:
            judge: LLMJudge instance (required when Ragas is not installed).
        """
        self._judge = judge

    async def compute(self, inp: EvalInput) -> EvalResult:
        if RAGAS_AVAILABLE:
            return await self._compute_ragas(inp)
        return await self._compute_llm_judge(inp)

    async def _compute_ragas(self, inp: EvalInput) -> EvalResult:
        def _run():
            dataset = _Dataset.from_dict({
                "question": [inp.question],
                "answer": [inp.answer],
                "contexts": [inp.contexts],
            })
            result = _ragas_evaluate(dataset, metrics=[_ragas_faithfulness])
            return result

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run)
        score = float(result["faithfulness"])
        return EvalResult(
            metric=self.name,
            score=score,
            reasoning="Computed via Ragas faithfulness metric.",
            raw={"ragas": True},
        )

    async def _compute_llm_judge(self, inp: EvalInput) -> EvalResult:
        if not self._judge:
            raise RuntimeError("LLMJudge required when Ragas is not installed.")
        prompt = _JUDGE_PROMPT.format(
            contexts="\n".join(f"- {c}" for c in inp.contexts),
            answer=inp.answer,
        )
        from core.inference.adapters.base import Message
        result = await self._judge.adapter.complete(
            messages=[Message(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=256,
        )
        try:
            parsed = json.loads(result.content)
            score = float(parsed.get("score", 0.5))
            reasoning = parsed.get("reasoning", "")
        except (json.JSONDecodeError, ValueError):
            score = 0.5
            reasoning = result.content[:200]

        return EvalResult(metric=self.name, score=max(0.0, min(1.0, score)), reasoning=reasoning)
