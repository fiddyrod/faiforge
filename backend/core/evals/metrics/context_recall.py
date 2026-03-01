import asyncio
import json

from .base import EvalInput, EvalResult, EvalMetric

try:
    from ragas.metrics import context_recall as _ragas_context_recall
    from ragas import evaluate as _ragas_evaluate
    from datasets import Dataset as _Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    _ragas_context_recall = None
    _ragas_evaluate = None
    _Dataset = None
    RAGAS_AVAILABLE = False

_JUDGE_PROMPT = (
    "You are evaluating context recall for a RAG system. "
    "Context recall measures whether the retrieved context contains all the information "
    "needed to answer the question correctly according to the ground truth answer.\n\n"
    "Question: {question}\n\nGround truth answer: {ground_truth}\n\n"
    "Retrieved context:\n{contexts}\n\n"
    "Rate context recall 0.0 (context missing all relevant info) to 1.0 (context contains all needed info). "
    'Return ONLY valid JSON: {{"score": <0.0-1.0>, "reasoning": "<1-2 sentences>"}}'
)


class ContextRecallMetric(EvalMetric):
    """Context recall: does the retrieved context cover the ground truth answer?

    Requires ground_truth in EvalInput. Uses Ragas when available, falls back to LLM-as-judge.
    """

    name = "context_recall"

    def __init__(self, judge=None):
        self._judge = judge

    async def compute(self, inp: EvalInput) -> EvalResult:
        if not inp.ground_truth:
            return EvalResult(
                metric=self.name,
                score=0.0,
                reasoning="context_recall requires ground_truth in EvalInput.",
            )
        if RAGAS_AVAILABLE:
            return await self._compute_ragas(inp)
        return await self._compute_llm_judge(inp)

    async def _compute_ragas(self, inp: EvalInput) -> EvalResult:
        def _run():
            dataset = _Dataset.from_dict({
                "question": [inp.question],
                "answer": [inp.answer],
                "contexts": [inp.contexts],
                "ground_truth": [inp.ground_truth],
            })
            result = _ragas_evaluate(dataset, metrics=[_ragas_context_recall])
            return result

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run)
        score = float(result["context_recall"])
        return EvalResult(
            metric=self.name,
            score=score,
            reasoning="Computed via Ragas context_recall metric.",
            raw={"ragas": True},
        )

    async def _compute_llm_judge(self, inp: EvalInput) -> EvalResult:
        if not self._judge:
            raise RuntimeError("LLMJudge required when Ragas is not installed.")
        prompt = _JUDGE_PROMPT.format(
            question=inp.question,
            ground_truth=inp.ground_truth,
            contexts="\n".join(f"[{i+1}] {c}" for i, c in enumerate(inp.contexts)),
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
