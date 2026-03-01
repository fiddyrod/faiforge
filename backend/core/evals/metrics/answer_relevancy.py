import asyncio
import json

from .base import EvalInput, EvalResult, EvalMetric

try:
    from ragas.metrics import answer_relevancy as _ragas_answer_relevancy
    from ragas import evaluate as _ragas_evaluate
    from datasets import Dataset as _Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    _ragas_answer_relevancy = None
    _ragas_evaluate = None
    _Dataset = None
    RAGAS_AVAILABLE = False

_JUDGE_PROMPT = (
    "You are evaluating the answer relevancy of an AI response. "
    "Answer relevancy measures how directly the answer addresses the question. "
    "A high score means the answer is focused and complete; a low score means it is off-topic or evasive.\n\n"
    "Question: {question}\n\nAnswer: {answer}\n\n"
    "Rate answer relevancy 0.0 (completely irrelevant) to 1.0 (perfectly relevant). "
    'Return ONLY valid JSON: {{"score": <0.0-1.0>, "reasoning": "<1-2 sentences>"}}'
)


class AnswerRelevancyMetric(EvalMetric):
    """Answer relevancy: does the answer directly address the question?

    Uses Ragas when available, falls back to LLM-as-judge.
    """

    name = "answer_relevancy"

    def __init__(self, judge=None):
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
            result = _ragas_evaluate(dataset, metrics=[_ragas_answer_relevancy])
            return result

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run)
        score = float(result["answer_relevancy"])
        return EvalResult(
            metric=self.name,
            score=score,
            reasoning="Computed via Ragas answer_relevancy metric.",
            raw={"ragas": True},
        )

    async def _compute_llm_judge(self, inp: EvalInput) -> EvalResult:
        if not self._judge:
            raise RuntimeError("LLMJudge required when Ragas is not installed.")
        prompt = _JUDGE_PROMPT.format(question=inp.question, answer=inp.answer)
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
