import json
import time
from dataclasses import dataclass
from typing import Optional

from ..observability import get_logger, log_with_context


@dataclass
class JudgeResult:
    score: float       # 0.0 – 10.0
    reasoning: str
    model: str
    latency_ms: float


class LLMJudge:
    """Generic LLM-as-judge evaluator.

    Uses any LLMAdapter to score a question/response pair.
    Works standalone (not just for RAG) — forks can wire it in
    for any quality evaluation task.

    Outputs JSON via response_format so scores are always parseable.
    Falls back to heuristic score extraction if the model doesn't
    respect the format.

    Usage:
        judge = LLMJudge(adapter=registry.get("gpt-4o-mini"))
        result = await judge.judge(question="...", response="...", context="...")
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an impartial AI quality evaluator. "
        "Given a question and a response (and optionally context that was used to generate it), "
        "rate the quality of the response on a scale of 0 to 10, where:\n"
        "  0-3 = poor (inaccurate, irrelevant, or harmful)\n"
        "  4-6 = acceptable (partially correct or incomplete)\n"
        "  7-9 = good (accurate, relevant, well-explained)\n"
        "  10  = excellent (perfectly accurate, comprehensive, and clear)\n"
        "Return ONLY valid JSON in this exact format: "
        '{"score": <number 0-10>, "reasoning": "<1-2 sentence explanation>"}'
    )

    def __init__(self, adapter, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.adapter = adapter
        self.system_prompt = system_prompt
        self.logger = get_logger()

    async def judge(
        self,
        question: str,
        response: str,
        context: str = "",
    ) -> JudgeResult:
        """Score a response with the LLM judge."""
        user_content = f"Question: {question}\n\nResponse: {response}"
        if context:
            user_content = f"Context: {context}\n\n{user_content}"

        from ..inference.adapters.base import Message, ResponseFormat

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_content),
        ]

        t0 = time.monotonic()
        try:
            completion = await self.adapter.complete(
                messages=messages,
                temperature=0.0,
                max_tokens=256,
                response_format=ResponseFormat(type="json_object"),
            )
            latency_ms = (time.monotonic() - t0) * 1000

            parsed = json.loads(completion.content)
            score = float(parsed.get("score", 5.0))
            score = max(0.0, min(10.0, score))
            reasoning = str(parsed.get("reasoning", ""))

        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: scan for first number in the response
            latency_ms = (time.monotonic() - t0) * 1000
            import re
            nums = re.findall(r"\b([0-9]|10)\b", completion.content)
            score = float(nums[0]) if nums else 5.0
            reasoning = completion.content[:200]

        log_with_context(
            self.logger, "info",
            f"LLM judge scored {score:.1f}/10",
            event="llm_judge_complete",
            score=score,
            model=completion.model,
            latency_ms=round(latency_ms, 1),
        )

        return JudgeResult(
            score=score,
            reasoning=reasoning,
            model=completion.model,
            latency_ms=round(latency_ms, 1),
        )
