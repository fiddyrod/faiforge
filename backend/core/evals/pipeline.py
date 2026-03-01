import asyncio
import time
from typing import Dict, List, Optional

from .metrics.base import EvalInput, EvalMetric, EvalResult
from ..observability import get_logger, log_with_context

# Default metric names for convenience
ALL_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


class RAGEvalPipeline:
    """Orchestrates multiple EvalMetric instances against a single EvalInput.

    Runs all metrics concurrently via asyncio.gather.

    Usage:
        pipeline = RAGEvalPipeline(metrics=[
            FaithfulnessMetric(judge=llm_judge),
            AnswerRelevancyMetric(judge=llm_judge),
        ])
        results = await pipeline.run(EvalInput(
            question="...", answer="...", contexts=[...]
        ))
        # results = {"faithfulness": EvalResult(...), "answer_relevancy": EvalResult(...)}

    Fork extension:
        Add any EvalMetric subclass to the metrics list — no other changes needed.
    """

    def __init__(self, metrics: List[EvalMetric]):
        self._metrics: Dict[str, EvalMetric] = {m.name: m for m in metrics}
        self.logger = get_logger()

    async def run(
        self,
        inp: EvalInput,
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, EvalResult]:
        """Run selected (or all) metrics concurrently.

        Args:
            inp: Eval input (question, answer, contexts, optional ground_truth).
            metric_names: Subset of metric names to run. Defaults to all registered metrics.

        Returns:
            Dict mapping metric name → EvalResult.
        """
        targets = metric_names or list(self._metrics.keys())
        selected = {n: self._metrics[n] for n in targets if n in self._metrics}

        t0 = time.monotonic()
        tasks = [metric.compute(inp) for metric in selected.values()]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        results: Dict[str, EvalResult] = {}
        for metric_name, result in zip(selected.keys(), raw_results):
            if isinstance(result, Exception):
                log_with_context(
                    self.logger, "warning",
                    f"Eval metric '{metric_name}' failed: {result}",
                    event="eval_metric_error",
                    metric=metric_name,
                    error=str(result),
                )
                results[metric_name] = EvalResult(
                    metric=metric_name,
                    score=0.0,
                    reasoning=f"Metric failed: {result}",
                )
            else:
                results[metric_name] = result

        log_with_context(
            self.logger, "info",
            f"RAG eval pipeline complete ({len(results)} metrics, {elapsed_ms}ms)",
            event="rag_eval_complete",
            metrics=list(results.keys()),
            latency_ms=elapsed_ms,
        )

        return results
