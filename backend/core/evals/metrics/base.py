from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalInput:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None  # required for context_recall


@dataclass
class EvalResult:
    metric: str         # "faithfulness", "answer_relevancy", etc.
    score: float        # 0.0 – 1.0 normalized
    reasoning: str
    raw: dict = field(default_factory=dict)  # adapter-specific passthrough


class EvalMetric(ABC):
    """Base class for all evaluation metrics.

    To add a custom metric in a fork:
        1. Subclass EvalMetric
        2. Set class-level `name`
        3. Implement `compute()`
        4. Pass instance to RAGEvalPipeline([..., MyMetric(judge)])
    """

    name: str  # class-level constant used as dict key in pipeline results

    @abstractmethod
    async def compute(self, inp: EvalInput) -> EvalResult:
        """Compute the metric for the given input.

        Should return an EvalResult with score in [0.0, 1.0].
        """
        ...
