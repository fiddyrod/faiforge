import asyncio
from typing import List

from .base import RerankerAdapter, RerankResult
from ...observability import get_logger, log_with_context

# sentence-transformers is optional
try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False


class CrossEncoderReranker(RerankerAdapter):
    """Reranker using a sentence-transformers CrossEncoder model.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - ~66MB download on first use
    - Fast CPU inference (~5-20ms for 10 candidates)
    - Significantly improves RAG answer quality

    Usage:
        reranker = CrossEncoderReranker()
        results = await reranker.rerank(query, search_results, top_k=3)
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 512):
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.max_length = max_length
        self.logger = get_logger()
        self._model: "_CrossEncoder | None" = None

    def _load_model(self) -> "_CrossEncoder":
        """Lazy-load the model on first use."""
        if self._model is None:
            log_with_context(
                self.logger, "info",
                f"Loading cross-encoder model: {self.model_name}",
                event="reranker_load", model=self.model_name
            )
            self._model = _CrossEncoder(self.model_name, max_length=self.max_length)
            log_with_context(
                self.logger, "info",
                "Cross-encoder model loaded",
                event="reranker_load_complete", model=self.model_name
            )
        return self._model

    async def rerank(self, query: str, results: list, top_k: int) -> List[RerankResult]:
        """Rerank results using cross-encoder scoring.

        Runs CPU inference in a thread pool to avoid blocking the event loop.
        """
        if not results:
            return []

        top_k = min(top_k, len(results))

        def _score() -> List[float]:
            model = self._load_model()
            pairs = [[query, r.content] for r in results]
            return model.predict(pairs).tolist()

        loop = asyncio.get_event_loop()
        scores: List[float] = await loop.run_in_executor(None, _score)

        reranked = [
            RerankResult(
                id=r.id,
                content=r.content,
                score=float(scores[i]),
                original_score=r.score,
                metadata=r.metadata,
            )
            for i, r in enumerate(results)
        ]
        reranked.sort(key=lambda x: x.score, reverse=True)

        log_with_context(
            self.logger, "info",
            f"Reranked {len(results)} results → top {top_k}",
            event="reranker_complete",
            model=self.model_name,
            candidates=len(results),
            top_k=top_k,
        )
        return reranked[:top_k]
