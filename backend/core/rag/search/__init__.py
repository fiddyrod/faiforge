"""
Hybrid Search Module

Combines BM25 keyword search with semantic vector search for improved retrieval.
Supports multiple fusion strategies: weighted sum, Reciprocal Rank Fusion (RRF).

Example usage:
    from core.rag.search import HybridSearcher, BM25Retriever, FusionMethod

    # Create hybrid searcher
    hybrid = HybridSearcher(
        vector_store=my_vector_store,
        embedding_adapter=my_embedding_adapter,
        semantic_weight=0.5,
        bm25_weight=0.5,
        fusion_method=FusionMethod.RRF
    )

    # Index documents for BM25
    await hybrid.index_documents(documents)

    # Search with hybrid retrieval
    results = await hybrid.search("machine learning frameworks", top_k=10)
"""

from .base import (
    SearchAdapter,
    ScoredResult,
    FusionMethod,
)

from .bm25 import BM25Retriever

from .hybrid import HybridSearcher, HybridSearchConfig

__all__ = [
    # Base interfaces
    "SearchAdapter",
    "ScoredResult",
    "FusionMethod",
    # Implementations
    "BM25Retriever",
    "HybridSearcher",
    "HybridSearchConfig",
]
