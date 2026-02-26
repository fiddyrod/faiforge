"""
Hybrid search combining BM25 and semantic vector search.

Hybrid search improves retrieval quality by combining the strengths of:
- BM25: Exact term matching, good for specific keywords and rare terms
- Semantic: Conceptual similarity, good for synonyms and paraphrasing
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import FusionMethod, SearchAdapter, ScoredResult
from .bm25 import BM25Config, BM25Retriever

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""
    # Score weights (should sum to 1.0)
    semantic_weight: float = 0.5
    bm25_weight: float = 0.5

    # Fusion method
    fusion_method: FusionMethod = FusionMethod.RRF

    # RRF parameter (only used with FusionMethod.RRF)
    rrf_k: int = 60  # Ranking constant for RRF

    # Retrieval parameters
    initial_k_multiplier: float = 2.0  # Fetch more candidates for re-ranking

    # BM25 config (optional override)
    bm25_config: Optional[BM25Config] = None


class HybridSearcher(SearchAdapter):
    """
    Hybrid search combining BM25 keyword search and semantic vector search.

    Uses configurable score fusion methods to combine results from both
    retrieval strategies, leveraging the strengths of each approach.

    Fusion Methods:
    - WEIGHTED_SUM: Linear combination of normalized scores
    - RRF (Reciprocal Rank Fusion): Rank-based fusion, robust to score scale
    - MAX: Take maximum score from either method
    - MIN: Take minimum (conservative, high precision)

    Example:
        hybrid = HybridSearcher(
            vector_store=chroma_store,
            embedding_adapter=openai_embedding,
            config=HybridSearchConfig(
                semantic_weight=0.6,
                bm25_weight=0.4,
                fusion_method=FusionMethod.RRF
            )
        )

        # Index documents for BM25 (vector store indexing done separately)
        await hybrid.index_documents(documents)

        # Search
        results = await hybrid.search("machine learning frameworks", top_k=10)
    """

    def __init__(
        self,
        vector_store,  # VectorStoreAdapter
        embedding_adapter,  # EmbeddingAdapter
        config: Optional[HybridSearchConfig] = None
    ):
        """
        Initialize hybrid searcher.

        Args:
            vector_store: Vector store adapter for semantic search
            embedding_adapter: Embedding adapter for query embedding
            config: Optional hybrid search configuration
        """
        self.vector_store = vector_store
        self.embedding_adapter = embedding_adapter
        self.config = config or HybridSearchConfig()

        # Initialize BM25 retriever
        self.bm25 = BM25Retriever(
            config=self.config.bm25_config or BM25Config()
        )

        self._indexed = False

    async def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Index documents for BM25 search.

        Note: Documents should already be added to the vector store.
        This method only builds the BM25 index.

        Args:
            documents: List of documents with 'id', 'content', and optional 'metadata'

        Returns:
            BM25 indexing statistics
        """
        start_time = time.time()

        result = await self.bm25.index_documents(documents)
        result["indexing_latency_ms"] = round((time.time() - start_time) * 1000, 2)

        self._indexed = True

        logger.info(
            f"Hybrid search BM25 index built: {result['total_documents']} docs, "
            f"{result['vocabulary_size']} terms, "
            f"{result['indexing_latency_ms']}ms"
        )

        return result

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredResult]:
        """
        Perform hybrid search combining BM25 and semantic results.

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of scored results with combined scores
        """
        start_time = time.time()

        # Fetch more candidates for re-ranking
        fetch_k = int(top_k * self.config.initial_k_multiplier)

        # Run both searches in parallel conceptually
        # (Python async, so sequential but non-blocking)
        bm25_results = await self._search_bm25(query, fetch_k, filters)
        semantic_results = await self._search_semantic(query, fetch_k, filters)

        # Fuse results
        if self.config.fusion_method == FusionMethod.RRF:
            fused = self._fuse_rrf(bm25_results, semantic_results, top_k)
        elif self.config.fusion_method == FusionMethod.WEIGHTED_SUM:
            fused = self._fuse_weighted_sum(bm25_results, semantic_results, top_k)
        elif self.config.fusion_method == FusionMethod.MAX:
            fused = self._fuse_max(bm25_results, semantic_results, top_k)
        elif self.config.fusion_method == FusionMethod.MIN:
            fused = self._fuse_min(bm25_results, semantic_results, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")

        latency_ms = (time.time() - start_time) * 1000

        logger.debug(
            f"Hybrid search: {len(fused)} results, "
            f"BM25={len(bm25_results)}, semantic={len(semantic_results)}, "
            f"fusion={self.config.fusion_method.value}, "
            f"latency={latency_ms:.1f}ms"
        )

        return fused

    async def _search_bm25(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[ScoredResult]:
        """Run BM25 search."""
        if not self._indexed:
            logger.warning("BM25 index not built, returning empty results")
            return []

        return await self.bm25.search(query, top_k, filters)

    async def _search_semantic(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[ScoredResult]:
        """Run semantic vector search."""
        # Embed query
        embedding_result = await self.embedding_adapter.embed_query(query)
        query_embedding = embedding_result.embeddings[0]

        # Search vector store
        search_response = await self.vector_store.search(
            query_embedding=query_embedding,
            limit=top_k,
            filters=filters
        )

        # Convert to ScoredResult
        results = []
        for sr in search_response.results:
            results.append(ScoredResult(
                id=sr.id,
                content=sr.content,
                score=sr.score,
                metadata=sr.metadata,
                semantic_score=sr.score
            ))

        return results

    def _fuse_rrf(
        self,
        bm25_results: List[ScoredResult],
        semantic_results: List[ScoredResult],
        top_k: int
    ) -> List[ScoredResult]:
        """
        Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each ranking list

        This method is robust to different score scales and performs well
        without requiring score normalization.
        """
        k = self.config.rrf_k

        # Build result map and compute RRF scores
        results_map: Dict[str, ScoredResult] = {}
        rrf_scores: Dict[str, float] = {}

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result.id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.config.bm25_weight / (k + rank)
            if doc_id not in results_map:
                results_map[doc_id] = result
            else:
                # Update BM25 score
                results_map[doc_id].bm25_score = result.bm25_score

        # Process semantic results
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result.id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.config.semantic_weight / (k + rank)
            if doc_id not in results_map:
                results_map[doc_id] = result
            else:
                # Update semantic score
                results_map[doc_id].semantic_score = result.semantic_score

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build final results
        fused = []
        for doc_id in sorted_ids[:top_k]:
            result = results_map[doc_id]
            result.score = rrf_scores[doc_id]
            fused.append(result)

        return fused

    def _fuse_weighted_sum(
        self,
        bm25_results: List[ScoredResult],
        semantic_results: List[ScoredResult],
        top_k: int
    ) -> List[ScoredResult]:
        """
        Weighted sum fusion with score normalization.

        Normalizes scores to [0,1] range and combines with weights.
        """
        # Normalize BM25 scores
        bm25_scores = [r.score for r in bm25_results] if bm25_results else [0]
        bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0

        # Normalize semantic scores
        sem_scores = [r.score for r in semantic_results] if semantic_results else [0]
        sem_min, sem_max = min(sem_scores), max(sem_scores)
        sem_range = sem_max - sem_min if sem_max > sem_min else 1.0

        # Build result map
        results_map: Dict[str, ScoredResult] = {}
        normalized_bm25: Dict[str, float] = {}
        normalized_semantic: Dict[str, float] = {}

        for result in bm25_results:
            norm_score = (result.score - bm25_min) / bm25_range
            normalized_bm25[result.id] = norm_score
            results_map[result.id] = result

        for result in semantic_results:
            norm_score = (result.score - sem_min) / sem_range
            normalized_semantic[result.id] = norm_score
            if result.id not in results_map:
                results_map[result.id] = result
            else:
                results_map[result.id].semantic_score = result.semantic_score

        # Compute weighted sum
        combined_scores: Dict[str, float] = {}
        all_ids = set(normalized_bm25.keys()) | set(normalized_semantic.keys())

        for doc_id in all_ids:
            bm25_norm = normalized_bm25.get(doc_id, 0)
            sem_norm = normalized_semantic.get(doc_id, 0)
            combined_scores[doc_id] = (
                self.config.bm25_weight * bm25_norm +
                self.config.semantic_weight * sem_norm
            )

        # Sort and return top_k
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        fused = []
        for doc_id in sorted_ids[:top_k]:
            result = results_map[doc_id]
            result.score = combined_scores[doc_id]
            fused.append(result)

        return fused

    def _fuse_max(
        self,
        bm25_results: List[ScoredResult],
        semantic_results: List[ScoredResult],
        top_k: int
    ) -> List[ScoredResult]:
        """
        Max fusion - take the maximum normalized score.

        Good for recall-oriented retrieval.
        """
        # Normalize scores first
        def normalize_scores(results: List[ScoredResult]) -> Dict[str, float]:
            if not results:
                return {}
            scores = [r.score for r in results]
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s > min_s else 1.0
            return {r.id: (r.score - min_s) / range_s for r in results}

        normalized_bm25 = normalize_scores(bm25_results)
        normalized_semantic = normalize_scores(semantic_results)

        # Build result map
        results_map: Dict[str, ScoredResult] = {}
        for result in bm25_results:
            results_map[result.id] = result
        for result in semantic_results:
            if result.id not in results_map:
                results_map[result.id] = result
            else:
                results_map[result.id].semantic_score = result.semantic_score

        # Compute max score
        max_scores: Dict[str, float] = {}
        all_ids = set(normalized_bm25.keys()) | set(normalized_semantic.keys())

        for doc_id in all_ids:
            bm25_norm = normalized_bm25.get(doc_id, 0)
            sem_norm = normalized_semantic.get(doc_id, 0)
            max_scores[doc_id] = max(bm25_norm, sem_norm)

        # Sort and return
        sorted_ids = sorted(max_scores.keys(), key=lambda x: max_scores[x], reverse=True)

        fused = []
        for doc_id in sorted_ids[:top_k]:
            result = results_map[doc_id]
            result.score = max_scores[doc_id]
            fused.append(result)

        return fused

    def _fuse_min(
        self,
        bm25_results: List[ScoredResult],
        semantic_results: List[ScoredResult],
        top_k: int
    ) -> List[ScoredResult]:
        """
        Min fusion - take the minimum normalized score.

        Good for precision-oriented retrieval (both methods must agree).
        """
        # Normalize scores first
        def normalize_scores(results: List[ScoredResult]) -> Dict[str, float]:
            if not results:
                return {}
            scores = [r.score for r in results]
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s > min_s else 1.0
            return {r.id: (r.score - min_s) / range_s for r in results}

        normalized_bm25 = normalize_scores(bm25_results)
        normalized_semantic = normalize_scores(semantic_results)

        # Only include documents found by BOTH methods
        common_ids = set(normalized_bm25.keys()) & set(normalized_semantic.keys())

        if not common_ids:
            # Fallback to semantic results if no overlap
            return semantic_results[:top_k]

        # Build result map
        results_map: Dict[str, ScoredResult] = {}
        for result in bm25_results:
            if result.id in common_ids:
                results_map[result.id] = result
        for result in semantic_results:
            if result.id in common_ids and result.id in results_map:
                results_map[result.id].semantic_score = result.semantic_score

        # Compute min score
        min_scores: Dict[str, float] = {}
        for doc_id in common_ids:
            bm25_norm = normalized_bm25[doc_id]
            sem_norm = normalized_semantic[doc_id]
            min_scores[doc_id] = min(bm25_norm, sem_norm)

        # Sort and return
        sorted_ids = sorted(min_scores.keys(), key=lambda x: min_scores[x], reverse=True)

        fused = []
        for doc_id in sorted_ids[:top_k]:
            result = results_map[doc_id]
            result.score = min_scores[doc_id]
            fused.append(result)

        return fused

    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid search statistics."""
        return {
            "bm25_indexed": self._indexed,
            "bm25_stats": self.bm25.get_stats(),
            "config": {
                "semantic_weight": self.config.semantic_weight,
                "bm25_weight": self.config.bm25_weight,
                "fusion_method": self.config.fusion_method.value,
                "rrf_k": self.config.rrf_k,
            }
        }
