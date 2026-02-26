import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
import uuid

from .embeddings import EmbeddingAdapter
from .chunking import ChunkingAdapter, Chunk
from .vector_stores import VectorStoreAdapter, Document, SearchResponse, SearchResult
from .search import HybridSearcher, HybridSearchConfig, FusionMethod, BM25Retriever
from ..observability import get_logger, log_with_context


class SearchMode(str, Enum):
    """Search mode for RAG queries."""
    SEMANTIC = "semantic"  # Pure vector similarity search
    BM25 = "bm25"  # Pure keyword search
    HYBRID = "hybrid"  # Combined BM25 + semantic


@dataclass
class HybridConfig:
    """Configuration for hybrid search in RAG pipeline."""
    enabled: bool = False
    semantic_weight: float = 0.5
    bm25_weight: float = 0.5
    fusion_method: str = "rrf"  # "rrf", "weighted_sum", "max", "min"
    rrf_k: int = 60


class RAGPipeline:
    """Orchestrates end-to-end RAG operations with optional hybrid search."""

    def __init__(
        self,
        embedding_adapter: EmbeddingAdapter,
        chunking_adapter: ChunkingAdapter,
        vector_store_adapter: VectorStoreAdapter,
        hybrid_config: Optional[HybridConfig] = None
    ):
        """
        Initialize RAG pipeline.

        Args:
            embedding_adapter: Adapter for generating embeddings
            chunking_adapter: Adapter for chunking documents
            vector_store_adapter: Adapter for vector storage/retrieval
            hybrid_config: Optional hybrid search configuration
        """
        self.embedding_adapter = embedding_adapter
        self.chunking_adapter = chunking_adapter
        self.vector_store = vector_store_adapter
        self.logger = get_logger()

        # Hybrid search setup
        self.hybrid_config = hybrid_config or HybridConfig()
        self._hybrid_searcher: Optional[HybridSearcher] = None
        self._bm25_retriever: Optional[BM25Retriever] = None

        if self.hybrid_config.enabled:
            self._init_hybrid_search()

    def _init_hybrid_search(self) -> None:
        """Initialize hybrid search components."""
        # Map fusion method string to enum
        fusion_map = {
            "rrf": FusionMethod.RRF,
            "weighted_sum": FusionMethod.WEIGHTED_SUM,
            "max": FusionMethod.MAX,
            "min": FusionMethod.MIN,
        }
        fusion_method = fusion_map.get(
            self.hybrid_config.fusion_method.lower(),
            FusionMethod.RRF
        )

        hybrid_search_config = HybridSearchConfig(
            semantic_weight=self.hybrid_config.semantic_weight,
            bm25_weight=self.hybrid_config.bm25_weight,
            fusion_method=fusion_method,
            rrf_k=self.hybrid_config.rrf_k,
        )

        self._hybrid_searcher = HybridSearcher(
            vector_store=self.vector_store,
            embedding_adapter=self.embedding_adapter,
            config=hybrid_search_config
        )

        # Also keep a standalone BM25 for BM25-only mode
        self._bm25_retriever = self._hybrid_searcher.bm25

        log_with_context(
            self.logger,
            "info",
            "Hybrid search initialized",
            event="rag_hybrid_init",
            semantic_weight=self.hybrid_config.semantic_weight,
            bm25_weight=self.hybrid_config.bm25_weight,
            fusion_method=fusion_method.value
        )

    async def ingest_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Ingest documents into RAG system.

        Workflow:
        1. Chunk documents using chunking adapter
        2. Generate embeddings for all chunks
        3. Create Document objects with embeddings
        4. Store in vector database

        Args:
            documents: List of documents with 'content' and optional 'metadata'
            batch_size: Batch size for vector store operations

        Returns:
            Dictionary with ingestion statistics:
                - documents_processed: Number of documents ingested
                - chunks_created: Total chunks created
                - embeddings_generated: Total embeddings generated
                - vector_store_documents: Documents added to vector store
                - total_latency_ms: Total processing time
                - chunking_latency_ms: Time spent chunking
                - embedding_latency_ms: Time spent generating embeddings
                - storage_latency_ms: Time spent storing in vector database
        """
        start_time = time.time()

        log_with_context(
            self.logger,
            "info",
            "Starting document ingestion",
            event="rag_ingest_start",
            document_count=len(documents),
            batch_size=batch_size
        )

        try:
            # Phase 1: Chunk documents
            chunking_start = time.time()
            all_chunks: List[Chunk] = []

            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})

                result = await self.chunking_adapter.chunk_text(content, metadata)
                all_chunks.extend(result.chunks)

            chunking_latency_ms = (time.time() - chunking_start) * 1000

            log_with_context(
                self.logger,
                "info",
                "Document chunking completed",
                event="rag_chunking_complete",
                chunks_created=len(all_chunks),
                latency_ms=round(chunking_latency_ms, 2)
            )

            # Phase 2: Generate embeddings for chunks
            embedding_start = time.time()
            chunk_texts = [chunk.content for chunk in all_chunks]

            embedding_result = await self.embedding_adapter.embed_documents(chunk_texts)
            embeddings = embedding_result.embeddings

            embedding_latency_ms = (time.time() - embedding_start) * 1000

            log_with_context(
                self.logger,
                "info",
                "Embedding generation completed",
                event="rag_embedding_complete",
                embeddings_generated=len(embeddings),
                latency_ms=round(embedding_latency_ms, 2)
            )

            # Phase 3: Create Document objects with embeddings
            vector_documents = []
            for chunk, embedding in zip(all_chunks, embeddings):
                doc = Document(
                    id=chunk.chunk_id,
                    content=chunk.content,
                    embedding=embedding,
                    metadata=chunk.metadata
                )
                vector_documents.append(doc)

            # Phase 4: Store in vector database
            storage_start = time.time()
            storage_result = await self.vector_store.add_documents(
                vector_documents,
                batch_size=batch_size
            )
            storage_latency_ms = (time.time() - storage_start) * 1000

            log_with_context(
                self.logger,
                "info",
                "Vector storage completed",
                event="rag_storage_complete",
                documents_stored=storage_result.get("documents_added", 0),
                latency_ms=round(storage_latency_ms, 2)
            )

            # Phase 5: Index for BM25 (if hybrid search enabled)
            bm25_latency_ms = 0.0
            bm25_indexed = 0

            if self.hybrid_config.enabled and self._hybrid_searcher:
                bm25_start = time.time()

                # Prepare documents for BM25 indexing
                bm25_docs = [
                    {
                        "id": chunk.chunk_id,
                        "content": chunk.content,
                        "metadata": chunk.metadata
                    }
                    for chunk in all_chunks
                ]

                bm25_result = await self._hybrid_searcher.index_documents(bm25_docs)
                bm25_indexed = bm25_result.get("indexed_count", 0)
                bm25_latency_ms = (time.time() - bm25_start) * 1000

                log_with_context(
                    self.logger,
                    "info",
                    "BM25 indexing completed",
                    event="rag_bm25_index_complete",
                    documents_indexed=bm25_indexed,
                    vocabulary_size=bm25_result.get("vocabulary_size", 0),
                    latency_ms=round(bm25_latency_ms, 2)
                )

            total_latency_ms = (time.time() - start_time) * 1000

            result = {
                "documents_processed": len(documents),
                "chunks_created": len(all_chunks),
                "embeddings_generated": len(embeddings),
                "vector_store_documents": storage_result.get("documents_added", 0),
                "bm25_indexed": bm25_indexed,
                "total_latency_ms": round(total_latency_ms, 2),
                "chunking_latency_ms": round(chunking_latency_ms, 2),
                "embedding_latency_ms": round(embedding_latency_ms, 2),
                "storage_latency_ms": round(storage_latency_ms, 2),
                "bm25_latency_ms": round(bm25_latency_ms, 2)
            }

            log_with_context(
                self.logger,
                "info",
                "Document ingestion completed successfully",
                event="rag_ingest_complete",
                **result
            )

            return result

        except Exception as e:
            total_latency_ms = (time.time() - start_time) * 1000

            log_with_context(
                self.logger,
                "error",
                f"Document ingestion failed: {str(e)}",
                event="rag_ingest_error",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(total_latency_ms, 2)
            )
            raise

    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        search_mode: Optional[SearchMode] = None
    ) -> SearchResponse:
        """
        Query the RAG system.

        Supports multiple search modes:
        - SEMANTIC: Pure vector similarity search (default if hybrid disabled)
        - BM25: Pure keyword-based search
        - HYBRID: Combined BM25 + semantic (default if hybrid enabled)

        Args:
            query_text: Natural language query
            top_k: Number of results to return
            filters: Optional metadata filters
            search_mode: Search mode (defaults based on hybrid_config)

        Returns:
            SearchResponse with results, scores, and metadata
        """
        start_time = time.time()

        # Determine search mode
        if search_mode is None:
            search_mode = (
                SearchMode.HYBRID
                if self.hybrid_config.enabled
                else SearchMode.SEMANTIC
            )

        log_with_context(
            self.logger,
            "info",
            "Starting RAG query",
            event="rag_query_start",
            query_length=len(query_text),
            top_k=top_k,
            has_filters=filters is not None,
            search_mode=search_mode.value
        )

        try:
            if search_mode == SearchMode.HYBRID and self._hybrid_searcher:
                return await self._query_hybrid(query_text, top_k, filters, start_time)
            elif search_mode == SearchMode.BM25 and self._bm25_retriever:
                return await self._query_bm25(query_text, top_k, filters, start_time)
            else:
                return await self._query_semantic(query_text, top_k, filters, start_time)

        except Exception as e:
            total_latency_ms = (time.time() - start_time) * 1000

            log_with_context(
                self.logger,
                "error",
                f"RAG query failed: {str(e)}",
                event="rag_query_error",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(total_latency_ms, 2)
            )
            raise

    async def _query_semantic(
        self,
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        start_time: float
    ) -> SearchResponse:
        """Execute semantic-only search."""
        # Phase 1: Embed query
        embedding_start = time.time()
        embedding_result = await self.embedding_adapter.embed_query(query_text)
        query_embedding = embedding_result.embeddings[0]
        embedding_latency_ms = (time.time() - embedding_start) * 1000

        log_with_context(
            self.logger,
            "info",
            "Query embedding generated",
            event="rag_query_embedding_complete",
            latency_ms=round(embedding_latency_ms, 2)
        )

        # Phase 2: Search vector store
        search_start = time.time()
        search_response = await self.vector_store.search(
            query_embedding=query_embedding,
            limit=top_k,
            filters=filters
        )
        search_latency_ms = (time.time() - search_start) * 1000

        total_latency_ms = (time.time() - start_time) * 1000

        log_with_context(
            self.logger,
            "info",
            "Semantic search completed",
            event="rag_query_complete",
            search_mode="semantic",
            results_found=len(search_response.results),
            total_latency_ms=round(total_latency_ms, 2),
            embedding_latency_ms=round(embedding_latency_ms, 2),
            search_latency_ms=round(search_latency_ms, 2)
        )

        return search_response

    async def _query_bm25(
        self,
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        start_time: float
    ) -> SearchResponse:
        """Execute BM25-only search."""
        search_start = time.time()
        bm25_results = await self._bm25_retriever.search(query_text, top_k, filters)
        search_latency_ms = (time.time() - search_start) * 1000

        total_latency_ms = (time.time() - start_time) * 1000

        # Convert to SearchResponse
        results = [
            SearchResult(
                id=r.id,
                content=r.content,
                score=r.score,
                metadata=r.metadata
            )
            for r in bm25_results
        ]

        log_with_context(
            self.logger,
            "info",
            "BM25 search completed",
            event="rag_query_complete",
            search_mode="bm25",
            results_found=len(results),
            total_latency_ms=round(total_latency_ms, 2),
            search_latency_ms=round(search_latency_ms, 2)
        )

        return SearchResponse(
            results=results,
            query_embedding=[],  # No embedding for BM25
            total_results=len(results),
            latency_ms=total_latency_ms
        )

    async def _query_hybrid(
        self,
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        start_time: float
    ) -> SearchResponse:
        """Execute hybrid search (BM25 + semantic)."""
        search_start = time.time()
        hybrid_results = await self._hybrid_searcher.search(query_text, top_k, filters)
        search_latency_ms = (time.time() - search_start) * 1000

        total_latency_ms = (time.time() - start_time) * 1000

        # Convert to SearchResponse
        results = [
            SearchResult(
                id=r.id,
                content=r.content,
                score=r.score,
                metadata={
                    **r.metadata,
                    "_semantic_score": r.semantic_score,
                    "_bm25_score": r.bm25_score
                }
            )
            for r in hybrid_results
        ]

        log_with_context(
            self.logger,
            "info",
            "Hybrid search completed",
            event="rag_query_complete",
            search_mode="hybrid",
            fusion_method=self.hybrid_config.fusion_method,
            results_found=len(results),
            total_latency_ms=round(total_latency_ms, 2),
            search_latency_ms=round(search_latency_ms, 2)
        )

        return SearchResponse(
            results=results,
            query_embedding=[],  # Embedding done internally
            total_results=len(results),
            latency_ms=total_latency_ms
        )

    async def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delete documents from the vector store.

        Args:
            ids: Optional list of document IDs to delete
            filters: Optional metadata filters for deletion

        Returns:
            Dictionary with deletion statistics
        """
        log_with_context(
            self.logger,
            "info",
            "Deleting documents",
            event="rag_delete_start",
            has_ids=ids is not None,
            has_filters=filters is not None
        )

        try:
            result = await self.vector_store.delete(ids=ids, filters=filters)

            log_with_context(
                self.logger,
                "info",
                "Documents deleted successfully",
                event="rag_delete_complete",
                **result
            )

            return result

        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"Document deletion failed: {str(e)}",
                event="rag_delete_error",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get RAG system statistics.

        Returns:
            Dictionary with system information:
                - embedding_model: Embedding model name
                - embedding_dimensions: Embedding vector dimensions
                - chunking_strategy: Chunking strategy name
                - vector_store: Vector store provider and stats
                - hybrid_search: Hybrid search stats (if enabled)
        """
        try:
            vector_store_info = await self.vector_store.get_collection_info()

            stats = {
                "embedding_model": self.embedding_adapter.model_name,
                "embedding_dimensions": self.embedding_adapter.dimensions,
                "chunking_strategy": self.chunking_adapter.strategy.value,
                "vector_store": vector_store_info,
                "hybrid_search": {
                    "enabled": self.hybrid_config.enabled,
                    "semantic_weight": self.hybrid_config.semantic_weight,
                    "bm25_weight": self.hybrid_config.bm25_weight,
                    "fusion_method": self.hybrid_config.fusion_method,
                }
            }

            # Add BM25 stats if enabled
            if self._hybrid_searcher:
                stats["hybrid_search"]["bm25_stats"] = self._hybrid_searcher.get_stats()

            return stats

        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"Failed to get RAG stats: {str(e)}",
                event="rag_stats_error",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
