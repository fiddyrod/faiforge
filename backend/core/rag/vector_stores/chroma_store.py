import time
from typing import List, Dict, Any, Optional

from .base import (
    VectorStoreAdapter,
    Document,
    SearchResult,
    SearchResponse,
    DistanceMetric
)
from ...observability import get_logger, log_with_context

# Optional import for ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None
    Settings = None
    CHROMA_AVAILABLE = False


class ChromaStore(VectorStoreAdapter):
    """Adapter for ChromaDB embedded vector database"""

    # Map our distance metrics to ChromaDB's format
    METRIC_MAP = {
        DistanceMetric.COSINE: "cosine",
        DistanceMetric.EUCLIDEAN: "l2",
        DistanceMetric.DOT_PRODUCT: "ip"
    }

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./data/chroma_db",
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        embedding_dimensions: int = 1536
    ):
        """
        Initialize ChromaDB store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
            distance_metric: Distance metric to use
            embedding_dimensions: Expected embedding dimensions
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )

        self._collection_name = collection_name
        self.persist_directory = persist_directory
        self._distance_metric = distance_metric
        self.embedding_dimensions = embedding_dimensions
        self.logger = get_logger()

        self.client = None
        self.collection = None

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def distance_metric(self) -> DistanceMetric:
        return self._distance_metric

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection"""

        log_with_context(
            self.logger,
            "info",
            "Initializing ChromaDB",
            event="vector_store_init",
            provider="chromadb",
            collection=self._collection_name,
            persist_directory=self.persist_directory
        )

        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )

            # Get or create collection with distance metric
            self.collection = self.client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": self.METRIC_MAP[self._distance_metric]}
            )

            log_with_context(
                self.logger,
                "info",
                "ChromaDB initialized",
                event="vector_store_init_complete",
                provider="chromadb",
                collection=self._collection_name,
                document_count=self.collection.count()
            )

        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"ChromaDB initialization failed: {str(e)}",
                event="vector_store_init_error",
                provider="chromadb",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Add documents in batches"""

        start_time = time.time()

        log_with_context(
            self.logger,
            "info",
            "Adding documents to ChromaDB",
            event="vector_store_add_start",
            provider="chromadb",
            document_count=len(documents),
            batch_size=batch_size
        )

        try:
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                self.collection.add(
                    ids=[doc.id for doc in batch],
                    embeddings=[doc.embedding for doc in batch],
                    documents=[doc.content for doc in batch],
                    metadatas=[doc.metadata for doc in batch]
                )

            latency_ms = (time.time() - start_time) * 1000

            log_with_context(
                self.logger,
                "info",
                "Documents added to ChromaDB",
                event="vector_store_add_complete",
                provider="chromadb",
                document_count=len(documents),
                latency_ms=round(latency_ms, 2),
                status="success"
            )

            return {
                "documents_added": len(documents),
                "latency_ms": round(latency_ms, 2)
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            log_with_context(
                self.logger,
                "error",
                f"ChromaDB add documents failed: {str(e)}",
                event="vector_store_add_error",
                provider="chromadb",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(latency_ms, 2)
            )
            raise

    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """Search using embedding vector"""

        start_time = time.time()

        log_with_context(
            self.logger,
            "info",
            "Starting ChromaDB search",
            event="vector_store_search_start",
            provider="chromadb",
            limit=limit,
            has_filters=filters is not None
        )

        try:
            # Build where clause for filters
            where_clause = filters if filters else None

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause
            )

            latency_ms = (time.time() - start_time) * 1000

            # Parse results
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    search_results.append(
                        SearchResult(
                            id=results["ids"][0][i],
                            content=results["documents"][0][i],
                            score=results["distances"][0][i],
                            metadata=results["metadatas"][0][i]
                        )
                    )

            log_with_context(
                self.logger,
                "info",
                "ChromaDB search completed",
                event="vector_store_search_complete",
                provider="chromadb",
                results_found=len(search_results),
                latency_ms=round(latency_ms, 2),
                status="success"
            )

            return SearchResponse(
                results=search_results,
                query_embedding=query_embedding,
                total_results=len(search_results),
                latency_ms=round(latency_ms, 2)
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            log_with_context(
                self.logger,
                "error",
                f"ChromaDB search failed: {str(e)}",
                event="vector_store_search_error",
                provider="chromadb",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(latency_ms, 2)
            )
            raise

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Delete documents by IDs or filters"""

        if not ids and not filters:
            raise ValueError("Must provide either ids or filters")

        try:
            if ids:
                self.collection.delete(ids=ids)
                return {"deleted_count": len(ids)}
            elif filters:
                self.collection.delete(where=filters)
                return {"deleted_by_filter": True}

        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"ChromaDB delete failed: {str(e)}",
                event="vector_store_delete_error",
                provider="chromadb",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def clear_collection(self) -> Dict[str, Any]:
        """Clear all documents from collection"""

        try:
            count_before = self.collection.count()
            self.client.delete_collection(name=self._collection_name)

            # Recreate collection
            self.collection = self.client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": self.METRIC_MAP[self._distance_metric]}
            )

            log_with_context(
                self.logger,
                "info",
                "ChromaDB collection cleared",
                event="vector_store_clear",
                provider="chromadb",
                documents_cleared=count_before
            )

            return {"documents_cleared": count_before}

        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"ChromaDB clear collection failed: {str(e)}",
                event="vector_store_clear_error",
                provider="chromadb",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics and info"""

        try:
            return {
                "name": self._collection_name,
                "document_count": self.collection.count(),
                "distance_metric": self._distance_metric.value,
                "embedding_dimensions": self.embedding_dimensions,
                "provider": "chromadb"
            }
        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"ChromaDB get collection info failed: {str(e)}",
                event="vector_store_info_error",
                provider="chromadb",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
