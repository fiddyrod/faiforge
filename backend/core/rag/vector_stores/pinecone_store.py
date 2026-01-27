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

# Optional import for Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    Pinecone = None
    ServerlessSpec = None
    PINECONE_AVAILABLE = False


class PineconeStore(VectorStoreAdapter):
    """Adapter for Pinecone managed vector database"""

    # Map our distance metrics to Pinecone's format
    METRIC_MAP = {
        DistanceMetric.COSINE: "cosine",
        DistanceMetric.EUCLIDEAN: "euclidean",
        DistanceMetric.DOT_PRODUCT: "dotproduct"
    }

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = 1536,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """
        Initialize Pinecone store.

        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            dimension: Embedding dimension size
            distance_metric: Distance metric to use
            cloud: Cloud provider ('aws', 'gcp', 'azure')
            region: Cloud region
        """
        if not PINECONE_AVAILABLE:
            raise ImportError(
                "pinecone-client not installed. Install with: pip install pinecone-client"
            )

        self._index_name = index_name
        self.dimension = dimension
        self._distance_metric = distance_metric
        self.cloud = cloud
        self.region = region
        self.logger = get_logger()

        self.pc = Pinecone(api_key=api_key)
        self.index = None

    @property
    def collection_name(self) -> str:
        return self._index_name

    @property
    def distance_metric(self) -> DistanceMetric:
        return self._distance_metric

    async def initialize(self) -> None:
        """Initialize Pinecone index"""

        log_with_context(
            self.logger,
            "info",
            "Initializing Pinecone",
            event="vector_store_init",
            provider="pinecone",
            index=self._index_name,
            dimension=self.dimension
        )

        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]

            if self._index_name not in index_names:
                # Create index
                self.pc.create_index(
                    name=self._index_name,
                    dimension=self.dimension,
                    metric=self.METRIC_MAP[self._distance_metric],
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region
                    )
                )
                self.logger.info(f"Created Pinecone index: {self._index_name}")

            # Connect to index
            self.index = self.pc.Index(self._index_name)

            # Get index stats
            stats = self.index.describe_index_stats()

            log_with_context(
                self.logger,
                "info",
                "Pinecone initialized",
                event="vector_store_init_complete",
                provider="pinecone",
                index=self._index_name,
                vector_count=stats.get("total_vector_count", 0)
            )

        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"Pinecone initialization failed: {str(e)}",
                event="vector_store_init_error",
                provider="pinecone",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Add documents in batches using upsert"""

        start_time = time.time()

        log_with_context(
            self.logger,
            "info",
            "Adding documents to Pinecone",
            event="vector_store_add_start",
            provider="pinecone",
            document_count=len(documents),
            batch_size=batch_size
        )

        try:
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                # Prepare vectors in Pinecone format
                vectors = [
                    {
                        "id": doc.id,
                        "values": doc.embedding,
                        "metadata": {
                            **doc.metadata,
                            "content": doc.content  # Store content in metadata
                        }
                    }
                    for doc in batch
                ]

                self.index.upsert(vectors=vectors)

            latency_ms = (time.time() - start_time) * 1000

            log_with_context(
                self.logger,
                "info",
                "Documents added to Pinecone",
                event="vector_store_add_complete",
                provider="pinecone",
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
                f"Pinecone add documents failed: {str(e)}",
                event="vector_store_add_error",
                provider="pinecone",
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
            "Starting Pinecone search",
            event="vector_store_search_start",
            provider="pinecone",
            limit=limit,
            has_filters=filters is not None
        )

        try:
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=limit,
                filter=filters,
                include_metadata=True
            )

            latency_ms = (time.time() - start_time) * 1000

            # Parse results
            search_results = []
            for match in results.matches:
                # Extract content from metadata
                content = match.metadata.pop("content", "")

                search_results.append(
                    SearchResult(
                        id=match.id,
                        content=content,
                        score=match.score,
                        metadata=match.metadata
                    )
                )

            log_with_context(
                self.logger,
                "info",
                "Pinecone search completed",
                event="vector_store_search_complete",
                provider="pinecone",
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
                f"Pinecone search failed: {str(e)}",
                event="vector_store_search_error",
                provider="pinecone",
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
                self.index.delete(ids=ids)
                return {"deleted_count": len(ids)}
            elif filters:
                self.index.delete(filter=filters)
                return {"deleted_by_filter": True}

        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"Pinecone delete failed: {str(e)}",
                event="vector_store_delete_error",
                provider="pinecone",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def clear_collection(self) -> Dict[str, Any]:
        """Clear all documents from index"""

        try:
            # Get stats before clearing
            stats = self.index.describe_index_stats()
            count_before = stats.get("total_vector_count", 0)

            # Delete all vectors
            self.index.delete(delete_all=True)

            log_with_context(
                self.logger,
                "info",
                "Pinecone index cleared",
                event="vector_store_clear",
                provider="pinecone",
                documents_cleared=count_before
            )

            return {"documents_cleared": count_before}

        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"Pinecone clear index failed: {str(e)}",
                event="vector_store_clear_error",
                provider="pinecone",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get index statistics and info"""

        try:
            stats = self.index.describe_index_stats()

            return {
                "name": self._index_name,
                "document_count": stats.get("total_vector_count", 0),
                "distance_metric": self._distance_metric.value,
                "embedding_dimensions": self.dimension,
                "provider": "pinecone",
                "namespaces": stats.get("namespaces", {})
            }
        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"Pinecone get index info failed: {str(e)}",
                event="vector_store_info_error",
                provider="pinecone",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
