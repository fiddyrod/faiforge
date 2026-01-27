import time
from typing import List, Dict, Any, Optional
import uuid

from .base import (
    VectorStoreAdapter,
    Document,
    SearchResult,
    SearchResponse,
    DistanceMetric
)
from ...observability import get_logger, log_with_context

# Optional import for Qdrant
try:
    from qdrant_client import QdrantClient, models
    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None
    models = None
    QDRANT_AVAILABLE = False


class QdrantStore(VectorStoreAdapter):
    """Adapter for Qdrant vector search engine"""

    # Map our distance metrics to Qdrant's format
    METRIC_MAP = {
        DistanceMetric.COSINE: "Cosine",
        DistanceMetric.EUCLIDEAN: "Euclid",
        DistanceMetric.DOT_PRODUCT: "Dot"
    }

    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        dimension: int = 1536,
        distance_metric: DistanceMetric = DistanceMetric.COSINE
    ):
        """
        Initialize Qdrant store.

        Args:
            collection_name: Name of the collection
            url: Qdrant server URL
            api_key: Optional API key for Qdrant Cloud
            dimension: Embedding dimension size
            distance_metric: Distance metric to use
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client not installed. Install with: pip install qdrant-client"
            )

        self._collection_name = collection_name
        self.url = url
        self.dimension = dimension
        self._distance_metric = distance_metric
        self.logger = get_logger()

        self.client = QdrantClient(url=url, api_key=api_key)

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def distance_metric(self) -> DistanceMetric:
        return self._distance_metric

    async def initialize(self) -> None:
        """Initialize Qdrant collection"""

        log_with_context(
            self.logger,
            "info",
            "Initializing Qdrant",
            event="vector_store_init",
            provider="qdrant",
            collection=self._collection_name,
            url=self.url
        )

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self._collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=self.dimension,
                        distance=getattr(models.Distance, self.METRIC_MAP[self._distance_metric])
                    )
                )
                self.logger.info(f"Created Qdrant collection: {self._collection_name}")

            # Get collection info
            info = self.client.get_collection(self._collection_name)

            log_with_context(
                self.logger,
                "info",
                "Qdrant initialized",
                event="vector_store_init_complete",
                provider="qdrant",
                collection=self._collection_name,
                vector_count=info.points_count
            )

        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"Qdrant initialization failed: {str(e)}",
                event="vector_store_init_error",
                provider="qdrant",
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
            "Adding documents to Qdrant",
            event="vector_store_add_start",
            provider="qdrant",
            document_count=len(documents),
            batch_size=batch_size
        )

        try:
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                # Prepare points in Qdrant format
                points = [
                    models.PointStruct(
                        id=doc.id,
                        vector=doc.embedding,
                        payload={
                            **doc.metadata,
                            "content": doc.content
                        }
                    )
                    for doc in batch
                ]

                self.client.upsert(
                    collection_name=self._collection_name,
                    points=points
                )

            latency_ms = (time.time() - start_time) * 1000

            log_with_context(
                self.logger,
                "info",
                "Documents added to Qdrant",
                event="vector_store_add_complete",
                provider="qdrant",
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
                f"Qdrant add documents failed: {str(e)}",
                event="vector_store_add_error",
                provider="qdrant",
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
            "Starting Qdrant search",
            event="vector_store_search_start",
            provider="qdrant",
            limit=limit,
            has_filters=filters is not None
        )

        try:
            # Build filter if provided
            query_filter = None
            if filters:
                # Convert filters to Qdrant filter format
                must_conditions = [
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filters.items()
                ]
                query_filter = models.Filter(must=must_conditions)

            # Search
            results = self.client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=query_filter,
                with_payload=True
            )

            latency_ms = (time.time() - start_time) * 1000

            # Parse results
            search_results = []
            for hit in results:
                # Extract content from payload
                content = hit.payload.pop("content", "")

                search_results.append(
                    SearchResult(
                        id=str(hit.id),
                        content=content,
                        score=hit.score,
                        metadata=hit.payload
                    )
                )

            log_with_context(
                self.logger,
                "info",
                "Qdrant search completed",
                event="vector_store_search_complete",
                provider="qdrant",
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
                f"Qdrant search failed: {str(e)}",
                event="vector_store_search_error",
                provider="qdrant",
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
                self.client.delete(
                    collection_name=self._collection_name,
                    points_selector=models.PointIdsList(points=ids)
                )
                return {"deleted_count": len(ids)}
            elif filters:
                # Convert filters to Qdrant filter format
                must_conditions = [
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filters.items()
                ]
                query_filter = models.Filter(must=must_conditions)

                self.client.delete(
                    collection_name=self._collection_name,
                    points_selector=models.FilterSelector(filter=query_filter)
                )
                return {"deleted_by_filter": True}

        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"Qdrant delete failed: {str(e)}",
                event="vector_store_delete_error",
                provider="qdrant",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def clear_collection(self) -> Dict[str, Any]:
        """Clear all documents from collection"""

        try:
            # Get count before clearing
            info = self.client.get_collection(self._collection_name)
            count_before = info.points_count

            # Delete collection and recreate
            self.client.delete_collection(self._collection_name)
            self.client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=getattr(models.Distance, self.METRIC_MAP[self._distance_metric])
                )
            )

            log_with_context(
                self.logger,
                "info",
                "Qdrant collection cleared",
                event="vector_store_clear",
                provider="qdrant",
                documents_cleared=count_before
            )

            return {"documents_cleared": count_before}

        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"Qdrant clear collection failed: {str(e)}",
                event="vector_store_clear_error",
                provider="qdrant",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics and info"""

        try:
            info = self.client.get_collection(self._collection_name)

            return {
                "name": self._collection_name,
                "document_count": info.points_count,
                "distance_metric": self._distance_metric.value,
                "embedding_dimensions": self.dimension,
                "provider": "qdrant",
                "status": info.status
            }
        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"Qdrant get collection info failed: {str(e)}",
                event="vector_store_info_error",
                provider="qdrant",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
