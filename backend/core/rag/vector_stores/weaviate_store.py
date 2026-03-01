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

# Optional import for Weaviate
try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    WEAVIATE_AVAILABLE = True
except ImportError:
    weaviate = None
    Configure = None
    Property = None
    DataType = None
    WEAVIATE_AVAILABLE = False


class WeaviateStore(VectorStoreAdapter):
    """Adapter for Weaviate vector database"""

    # Map our distance metrics to Weaviate's format
    METRIC_MAP = {
        DistanceMetric.COSINE: "cosine",
        DistanceMetric.EUCLIDEAN: "l2-squared",
        DistanceMetric.DOT_PRODUCT: "dot"
    }

    def __init__(
        self,
        class_name: str,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        dimension: int = 1536,
        distance_metric: DistanceMetric = DistanceMetric.COSINE
    ):
        """
        Initialize Weaviate store.

        Args:
            class_name: Name of the Weaviate class
            url: Weaviate server URL
            api_key: Optional API key for Weaviate Cloud
            dimension: Embedding dimension size
            distance_metric: Distance metric to use
        """
        if not WEAVIATE_AVAILABLE:
            raise ImportError(
                "weaviate-client not installed. Install with: pip install weaviate-client"
            )

        self._class_name = class_name
        self.url = url
        self.dimension = dimension
        self._distance_metric = distance_metric
        self.logger = get_logger()

        # Initialize client
        if api_key:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=weaviate.auth.AuthApiKey(api_key)
            )
        else:
            self.client = weaviate.connect_to_local(host=url)

        self.collection = None

    @property
    def collection_name(self) -> str:
        return self._class_name

    @property
    def distance_metric(self) -> DistanceMetric:
        return self._distance_metric

    async def initialize(self) -> None:
        """Initialize Weaviate class/schema"""

        log_with_context(
            self.logger,
            "info",
            "Initializing Weaviate",
            event="vector_store_init",
            provider="weaviate",
            class_name=self._class_name,
            url=self.url
        )

        try:
            # Check if class exists
            if not self.client.collections.exists(self._class_name):
                # Create class with schema
                self.client.collections.create(
                    name=self._class_name,
                    vectorizer_config=Configure.Vectorizer.none(),
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=self.METRIC_MAP[self._distance_metric]
                    ),
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                    ]
                )
                self.logger.info(f"Created Weaviate class: {self._class_name}")

            # Get collection
            self.collection = self.client.collections.get(self._class_name)

            # Get collection stats
            agg = self.collection.aggregate.over_all(total_count=True)

            log_with_context(
                self.logger,
                "info",
                "Weaviate initialized",
                event="vector_store_init_complete",
                provider="weaviate",
                class_name=self._class_name,
                object_count=agg.total_count
            )

        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"Weaviate initialization failed: {str(e)}",
                event="vector_store_init_error",
                provider="weaviate",
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
            "Adding documents to Weaviate",
            event="vector_store_add_start",
            provider="weaviate",
            document_count=len(documents),
            batch_size=batch_size
        )

        try:
            # Process in batches
            with self.collection.batch.dynamic() as batch:
                for doc in documents:
                    # Prepare properties
                    properties = {
                        "content": doc.content,
                        **doc.metadata
                    }

                    # Add object with vector
                    batch.add_object(
                        properties=properties,
                        uuid=doc.id,
                        vector=doc.embedding
                    )

            latency_ms = (time.time() - start_time) * 1000

            log_with_context(
                self.logger,
                "info",
                "Documents added to Weaviate",
                event="vector_store_add_complete",
                provider="weaviate",
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
                f"Weaviate add documents failed: {str(e)}",
                event="vector_store_add_error",
                provider="weaviate",
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
            "Starting Weaviate search",
            event="vector_store_search_start",
            provider="weaviate",
            limit=limit,
            has_filters=filters is not None
        )

        try:
            # Build query
            query = self.collection.query.near_vector(
                near_vector=query_embedding,
                limit=limit,
                return_metadata=['distance']
            )

            # Add filters if provided
            if filters:
                # Simple equality filters
                where_conditions = []
                for key, value in filters.items():
                    where_conditions.append({
                        "path": [key],
                        "operator": "Equal",
                        "valueText": str(value) if not isinstance(value, str) else value
                    })

                if len(where_conditions) == 1:
                    query = query.where(where_conditions[0])
                else:
                    query = query.where({
                        "operator": "And",
                        "operands": where_conditions
                    })

            # Execute query
            results = query.objects

            latency_ms = (time.time() - start_time) * 1000

            # Parse results
            search_results = []
            for obj in results:
                content = obj.properties.get("content", "")
                metadata = {k: v for k, v in obj.properties.items() if k != "content"}

                search_results.append(
                    SearchResult(
                        id=str(obj.uuid),
                        content=content,
                        score=obj.metadata.distance,
                        metadata=metadata
                    )
                )

            log_with_context(
                self.logger,
                "info",
                "Weaviate search completed",
                event="vector_store_search_complete",
                provider="weaviate",
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
                f"Weaviate search failed: {str(e)}",
                event="vector_store_search_error",
                provider="weaviate",
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
                for doc_id in ids:
                    self.collection.data.delete_by_id(uuid=doc_id)
                return {"deleted_count": len(ids)}
            elif filters:
                # Build where filter
                where_conditions = []
                for key, value in filters.items():
                    where_conditions.append({
                        "path": [key],
                        "operator": "Equal",
                        "valueText": str(value) if not isinstance(value, str) else value
                    })

                where_filter = where_conditions[0] if len(where_conditions) == 1 else {
                    "operator": "And",
                    "operands": where_conditions
                }

                result = self.collection.data.delete_many(where=where_filter)
                return {"deleted_count": result.successful}

        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"Weaviate delete failed: {str(e)}",
                event="vector_store_delete_error",
                provider="weaviate",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """List documents using Weaviate's fetch_objects API."""
        try:
            agg = self.collection.aggregate.over_all(total_count=True)
            total = agg.total_count or 0

            # Build optional where filter
            where_filter = None
            if filters and weaviate:
                from weaviate.classes.query import Filter
                conditions = [Filter.by_property(k).equal(v) for k, v in filters.items()]
                where_filter = conditions[0] if len(conditions) == 1 else Filter.all_of(conditions)

            result = self.collection.query.fetch_objects(
                limit=limit,
                offset=offset,
                filters=where_filter,
                include_vector=False,
            )
            docs = []
            for obj in result.objects:
                props = dict(obj.properties)
                docs.append({
                    "id": str(obj.uuid),
                    "content": str(props.pop("content", ""))[:300],
                    "metadata": props,
                })
            return {"documents": docs, "total": total, "limit": limit, "offset": offset}
        except Exception as e:
            log_with_context(
                self.logger, "error", f"Weaviate list documents failed: {str(e)}",
                event="vector_store_list_error", provider="weaviate", error=str(e)
            )
            raise

    async def clear_collection(self) -> Dict[str, Any]:
        """Clear all documents from class"""

        try:
            # Get count before clearing
            agg = self.collection.aggregate.over_all(total_count=True)
            count_before = agg.total_count

            # Delete class and recreate
            self.client.collections.delete(self._class_name)

            self.client.collections.create(
                name=self._class_name,
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=self.METRIC_MAP[self._distance_metric]
                ),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                ]
            )

            self.collection = self.client.collections.get(self._class_name)

            log_with_context(
                self.logger,
                "info",
                "Weaviate class cleared",
                event="vector_store_clear",
                provider="weaviate",
                documents_cleared=count_before
            )

            return {"documents_cleared": count_before}

        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"Weaviate clear class failed: {str(e)}",
                event="vector_store_clear_error",
                provider="weaviate",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get class statistics and info"""

        try:
            agg = self.collection.aggregate.over_all(total_count=True)

            return {
                "name": self._class_name,
                "document_count": agg.total_count,
                "distance_metric": self._distance_metric.value,
                "embedding_dimensions": self.dimension,
                "provider": "weaviate"
            }
        except Exception as e:
            log_with_context(
                self.logger,
                "error",
                f"Weaviate get class info failed: {str(e)}",
                event="vector_store_info_error",
                provider="weaviate",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def __del__(self):
        """Cleanup: close Weaviate client connection"""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
        except:
            pass
