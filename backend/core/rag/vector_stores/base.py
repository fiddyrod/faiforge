from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class DistanceMetric(str, Enum):
    """Distance metric types"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class Document:
    """Document to store in vector database"""
    id: str                              # Unique document ID
    content: str                         # Text content
    embedding: List[float]               # Embedding vector
    metadata: Dict[str, Any]             # Metadata (searchable)


@dataclass
class SearchResult:
    """Single search result"""
    id: str                              # Document ID
    content: str                         # Document content
    score: float                         # Similarity score
    metadata: Dict[str, Any]             # Document metadata


@dataclass
class SearchResponse:
    """Response from similarity search"""
    results: List[SearchResult]          # Search results
    query_embedding: List[float]         # Query embedding used
    total_results: int                   # Total results found
    latency_ms: float                    # Search latency


class VectorStoreAdapter(ABC):
    """Base class for all vector store adapters"""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize connection and create collection if needed"""
        pass

    @abstractmethod
    async def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Add documents to vector store.

        Args:
            documents: List of documents to add
            batch_size: Batch size for bulk operations

        Returns:
            Dict with operation metadata (count, latency, etc.)
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """
        Similarity search using embedding.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum results to return
            filters: Optional metadata filters

        Returns:
            SearchResponse with results and metadata
        """
        pass

    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delete documents by IDs or filters.

        Args:
            ids: Optional list of document IDs
            filters: Optional metadata filters

        Returns:
            Dict with deletion metadata
        """
        pass

    @abstractmethod
    async def clear_collection(self) -> Dict[str, Any]:
        """Clear all documents from collection"""
        pass

    @abstractmethod
    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """List documents with their IDs and metadata (no embeddings).

        Returns:
            Dict with keys: documents (list of {id, content, metadata}),
            total (total count in collection), limit, offset.
        """
        pass

    @abstractmethod
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics and info"""
        pass

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Return collection/index name"""
        pass

    @property
    @abstractmethod
    def distance_metric(self) -> DistanceMetric:
        """Return configured distance metric"""
        pass
