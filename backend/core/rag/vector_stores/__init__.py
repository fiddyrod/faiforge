from .base import (
    VectorStoreAdapter,
    Document,
    SearchResult,
    SearchResponse,
    DistanceMetric
)

# Optional imports for vector stores
# ChromaDB
try:
    from .chroma_store import ChromaStore, CHROMA_AVAILABLE
except ImportError:
    ChromaStore = None
    CHROMA_AVAILABLE = False

# Pinecone
try:
    from .pinecone_store import PineconeStore, PINECONE_AVAILABLE
except ImportError:
    PineconeStore = None
    PINECONE_AVAILABLE = False

# Qdrant
try:
    from .qdrant_store import QdrantStore, QDRANT_AVAILABLE
except ImportError:
    QdrantStore = None
    QDRANT_AVAILABLE = False

# Weaviate
try:
    from .weaviate_store import WeaviateStore, WEAVIATE_AVAILABLE
except ImportError:
    WeaviateStore = None
    WEAVIATE_AVAILABLE = False

__all__ = [
    "VectorStoreAdapter",
    "Document",
    "SearchResult",
    "SearchResponse",
    "DistanceMetric",
    "ChromaStore",
    "PineconeStore",
    "QdrantStore",
    "WeaviateStore",
    "CHROMA_AVAILABLE",
    "PINECONE_AVAILABLE",
    "QDRANT_AVAILABLE",
    "WEAVIATE_AVAILABLE"
]
