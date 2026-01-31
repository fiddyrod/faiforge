"""
RAG (Retrieval-Augmented Generation) System

A comprehensive RAG implementation with adapter pattern for:
- Embeddings: OpenAI, HuggingFace
- Chunking: Recursive, Semantic, Fixed-size, Token-based
- Vector Stores: ChromaDB, Pinecone, Qdrant, Weaviate

Example usage:
    from core.rag import load_rag_registry, RAGPipeline

    # Load RAG components from configuration
    registry = await load_rag_registry(
        config_path="core/config/rag.yaml",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create pipeline
    pipeline = RAGPipeline(
        embedding_adapter=registry.get_embedding("openai-small"),
        chunking_adapter=registry.get_chunker("recursive-default"),
        vector_store_adapter=registry.get_vector_store("chroma-local")
    )

    # Ingest documents
    documents = [
        {
            "content": "Python is a high-level programming language.",
            "metadata": {"source": "docs", "category": "programming"}
        }
    ]
    result = await pipeline.ingest_documents(documents)

    # Query
    response = await pipeline.query("What is Python?", top_k=3)
    for result in response.results:
        print(f"Score: {result.score}, Content: {result.content}")
"""

# Core interfaces
from .embeddings.base import (
    EmbeddingAdapter,
    EmbeddingResult
)

from .chunking.base import (
    ChunkingAdapter,
    Chunk,
    ChunkingResult,
    ChunkingStrategy
)

from .vector_stores.base import (
    VectorStoreAdapter,
    Document,
    SearchResult,
    SearchResponse,
    DistanceMetric
)

# Registry and pipeline
from .registry import (
    RAGRegistry,
    load_rag_registry
)

from .pipeline import RAGPipeline

# Implementations (for direct instantiation if needed)
from .embeddings import (
    OpenAIEmbedding,
    HuggingFaceEmbedding,
    SENTENCE_TRANSFORMERS_AVAILABLE
)

from .chunking import (
    RecursiveChunker,
    FixedSizeChunker,
    TokenChunker,
    SemanticChunker
)

from .vector_stores import (
    ChromaStore,
    PineconeStore,
    QdrantStore,
    WeaviateStore,
    CHROMA_AVAILABLE,
    PINECONE_AVAILABLE,
    QDRANT_AVAILABLE,
    WEAVIATE_AVAILABLE
)

__all__ = [
    # Core protocols
    "EmbeddingAdapter",
    "EmbeddingResult",
    "ChunkingAdapter",
    "Chunk",
    "ChunkingResult",
    "ChunkingStrategy",
    "VectorStoreAdapter",
    "Document",
    "SearchResult",
    "SearchResponse",
    "DistanceMetric",

    # Registry and pipeline
    "RAGRegistry",
    "load_rag_registry",
    "RAGPipeline",

    # Embedding implementations
    "OpenAIEmbedding",
    "HuggingFaceEmbedding",
    "SENTENCE_TRANSFORMERS_AVAILABLE",

    # Chunking implementations
    "RecursiveChunker",
    "FixedSizeChunker",
    "TokenChunker",
    "SemanticChunker",

    # Vector store implementations
    "ChromaStore",
    "PineconeStore",
    "QdrantStore",
    "WeaviateStore",
    "CHROMA_AVAILABLE",
    "PINECONE_AVAILABLE",
    "QDRANT_AVAILABLE",
    "WEAVIATE_AVAILABLE",
]
