import yaml
from typing import Dict, Optional, Any
from pathlib import Path
import logging

from .embeddings import (
    EmbeddingAdapter,
    OpenAIEmbedding,
    HuggingFaceEmbedding,
    SENTENCE_TRANSFORMERS_AVAILABLE
)
from .chunking import (
    ChunkingAdapter,
    RecursiveChunker,
    FixedSizeChunker,
    TokenChunker,
    SemanticChunker
)
from .vector_stores import (
    VectorStoreAdapter,
    ChromaStore,
    PineconeStore,
    QdrantStore,
    WeaviateStore,
    CHROMA_AVAILABLE,
    PINECONE_AVAILABLE,
    QDRANT_AVAILABLE,
    WEAVIATE_AVAILABLE
)

logger = logging.getLogger(__name__)


class RAGRegistry:
    """Unified registry for all RAG components"""

    def __init__(self):
        self._embeddings: Dict[str, EmbeddingAdapter] = {}
        self._chunkers: Dict[str, ChunkingAdapter] = {}
        self._vector_stores: Dict[str, VectorStoreAdapter] = {}

    # Embedding methods
    def register_embedding(self, name: str, adapter: EmbeddingAdapter):
        """Register an embedding adapter"""
        self._embeddings[name] = adapter

    def get_embedding(self, name: str) -> EmbeddingAdapter:
        """Get embedding adapter by name"""
        if name not in self._embeddings:
            available = ", ".join(self._embeddings.keys())
            raise ValueError(f"Embedding '{name}' not found. Available: {available}")
        return self._embeddings[name]

    def list_embeddings(self) -> list:
        """List all embedding adapter names"""
        return list(self._embeddings.keys())

    # Chunking methods
    def register_chunker(self, name: str, adapter: ChunkingAdapter):
        """Register a chunking adapter"""
        self._chunkers[name] = adapter

    def get_chunker(self, name: str) -> ChunkingAdapter:
        """Get chunking adapter by name"""
        if name not in self._chunkers:
            available = ", ".join(self._chunkers.keys())
            raise ValueError(f"Chunker '{name}' not found. Available: {available}")
        return self._chunkers[name]

    def list_chunkers(self) -> list:
        """List all chunking adapter names"""
        return list(self._chunkers.keys())

    # Vector store methods
    def register_vector_store(self, name: str, adapter: VectorStoreAdapter):
        """Register a vector store adapter"""
        self._vector_stores[name] = adapter

    def get_vector_store(self, name: str) -> VectorStoreAdapter:
        """Get vector store adapter by name"""
        if name not in self._vector_stores:
            available = ", ".join(self._vector_stores.keys())
            raise ValueError(f"Vector store '{name}' not found. Available: {available}")
        return self._vector_stores[name]

    def list_vector_stores(self) -> list:
        """List all vector store adapter names"""
        return list(self._vector_stores.keys())


async def load_rag_registry(
    config_path: str,
    openai_api_key: Optional[str] = None,
    pinecone_api_key: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    weaviate_api_key: Optional[str] = None
) -> RAGRegistry:
    """
    Load RAG components from YAML configuration.

    Args:
        config_path: Path to rag.yaml
        openai_api_key: OpenAI API key (for embeddings/semantic chunking)
        pinecone_api_key: Pinecone API key (optional)
        qdrant_api_key: Qdrant API key (optional, for cloud)
        weaviate_api_key: Weaviate API key (optional, for cloud)
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    registry = RAGRegistry()

    # Load embeddings
    for name, embed_config in config.get("embeddings", {}).items():
        adapter_type = embed_config["adapter"]

        if adapter_type == "openai":
            if not openai_api_key:
                logger.warning(f"Skipping embedding '{name}' - OPENAI_API_KEY not set")
                continue

            adapter = OpenAIEmbedding(
                api_key=openai_api_key,
                model=embed_config["model"]
            )
            registry.register_embedding(name, adapter)
            logger.info(f"Registered OpenAI embedding: {name}")

        elif adapter_type == "huggingface":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning(
                    f"Skipping embedding '{name}' - sentence-transformers not installed"
                )
                continue

            adapter = HuggingFaceEmbedding(
                model=embed_config["model"],
                device=embed_config.get("device")
            )
            registry.register_embedding(name, adapter)
            logger.info(f"Registered HuggingFace embedding: {name}")

        else:
            logger.warning(f"Unknown embedding adapter type '{adapter_type}' for '{name}'")

    # Load chunkers
    for name, chunk_config in config.get("chunking", {}).items():
        strategy = chunk_config["strategy"]

        if strategy == "recursive":
            adapter = RecursiveChunker(
                chunk_size=chunk_config.get("chunk_size", 1000),
                chunk_overlap=chunk_config.get("chunk_overlap", 200)
            )
            registry.register_chunker(name, adapter)
            logger.info(f"Registered recursive chunker: {name}")

        elif strategy == "semantic":
            # Semantic chunker needs embedding adapter
            embedding_name = chunk_config.get("embedding_adapter")
            if not embedding_name:
                logger.warning(f"Skipping chunker '{name}' - no embedding_adapter specified")
                continue

            try:
                embedding_adapter = registry.get_embedding(embedding_name)
                adapter = SemanticChunker(
                    embedding_adapter=embedding_adapter,
                    similarity_threshold=chunk_config.get("similarity_threshold", 0.5),
                    max_chunk_size=chunk_config.get("max_chunk_size", 1500)
                )
                registry.register_chunker(name, adapter)
                logger.info(f"Registered semantic chunker: {name}")
            except ValueError as e:
                logger.warning(f"Skipping chunker '{name}' - {str(e)}")

        elif strategy == "fixed_size":
            adapter = FixedSizeChunker(
                chunk_size=chunk_config.get("chunk_size", 1000),
                chunk_overlap=chunk_config.get("chunk_overlap", 100)
            )
            registry.register_chunker(name, adapter)
            logger.info(f"Registered fixed-size chunker: {name}")

        elif strategy == "token":
            adapter = TokenChunker(
                chunk_size=chunk_config.get("chunk_size", 512),
                chunk_overlap=chunk_config.get("chunk_overlap", 50),
                encoding_name=chunk_config.get("encoding", "cl100k_base")
            )
            registry.register_chunker(name, adapter)
            logger.info(f"Registered token chunker: {name}")

        else:
            logger.warning(f"Unknown chunking strategy '{strategy}' for '{name}'")

    # Load vector stores
    for name, store_config in config.get("vector_stores", {}).items():
        provider = store_config["provider"]

        if provider == "chromadb":
            if not CHROMA_AVAILABLE:
                logger.warning(f"Skipping vector store '{name}' - chromadb not installed")
                continue

            adapter = ChromaStore(
                collection_name=store_config["collection_name"],
                persist_directory=store_config.get("persist_directory", "./data/chroma_db"),
                embedding_dimensions=store_config.get("dimension", 1536)
            )
            await adapter.initialize()
            registry.register_vector_store(name, adapter)
            logger.info(f"Registered ChromaDB vector store: {name}")

        elif provider == "pinecone":
            if not PINECONE_AVAILABLE:
                logger.warning(f"Skipping vector store '{name}' - pinecone-client not installed")
                continue

            if not pinecone_api_key:
                logger.warning(f"Skipping vector store '{name}' - PINECONE_API_KEY not set")
                continue

            adapter = PineconeStore(
                api_key=pinecone_api_key,
                index_name=store_config["index_name"],
                dimension=store_config.get("dimension", 1536),
                cloud=store_config.get("cloud", "aws"),
                region=store_config.get("region", "us-east-1")
            )
            await adapter.initialize()
            registry.register_vector_store(name, adapter)
            logger.info(f"Registered Pinecone vector store: {name}")

        elif provider == "qdrant":
            if not QDRANT_AVAILABLE:
                logger.warning(f"Skipping vector store '{name}' - qdrant-client not installed")
                continue

            adapter = QdrantStore(
                collection_name=store_config["collection_name"],
                url=store_config.get("url", "http://localhost:6333"),
                api_key=qdrant_api_key,
                dimension=store_config.get("dimension", 1536)
            )
            await adapter.initialize()
            registry.register_vector_store(name, adapter)
            logger.info(f"Registered Qdrant vector store: {name}")

        elif provider == "weaviate":
            if not WEAVIATE_AVAILABLE:
                logger.warning(f"Skipping vector store '{name}' - weaviate-client not installed")
                continue

            adapter = WeaviateStore(
                class_name=store_config["class_name"],
                url=store_config.get("url", "http://localhost:8080"),
                api_key=weaviate_api_key,
                dimension=store_config.get("dimension", 1536)
            )
            await adapter.initialize()
            registry.register_vector_store(name, adapter)
            logger.info(f"Registered Weaviate vector store: {name}")

        else:
            logger.warning(f"Unknown vector store provider '{provider}' for '{name}'")

    logger.info(
        f"RAG Registry loaded: "
        f"{len(registry.list_embeddings())} embeddings, "
        f"{len(registry.list_chunkers())} chunkers, "
        f"{len(registry.list_vector_stores())} vector stores"
    )

    return registry
