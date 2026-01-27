import time
from typing import List, Dict, Any, Optional
import uuid

from .embeddings import EmbeddingAdapter
from .chunking import ChunkingAdapter, Chunk
from .vector_stores import VectorStoreAdapter, Document, SearchResponse
from ..observability import get_logger, log_with_context


class RAGPipeline:
    """Orchestrates end-to-end RAG operations"""

    def __init__(
        self,
        embedding_adapter: EmbeddingAdapter,
        chunking_adapter: ChunkingAdapter,
        vector_store_adapter: VectorStoreAdapter
    ):
        """
        Initialize RAG pipeline.

        Args:
            embedding_adapter: Adapter for generating embeddings
            chunking_adapter: Adapter for chunking documents
            vector_store_adapter: Adapter for vector storage/retrieval
        """
        self.embedding_adapter = embedding_adapter
        self.chunking_adapter = chunking_adapter
        self.vector_store = vector_store_adapter
        self.logger = get_logger()

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

            total_latency_ms = (time.time() - start_time) * 1000

            result = {
                "documents_processed": len(documents),
                "chunks_created": len(all_chunks),
                "embeddings_generated": len(embeddings),
                "vector_store_documents": storage_result.get("documents_added", 0),
                "total_latency_ms": round(total_latency_ms, 2),
                "chunking_latency_ms": round(chunking_latency_ms, 2),
                "embedding_latency_ms": round(embedding_latency_ms, 2),
                "storage_latency_ms": round(storage_latency_ms, 2)
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
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """
        Query the RAG system.

        Workflow:
        1. Embed query text
        2. Search vector store with query embedding
        3. Return results with similarity scores

        Args:
            query_text: Natural language query
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            SearchResponse with results, scores, and metadata
        """
        start_time = time.time()

        log_with_context(
            self.logger,
            "info",
            "Starting RAG query",
            event="rag_query_start",
            query_length=len(query_text),
            top_k=top_k,
            has_filters=filters is not None
        )

        try:
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
                "RAG query completed successfully",
                event="rag_query_complete",
                results_found=len(search_response.results),
                total_latency_ms=round(total_latency_ms, 2),
                embedding_latency_ms=round(embedding_latency_ms, 2),
                search_latency_ms=round(search_latency_ms, 2)
            )

            return search_response

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
        """
        try:
            vector_store_info = await self.vector_store.get_collection_info()

            return {
                "embedding_model": self.embedding_adapter.model_name,
                "embedding_dimensions": self.embedding_adapter.dimensions,
                "chunking_strategy": self.chunking_adapter.strategy.value,
                "vector_store": vector_store_info
            }

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
