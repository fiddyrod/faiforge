"""
BM25 keyword search retriever.

Implements the Okapi BM25 algorithm for sparse keyword-based retrieval.
BM25 is a ranking function that considers term frequency, document frequency,
and document length normalization.
"""

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .base import SearchAdapter, ScoredResult

logger = logging.getLogger(__name__)


@dataclass
class BM25Config:
    """Configuration for BM25 retriever."""
    k1: float = 1.5  # Term frequency saturation parameter
    b: float = 0.75  # Document length normalization (0=none, 1=full)
    epsilon: float = 0.25  # Floor for IDF smoothing
    # Tokenization
    lowercase: bool = True
    remove_punctuation: bool = True
    min_token_length: int = 2
    # Stopwords (optional)
    stopwords: Set[str] = field(default_factory=lambda: {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "it", "that", "this",
        "as", "be", "are", "was", "were", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "can", "not", "no", "so", "if", "then", "than",
        "when", "where", "which", "who", "what", "how", "why"
    })


@dataclass
class IndexedDocument:
    """Document stored in the BM25 index."""
    id: str
    content: str
    tokens: List[str]
    token_freqs: Dict[str, int]
    metadata: Dict[str, Any]


class BM25Retriever(SearchAdapter):
    """
    BM25 (Best Matching 25) sparse retriever.

    Implements the Okapi BM25 ranking function for keyword-based retrieval.
    Useful for exact term matching and as part of hybrid search.

    Features:
    - Inverted index for fast term lookup
    - Configurable tokenization and stopwords
    - IDF smoothing to handle rare/unseen terms
    - Document length normalization

    Example:
        retriever = BM25Retriever()
        await retriever.index_documents([
            {"id": "1", "content": "Python is a programming language"},
            {"id": "2", "content": "Java is also a programming language"}
        ])
        results = await retriever.search("python programming", top_k=5)
    """

    def __init__(self, config: Optional[BM25Config] = None):
        """
        Initialize BM25 retriever.

        Args:
            config: Optional BM25 configuration
        """
        self.config = config or BM25Config()

        # Document storage
        self._documents: Dict[str, IndexedDocument] = {}

        # Inverted index: term -> set of doc_ids
        self._inverted_index: Dict[str, Set[str]] = defaultdict(set)

        # Document frequency: term -> number of documents containing term
        self._doc_freq: Dict[str, int] = defaultdict(int)

        # Corpus statistics
        self._total_docs = 0
        self._avg_doc_length = 0.0
        self._total_tokens = 0

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if self.config.lowercase:
            text = text.lower()

        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        tokens = text.split()

        # Filter tokens
        tokens = [
            t for t in tokens
            if len(t) >= self.config.min_token_length
            and t not in self.config.stopwords
        ]

        return tokens

    def _compute_idf(self, term: str) -> float:
        """
        Compute Inverse Document Frequency for a term.

        Uses smoothed IDF: log((N - n + 0.5) / (n + 0.5))

        Args:
            term: The term to compute IDF for

        Returns:
            IDF value
        """
        n = self._doc_freq.get(term, 0)
        N = self._total_docs

        if N == 0:
            return 0.0

        # Standard BM25 IDF with smoothing
        idf = math.log((N - n + 0.5) / (n + 0.5) + 1.0)

        # Apply floor for very common terms
        return max(idf, self.config.epsilon)

    def _score_document(
        self,
        query_tokens: List[str],
        doc: IndexedDocument
    ) -> float:
        """
        Compute BM25 score for a document given query.

        Args:
            query_tokens: Tokenized query terms
            doc: Document to score

        Returns:
            BM25 score
        """
        score = 0.0
        doc_length = len(doc.tokens)

        for term in query_tokens:
            if term not in doc.token_freqs:
                continue

            tf = doc.token_freqs[term]
            idf = self._compute_idf(term)

            # BM25 term score formula
            # score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |D|/avgdl))
            numerator = tf * (self.config.k1 + 1)
            length_norm = 1 - self.config.b + self.config.b * (
                doc_length / self._avg_doc_length if self._avg_doc_length > 0 else 1
            )
            denominator = tf + self.config.k1 * length_norm

            score += idf * (numerator / denominator)

        return score

    async def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Index documents for BM25 search.

        Args:
            documents: List of documents with 'id', 'content', and optional 'metadata'

        Returns:
            Indexing statistics
        """
        indexed_count = 0
        skipped_count = 0

        for doc in documents:
            doc_id = doc.get("id")
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            if not doc_id or not content:
                skipped_count += 1
                continue

            # Tokenize
            tokens = self._tokenize(content)

            # Compute token frequencies
            token_freqs: Dict[str, int] = defaultdict(int)
            for token in tokens:
                token_freqs[token] += 1

            # Create indexed document
            indexed_doc = IndexedDocument(
                id=doc_id,
                content=content,
                tokens=tokens,
                token_freqs=dict(token_freqs),
                metadata=metadata
            )

            # Check if document already exists (update case)
            if doc_id in self._documents:
                old_doc = self._documents[doc_id]
                # Remove old document's contribution to stats
                self._total_tokens -= len(old_doc.tokens)
                for term in old_doc.token_freqs:
                    self._inverted_index[term].discard(doc_id)
                    self._doc_freq[term] = max(0, self._doc_freq[term] - 1)
            else:
                self._total_docs += 1

            # Store document
            self._documents[doc_id] = indexed_doc
            self._total_tokens += len(tokens)

            # Update inverted index and doc freq
            for term in token_freqs:
                self._inverted_index[term].add(doc_id)
                if doc_id not in self._inverted_index[term]:
                    self._doc_freq[term] += 1
                else:
                    # Term was already counted for this doc
                    pass

            # Actually update doc_freq correctly
            for term in set(tokens):
                if doc_id not in self._inverted_index[term]:
                    self._doc_freq[term] += 1
                self._inverted_index[term].add(doc_id)

            indexed_count += 1

        # Update average document length
        self._avg_doc_length = (
            self._total_tokens / self._total_docs
            if self._total_docs > 0 else 0
        )

        logger.info(
            f"BM25 indexed {indexed_count} documents, "
            f"skipped {skipped_count}, "
            f"total {self._total_docs} docs, "
            f"vocabulary size {len(self._inverted_index)}"
        )

        return {
            "indexed_count": indexed_count,
            "skipped_count": skipped_count,
            "total_documents": self._total_docs,
            "vocabulary_size": len(self._inverted_index),
            "avg_document_length": round(self._avg_doc_length, 2)
        }

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredResult]:
        """
        Search documents using BM25.

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of scored results ordered by BM25 score
        """
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Find candidate documents (any doc containing at least one query term)
        candidate_ids: Set[str] = set()
        for term in query_tokens:
            candidate_ids.update(self._inverted_index.get(term, set()))

        if not candidate_ids:
            return []

        # Score candidates
        scored_docs: List[tuple] = []
        for doc_id in candidate_ids:
            doc = self._documents[doc_id]

            # Apply metadata filters
            if filters:
                match = all(
                    doc.metadata.get(k) == v
                    for k, v in filters.items()
                )
                if not match:
                    continue

            score = self._score_document(query_tokens, doc)
            scored_docs.append((doc_id, doc, score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[2], reverse=True)

        # Return top_k results
        results = []
        for doc_id, doc, score in scored_docs[:top_k]:
            results.append(ScoredResult(
                id=doc_id,
                content=doc.content,
                score=score,
                metadata=doc.metadata,
                bm25_score=score
            ))

        return results

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the index.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if document was found and removed
        """
        if doc_id not in self._documents:
            return False

        doc = self._documents[doc_id]

        # Update statistics
        self._total_docs -= 1
        self._total_tokens -= len(doc.tokens)

        # Update inverted index and doc freq
        for term in doc.token_freqs:
            self._inverted_index[term].discard(doc_id)
            self._doc_freq[term] = max(0, self._doc_freq[term] - 1)
            # Clean up empty terms
            if not self._inverted_index[term]:
                del self._inverted_index[term]
                del self._doc_freq[term]

        # Remove document
        del self._documents[doc_id]

        # Update average
        self._avg_doc_length = (
            self._total_tokens / self._total_docs
            if self._total_docs > 0 else 0
        )

        return True

    def clear(self) -> None:
        """Clear all indexed documents."""
        self._documents.clear()
        self._inverted_index.clear()
        self._doc_freq.clear()
        self._total_docs = 0
        self._avg_doc_length = 0.0
        self._total_tokens = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_documents": self._total_docs,
            "vocabulary_size": len(self._inverted_index),
            "total_tokens": self._total_tokens,
            "avg_document_length": round(self._avg_doc_length, 2),
            "config": {
                "k1": self.config.k1,
                "b": self.config.b,
                "stopwords_count": len(self.config.stopwords)
            }
        }
