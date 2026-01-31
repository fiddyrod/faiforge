from .base import EmbeddingAdapter, EmbeddingResult

# Always available
from .openai_embedding import OpenAIEmbedding

# Optional HuggingFace import (requires sentence-transformers)
try:
    from .huggingface_embedding import HuggingFaceEmbedding
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HuggingFaceEmbedding = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

__all__ = [
    "EmbeddingAdapter",
    "EmbeddingResult",
    "OpenAIEmbedding",
    "HuggingFaceEmbedding",
    "SENTENCE_TRANSFORMERS_AVAILABLE"
]
