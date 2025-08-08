"""
Vector store package for the OMEN platform.

This package provides functionality for embedding, storing, and searching
metadata using vector representations.
"""

from omen.vectorstore.models import (
    MetadataDocument,
    MetadataSource,
    MetadataType,
    SearchQuery,
    SearchResult,
)
from omen.vectorstore.embedding import (
    EmbeddingProvider,
    SentenceTransformerProvider,
    OpenAIProvider,
    get_embedding_provider,
)
from omen.vectorstore.store import VectorStore
from omen.vectorstore.indexer import QdrantIndexer
from omen.vectorstore.search import VectorSearch
from omen.vectorstore.processor import MetadataProcessor
from omen.vectorstore.vectorizer import Vectorizer
from omen.vectorstore.hybrid_search import HybridSearch

__all__ = [
    "MetadataDocument",
    "MetadataSource",
    "MetadataType",
    "SearchQuery",
    "SearchResult",
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    "OpenAIProvider",
    "get_embedding_provider",
    "VectorStore",
    "QdrantIndexer",
    "VectorSearch",
    "HybridSearch",
    "MetadataProcessor",
    "Vectorizer",
]
