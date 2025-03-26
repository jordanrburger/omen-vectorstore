"""
Vector database integration for the OMEN platform.
"""

from omen.vectorstore.embedding import (
    EmbeddingProvider,
    SentenceTransformerProvider,
    OpenAIProvider,
    get_embedding_provider,
)
from omen.vectorstore.indexer import QdrantIndexer
from omen.vectorstore.models import (
    MetadataType,
    MetadataSource,
    MetadataDocument,
    SearchQuery,
    SearchResult,
)
from omen.vectorstore.processor import MetadataProcessor
from omen.vectorstore.search import VectorSearch
from omen.vectorstore.vectorizer import Vectorizer

__all__ = [
    # Embedding providers
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    "OpenAIProvider",
    "get_embedding_provider",
    
    # Indexer
    "QdrantIndexer",
    
    # Models
    "MetadataType",
    "MetadataSource",
    "MetadataDocument",
    "SearchQuery",
    "SearchResult",
    
    # Processing
    "MetadataProcessor",
    
    # Search
    "VectorSearch",
    
    # Vectorization
    "Vectorizer",
]
