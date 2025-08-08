"""
Storage abstraction layer for OMEN.

This package provides a unified interface for vector and ontology storage,
allowing different backend implementations to be used interchangeably.
"""

from omen.storage.base import (
    VectorStoreBackend,
    OntologyStoreBackend,
    UnifiedStorageBackend,
)
from omen.storage.factory import StorageFactory
from omen.storage.vector.qdrant import QdrantVectorStore
from omen.storage.ontology.rdflib import RDFLibStore

__all__ = [
    # Base classes
    "VectorStoreBackend",
    "OntologyStoreBackend",
    "UnifiedStorageBackend",
    
    # Factory
    "StorageFactory",
    
    # Vector store implementations
    "QdrantVectorStore",
    
    # Ontology store implementations
    "RDFLibStore",
] 