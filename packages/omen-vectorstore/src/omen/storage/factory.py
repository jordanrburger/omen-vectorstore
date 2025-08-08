"""
Factory for creating storage backend instances.
"""

from typing import Optional, Type, Dict, Any
from pathlib import Path

from omen.core import get_logger, load_settings
from omen.storage.base import VectorStoreBackend, OntologyStoreBackend, UnifiedStorageBackend
from omen.storage.vector.qdrant import QdrantVectorStore
from omen.storage.ontology.rdflib import RDFLibStore

logger = get_logger(__name__)
settings = load_settings()


class StorageFactory:
    """Factory for creating storage backend instances."""
    
    # Registry of available backends
    _vector_backends: Dict[str, Type[VectorStoreBackend]] = {
        "qdrant": QdrantVectorStore,
    }
    
    _ontology_backends: Dict[str, Type[OntologyStoreBackend]] = {
        "rdflib": RDFLibStore,
    }
    
    _unified_backends: Dict[str, Type[UnifiedStorageBackend]] = {
        # To be added when implementing unified backends
    }
    
    @classmethod
    def register_vector_backend(
        cls,
        name: str,
        backend_class: Type[VectorStoreBackend]
    ) -> None:
        """Register a new vector store backend.
        
        Args:
            name: Name of the backend
            backend_class: Backend class to register
        """
        cls._vector_backends[name.lower()] = backend_class
        logger.info(f"Registered vector store backend: {name}")
    
    @classmethod
    def register_ontology_backend(
        cls,
        name: str,
        backend_class: Type[OntologyStoreBackend]
    ) -> None:
        """Register a new ontology store backend.
        
        Args:
            name: Name of the backend
            backend_class: Backend class to register
        """
        cls._ontology_backends[name.lower()] = backend_class
        logger.info(f"Registered ontology store backend: {name}")
    
    @classmethod
    def register_unified_backend(
        cls,
        name: str,
        backend_class: Type[UnifiedStorageBackend]
    ) -> None:
        """Register a new unified store backend.
        
        Args:
            name: Name of the backend
            backend_class: Backend class to register
        """
        cls._unified_backends[name.lower()] = backend_class
        logger.info(f"Registered unified store backend: {name}")
    
    @classmethod
    def create_vector_store(
        cls,
        backend_type: str = "qdrant",
        **kwargs
    ) -> VectorStoreBackend:
        """Create a vector store backend instance.
        
        Args:
            backend_type: Type of backend to create
            **kwargs: Backend-specific configuration
            
        Returns:
            Vector store backend instance
            
        Raises:
            ValueError: If backend type is not registered
        """
        backend_class = cls._vector_backends.get(backend_type.lower())
        if not backend_class:
            raise ValueError(
                f"Unknown vector store backend: {backend_type}. "
                f"Available backends: {list(cls._vector_backends.keys())}"
            )
        
        return backend_class(**kwargs)
    
    @classmethod
    def create_ontology_store(
        cls,
        backend_type: str = "rdflib",
        **kwargs
    ) -> OntologyStoreBackend:
        """Create an ontology store backend instance.
        
        Args:
            backend_type: Type of backend to create
            **kwargs: Backend-specific configuration
            
        Returns:
            Ontology store backend instance
            
        Raises:
            ValueError: If backend type is not registered
        """
        backend_class = cls._ontology_backends.get(backend_type.lower())
        if not backend_class:
            raise ValueError(
                f"Unknown ontology store backend: {backend_type}. "
                f"Available backends: {list(cls._ontology_backends.keys())}"
            )
        
        return backend_class(**kwargs)
    
    @classmethod
    def create_unified_store(
        cls,
        backend_type: str,
        **kwargs
    ) -> UnifiedStorageBackend:
        """Create a unified store backend instance.
        
        Args:
            backend_type: Type of backend to create
            **kwargs: Backend-specific configuration
            
        Returns:
            Unified store backend instance
            
        Raises:
            ValueError: If backend type is not registered
        """
        backend_class = cls._unified_backends.get(backend_type.lower())
        if not backend_class:
            raise ValueError(
                f"Unknown unified store backend: {backend_type}. "
                f"Available backends: {list(cls._unified_backends.keys())}"
            )
        
        return backend_class(**kwargs)
    
    @classmethod
    def get_available_backends(cls) -> Dict[str, Dict[str, str]]:
        """Get information about available backends.
        
        Returns:
            Dictionary with backend types and their descriptions
        """
        return {
            "vector": {
                name: backend.__doc__ or "No description available"
                for name, backend in cls._vector_backends.items()
            },
            "ontology": {
                name: backend.__doc__ or "No description available"
                for name, backend in cls._ontology_backends.items()
            },
            "unified": {
                name: backend.__doc__ or "No description available"
                for name, backend in cls._unified_backends.items()
            }
        } 