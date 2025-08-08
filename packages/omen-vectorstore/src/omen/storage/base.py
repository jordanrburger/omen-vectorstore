"""
Base abstract classes for OMEN storage backends.

This module defines the interface that all storage backends must implement,
providing a consistent API for both vector and ontology storage.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from uuid import UUID

from omen.vectorstore.models import MetadataDocument, SearchResult, SearchQuery
from omen.ontology.models import Entity, Relationship, Triple


class VectorStoreBackend(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    def store_document(self, document: MetadataDocument) -> str:
        """Store a document with its vector embedding.
        
        Args:
            document: Document to store with embedding
            
        Returns:
            ID of the stored document
        """
        pass

    @abstractmethod
    def store_documents(self, documents: List[MetadataDocument]) -> List[str]:
        """Store multiple documents with their vector embeddings.
        
        Args:
            documents: List of documents to store
            
        Returns:
            List of stored document IDs
        """
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[MetadataDocument]:
        """Retrieve a document by ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if document was deleted, False otherwise
        """
        pass

    @abstractmethod
    def search(
        self,
        query: Union[str, SearchQuery],
        limit: int = 10,
        offset: int = 0,
        **kwargs
    ) -> List[SearchResult]:
        """Search for similar documents.
        
        Args:
            query: Search query (text or SearchQuery object)
            limit: Maximum number of results
            offset: Number of results to skip
            **kwargs: Additional backend-specific parameters
            
        Returns:
            List of search results
        """
        pass

    @abstractmethod
    def update_metadata(
        self,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for a document.
        
        Args:
            document_id: ID of the document
            metadata: New metadata dictionary
            
        Returns:
            True if update was successful, False otherwise
        """
        pass


class OntologyStoreBackend(ABC):
    """Abstract base class for ontology storage backends."""

    @abstractmethod
    def store_entity(self, entity: Entity) -> str:
        """Store an entity in the ontology.
        
        Args:
            entity: Entity to store
            
        Returns:
            ID of the stored entity
        """
        pass

    @abstractmethod
    def store_relationship(self, relationship: Relationship) -> str:
        """Store a relationship in the ontology.
        
        Args:
            relationship: Relationship to store
            
        Returns:
            ID of the stored relationship
        """
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Entity if found, None otherwise
        """
        pass

    @abstractmethod
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Retrieve a relationship by ID.
        
        Args:
            relationship_id: ID of the relationship
            
        Returns:
            Relationship if found, None otherwise
        """
        pass

    @abstractmethod
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if entity was deleted, False otherwise
        """
        pass

    @abstractmethod
    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship.
        
        Args:
            relationship_id: ID of the relationship to delete
            
        Returns:
            True if relationship was deleted, False otherwise
        """
        pass

    @abstractmethod
    def query(
        self,
        query: str,
        query_type: str = "sparql",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute a query against the ontology.
        
        Args:
            query: Query string (SPARQL, Cypher, etc.)
            query_type: Type of query language
            **kwargs: Additional backend-specific parameters
            
        Returns:
            List of query results
        """
        pass

    @abstractmethod
    def get_connected_entities(
        self,
        entity_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "both"
    ) -> List[Tuple[Entity, Relationship]]:
        """Get entities connected to the given entity.
        
        Args:
            entity_id: ID of the source entity
            relationship_types: Optional list of relationship types to filter by
            direction: "in", "out", or "both"
            
        Returns:
            List of (entity, connecting_relationship) tuples
        """
        pass


class UnifiedStorageBackend(VectorStoreBackend, OntologyStoreBackend):
    """Abstract base class for unified vector and ontology storage backends."""

    @abstractmethod
    def semantic_graph_search(
        self,
        query: str,
        start_nodes: Optional[List[str]] = None,
        max_distance: int = 2,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform a combined semantic and graph search.
        
        This method should leverage both vector similarity and graph structure
        to find relevant results.
        
        Args:
            query: Search query text
            start_nodes: Optional list of entity IDs to start search from
            max_distance: Maximum graph distance to traverse
            limit: Maximum number of results
            
        Returns:
            List of results combining semantic and graph relevance
        """
        pass 