"""
Ontology manager for OMEN metadata.

This module provides the main interface for building, querying,
and managing the ontology.
"""

import json
import os
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path

from omen.core import get_logger, AppSettings
from omen.ontology.models import Entity, Relationship, Triple, EntityType, RelationshipType
from omen.ontology.rdf_store import RDFStore

logger = get_logger(__name__)


class OntologyManager:
    """
    Manager for the metadata ontology.
    
    This class provides methods for building and querying the ontology,
    and serves as the main interface for the ontology module.
    """
    
    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize the ontology manager.
        
        Args:
            state_dir: Directory for storing ontology state
        """
        self.state_dir = state_dir or AppSettings.ontology.storage_path
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.entity_types: Set[EntityType] = set()
        self.relationship_types: Set[RelationshipType] = set()
        
        # Initialize TripleStore with storage path
        storage_path = self.state_dir / "ontology.ttl"
        self.triple_store = RDFStore(storage_path=storage_path)
        
        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the ontology.
        
        Args:
            entity: The entity to add
        """
        self.entities[entity.id] = entity
        self.entity_types.add(entity.type)
        self.triple_store.add_entity(entity)
    
    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship to the ontology.
        
        Args:
            relationship: The relationship to add
        """
        self.relationships[relationship.id] = relationship
        self.relationship_types.add(relationship.type)
        self.triple_store.add_relationship(relationship)
    
    def add_triple(self, triple: Triple) -> None:
        """
        Add a triple to the ontology.
        
        Args:
            triple: The triple to add
        """
        self.triple_store.add_triple(triple)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: ID of the entity to retrieve
            
        Returns:
            The entity if found, None otherwise
        """
        return self.entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: Type of entities to retrieve
            
        Returns:
            List of entities
        """
        return [e for e in self.entities.values() if e.type == entity_type]
    
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """
        Get a relationship by ID.
        
        Args:
            relationship_id: ID of the relationship to retrieve
            
        Returns:
            The relationship if found, None otherwise
        """
        return self.relationships.get(relationship_id)
    
    def get_relationships_by_type(self, relationship_type: RelationshipType) -> List[Relationship]:
        """
        Get all relationships of a specific type.
        
        Args:
            relationship_type: Type of relationships to retrieve
            
        Returns:
            List of relationships
        """
        return [r for r in self.relationships.values() if r.type == relationship_type]
    
    def get_relationships_for_entity(
        self, entity_id: str, direction: str = "both"
    ) -> List[Relationship]:
        """
        Get all relationships for a specific entity.
        
        Args:
            entity_id: ID of the entity
            direction: "in" for incoming, "out" for outgoing, "both" for both
            
        Returns:
            List of relationships
        """
        if direction == "out":
            return [r for r in self.relationships.values() if r.source_id == entity_id]
        elif direction == "in":
            return [r for r in self.relationships.values() if r.target_id == entity_id]
        else:
            return [r for r in self.relationships.values() 
                    if r.source_id == entity_id or r.target_id == entity_id]
    
    def get_connected_entities(self, entity_id: str, direction: str = "both") -> List[Entity]:
        """
        Get all entities connected to the given entity.
        
        Args:
            entity_id: ID of the entity
            direction: "in" for incoming, "out" for outgoing, "both" for both
            
        Returns:
            List of connected entities
        """
        relationships = self.get_relationships_for_entity(entity_id, direction)
        
        if direction == "out":
            connected_ids = [r.target_id for r in relationships]
        elif direction == "in":
            connected_ids = [r.source_id for r in relationships]
        else:
            connected_ids = [r.target_id if r.source_id == entity_id else r.source_id 
                             for r in relationships]
        
        return [self.entities[eid] for eid in connected_ids if eid in self.entities]
    
    def entity_exists(self, entity_id: str) -> bool:
        """
        Check if an entity exists.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            True if the entity exists, False otherwise
        """
        return entity_id in self.entities
    
    def relationship_exists(self, relationship_id: str) -> bool:
        """
        Check if a relationship exists.
        
        Args:
            relationship_id: ID of the relationship
            
        Returns:
            True if the relationship exists, False otherwise
        """
        return relationship_id in self.relationships
    
    def query_sparql(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL query on the ontology.
        
        Args:
            query: SPARQL query string
            
        Returns:
            List of results as dictionaries
        """
        return self.triple_store.query_sparql(query)
    
    def save_state(self) -> None:
        """Save the ontology state to disk."""
        try:
            # Save entities
            entities_path = self.state_dir / "entities.json"
            with open(entities_path, "w") as f:
                entities_dict = {
                    key: entity.to_dict() for key, entity in self.entities.items()
                }
                json.dump(entities_dict, f, indent=2)
            
            # Save relationships
            relationships_path = self.state_dir / "relationships.json"
            with open(relationships_path, "w") as f:
                relationships_dict = {
                    key: rel.to_dict() for key, rel in self.relationships.items()
                }
                json.dump(relationships_dict, f, indent=2)
            
            # Save RDF triples using TripleStore
            self.triple_store.save()
            
            logger.info(f"Saved ontology state to {self.state_dir}")
        except Exception as e:
            logger.error(f"Error saving ontology state: {e}")
    
    def load_state(self) -> None:
        """Load the ontology state from disk."""
        try:
            # Load entities
            entities_path = self.state_dir / "entities.json"
            if entities_path.exists():
                with open(entities_path, "r") as f:
                    entities_dict = json.load(f)
                    self.entities = {
                        key: Entity.from_dict(entity_data) 
                        for key, entity_data in entities_dict.items()
                    }
                    self.entity_types = {entity.type for entity in self.entities.values()}
            
            # Load relationships
            relationships_path = self.state_dir / "relationships.json"
            if relationships_path.exists():
                with open(relationships_path, "r") as f:
                    relationships_dict = json.load(f)
                    self.relationships = {
                        key: Relationship.from_dict(rel_data) 
                        for key, rel_data in relationships_dict.items()
                    }
                    self.relationship_types = {rel.type for rel in self.relationships.values()}
            
            # Load RDF triples using TripleStore
            self.triple_store.load()
            
            logger.info(f"Loaded ontology state from {self.state_dir}")
        except Exception as e:
            logger.error(f"Error loading ontology state: {e}")
    
    def clear(self) -> None:
        """Clear the ontology."""
        self.entities = {}
        self.relationships = {}
        self.entity_types = set()
        self.relationship_types = set()
        self.triple_store.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ontology.
        
        Returns:
            Dictionary of statistics
        """
        entity_counts = {}
        for entity_type in self.entity_types:
            count = len(self.get_entities_by_type(entity_type))
            entity_counts[entity_type.value] = count
        
        relationship_counts = {}
        for rel_type in self.relationship_types:
            count = len(self.get_relationships_by_type(rel_type))
            relationship_counts[rel_type.value] = count
        
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_types": entity_counts,
            "relationship_types": relationship_counts,
            "triple_count": len(self.triple_store.get_triples())
        }
    
    def to_networkx(self):
        """
        Convert the ontology to a NetworkX graph.
        
        Returns:
            NetworkX graph
        """
        return self.triple_store.to_networkx()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ontology to a dictionary.
        
        Returns:
            Dictionary representation of the ontology
        """
        return {
            "entities": {
                key: entity.to_dict() for key, entity in self.entities.items()
            },
            "relationships": {
                key: rel.to_dict() for key, rel in self.relationships.items()
            }
        }
    
    def from_dict(self, data: dict) -> None:
        """
        Load ontology data from a dictionary.
        
        Args:
            data: Dictionary representation of the ontology
        """        
        # Clear existing data
        self.clear()
        
        # Load entities
        if "entities" in data:
            for entity_id, entity_data in data["entities"].items():
                entity = Entity.from_dict(entity_data)
                self.entities[entity_id] = entity
            self.entity_types = {entity.type for entity in self.entities.values()}
        
        # Load relationships
        if "relationships" in data:
            for rel_id, rel_data in data["relationships"].items():
                relationship = Relationship.from_dict(rel_data)
                self.relationships[rel_id] = relationship
            self.relationship_types = {rel.type for rel in self.relationships.values()}
        
        # Reload triples into the triple store
        for entity in self.entities.values():
            self.triple_store.add_entity(entity)
            
        for relationship in self.relationships.values():
            self.triple_store.add_relationship(relationship) 