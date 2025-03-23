"""
Core models for the ontology module.

This module defines the basic RDF triple structure and related classes.
"""

import uuid
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, field


class EntityType(str, Enum):
    """Enum for entity types in the Keboola ontology."""
    TABLE = "table"
    COLUMN = "column"
    BUCKET = "bucket"
    CONFIGURATION = "configuration"
    TRANSFORMATION = "transformation"
    BLOCK = "block"
    ORCHESTRATION = "orchestration"
    TASK = "task"
    COMPONENT = "component"
    PROJECT = "project"
    GENERIC = "generic"


class RelationshipType(str, Enum):
    """Enum for relationship types in the Keboola ontology."""
    HAS_COLUMN = "hasColumn"
    BELONGS_TO = "belongsTo"
    DEPENDS_ON = "dependsOn"
    INPUTS_FROM = "inputsFrom"
    OUTPUTS_TO = "outputsTo"
    PART_OF = "partOf"
    LINKED_TO = "linkedTo"
    CREATED_BY = "createdBy"
    TRIGGERS = "triggers"
    CONTAINS = "contains"
    GENERIC = "generic"


@dataclass
class Entity:
    """Represents an entity in the ontology."""
    id: str
    type: EntityType
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_metadata(cls, entity_id: str, entity_type: EntityType, 
                     name: str, metadata: Dict[str, Any]) -> "Entity":
        """Create an Entity instance from metadata."""
        return cls(
            id=entity_id,
            type=entity_type,
            name=name,
            metadata=metadata,
            properties={}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "metadata": self.metadata,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create an Entity instance from a dictionary."""
        return cls(
            id=data["id"],
            type=EntityType(data["type"]),
            name=data["name"],
            metadata=data.get("metadata", {}),
            properties=data.get("properties", {})
        )


@dataclass
class Relationship:
    """Represents a relationship between entities in the ontology."""
    id: str
    type: RelationshipType
    source_id: str
    target_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, rel_type: RelationshipType, source_id: str, 
              target_id: str, metadata: Optional[Dict[str, Any]] = None) -> "Relationship":
        """Create a new relationship with a generated ID."""
        rel_id = f"{rel_type.value}_{source_id}_{target_id}_{str(uuid.uuid4())[:8]}"
        return cls(
            id=rel_id,
            type=rel_type,
            source_id=source_id,
            target_id=target_id,
            metadata=metadata or {},
            properties={}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "metadata": self.metadata,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """Create a Relationship instance from a dictionary."""
        return cls(
            id=data["id"],
            type=RelationshipType(data["type"]),
            source_id=data["source_id"],
            target_id=data["target_id"],
            metadata=data.get("metadata", {}),
            properties=data.get("properties", {})
        )


@dataclass
class Triple:
    """
    Represents an RDF triple in the ontology.
    
    A triple consists of subject, predicate, and object, representing
    the statement "subject predicate object".
    """
    subject: str  # Entity ID
    predicate: str  # Relationship type
    object: str  # Entity ID or literal value
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert triple to dictionary."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Triple":
        """Create a Triple instance from a dictionary."""
        return cls(
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_relationship(cls, relationship: Relationship) -> "Triple":
        """Create a Triple from a Relationship."""
        return cls(
            subject=relationship.source_id,
            predicate=relationship.type.value,
            object=relationship.target_id,
            metadata=relationship.metadata
        )
    
    def __str__(self) -> str:
        """Return string representation of the triple."""
        return f"({self.subject}, {self.predicate}, {self.object})" 