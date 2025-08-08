"""
Models for the OMEN ontology system.
"""
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from pydantic import BaseModel, Field

from omen.core.utils import generate_uuid


class EntityType(str, Enum):
    """Type of entity in the ontology."""
    TABLE = "table"
    COLUMN = "column"
    BUCKET = "bucket"
    CONFIGURATION = "configuration"
    TRANSFORMATION = "transformation"
    BLOCK = "block"
    COMPONENT = "component"
    ORCHESTRATION = "orchestration"
    JOB = "job"
    TOKEN = "token"
    USER = "user"
    BRANCH = "branch"
    WORKSPACE = "workspace"
    ORGANIZATION = "organization"
    PROJECT = "project"
    CUSTOM = "custom"


class RelationshipType(str, Enum):
    """Type of relationship in the ontology."""
    HAS_COLUMN = "hasColumn"
    BELONGS_TO = "belongsTo"
    DEPENDS_ON = "dependsOn"
    INPUTS_FROM = "inputsFrom"
    OUTPUTS_TO = "outputsTo"
    CREATED_BY = "createdBy"
    MODIFIED_BY = "modifiedBy"
    REFERENCES = "references"
    RELATED_TO = "relatedTo"
    PART_OF = "partOf"
    CONTAINS = "contains"
    TRIGGERS = "triggers"
    TRIGGERED_BY = "triggeredBy"
    CUSTOM = "custom"


class Entity(BaseModel):
    """An entity in the ontology."""
    
    id: str = Field(default_factory=generate_uuid, description="Unique identifier")
    name: str = Field(..., description="Entity name")
    type: EntityType = Field(..., description="Entity type")
    description: Optional[str] = Field(None, description="Optional description")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create entity from dictionary representation."""
        entity_type = data.get("type")
        if isinstance(entity_type, str):
            data["type"] = EntityType(entity_type)
        
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            data["created_at"] = datetime.fromisoformat(created_at)
            
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            data["updated_at"] = datetime.fromisoformat(updated_at)
            
        return cls(**data)


class Relationship(BaseModel):
    """A relationship between entities in the ontology."""
    
    id: str = Field(default_factory=generate_uuid, description="Unique identifier")
    type: RelationshipType = Field(..., description="Relationship type")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """Create relationship from dictionary representation."""
        rel_type = data.get("type")
        if isinstance(rel_type, str):
            data["type"] = RelationshipType(rel_type)
            
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            data["created_at"] = datetime.fromisoformat(created_at)
            
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            data["updated_at"] = datetime.fromisoformat(updated_at)
            
        return cls(**data)


class Triple(BaseModel):
    """A semantic triple (subject-predicate-object) in the ontology."""
    
    subject: str = Field(..., description="Subject entity ID")
    predicate: str = Field(..., description="Predicate (relationship type)")
    object: str = Field(..., description="Object entity ID")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class PropertyDefinition(BaseModel):
    """Definition of a property for an entity type."""
    
    name: str = Field(..., description="Property name")
    description: str = Field(..., description="Property description")
    data_type: str = Field(..., description="Data type")
    required: bool = Field(False, description="Whether the property is required")
    default_value: Optional[Any] = Field(None, description="Default value")
    enum_values: Optional[List[str]] = Field(None, description="Allowed values for enum types")


class EntityTypeDefinition(BaseModel):
    """Definition of an entity type in the ontology schema."""
    
    type: EntityType = Field(..., description="Entity type")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Description")
    properties: Dict[str, PropertyDefinition] = Field(..., description="Property definitions")
    required_properties: List[str] = Field(default_factory=list, description="Required properties")


class RelationshipTypeDefinition(BaseModel):
    """Definition of a relationship type in the ontology schema."""
    
    type: RelationshipType = Field(..., description="Relationship type")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Description")
    source_types: List[EntityType] = Field(..., description="Valid source entity types")
    target_types: List[EntityType] = Field(..., description="Valid target entity types")
    properties: Dict[str, PropertyDefinition] = Field(default_factory=dict, description="Property definitions")
    required_properties: List[str] = Field(default_factory=list, description="Required properties")
    is_directed: bool = Field(True, description="Whether the relationship is directed") 