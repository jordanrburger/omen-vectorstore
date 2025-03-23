"""
RDF Schema and ontology structure for Keboola metadata.

This module defines the formal schema for the ontology, including
entity and relationship types, property definitions, and validation rules.
"""

import logging
import re
import uuid
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from app.ontology.models import EntityType, RelationshipType, Entity, Relationship, Triple

logger = logging.getLogger(__name__)


@dataclass
class PropertyDefinition:
    """Definition of a property in the ontology schema."""
    name: str
    description: str
    data_type: str  # string, integer, float, boolean, date, datetime
    required: bool = False
    default_value: Optional[Any] = None
    enum_values: Optional[List[Any]] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    pattern: Optional[str] = None
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against this property definition.
        
        Args:
            value: The value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if required
        if self.required and value is None:
            return False, f"Property '{self.name}' is required"
        
        # If not required and None, it's valid
        if not self.required and value is None:
            return True, None
        
        # Check data type
        if self.data_type == "string":
            if not isinstance(value, str):
                return False, f"Property '{self.name}' must be a string"
            if self.pattern and not re.match(self.pattern, value):
                return False, f"Property '{self.name}' must match pattern {self.pattern}"
        elif self.data_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"Property '{self.name}' must be an integer"
            if self.min_value is not None and value < self.min_value:
                return False, f"Property '{self.name}' must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Property '{self.name}' must be <= {self.max_value}"
        elif self.data_type == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"Property '{self.name}' must be a number"
            if self.min_value is not None and value < self.min_value:
                return False, f"Property '{self.name}' must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Property '{self.name}' must be <= {self.max_value}"
        elif self.data_type == "boolean":
            if not isinstance(value, bool):
                return False, f"Property '{self.name}' must be a boolean"
        elif self.data_type == "date" or self.data_type == "datetime":
            if not isinstance(value, str):
                return False, f"Property '{self.name}' must be a string in ISO format"
            try:
                if self.data_type == "date":
                    datetime.fromisoformat(value).date()
                else:
                    datetime.fromisoformat(value)
            except ValueError:
                return False, f"Property '{self.name}' must be a valid ISO date/datetime"
        
        # Check enum values
        if self.enum_values is not None and value not in self.enum_values:
            return False, f"Property '{self.name}' must be one of {self.enum_values}"
        
        return True, None


@dataclass
class EntityTypeDefinition:
    """Definition of an entity type in the ontology schema."""
    type: EntityType
    name: str
    description: str
    properties: Dict[str, PropertyDefinition] = field(default_factory=dict)
    required_properties: List[str] = field(default_factory=list)
    allowed_relationships: Dict[RelationshipType, List[EntityType]] = field(default_factory=dict)
    
    def validate_entity(self, entity: Entity) -> Tuple[bool, List[str]]:
        """
        Validate an entity against this definition.
        
        Args:
            entity: The entity to validate
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Check entity type
        if entity.type != self.type:
            errors.append(f"Entity type mismatch: expected {self.type}, got {entity.type}")
            return False, errors
        
        # Check required properties
        for prop_name in self.required_properties:
            if prop_name not in entity.properties:
                errors.append(f"Required property '{prop_name}' is missing")
        
        # Validate properties
        for prop_name, prop_value in entity.properties.items():
            if prop_name in self.properties:
                is_valid, error = self.properties[prop_name].validate(prop_value)
                if not is_valid:
                    errors.append(error)
        
        return len(errors) == 0, errors


@dataclass
class RelationshipTypeDefinition:
    """Definition of a relationship type in the ontology schema."""
    type: RelationshipType
    name: str
    description: str
    source_types: List[EntityType]
    target_types: List[EntityType]
    properties: Dict[str, PropertyDefinition] = field(default_factory=dict)
    required_properties: List[str] = field(default_factory=list)
    cardinality: str = "many-to-many"  # one-to-one, one-to-many, many-to-one, many-to-many
    
    def validate_relationship(self, relationship: Relationship, entities: Dict[str, Entity]) -> Tuple[bool, List[str]]:
        """
        Validate a relationship against this definition.
        
        Args:
            relationship: The relationship to validate
            entities: Dictionary of entity_id -> Entity
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Check relationship type
        if relationship.type != self.type:
            errors.append(f"Relationship type mismatch: expected {self.type}, got {relationship.type}")
            return False, errors
        
        # Check source and target entities exist
        source_entity = entities.get(relationship.source_id)
        target_entity = entities.get(relationship.target_id)
        
        if not source_entity:
            errors.append(f"Source entity '{relationship.source_id}' does not exist")
        if not target_entity:
            errors.append(f"Target entity '{relationship.target_id}' does not exist")
        
        if not source_entity or not target_entity:
            return False, errors
        
        # Check source entity type
        if source_entity.type not in self.source_types:
            errors.append(f"Source entity type '{source_entity.type}' is not allowed for relationship {self.type}")
        
        # Check target entity type
        if target_entity.type not in self.target_types:
            errors.append(f"Target entity type '{target_entity.type}' is not allowed for relationship {self.type}")
        
        # Check required properties
        for prop_name in self.required_properties:
            if prop_name not in relationship.properties:
                errors.append(f"Required property '{prop_name}' is missing")
        
        # Validate properties
        for prop_name, prop_value in relationship.properties.items():
            if prop_name in self.properties:
                is_valid, error = self.properties[prop_name].validate(prop_value)
                if not is_valid:
                    errors.append(error)
        
        return len(errors) == 0, errors


class KeboolaOntologySchema:
    """Schema definition for the Keboola metadata ontology."""
    
    def __init__(self):
        """Initialize the ontology schema with entity and relationship type definitions."""
        self.entity_types: Dict[EntityType, EntityTypeDefinition] = {}
        self.relationship_types: Dict[RelationshipType, RelationshipTypeDefinition] = {}
        self._define_schema()
    
    def _define_schema(self):
        """Define the ontology schema."""
        # Define common properties
        name_prop = PropertyDefinition(
            name="name",
            description="Human-readable name",
            data_type="string",
            required=True
        )
        
        description_prop = PropertyDefinition(
            name="description",
            description="Human-readable description",
            data_type="string",
            required=False
        )
        
        created_at_prop = PropertyDefinition(
            name="created_at",
            description="Creation timestamp",
            data_type="datetime",
            required=False
        )
        
        updated_at_prop = PropertyDefinition(
            name="updated_at",
            description="Last update timestamp",
            data_type="datetime",
            required=False
        )
        
        # Define entity types
        
        # Project
        project_def = EntityTypeDefinition(
            type=EntityType.PROJECT,
            name="Project",
            description="Keboola project containing components, data, and configurations",
            properties={
                "name": name_prop,
                "description": description_prop,
                "project_id": PropertyDefinition(
                    name="project_id",
                    description="Unique project identifier",
                    data_type="string",
                    required=True
                ),
                "created_at": created_at_prop,
                "updated_at": updated_at_prop,
                "region": PropertyDefinition(
                    name="region",
                    description="Region where the project is hosted",
                    data_type="string",
                    required=False
                )
            },
            required_properties=["name", "project_id"]
        )
        self.entity_types[EntityType.PROJECT] = project_def
        
        # Bucket
        bucket_def = EntityTypeDefinition(
            type=EntityType.BUCKET,
            name="Bucket",
            description="Storage bucket containing tables",
            properties={
                "name": name_prop,
                "description": description_prop,
                "bucket_id": PropertyDefinition(
                    name="bucket_id",
                    description="Unique bucket identifier",
                    data_type="string",
                    required=True
                ),
                "stage": PropertyDefinition(
                    name="stage",
                    description="Storage stage (in/out)",
                    data_type="string",
                    required=True,
                    enum_values=["in", "out"]
                ),
                "created_at": created_at_prop,
                "updated_at": updated_at_prop
            },
            required_properties=["name", "bucket_id", "stage"]
        )
        self.entity_types[EntityType.BUCKET] = bucket_def
        
        # Table
        table_def = EntityTypeDefinition(
            type=EntityType.TABLE,
            name="Table",
            description="Storage table containing data",
            properties={
                "name": name_prop,
                "description": description_prop,
                "table_id": PropertyDefinition(
                    name="table_id",
                    description="Unique table identifier",
                    data_type="string",
                    required=True
                ),
                "primary_key": PropertyDefinition(
                    name="primary_key",
                    description="List of primary key columns",
                    data_type="string",  # Serialized JSON array
                    required=False
                ),
                "row_count": PropertyDefinition(
                    name="row_count",
                    description="Number of rows in the table",
                    data_type="integer",
                    required=False,
                    min_value=0
                ),
                "data_size_bytes": PropertyDefinition(
                    name="data_size_bytes",
                    description="Size of the table data in bytes",
                    data_type="integer",
                    required=False,
                    min_value=0
                ),
                "is_alias": PropertyDefinition(
                    name="is_alias",
                    description="Whether the table is an alias",
                    data_type="boolean",
                    required=False
                ),
                "created_at": created_at_prop,
                "updated_at": updated_at_prop
            },
            required_properties=["name", "table_id"]
        )
        self.entity_types[EntityType.TABLE] = table_def
        
        # Column
        column_def = EntityTypeDefinition(
            type=EntityType.COLUMN,
            name="Column",
            description="Table column",
            properties={
                "name": name_prop,
                "description": description_prop,
                "data_type": PropertyDefinition(
                    name="data_type",
                    description="Data type of the column",
                    data_type="string",
                    required=False
                ),
                "nullable": PropertyDefinition(
                    name="nullable",
                    description="Whether the column allows null values",
                    data_type="boolean",
                    required=False
                ),
                "is_primary_key": PropertyDefinition(
                    name="is_primary_key",
                    description="Whether the column is part of the primary key",
                    data_type="boolean",
                    required=False
                ),
                "statistics": PropertyDefinition(
                    name="statistics",
                    description="Column statistics (min, max, avg, etc.)",
                    data_type="string",  # Serialized JSON object
                    required=False
                ),
                "data_quality": PropertyDefinition(
                    name="data_quality",
                    description="Data quality metrics",
                    data_type="string",  # Serialized JSON object
                    required=False
                )
            },
            required_properties=["name"]
        )
        self.entity_types[EntityType.COLUMN] = column_def
        
        # Component
        component_def = EntityTypeDefinition(
            type=EntityType.COMPONENT,
            name="Component",
            description="Keboola component (extractor, writer, transformation, etc.)",
            properties={
                "name": name_prop,
                "description": description_prop,
                "component_id": PropertyDefinition(
                    name="component_id",
                    description="Unique component identifier",
                    data_type="string",
                    required=True
                ),
                "component_type": PropertyDefinition(
                    name="component_type",
                    description="Type of component",
                    data_type="string",
                    required=True,
                    enum_values=["extractor", "writer", "transformation", "application", "other"]
                ),
                "vendor": PropertyDefinition(
                    name="vendor",
                    description="Component vendor",
                    data_type="string",
                    required=False
                )
            },
            required_properties=["name", "component_id", "component_type"]
        )
        self.entity_types[EntityType.COMPONENT] = component_def
        
        # Configuration
        config_def = EntityTypeDefinition(
            type=EntityType.CONFIGURATION,
            name="Configuration",
            description="Component configuration",
            properties={
                "name": name_prop,
                "description": description_prop,
                "config_id": PropertyDefinition(
                    name="config_id",
                    description="Unique configuration identifier",
                    data_type="string",
                    required=True
                ),
                "version": PropertyDefinition(
                    name="version",
                    description="Configuration version",
                    data_type="integer",
                    required=False,
                    min_value=1
                ),
                "created_at": created_at_prop,
                "updated_at": updated_at_prop,
                "is_disabled": PropertyDefinition(
                    name="is_disabled",
                    description="Whether the configuration is disabled",
                    data_type="boolean",
                    required=False
                )
            },
            required_properties=["name", "config_id"]
        )
        self.entity_types[EntityType.CONFIGURATION] = config_def
        
        # Transformation
        transformation_def = EntityTypeDefinition(
            type=EntityType.TRANSFORMATION,
            name="Transformation",
            description="Data transformation",
            properties={
                "name": name_prop,
                "description": description_prop,
                "transformation_id": PropertyDefinition(
                    name="transformation_id",
                    description="Unique transformation identifier",
                    data_type="string",
                    required=True
                ),
                "type": PropertyDefinition(
                    name="type",
                    description="Transformation type",
                    data_type="string",
                    required=True,
                    enum_values=["python", "r", "sql"]
                ),
                "backend": PropertyDefinition(
                    name="backend",
                    description="Transformation backend",
                    data_type="string",
                    required=False,
                    enum_values=["snowflake", "redshift", "synapse", "bigquery"]
                ),
                "created_at": created_at_prop,
                "updated_at": updated_at_prop
            },
            required_properties=["name", "transformation_id", "type"]
        )
        self.entity_types[EntityType.TRANSFORMATION] = transformation_def
        
        # Block
        block_def = EntityTypeDefinition(
            type=EntityType.BLOCK,
            name="Block",
            description="Transformation code block",
            properties={
                "name": name_prop,
                "description": description_prop,
                "block_id": PropertyDefinition(
                    name="block_id",
                    description="Unique block identifier",
                    data_type="string",
                    required=True
                ),
                "code": PropertyDefinition(
                    name="code",
                    description="Block code",
                    data_type="string",
                    required=True
                ),
                "order": PropertyDefinition(
                    name="order",
                    description="Block execution order",
                    data_type="integer",
                    required=False,
                    min_value=0
                )
            },
            required_properties=["name", "block_id", "code"]
        )
        self.entity_types[EntityType.BLOCK] = block_def
        
        # Orchestration
        orchestration_def = EntityTypeDefinition(
            type=EntityType.ORCHESTRATION,
            name="Orchestration",
            description="Workflow orchestration",
            properties={
                "name": name_prop,
                "description": description_prop,
                "orchestration_id": PropertyDefinition(
                    name="orchestration_id",
                    description="Unique orchestration identifier",
                    data_type="string",
                    required=True
                ),
                "schedule": PropertyDefinition(
                    name="schedule",
                    description="Orchestration schedule (cron expression)",
                    data_type="string",
                    required=False
                ),
                "is_enabled": PropertyDefinition(
                    name="is_enabled",
                    description="Whether the orchestration is enabled",
                    data_type="boolean",
                    required=False
                ),
                "created_at": created_at_prop,
                "updated_at": updated_at_prop
            },
            required_properties=["name", "orchestration_id"]
        )
        self.entity_types[EntityType.ORCHESTRATION] = orchestration_def
        
        # Task
        task_def = EntityTypeDefinition(
            type=EntityType.TASK,
            name="Task",
            description="Orchestration task",
            properties={
                "name": name_prop,
                "description": description_prop,
                "task_id": PropertyDefinition(
                    name="task_id",
                    description="Unique task identifier",
                    data_type="string",
                    required=True
                ),
                "phase": PropertyDefinition(
                    name="phase",
                    description="Task execution phase",
                    data_type="integer",
                    required=False,
                    min_value=0
                ),
                "is_active": PropertyDefinition(
                    name="is_active",
                    description="Whether the task is active",
                    data_type="boolean",
                    required=False
                ),
                "continue_on_failure": PropertyDefinition(
                    name="continue_on_failure",
                    description="Whether to continue execution on task failure",
                    data_type="boolean",
                    required=False
                )
            },
            required_properties=["name", "task_id"]
        )
        self.entity_types[EntityType.TASK] = task_def
        
        # Define relationship types
        
        # Project contains Bucket
        project_contains_bucket = RelationshipTypeDefinition(
            type=RelationshipType.CONTAINS,
            name="Contains",
            description="Project contains Bucket",
            source_types=[EntityType.PROJECT],
            target_types=[EntityType.BUCKET],
            cardinality="one-to-many"
        )
        self.relationship_types[RelationshipType.CONTAINS] = project_contains_bucket
        
        # Bucket contains Table
        bucket_contains_table = RelationshipTypeDefinition(
            type=RelationshipType.CONTAINS,
            name="Contains",
            description="Bucket contains Table",
            source_types=[EntityType.BUCKET],
            target_types=[EntityType.TABLE],
            cardinality="one-to-many"
        )
        self.relationship_types[RelationshipType.CONTAINS] = bucket_contains_table
        
        # Table has Column
        table_has_column = RelationshipTypeDefinition(
            type=RelationshipType.HAS_COLUMN,
            name="HasColumn",
            description="Table has Column",
            source_types=[EntityType.TABLE],
            target_types=[EntityType.COLUMN],
            cardinality="one-to-many"
        )
        self.relationship_types[RelationshipType.HAS_COLUMN] = table_has_column
        
        # Column belongs to Table
        column_belongs_to_table = RelationshipTypeDefinition(
            type=RelationshipType.BELONGS_TO,
            name="BelongsTo",
            description="Column belongs to Table",
            source_types=[EntityType.COLUMN],
            target_types=[EntityType.TABLE],
            cardinality="many-to-one"
        )
        self.relationship_types[RelationshipType.BELONGS_TO] = column_belongs_to_table
        
        # Transformation inputs from Table
        transformation_inputs_from_table = RelationshipTypeDefinition(
            type=RelationshipType.INPUTS_FROM,
            name="InputsFrom",
            description="Transformation inputs from Table",
            source_types=[EntityType.TRANSFORMATION],
            target_types=[EntityType.TABLE],
            cardinality="many-to-many"
        )
        self.relationship_types[RelationshipType.INPUTS_FROM] = transformation_inputs_from_table
        
        # Transformation outputs to Table
        transformation_outputs_to_table = RelationshipTypeDefinition(
            type=RelationshipType.OUTPUTS_TO,
            name="OutputsTo",
            description="Transformation outputs to Table",
            source_types=[EntityType.TRANSFORMATION],
            target_types=[EntityType.TABLE],
            cardinality="many-to-many"
        )
        self.relationship_types[RelationshipType.OUTPUTS_TO] = transformation_outputs_to_table
        
        # Component has Configuration
        component_has_configuration = RelationshipTypeDefinition(
            type=RelationshipType.CONTAINS,
            name="Contains",
            description="Component has Configuration",
            source_types=[EntityType.COMPONENT],
            target_types=[EntityType.CONFIGURATION],
            cardinality="one-to-many"
        )
        self.relationship_types[RelationshipType.CONTAINS] = component_has_configuration
        
        # Configuration belongs to Component
        configuration_belongs_to_component = RelationshipTypeDefinition(
            type=RelationshipType.BELONGS_TO,
            name="BelongsTo",
            description="Configuration belongs to Component",
            source_types=[EntityType.CONFIGURATION],
            target_types=[EntityType.COMPONENT],
            cardinality="many-to-one"
        )
        self.relationship_types[RelationshipType.BELONGS_TO] = configuration_belongs_to_component
        
        # Transformation contains Block
        transformation_contains_block = RelationshipTypeDefinition(
            type=RelationshipType.CONTAINS,
            name="Contains",
            description="Transformation contains Block",
            source_types=[EntityType.TRANSFORMATION],
            target_types=[EntityType.BLOCK],
            cardinality="one-to-many"
        )
        self.relationship_types[RelationshipType.CONTAINS] = transformation_contains_block
        
        # Block part of Transformation
        block_part_of_transformation = RelationshipTypeDefinition(
            type=RelationshipType.PART_OF,
            name="PartOf",
            description="Block is part of Transformation",
            source_types=[EntityType.BLOCK],
            target_types=[EntityType.TRANSFORMATION],
            cardinality="many-to-one"
        )
        self.relationship_types[RelationshipType.PART_OF] = block_part_of_transformation
        
        # Block depends on Block
        block_depends_on_block = RelationshipTypeDefinition(
            type=RelationshipType.DEPENDS_ON,
            name="DependsOn",
            description="Block depends on Block",
            source_types=[EntityType.BLOCK],
            target_types=[EntityType.BLOCK],
            cardinality="many-to-many"
        )
        self.relationship_types[RelationshipType.DEPENDS_ON] = block_depends_on_block
        
        # Orchestration contains Task
        orchestration_contains_task = RelationshipTypeDefinition(
            type=RelationshipType.CONTAINS,
            name="Contains",
            description="Orchestration contains Task",
            source_types=[EntityType.ORCHESTRATION],
            target_types=[EntityType.TASK],
            cardinality="one-to-many"
        )
        self.relationship_types[RelationshipType.CONTAINS] = orchestration_contains_task
        
        # Task part of Orchestration
        task_part_of_orchestration = RelationshipTypeDefinition(
            type=RelationshipType.PART_OF,
            name="PartOf",
            description="Task is part of Orchestration",
            source_types=[EntityType.TASK],
            target_types=[EntityType.ORCHESTRATION],
            cardinality="many-to-one"
        )
        self.relationship_types[RelationshipType.PART_OF] = task_part_of_orchestration
        
        # Task triggers Configuration
        task_triggers_configuration = RelationshipTypeDefinition(
            type=RelationshipType.TRIGGERS,
            name="Triggers",
            description="Task triggers Configuration",
            source_types=[EntityType.TASK],
            target_types=[EntityType.CONFIGURATION],
            cardinality="many-to-many"
        )
        self.relationship_types[RelationshipType.TRIGGERS] = task_triggers_configuration
        
        # Task depends on Task
        task_depends_on_task = RelationshipTypeDefinition(
            type=RelationshipType.DEPENDS_ON,
            name="DependsOn",
            description="Task depends on Task",
            source_types=[EntityType.TASK],
            target_types=[EntityType.TASK],
            cardinality="many-to-many"
        )
        self.relationship_types[RelationshipType.DEPENDS_ON] = task_depends_on_task
        
        # Table linked to Table (for aliases)
        table_linked_to_table = RelationshipTypeDefinition(
            type=RelationshipType.LINKED_TO,
            name="LinkedTo",
            description="Table is linked to Table (for aliases)",
            source_types=[EntityType.TABLE],
            target_types=[EntityType.TABLE],
            cardinality="many-to-one"
        )
        self.relationship_types[RelationshipType.LINKED_TO] = table_linked_to_table
        
        # Table created by Transformation
        table_created_by_transformation = RelationshipTypeDefinition(
            type=RelationshipType.CREATED_BY,
            name="CreatedBy",
            description="Table created by Transformation",
            source_types=[EntityType.TABLE],
            target_types=[EntityType.TRANSFORMATION],
            cardinality="many-to-one"
        )
        self.relationship_types[RelationshipType.CREATED_BY] = table_created_by_transformation
    
    def validate_entity(self, entity: Entity) -> Tuple[bool, List[str]]:
        """
        Validate an entity against the schema.
        
        Args:
            entity: The entity to validate
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        if entity.type not in self.entity_types:
            return False, [f"Unknown entity type: {entity.type}"]
        
        entity_def = self.entity_types[entity.type]
        return entity_def.validate_entity(entity)
    
    def validate_relationship(self, relationship: Relationship, entities: Dict[str, Entity]) -> Tuple[bool, List[str]]:
        """
        Validate a relationship against the schema.
        
        Args:
            relationship: The relationship to validate
            entities: Dictionary of entity_id -> Entity
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        if relationship.type not in self.relationship_types:
            return False, [f"Unknown relationship type: {relationship.type}"]
        
        rel_def = self.relationship_types[relationship.type]
        return rel_def.validate_relationship(relationship, entities)
    
    def validate_triple(self, triple: Triple, entities: Dict[str, Entity]) -> Tuple[bool, List[str]]:
        """
        Validate a triple against the schema.
        
        Args:
            triple: The triple to validate
            entities: Dictionary of entity_id -> Entity
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Check subject exists
        if triple.subject not in entities:
            errors.append(f"Subject entity '{triple.subject}' does not exist")
            return False, errors
        
        # Check if predicate is a valid relationship type
        try:
            rel_type = RelationshipType(triple.predicate)
        except ValueError:
            errors.append(f"Unknown predicate type: {triple.predicate}")
            return False, errors
        
        # Check object exists if it's an entity ID
        if triple.object in entities:
            # Create a temporary relationship to validate
            temp_rel = Relationship(
                id=f"temp_{uuid.uuid4()}",
                type=rel_type,
                source_id=triple.subject,
                target_id=triple.object,
                metadata=triple.metadata
            )
            
            # Validate the relationship
            is_valid, rel_errors = self.validate_relationship(temp_rel, entities)
            if not is_valid:
                errors.extend(rel_errors)
        
        return len(errors) == 0, errors
    
    def get_allowed_relationships(self, entity_type: EntityType) -> Dict[RelationshipType, List[EntityType]]:
        """
        Get allowed relationships for an entity type.
        
        Args:
            entity_type: The entity type
            
        Returns:
            Dictionary of relationship_type -> list of target entity types
        """
        if entity_type not in self.entity_types:
            return {}
        
        allowed = {}
        for rel_type, rel_def in self.relationship_types.items():
            if entity_type in rel_def.source_types:
                allowed[rel_type] = rel_def.target_types
        
        return allowed
    
    def get_entity_type_definition(self, entity_type: EntityType) -> Optional[EntityTypeDefinition]:
        """
        Get the definition for an entity type.
        
        Args:
            entity_type: The entity type
            
        Returns:
            EntityTypeDefinition if found, None otherwise
        """
        return self.entity_types.get(entity_type)
    
    def get_relationship_type_definition(self, rel_type: RelationshipType) -> Optional[RelationshipTypeDefinition]:
        """
        Get the definition for a relationship type.
        
        Args:
            rel_type: The relationship type
            
        Returns:
            RelationshipTypeDefinition if found, None otherwise
        """
        return self.relationship_types.get(rel_type)


# Create a default schema instance
default_schema = KeboolaOntologySchema()


class SchemaValidator:
    """Utility class for validating ontology data against a schema."""
    
    def __init__(self, schema=None):
        """
        Initialize the schema validator.
        
        Args:
            schema: Optional schema to use (defaults to importing the default schema)
        """
        if schema is None:
            # Import here to avoid circular imports
            from app.ontology.schema_definition import default_schema
            self.schema = default_schema
        else:
            self.schema = schema
    
    def validate_entity(self, entity: Entity) -> Tuple[bool, List[str]]:
        """
        Validate an entity against the schema.
        
        Args:
            entity: The entity to validate
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        return self.schema.validate_entity(entity)
    
    def validate_relationship(self, relationship: Relationship, entities: Dict[str, Entity]) -> Tuple[bool, List[str]]:
        """
        Validate a relationship against the schema.
        
        Args:
            relationship: The relationship to validate
            entities: Dictionary of entity_id -> Entity
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        return self.schema.validate_relationship(relationship, entities)
    
    def validate_triple(self, triple: Triple, entities: Dict[str, Entity]) -> Tuple[bool, List[str]]:
        """
        Validate a triple against the schema.
        
        Args:
            triple: The triple to validate
            entities: Dictionary of entity_id -> Entity
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        return self.schema.validate_triple(triple, entities)
    
    def get_allowed_relationships(self, entity_type: EntityType) -> Dict[RelationshipType, List[EntityType]]:
        """
        Get allowed relationships for an entity type.
        
        Args:
            entity_type: The entity type
            
        Returns:
            Dictionary of relationship_type -> list of target entity types
        """
        return self.schema.get_allowed_relationships(entity_type)
    
    def validate_ontology(self, entities: Dict[str, Entity], relationships: Dict[str, Relationship]) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate an entire ontology against the schema.
        
        Args:
            entities: Dictionary of entity_id -> Entity
            relationships: Dictionary of relationship_id -> Relationship
            
        Returns:
            Tuple of (is_valid, dict_of_errors)
        """
        all_valid = True
        errors = {
            "entities": {},
            "relationships": {}
        }
        
        # Validate entities
        for entity_id, entity in entities.items():
            is_valid, entity_errors = self.validate_entity(entity)
            if not is_valid:
                all_valid = False
                errors["entities"][entity_id] = entity_errors
        
        # Validate relationships
        for rel_id, rel in relationships.items():
            is_valid, rel_errors = self.validate_relationship(rel, entities)
            if not is_valid:
                all_valid = False
                errors["relationships"][rel_id] = rel_errors
        
        return all_valid, errors


# Create a default validator instance
default_validator = SchemaValidator() 