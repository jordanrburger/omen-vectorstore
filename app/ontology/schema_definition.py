"""
Keboola ontology schema definition.

This module defines the Keboola ontology schema instance with all entity
and relationship type definitions.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple

from app.ontology.models import EntityType, RelationshipType, Entity, Relationship, Triple
from app.ontology.schema import PropertyDefinition, EntityTypeDefinition, RelationshipTypeDefinition

logger = logging.getLogger(__name__)


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