"""
Ontology and Action Graph module for Keboola metadata.

This module provides functionality for creating, managing, and querying
an ontology and action graph based on Keboola metadata. It includes
classes for representing entities, relationships, triples, and actions.
"""

from app.ontology.models import (
    Triple,
    Entity,
    Relationship,
    EntityType,
    RelationshipType
)

from app.ontology.manager import OntologyManager
from app.ontology.storage import TripleStore
from app.ontology.schema import (
    PropertyDefinition,
    EntityTypeDefinition,
    RelationshipTypeDefinition,
    SchemaValidator
)
from app.ontology.schema_definition import (
    KeboolaOntologySchema,
    default_schema
)
from app.ontology.builder import OntologyBuilder
from app.ontology.action_graph import (
    Action,
    ActionType,
    ActionGraph,
    ActionGraphBuilder
)

__all__ = [
    # Models
    "Triple",
    "Entity",
    "Relationship",
    "EntityType",
    "RelationshipType",
    
    # Manager and storage
    "OntologyManager",
    "TripleStore",
    
    # Schema
    "PropertyDefinition",
    "EntityTypeDefinition",
    "RelationshipTypeDefinition",
    "SchemaValidator",
    "KeboolaOntologySchema",
    "default_schema",
    
    # Builder
    "OntologyBuilder",
    
    # Action Graph
    "Action",
    "ActionType",
    "ActionGraph",
    "ActionGraphBuilder"
] 