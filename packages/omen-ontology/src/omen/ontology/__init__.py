"""
Ontology system for the OMEN platform.
"""

from omen.ontology.models import (
    Entity,
    EntityType,
    EntityTypeDefinition,
    PropertyDefinition,
    Relationship,
    RelationshipType,
    RelationshipTypeDefinition,
    Triple,
)
from omen.ontology.manager import OntologyManager
from omen.ontology.rdf_store import RDFStore

__all__ = [
    # Models
    "Entity",
    "EntityType",
    "EntityTypeDefinition",
    "PropertyDefinition",
    "Relationship",
    "RelationshipType",
    "RelationshipTypeDefinition",
    "Triple",
    
    # Ontology management
    "OntologyManager",
    "RDFStore",
]
