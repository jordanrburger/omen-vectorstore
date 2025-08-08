"""
API endpoints for ontology operations.
"""

from typing import List, Dict, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Query, Depends, Body, Path
from pydantic import BaseModel, Field

from omen.core import get_logger
from omen.ontology import (
    Entity,
    EntityType,
    Relationship,
    RelationshipType,
    OntologyManager,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/ontology", tags=["ontology"])

# Pydantic models for request/response
class EntityResponse(BaseModel):
    """Response model for entity operations."""
    id: str
    name: str
    type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class RelationshipResponse(BaseModel):
    """Response model for relationship operations."""
    id: str
    type: str
    source_id: str
    target_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class EntityCreate(BaseModel):
    """Request model for entity creation."""
    name: str
    type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class RelationshipCreate(BaseModel):
    """Request model for relationship creation."""
    type: str
    source_id: str
    target_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class SPARQLQuery(BaseModel):
    """Request model for SPARQL queries."""
    query: str


class StatsResponse(BaseModel):
    """Response model for ontology statistics."""
    total_entities: int
    total_relationships: int
    entity_types: Dict[str, int]
    relationship_types: Dict[str, int]
    triple_count: int


# Dependency for getting the ontology manager
def get_ontology_manager():
    """Get or create the OntologyManager instance."""
    manager = OntologyManager()
    manager.load_state()
    return manager


@router.get("/entities", response_model=List[EntityResponse])
async def list_entities(
    entity_type: Optional[str] = None,
    manager: OntologyManager = Depends(get_ontology_manager)
):
    """List all entities, optionally filtered by type."""
    try:
        if entity_type:
            try:
                enum_type = EntityType(entity_type)
                entities = manager.get_entities_by_type(enum_type)
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid entity type: {entity_type}"
                )
        else:
            entities = list(manager.entities.values())
        
        return [
            EntityResponse(
                id=entity.id,
                name=entity.name,
                type=entity.type.value,
                description=entity.description,
                properties=entity.properties
            )
            for entity in entities
        ]
    except Exception as e:
        logger.error(f"Error listing entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: str = Path(..., description="Entity ID"),
    manager: OntologyManager = Depends(get_ontology_manager)
):
    """Get a specific entity by ID."""
    entity = manager.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    
    return EntityResponse(
        id=entity.id,
        name=entity.name,
        type=entity.type.value,
        description=entity.description,
        properties=entity.properties
    )


@router.post("/entities", response_model=EntityResponse)
async def create_entity(
    entity_data: EntityCreate,
    manager: OntologyManager = Depends(get_ontology_manager)
):
    """Create a new entity."""
    try:
        entity_type = EntityType(entity_data.type)
        
        entity = Entity(
            name=entity_data.name,
            type=entity_type,
            description=entity_data.description,
            properties=entity_data.properties
        )
        
        manager.add_entity(entity)
        manager.save_state()
        
        return EntityResponse(
            id=entity.id,
            name=entity.name,
            type=entity.type.value,
            description=entity.description,
            properties=entity.properties
        )
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid entity type: {entity_data.type}"
        )
    except Exception as e:
        logger.error(f"Error creating entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relationships", response_model=List[RelationshipResponse])
async def list_relationships(
    relationship_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    direction: Optional[str] = "both",
    manager: OntologyManager = Depends(get_ontology_manager)
):
    """List all relationships, optionally filtered by type or entity."""
    try:
        if relationship_type and entity_id:
            # Filter by both type and entity
            try:
                enum_type = RelationshipType(relationship_type)
                all_relationships = manager.get_relationships_for_entity(entity_id, direction)
                relationships = [r for r in all_relationships if r.type == enum_type]
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid relationship type: {relationship_type}"
                )
        elif relationship_type:
            # Filter by type only
            try:
                enum_type = RelationshipType(relationship_type)
                relationships = manager.get_relationships_by_type(enum_type)
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid relationship type: {relationship_type}"
                )
        elif entity_id:
            # Filter by entity only
            relationships = manager.get_relationships_for_entity(entity_id, direction)
        else:
            # No filters
            relationships = list(manager.relationships.values())
        
        return [
            RelationshipResponse(
                id=rel.id,
                type=rel.type.value,
                source_id=rel.source_id,
                target_id=rel.target_id,
                properties=rel.properties
            )
            for rel in relationships
        ]
    except Exception as e:
        logger.error(f"Error listing relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relationships/{relationship_id}", response_model=RelationshipResponse)
async def get_relationship(
    relationship_id: str = Path(..., description="Relationship ID"),
    manager: OntologyManager = Depends(get_ontology_manager)
):
    """Get a specific relationship by ID."""
    relationship = manager.get_relationship(relationship_id)
    if not relationship:
        raise HTTPException(status_code=404, detail=f"Relationship {relationship_id} not found")
    
    return RelationshipResponse(
        id=relationship.id,
        type=relationship.type.value,
        source_id=relationship.source_id,
        target_id=relationship.target_id,
        properties=relationship.properties
    )


@router.post("/relationships", response_model=RelationshipResponse)
async def create_relationship(
    relationship_data: RelationshipCreate,
    manager: OntologyManager = Depends(get_ontology_manager)
):
    """Create a new relationship."""
    try:
        # Check if source and target entities exist
        if not manager.entity_exists(relationship_data.source_id):
            raise HTTPException(
                status_code=404, 
                detail=f"Source entity {relationship_data.source_id} not found"
            )
        
        if not manager.entity_exists(relationship_data.target_id):
            raise HTTPException(
                status_code=404, 
                detail=f"Target entity {relationship_data.target_id} not found"
            )
        
        # Create relationship
        rel_type = RelationshipType(relationship_data.type)
        
        relationship = Relationship(
            type=rel_type,
            source_id=relationship_data.source_id,
            target_id=relationship_data.target_id,
            properties=relationship_data.properties
        )
        
        manager.add_relationship(relationship)
        manager.save_state()
        
        return RelationshipResponse(
            id=relationship.id,
            type=relationship.type.value,
            source_id=relationship.source_id,
            target_id=relationship.target_id,
            properties=relationship.properties
        )
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid relationship type: {relationship_data.type}"
        )
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=List[Dict[str, Any]])
async def query_sparql(
    query_data: SPARQLQuery,
    manager: OntologyManager = Depends(get_ontology_manager)
):
    """Execute a SPARQL query on the ontology."""
    try:
        results = manager.query_sparql(query_data.query)
        return results
    except Exception as e:
        logger.error(f"Error executing SPARQL query: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    manager: OntologyManager = Depends(get_ontology_manager)
):
    """Get statistics about the ontology."""
    try:
        stats = manager.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_ontology(
    manager: OntologyManager = Depends(get_ontology_manager)
):
    """Clear the entire ontology."""
    try:
        manager.clear()
        manager.save_state()
        return {"message": "Ontology cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing ontology: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 