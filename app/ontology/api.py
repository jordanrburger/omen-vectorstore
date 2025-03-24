"""
API endpoints for ontology and action graph operations.
"""

from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from .models import Entity, Relationship
from .schema import EntityType, RelationshipType
from .rdf_store import RDFStore
from .action_graph import ActionGraph
from .llm_utils import (
    SemanticMatcher,
    NLQueryConverter,
    OntologyExplainer,
    OntologyValidator
)

router = APIRouter(prefix="/ontology", tags=["ontology"])

# Pydantic models for request/response
class EntityResponse(BaseModel):
    """Response model for entity operations."""
    id: str
    name: str
    type: EntityType
    properties: Dict

class RelationshipResponse(BaseModel):
    """Response model for relationship operations."""
    id: str
    type: RelationshipType
    source_id: str
    target_id: str
    properties: Dict

class ActionResponse(BaseModel):
    """Response model for action operations."""
    id: str
    type: str
    source_id: str
    target_id: str
    description: str
    properties: Dict

class NLQueryRequest(BaseModel):
    """Request model for natural language queries."""
    query: str

class SemanticSearchRequest(BaseModel):
    """Request model for semantic search."""
    query: str
    top_k: Optional[int] = 5
    threshold: Optional[float] = 0.7

class ActionChainRequest(BaseModel):
    """Request model for action chain queries."""
    source_id: str
    target_id: str

# Dependency injection
async def get_rdf_store() -> RDFStore:
    """Get RDF store instance."""
    store = RDFStore()
    
    # Initialize with test data if empty
    if not store.find_all_entities():
        # Add test bucket
        bucket = Entity(
            id="test_bucket_1",
            name="Test Bucket",
            type=EntityType.BUCKET,
            properties={
                "description": "A test bucket for demonstration",
                "stage": "in"
            }
        )
        store.add_entity(bucket)
        
        # Add test table
        table = Entity(
            id="test_table_1",
            name="Test Table",
            type=EntityType.TABLE,
            properties={
                "description": "A test table for demonstration",
                "columns": ["id", "name", "value"],
                "row_count": 1000
            }
        )
        store.add_entity(table)
        
        # Add test transformation
        transformation = Entity(
            id="test_transformation_1",
            name="Test Transformation",
            type=EntityType.TRANSFORMATION,
            properties={
                "description": "A test transformation for demonstration",
                "type": "python",
                "code": "# Sample code\ndf = pd.read_csv('input.csv')"
            }
        )
        store.add_entity(transformation)
        
        # Add relationships
        contains_rel = Relationship(
            id="test_rel_1",
            type=RelationshipType.CONTAINS,
            source_id="test_bucket_1",
            target_id="test_table_1",
            properties={"description": "Bucket contains table"}
        )
        store.add_relationship(contains_rel)
        
        inputs_rel = Relationship(
            id="test_rel_2",
            type=RelationshipType.INPUTS_FROM,
            source_id="test_transformation_1",
            target_id="test_table_1",
            properties={"description": "Transformation reads from table"}
        )
        store.add_relationship(inputs_rel)
    
    return store

async def get_action_graph() -> ActionGraph:
    """Get action graph instance."""
    # This would be properly initialized with your configuration
    return ActionGraph()

async def get_semantic_matcher() -> SemanticMatcher:
    """Get semantic matcher instance."""
    return SemanticMatcher()

async def get_nl_converter(rdf_store: RDFStore = Depends(get_rdf_store)) -> NLQueryConverter:
    """Get natural language query converter instance."""
    return NLQueryConverter(rdf_store)

async def get_explainer(rdf_store: RDFStore = Depends(get_rdf_store)) -> OntologyExplainer:
    """Get ontology explainer instance."""
    return OntologyExplainer(rdf_store)

async def get_validator(rdf_store: RDFStore = Depends(get_rdf_store)) -> OntologyValidator:
    """Get ontology validator instance."""
    return OntologyValidator(rdf_store)

# Entity endpoints
@router.get("/entities", response_model=List[EntityResponse])
async def list_entities(
    type: Optional[EntityType] = None,
    rdf_store: RDFStore = Depends(get_rdf_store)
):
    """List all entities, optionally filtered by type."""
    entities = rdf_store.find_entities_by_type(type) if type else rdf_store.find_all_entities()
    return [EntityResponse(**entity.dict()) for entity in entities]

@router.get("/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: str,
    rdf_store: RDFStore = Depends(get_rdf_store)
):
    """Get a specific entity by ID."""
    entity = rdf_store.find_entity_by_id(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    return EntityResponse(**entity.dict())

# Relationship endpoints
@router.get("/relationships", response_model=List[RelationshipResponse])
async def list_relationships(
    type: Optional[RelationshipType] = None,
    rdf_store: RDFStore = Depends(get_rdf_store)
):
    """List all relationships, optionally filtered by type."""
    relationships = rdf_store.find_relationships_by_type(type) if type else rdf_store.find_all_relationships()
    return [RelationshipResponse(**rel.dict()) for rel in relationships]

@router.get("/relationships/{relationship_id}", response_model=RelationshipResponse)
async def get_relationship(
    relationship_id: str,
    rdf_store: RDFStore = Depends(get_rdf_store)
):
    """Get a specific relationship by ID."""
    relationship = rdf_store.find_relationship_by_id(relationship_id)
    if not relationship:
        raise HTTPException(status_code=404, detail="Relationship not found")
    return RelationshipResponse(**relationship.dict())

# Action graph endpoints
@router.get("/actions", response_model=List[ActionResponse])
async def list_actions(
    action_graph: ActionGraph = Depends(get_action_graph)
):
    """List all actions in the action graph."""
    actions = action_graph.get_all_actions()
    return [ActionResponse(**action) for action in actions]

@router.get("/actions/{action_id}", response_model=ActionResponse)
async def get_action(
    action_id: str,
    action_graph: ActionGraph = Depends(get_action_graph)
):
    """Get a specific action by ID."""
    action = action_graph.get_action(action_id)
    if not action:
        raise HTTPException(status_code=404, detail="Action not found")
    return ActionResponse(**action)

@router.get("/action-chains", response_model=List[ActionResponse])
async def get_action_chain(
    source_id: str,
    target_id: str,
    action_graph: ActionGraph = Depends(get_action_graph)
):
    """Get the action chain between two entities."""
    actions = action_graph.get_action_chain(source_id, target_id)
    if not actions:
        raise HTTPException(status_code=404, detail="No action chain found")
    return [ActionResponse(**action) for action in actions]

# Semantic search endpoint
@router.post("/search", response_model=List[EntityResponse])
async def semantic_search(
    request: SemanticSearchRequest,
    rdf_store: RDFStore = Depends(get_rdf_store),
    semantic_matcher: SemanticMatcher = Depends(get_semantic_matcher)
):
    """Find entities semantically similar to the query."""
    entities = rdf_store.find_all_entities()
    matches = semantic_matcher.find_similar_entities(
        request.query,
        entities,
        top_k=request.top_k,
        threshold=request.threshold
    )
    return [EntityResponse(**entity.dict()) for entity, _ in matches]

# Natural language query endpoint
@router.post("/query", response_model=List[Dict])
async def natural_language_query(
    request: NLQueryRequest,
    nl_converter: NLQueryConverter = Depends(get_nl_converter)
):
    """Execute a natural language query."""
    return nl_converter.execute_nl_query(request.query)

# Explanation endpoints
@router.get("/explain/relationship/{relationship_id}")
async def explain_relationship(
    relationship_id: str,
    rdf_store: RDFStore = Depends(get_rdf_store),
    explainer: OntologyExplainer = Depends(get_explainer)
):
    """Generate a natural language explanation of a relationship."""
    relationship = rdf_store.find_relationship_by_id(relationship_id)
    if not relationship:
        raise HTTPException(status_code=404, detail="Relationship not found")
    
    source = rdf_store.find_entity_by_id(relationship.source_id)
    target = rdf_store.find_entity_by_id(relationship.target_id)
    
    if not source or not target:
        raise HTTPException(status_code=404, detail="Source or target entity not found")
    
    return {"explanation": explainer.explain_relationship(source, target, relationship)}

@router.get("/explain/action-chain")
async def explain_action_chain(
    source_id: str,
    target_id: str,
    action_graph: ActionGraph = Depends(get_action_graph),
    rdf_store: RDFStore = Depends(get_rdf_store),
    explainer: OntologyExplainer = Depends(get_explainer)
):
    """Generate a natural language explanation of an action chain."""
    source = rdf_store.find_entity_by_id(source_id)
    target = rdf_store.find_entity_by_id(target_id)
    
    if not source or not target:
        raise HTTPException(status_code=404, detail="Source or target entity not found")
    
    actions = action_graph.get_action_chain(source_id, target_id)
    if not actions:
        raise HTTPException(status_code=404, detail="No action chain found")
    
    return {"explanation": explainer.explain_action_chain(source, target, actions)}

# Validation endpoints
@router.get("/validate/entity/{entity_id}")
async def validate_entity(
    entity_id: str,
    rdf_store: RDFStore = Depends(get_rdf_store),
    validator: OntologyValidator = Depends(get_validator)
):
    """Validate an entity."""
    entity = rdf_store.find_entity_by_id(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    issues = validator.validate_entity(entity)
    return {"valid": len(issues) == 0, "issues": issues}

@router.get("/validate/relationship/{relationship_id}")
async def validate_relationship(
    relationship_id: str,
    rdf_store: RDFStore = Depends(get_rdf_store),
    validator: OntologyValidator = Depends(get_validator)
):
    """Validate a relationship."""
    relationship = rdf_store.find_relationship_by_id(relationship_id)
    if not relationship:
        raise HTTPException(status_code=404, detail="Relationship not found")
    
    source = rdf_store.find_entity_by_id(relationship.source_id)
    target = rdf_store.find_entity_by_id(relationship.target_id)
    
    if not source or not target:
        raise HTTPException(status_code=404, detail="Source or target entity not found")
    
    issues = validator.validate_relationship(relationship, source, target)
    return {"valid": len(issues) == 0, "issues": issues} 