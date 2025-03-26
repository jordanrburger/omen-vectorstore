"""
API endpoints for hybrid search operations.
"""

from typing import List, Dict, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Query, Depends, Body, Path
from pydantic import BaseModel, Field

from omen.core import get_logger
from omen.vectorstore import (
    MetadataType,
    MetadataDocument,
    SearchQuery,
    VectorSearch,
    HybridSearch,
    QdrantIndexer,
    get_embedding_provider,
)
from omen.ontology import OntologyManager

logger = get_logger(__name__)

router = APIRouter(prefix="/hybrid", tags=["hybrid"])


# Pydantic models for request/response
class HybridSearchRequest(BaseModel):
    """Request model for hybrid search operations."""
    query: str = Field(..., description="Search query text")
    limit: int = Field(10, description="Maximum number of results to return")
    offset: int = Field(0, description="Offset for pagination")
    vector_weight: float = Field(0.7, description="Weight for vector search (0-1)")
    semantic_weight: float = Field(0.3, description="Weight for semantic search (0-1)")
    type_filter: Optional[List[str]] = Field(None, description="Filter by metadata type")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Filter by metadata fields")
    include_related: bool = Field(False, description="Include related entities in results")
    related_depth: int = Field(1, description="Maximum depth for related entities")


class RelatedEntity(BaseModel):
    """Model for related entity information."""
    source_id: str
    target_id: str
    target_name: Optional[str] = None
    relationship: str
    direction: str
    path_length: int
    path: List[Dict[str, Any]] = Field(default_factory=list)


class HybridSearchResponse(BaseModel):
    """Response model for hybrid search results."""
    id: str
    content: str
    score: float
    vector_score: Optional[float] = None
    semantic_score: Optional[float] = None
    source: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    related_entities: List[RelatedEntity] = Field(default_factory=list)


class HybridRecommendationRequest(BaseModel):
    """Request model for hybrid recommendations."""
    entity_id: str = Field(..., description="Entity ID to recommend for")
    limit: int = Field(10, description="Maximum number of recommendations")
    vector_weight: float = Field(0.5, description="Weight for vector recommendations (0-1)")
    semantic_weight: float = Field(0.5, description="Weight for semantic recommendations (0-1)")


class RelationshipPath(BaseModel):
    """Model for relationship path information."""
    id: str
    name: str
    rel: str


class HybridRecommendationResponse(BaseModel):
    """Response model for hybrid recommendations."""
    id: str
    title: str
    content: str
    type: Optional[str] = None
    vector_score: float
    semantic_score: float
    final_score: float
    relationship_path: Optional[List[RelationshipPath]] = None


# Dependency for getting the hybrid search
def get_hybrid_search():
    """Get or create the HybridSearch instance."""
    indexer = QdrantIndexer()
    embedding_provider = get_embedding_provider()
    vector_search = VectorSearch(indexer=indexer, embedding_provider=embedding_provider)
    ontology_manager = OntologyManager()
    ontology_manager.load_state()
    
    return HybridSearch(
        vector_search=vector_search,
        ontology_manager=ontology_manager
    )


@router.post("/search", response_model=List[HybridSearchResponse])
async def hybrid_search(
    search_request: HybridSearchRequest,
    search_engine: HybridSearch = Depends(get_hybrid_search)
):
    """Search using hybrid vector and semantic techniques."""
    try:
        # Convert type filter strings to enum values
        type_filter = None
        if search_request.type_filter:
            try:
                type_filter = [MetadataType(t) for t in search_request.type_filter]
            except ValueError as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid metadata type: {str(e)}"
                )
        
        # Perform hybrid search
        results = search_engine.search(
            query=search_request.query,
            limit=search_request.limit,
            offset=search_request.offset,
            type_filter=type_filter,
            metadata_filter=search_request.metadata_filter,
            vector_weight=search_request.vector_weight,
            semantic_weight=search_request.semantic_weight,
            include_related=search_request.include_related,
            max_related_depth=search_request.related_depth
        )
        
        # Convert to response format
        responses = []
        for result in results:
            # Convert related entities if available
            related_entities = []
            if result.related_entities:
                for rel in result.related_entities:
                    related_entities.append(RelatedEntity(**rel))
            
            responses.append(HybridSearchResponse(
                id=result.document.id,
                content=result.document.content,
                score=result.score,
                vector_score=result.vector_score,
                semantic_score=result.semantic_score,
                source=result.document.source.dict(),
                metadata=result.document.metadata,
                related_entities=related_entities
            ))
        
        return responses
    except Exception as e:
        logger.error(f"Error performing hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend", response_model=List[HybridRecommendationResponse])
async def hybrid_recommend(
    request: HybridRecommendationRequest,
    search_engine: HybridSearch = Depends(get_hybrid_search)
):
    """Generate hybrid recommendations for an entity."""
    try:
        # Check if entity exists
        if not search_engine.ontology_manager.entity_exists(request.entity_id):
            raise HTTPException(
                status_code=404,
                detail=f"Entity with ID '{request.entity_id}' not found"
            )
        
        # Get recommendations
        recommendations = search_engine.hybrid_recommendation(
            entity_id=request.entity_id,
            limit=request.limit,
            vector_weight=request.vector_weight,
            semantic_weight=request.semantic_weight
        )
        
        if not recommendations:
            return []
        
        # Convert to response format
        responses = []
        for rec in recommendations:
            # Convert relationship path if available
            relationship_path = None
            if rec.get('relationship_path'):
                relationship_path = [
                    RelationshipPath(**step)
                    for step in rec['relationship_path']
                ]
            
            responses.append(HybridRecommendationResponse(
                id=rec['id'],
                title=rec['title'],
                content=rec['content'],
                type=rec['type'],
                vector_score=rec['vector_score'],
                semantic_score=rec['semantic_score'],
                final_score=rec['final_score'],
                relationship_path=relationship_path
            ))
        
        return responses
    except Exception as e:
        logger.error(f"Error generating hybrid recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entity-recommendations/{entity_id}", response_model=List[Dict[str, Any]])
async def entity_recommendations(
    entity_id: str = Path(..., description="Entity ID"),
    limit: int = Query(10, description="Maximum number of recommendations"),
    include_paths: bool = Query(True, description="Include relationship paths"),
    search_engine: HybridSearch = Depends(get_hybrid_search)
):
    """Get recommendations based on ontology relationships."""
    try:
        # Check if entity exists
        if not search_engine.ontology_manager.entity_exists(entity_id):
            raise HTTPException(
                status_code=404,
                detail=f"Entity with ID '{entity_id}' not found"
            )
        
        # Get recommendations
        recommendations = search_engine.entity_based_recommendation(
            entity_id=entity_id,
            limit=limit,
            include_paths=include_paths
        )
        
        return recommendations
    except Exception as e:
        logger.error(f"Error getting entity recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 