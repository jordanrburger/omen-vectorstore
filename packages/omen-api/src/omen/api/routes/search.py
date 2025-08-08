"""
API endpoints for vector search operations.
"""

from typing import List, Dict, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Query, Depends, Body, Path
from pydantic import BaseModel, Field

from omen.core import get_logger
from omen.vectorstore import (
    MetadataType,
    MetadataSource,
    MetadataDocument,
    SearchQuery,
    VectorSearch,
    QdrantIndexer,
    get_embedding_provider,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

# Pydantic models for request/response
class SearchRequest(BaseModel):
    """Request model for search operations."""
    query: str = Field(..., description="Search query text")
    limit: int = Field(10, description="Maximum number of results to return")
    offset: int = Field(0, description="Offset for pagination")
    type_filter: Optional[List[str]] = Field(None, description="Filter by metadata type")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Filter by metadata fields")


class SearchResponse(BaseModel):
    """Response model for search results."""
    id: str
    content: str
    score: float
    source: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentRequest(BaseModel):
    """Request model for document creation."""
    content: str = Field(..., description="Text content to vectorize")
    source: Dict[str, Any] = Field(..., description="Source information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    id: str
    content: str
    source: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Dependency for getting the vector search
def get_vector_search():
    """Get or create the VectorSearch instance."""
    indexer = QdrantIndexer()
    embedding_provider = get_embedding_provider()
    return VectorSearch(indexer=indexer, embedding_provider=embedding_provider)


@router.post("/", response_model=List[SearchResponse])
async def search(
    search_request: SearchRequest,
    search_engine: VectorSearch = Depends(get_vector_search)
):
    """Search for documents."""
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
        
        # Perform search
        results = search_engine.search(
            query=search_request.query,
            limit=search_request.limit,
            offset=search_request.offset,
            type_filter=type_filter,
            metadata_filter=search_request.metadata_filter
        )
        
        # Convert to response format
        return [
            SearchResponse(
                id=result.document.id,
                content=result.document.content,
                score=result.score,
                source=result.document.source.dict(),
                metadata=result.document.metadata
            )
            for result in results
        ]
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents", response_model=DocumentResponse)
async def create_document(
    document_request: DocumentRequest,
    search_engine: VectorSearch = Depends(get_vector_search)
):
    """Create and index a new document."""
    try:
        # Convert source dictionary to MetadataSource
        try:
            # Ensure type is valid
            source_type = MetadataType(document_request.source.get("type"))
            source_data = {**document_request.source, "type": source_type}
            source = MetadataSource.model_validate(source_data)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid source data: {str(e)}"
            )
        
        # Create document
        document = MetadataDocument(
            content=document_request.content,
            source=source,
            metadata=document_request.metadata
        )
        
        # Index document
        search_engine.indexer.index_document(document)
        
        return DocumentResponse(
            id=document.id,
            content=document.content,
            source=document.source.dict(),
            metadata=document.metadata
        )
    except Exception as e:
        logger.error(f"Error creating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str = Path(..., description="Document ID"),
    search_engine: VectorSearch = Depends(get_vector_search)
):
    """Get a specific document by ID."""
    # Note: This is a placeholder as we don't have a direct way to retrieve documents by ID
    # In a real implementation, we would need to add this functionality to the indexer
    raise HTTPException(
        status_code=501, 
        detail="Direct document retrieval not implemented yet"
    )


@router.get("/by-type/{metadata_type}", response_model=List[DocumentResponse])
async def get_documents_by_type(
    metadata_type: str = Path(..., description="Metadata type"),
    limit: int = Query(100, description="Maximum number of results"),
    offset: int = Query(0, description="Offset for pagination"),
    search_engine: VectorSearch = Depends(get_vector_search)
):
    """Get documents by metadata type."""
    try:
        # Convert type string to enum value
        try:
            type_enum = MetadataType(metadata_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid metadata type: {metadata_type}"
            )
        
        # Get documents by type
        documents = search_engine.get_by_metadata_type(
            metadata_type=type_enum,
            limit=limit,
            offset=offset
        )
        
        # Convert to response format
        return [
            DocumentResponse(
                id=doc.id,
                content=doc.content,
                source=doc.source.dict(),
                metadata=doc.metadata
            )
            for doc in documents
        ]
    except Exception as e:
        logger.error(f"Error retrieving documents by type: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str = Path(..., description="Document ID"),
    search_engine: VectorSearch = Depends(get_vector_search)
):
    """Delete a document by ID."""
    try:
        success = search_engine.indexer.delete_document(document_id)
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Document {document_id} not found or could not be deleted"
            )
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 