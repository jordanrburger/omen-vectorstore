"""
Data models for the OMEN vectorstore.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from omen.core.utils import generate_uuid


class MetadataType(str, Enum):
    """Enumeration of metadata types."""
    
    TABLE = "table"
    COLUMN = "column"
    BUCKET = "bucket"
    CONFIGURATION = "configuration"
    TRANSFORMATION = "transformation"
    TRANSFORMATION_CODE = "transformation_code"
    COMPONENT = "component"
    ORCHESTRATION = "orchestration"
    JOB = "job"
    TOKEN = "token"
    USER = "user"
    BRANCH = "branch"
    ORGANIZATION = "organization"
    PROJECT = "project"
    CUSTOM = "custom"


class MetadataSource(BaseModel):
    """Source information for metadata."""
    
    id: str = Field(..., description="Identifier in the source system")
    type: MetadataType = Field(..., description="Type of metadata")
    url: Optional[str] = Field(None, description="URL to the source in the Keboola UI")
    project_id: Optional[str] = Field(None, description="Project ID in Keboola")
    branch_id: Optional[str] = Field(None, description="Branch ID in Keboola")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class MetadataDocument(BaseModel):
    """A document containing metadata to be vectorized."""
    
    id: str = Field(default_factory=generate_uuid, description="Unique identifier")
    source: MetadataSource = Field(..., description="Source information")
    content: str = Field(..., description="Text content to vectorize")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    vector: Optional[List[float]] = Field(None, description="Vector embedding of the content")
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to a payload for vector storage."""
        return {
            "id": self.id,
            "source": self.source.dict(),
            "metadata": self.metadata,
            "content": self.content,
        }


class SearchQuery(BaseModel):
    """A search query."""
    
    query: str = Field(..., description="Search query text")
    limit: int = Field(10, description="Maximum number of results to return")
    offset: int = Field(0, description="Offset for pagination")
    filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter")
    type_filter: Optional[List[MetadataType]] = Field(
        None, description="Filter by metadata type"
    )


class SearchResult(BaseModel):
    """A search result."""
    
    document: MetadataDocument = Field(..., description="The matching document")
    score: float = Field(..., description="Search score (0-1)")
    
    class Config:
        arbitrary_types_allowed = True 