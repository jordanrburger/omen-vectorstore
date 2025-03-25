"""
Search functionality for the OMEN vectorstore.
"""
from typing import Dict, List, Optional, Union, Any

from omen.core import get_logger
from omen.vectorstore.embedding import EmbeddingProvider, get_embedding_provider
from omen.vectorstore.indexer import QdrantIndexer
from omen.vectorstore.models import MetadataDocument, SearchQuery, SearchResult, MetadataType

logger = get_logger(__name__)


class VectorSearch:
    """Vector-based semantic search for metadata documents."""

    def __init__(
        self,
        indexer: Optional[QdrantIndexer] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        """Initialize the search engine.
        
        Args:
            indexer: Indexer for accessing the vector database
            embedding_provider: Provider for generating embeddings
        """
        self.indexer = indexer or QdrantIndexer()
        self.embedding_provider = embedding_provider or get_embedding_provider()

    def search(
        self,
        query: Union[str, SearchQuery],
        limit: int = 10,
        offset: int = 0,
        type_filter: Optional[List[MetadataType]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for documents similar to the query.
        
        Args:
            query: Query text or SearchQuery object
            limit: Maximum number of results to return
            offset: Offset for pagination
            type_filter: Filter results by metadata type
            metadata_filter: Filter results by metadata fields
            
        Returns:
            List of search results
        """
        # Handle string query
        if isinstance(query, str):
            search_query = SearchQuery(
                query=query,
                limit=limit,
                offset=offset,
                type_filter=type_filter,
                filter=metadata_filter
            )
        else:
            # Use provided SearchQuery, but override parameters if provided
            search_query = query
            if limit is not None:
                search_query.limit = limit
            if offset is not None:
                search_query.offset = offset
            if type_filter is not None:
                search_query.type_filter = type_filter
            if metadata_filter is not None:
                search_query.filter = metadata_filter
        
        # Perform search
        results = self.indexer.search(
            query=search_query,
            embedding_provider=self.embedding_provider
        )
        
        return results

    def find_similar(
        self,
        document: Union[str, MetadataDocument],
        limit: int = 10,
        exclude_self: bool = True,
        type_filter: Optional[List[MetadataType]] = None,
    ) -> List[SearchResult]:
        """Find documents similar to a reference document.
        
        Args:
            document: Reference document or document ID
            limit: Maximum number of results to return
            exclude_self: Whether to exclude the reference document from results
            type_filter: Filter results by metadata type
            
        Returns:
            List of similar documents
        """
        # Handle string document ID
        if isinstance(document, str):
            # TODO: Implement document retrieval by ID
            raise NotImplementedError("Finding similar documents by ID is not yet implemented")
        
        # Use document content as query
        search_query = SearchQuery(
            query=document.content,
            limit=limit + (1 if exclude_self else 0),  # Add 1 to account for self-match
            type_filter=type_filter,
        )
        
        # Perform search
        results = self.indexer.search(
            query=search_query,
            embedding_provider=self.embedding_provider
        )
        
        # Filter out self if needed
        if exclude_self:
            results = [r for r in results if r.document.id != document.id]
            
        # Limit results
        return results[:limit]

    def get_by_metadata_type(
        self,
        metadata_type: MetadataType,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MetadataDocument]:
        """Get documents by metadata type.
        
        Args:
            metadata_type: Type of metadata to retrieve
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            List of documents
        """
        # Create dummy query
        search_query = SearchQuery(
            query="",  # Empty query will match based on filter only
            limit=limit,
            offset=offset,
            type_filter=[metadata_type],
        )
        
        # Perform search
        results = self.indexer.search(
            query=search_query,
            embedding_provider=self.embedding_provider
        )
        
        # Extract documents
        return [result.document for result in results] 