"""
High-level interface to vector store functionality.
"""
from typing import List, Optional, Dict, Any, Union

from omen.vectorstore.indexer import QdrantIndexer
from omen.vectorstore.models import MetadataDocument, SearchResult, SearchQuery, MetadataSource, MetadataType
from omen.vectorstore.embedding import get_embedding_provider

class VectorStore:
    """High-level interface to vector store functionality."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize VectorStore with configuration."""
        self.indexer = QdrantIndexer(
            host=host,
            port=port,
            collection_name=collection_name,
            api_key=api_key,
        )
        self.embedding_provider = get_embedding_provider()

    def add_document(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a document to the vector store.

        Args:
            id: Unique identifier for the document
            text: Text content of the document
            metadata: Optional metadata for the document

        Returns:
            ID of the stored document
        """
        # Create metadata document
        doc = MetadataDocument(
            id=id,
            content=text,
            source=MetadataSource(
                type=MetadataType.CUSTOM,
                id=id,
                project_id="example",
                branch_id="main",
            ),
            metadata=metadata or {},
        )

        # Index the document
        return self.indexer.index_document(doc, embedding_provider=self.embedding_provider)

    def search(
        self,
        query: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            metadata_filter: Optional metadata filters to apply

        Returns:
            List of search results
        """
        # Create search query
        search_query = SearchQuery(
            query=query,
            limit=limit,
            filter=metadata_filter,
        )

        # Perform search
        return self.indexer.search(search_query) 