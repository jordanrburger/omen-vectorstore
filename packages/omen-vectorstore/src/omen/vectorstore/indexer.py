"""
Indexer for vector storage of metadata into Qdrant.
"""
import uuid
from typing import Dict, List, Optional, Any, Union, Callable

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from omen.core import get_logger, load_settings, BatchProcessor, StateManager
from omen.vectorstore.embedding import EmbeddingProvider, get_embedding_provider
from omen.vectorstore.models import MetadataDocument, SearchQuery, SearchResult

logger = get_logger(__name__)
settings = load_settings()
batch_processor = BatchProcessor()
state_manager = StateManager()


class QdrantIndexer:
    """Indexes metadata into Qdrant with vector embeddings."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        vector_size: Optional[int] = None,
        prefer_grpc: Optional[bool] = None,
    ):
        """Initialize QdrantIndexer with collection name and configuration."""
        self.collection_name = collection_name or settings.qdrant.collection_name
        self.vector_size = vector_size or settings.openai.embedding_dimension
        self.client = QdrantClient(
            host=host or settings.qdrant.host,
            port=port or settings.qdrant.port,
            api_key=api_key or (getattr(settings.qdrant, 'api_key', None)),
            prefer_grpc=prefer_grpc if prefer_grpc is not None else getattr(settings.qdrant, 'prefer_grpc', False),
            https=bool(getattr(settings.qdrant, 'https', False))
        )
        self.ensure_collection()
        self.indexed_source_ids = self._load_indexed_source_ids()
        
    def _load_indexed_source_ids(self) -> Dict[str, Dict[str, str]]:
        """
        Load a mapping of already indexed source IDs to document IDs.
        This helps prevent duplicate documents when indexing.
        
        Returns:
            Dictionary mapping source type and ID to document ID
        """
        indexed_ids = {}
        try:
            # Query for all documents in the collection
            scroll_results = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=1000,  # Process in batches
            )
            
            total_found = 0
            while scroll_results[0]:
                batch, next_offset = scroll_results
                
                for point in batch:
                    if (not hasattr(point, 'payload') or 
                        not point.payload or 
                        'source' not in point.payload):
                        continue
                        
                    source = point.payload.get('source', {})
                    source_type = source.get('type')
                    source_id = source.get('id')
                    project_id = source.get('project_id')
                    
                    if source_type and source_id:
                        # Create a compound key with project_id to allow same ID in different projects
                        project_prefix = f"{project_id}_" if project_id else ""
                        key = f"{source_type}_{project_prefix}{source_id}"
                        indexed_ids[key] = str(point.id)
                        total_found += 1
                
                if not next_offset:
                    break
                    
                # Get next batch
                scroll_results = self.client.scroll(
                    collection_name=self.collection_name,
                    with_payload=True,
                    limit=1000,
                    offset=next_offset
                )
                
            logger.info(f"Loaded {total_found} indexed source IDs from collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error loading indexed source IDs: {e}")
        
        return indexed_ids

    def ensure_collection(self) -> None:
        """Ensure the collection exists, creating it if necessary."""
        try:
            collections_list = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections_list]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection {self.collection_name}")
                logger.debug(f"Vector size: {self.vector_size}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size, 
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=10000
                    )
                )
                
                # Create necessary payload indexes for faster filtering
                self._create_payload_indexes()
                
                logger.info(f"Created collection {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                collection_info = self.client.get_collection(self.collection_name)
                logger.debug(f"Collection vector size: {collection_info.config.params.vectors.size}")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise

    def _create_payload_indexes(self) -> None:
        """Create payload indexes for faster filtering."""
        try:
            # Index for metadata_type filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            # Indexes for source filters
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source.type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source.project_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source.branch_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            # Additional indexes for common metadata fields
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.bucket_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.project_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            logger.info(f"Created payload indexes for collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating payload indexes: {e}")
            raise

    def index_document(
        self, 
        document: MetadataDocument,
        embedding_provider: Optional[EmbeddingProvider] = None,
        max_content_length: int = 8000,  # Limit document content size
    ) -> str:
        """
        Index a single document with its embedding.
        
        Args:
            document: The document to index
            embedding_provider: Provider for generating embeddings
            max_content_length: Maximum length for document content
            
        Returns:
            String ID of the indexed document
        """
        if embedding_provider is None:
            embedding_provider = get_embedding_provider()
        
        # Check if this document's source is already indexed
        source_type = document.source.type.value
        source_id = document.source.id
        project_id = document.source.project_id or ""
        
        # Create a compound key with project_id to allow same ID in different projects
        project_prefix = f"{project_id}_" if project_id else ""
        compound_key = f"{source_type}_{project_prefix}{source_id}"
        
        # If already indexed, use the existing document ID to update it
        if compound_key in self.indexed_source_ids:
            document.id = self.indexed_source_ids[compound_key]
            logger.debug(f"Updating existing document {document.id} for source {source_type}/{source_id} in project {project_id}")
        else:
            logger.debug(f"Creating new document for source {source_type}/{source_id} in project {project_id}")
        
        # Truncate document content if needed
        content = document.content
        if len(content) > max_content_length:
            logger.warning(f"Document content is too large ({len(content)} chars), truncating to {max_content_length} chars")
            content = content[:max_content_length]
            
            # Create a copy of the document with truncated content
            doc_copy = document.model_copy()
            doc_copy.content = content
            document = doc_copy
        
        # Generate embedding for the document content
        embeddings = embedding_provider.embed([content])
        logger.debug(f"Generated embeddings with shape: {len(embeddings)}x{len(embeddings[0])}")
        
        # Create point
        point = models.PointStruct(
            id=document.id,
            vector=embeddings[0],
            payload=document.to_payload()
        )
        
        # Upsert point
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True
            )
            
            # Update our in-memory tracking
            self.indexed_source_ids[compound_key] = document.id
            
            # Mark as indexed in state manager
            state_manager.mark_indexed(
                item_type=document.source.type.value, 
                item_id=document.source.id,
                metadata={
                    "document_id": document.id,
                    "project_id": project_id,
                    "collection": self.collection_name
                }
            )
            
            return document.id
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            raise

    def index_documents(
        self,
        documents: List[MetadataDocument],
        embedding_provider: Optional[EmbeddingProvider] = None,
        batch_size: Optional[int] = None,
        max_content_length: int = 8000,  # Limit document content size
    ) -> List[str]:
        """
        Index multiple documents with their embeddings.
        
        Args:
            documents: List of documents to index
            embedding_provider: Provider for generating embeddings
            batch_size: Size of batches for processing
            max_content_length: Maximum length for document content
            
        Returns:
            List of indexed document IDs
        """
        if not documents:
            return []
            
        if embedding_provider is None:
            embedding_provider = get_embedding_provider()
        
        # Pre-process documents to check for duplicates and assign IDs 
        for doc in documents:
            source_type = doc.source.type.value
            source_id = doc.source.id
            project_id = doc.source.project_id or ""
            
            # Create a compound key with project_id to allow same ID in different projects
            project_prefix = f"{project_id}_" if project_id else ""
            compound_key = f"{source_type}_{project_prefix}{source_id}"
            
            # If already indexed, use the existing document ID to update it
            if compound_key in self.indexed_source_ids:
                doc.id = self.indexed_source_ids[compound_key]
        
        # True batching for embeddings and upserts
        batch_size = batch_size or 64
        document_ids: List[str] = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                result_ids = self._process_document_batch(
                    documents=batch,
                    embedding_provider=embedding_provider,
                    max_content_length=max_content_length,
                )
                document_ids.extend(result_ids)
                # Update in-memory tracking for the batch
                for doc in batch:
                    source_type = doc.source.type.value
                    source_id = doc.source.id
                    project_id = doc.source.project_id or ""
                    project_prefix = f"{project_id}_" if project_id else ""
                    compound_key = f"{source_type}_{project_prefix}{source_id}"
                    self.indexed_source_ids[compound_key] = doc.id
            except Exception as e:
                logger.error(f"Error indexing batch starting at {i}: {e}")
                raise
        return document_ids

    def _process_document_batch(
        self, 
        documents: List[MetadataDocument],
        embedding_provider: EmbeddingProvider,
        max_content_length: int = 8000,
    ) -> List[str]:
        """Process a batch of documents."""
        if not documents:
            return []
        
        # Prepare documents with truncated content if needed
        processed_docs = []
        contents = []
        
        for doc in documents:
            content = doc.content
            if len(content) > max_content_length:
                logger.warning(f"Document content is too large ({len(content)} chars), truncating to {max_content_length} chars")
                content = content[:max_content_length]
                
                # Create a copy of the document with truncated content
                doc_copy = doc.model_copy()
                doc_copy.content = content
                processed_docs.append(doc_copy)
            else:
                processed_docs.append(doc)
                
            contents.append(content)
        
        # Generate embeddings
        embeddings = embedding_provider.embed(contents)
        
        # Create points
        points = [
            models.PointStruct(
                id=doc.id,
                vector=embedding,
                payload=doc.to_payload()
            )
            for doc, embedding in zip(processed_docs, embeddings)
        ]
        
        # Upsert points
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
            # Mark documents as indexed
            for doc in processed_docs:
                state_manager.mark_indexed(
                    item_type=doc.source.type.value,
                    item_id=doc.source.id,
                    metadata={
                        "document_id": doc.id,
                        "project_id": doc.source.project_id,
                        "collection": self.collection_name
                    }
                )
            
            return [doc.id for doc in processed_docs]
        except Exception as e:
            logger.error(f"Error indexing document batch: {e}")
            raise
    
    def search(
        self, 
        query: Union[str, SearchQuery],
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> List[SearchResult]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query text or SearchQuery object
            embedding_provider: Provider for generating embeddings
            
        Returns:
            List of search results
        """
        if embedding_provider is None:
            embedding_provider = get_embedding_provider()
        
        # Handle different query types
        if isinstance(query, str):
            search_query = SearchQuery(query=query)
        else:
            search_query = query
        
        # Generate embedding for the query
        query_embedding = embedding_provider.embed([search_query.query])[0]
        
        # Create filter if needed
        search_filter = None
        if search_query.filter or search_query.type_filter:
            filter_conditions = []
            
            # Add metadata filter if provided
            if search_query.filter:
                for key, value in search_query.filter.items():
                    if key.startswith('metadata.'):
                        # Direct filter on metadata fields
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
                    elif key == 'project_id':
                        # Special handling for project_id
                        filter_conditions.append(
                            models.FieldCondition(
                                key="source.project_id",
                                match=models.MatchValue(value=value)
                            )
                        )
                    else:
                        # Default placement under metadata
                        filter_conditions.append(
                            models.FieldCondition(
                                key=f"metadata.{key}",
                                match=models.MatchValue(value=value)
                            )
                        )
            
            # Add type filter if provided
            if search_query.type_filter:
                filter_conditions.append(
                    models.FieldCondition(
                        key="source.type",
                        match=models.MatchAny(any=[t.value for t in search_query.type_filter])
                    )
                )
            
            search_filter = models.Filter(
                must=filter_conditions
            )
        
        # Perform search
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=search_query.limit,
                offset=search_query.offset,
                with_payload=True
            )
            
            # Convert to SearchResult objects
            results = []
            for hit in search_results:
                # Convert payload back to MetadataDocument
                document = MetadataDocument.model_validate(hit.payload)
                
                # Create SearchResult
                result = SearchResult(
                    document=document,
                    score=hit.score
                )
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the index.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[document_id]
                ),
                wait=True
            )
            
            # Update in-memory tracking by finding and removing
            # the entry for this document ID
            to_remove = []
            for compound_key in self.indexed_source_ids.keys():
                if self.indexed_source_ids[compound_key] == document_id:
                    to_remove.append(compound_key)
            
            for key in to_remove:
                del self.indexed_source_ids[key]
                
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

    def delete_project_documents(self, project_id: str) -> int:
        """
        Delete all documents belonging to a specific project.
        
        Args:
            project_id: Project ID to delete documents for
            
        Returns:
            Number of documents deleted
        """
        try:
            # Create filter for documents with this project ID
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source.project_id",
                        match=models.MatchValue(value=project_id)
                    )
                ]
            )
            
            # Count documents to delete
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=search_filter
            )
            doc_count = count_result.count
            
            if doc_count == 0:
                logger.info(f"No documents found for project {project_id}")
                return 0
                
            logger.info(f"Deleting {doc_count} documents for project {project_id}")
            
            # Delete documents
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=search_filter
                ),
                wait=True
            )
            
            # Update in-memory tracking by finding and removing
            # entries for this project
            to_remove = []
            for compound_key in self.indexed_source_ids.keys():
                if f"{project_id}_" in compound_key:
                    to_remove.append(compound_key)
            
            for key in to_remove:
                del self.indexed_source_ids[key]
                
            return doc_count
        except Exception as e:
            logger.error(f"Error deleting project documents: {e}")
            return 0 