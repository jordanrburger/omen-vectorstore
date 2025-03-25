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
            api_key=api_key or settings.qdrant.api_key if hasattr(settings.qdrant, 'api_key') else None,
            prefer_grpc=prefer_grpc or settings.qdrant.prefer_grpc if hasattr(settings.qdrant, 'prefer_grpc') else False,
            https=False  # Disable HTTPS for local connections
        )
        self.ensure_collection()

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
            
            logger.info(f"Created payload indexes for collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating payload indexes: {e}")
            raise

    def index_document(
        self, 
        document: MetadataDocument,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> str:
        """
        Index a single document with its embedding.
        
        Args:
            document: The document to index
            embedding_provider: Provider for generating embeddings
            
        Returns:
            String ID of the indexed document
        """
        if embedding_provider is None:
            embedding_provider = get_embedding_provider()
        
        # Generate embedding for the document content
        embeddings = embedding_provider.embed([document.content])
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
            
            # Mark as indexed in state manager
            state_manager.mark_indexed(
                item_type=document.source.type.value, 
                item_id=document.source.id,
                metadata={"document_id": document.id}
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
    ) -> List[str]:
        """
        Index multiple documents with their embeddings.
        
        Args:
            documents: List of documents to index
            embedding_provider: Provider for generating embeddings
            batch_size: Size of batches for processing
            
        Returns:
            List of indexed document IDs
        """
        if not documents:
            return []
            
        if embedding_provider is None:
            embedding_provider = get_embedding_provider()
        
        # Process in batches
        results = batch_processor.process(
            items=documents,
            process_fn=lambda doc: self._process_document_batch(
                [doc], embedding_provider
            ),
            desc=f"Indexing {len(documents)} documents",
            show_progress=True
        )
        
        # Extract document IDs from results
        document_ids = []
        for doc, result, error in results:
            if error is None and result:
                document_ids.extend(result)
        
        return document_ids

    def _process_document_batch(
        self, 
        documents: List[MetadataDocument],
        embedding_provider: EmbeddingProvider
    ) -> List[str]:
        """Process a batch of documents."""
        if not documents:
            return []
        
        # Extract contents for embedding
        contents = [doc.content for doc in documents]
        
        # Generate embeddings
        embeddings = embedding_provider.embed(contents)
        
        # Create points
        points = [
            models.PointStruct(
                id=doc.id,
                vector=embedding,
                payload=doc.to_payload()
            )
            for doc, embedding in zip(documents, embeddings)
        ]
        
        # Upsert points
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
            # Mark documents as indexed
            for doc in documents:
                state_manager.mark_indexed(
                    item_type=doc.source.type.value,
                    item_id=doc.source.id,
                    metadata={"document_id": doc.id}
                )
            
            return [doc.id for doc in documents]
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
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
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
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False 