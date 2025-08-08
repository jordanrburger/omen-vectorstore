"""
Qdrant vector store backend implementation.
"""

from typing import List, Dict, Any, Optional, Union
from qdrant_client import QdrantClient, models

from omen.core import get_logger, load_settings
from omen.vectorstore.models import MetadataDocument, SearchQuery, SearchResult
from omen.storage.base import VectorStoreBackend

logger = get_logger(__name__)
settings = load_settings()


class QdrantVectorStore(VectorStoreBackend):
    """Qdrant implementation of vector storage backend."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        vector_size: Optional[int] = None,
        prefer_grpc: Optional[bool] = None,
    ):
        """Initialize QdrantVectorStore with configuration."""
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
                self._create_payload_indexes()
                logger.info(f"Created collection {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise

    def _create_payload_indexes(self) -> None:
        """Create necessary payload indexes for faster filtering."""
        try:
            # Create indexes for commonly filtered fields
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
        except Exception as e:
            logger.error(f"Error creating payload indexes: {e}")

    def store_document(self, document: MetadataDocument) -> str:
        """Store a single document."""
        try:
            point = models.PointStruct(
                id=document.id,
                vector=document.embedding,
                payload=document.to_payload()
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True
            )
            
            return document.id
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            raise

    def store_documents(self, documents: List[MetadataDocument]) -> List[str]:
        """Store multiple documents."""
        try:
            points = [
                models.PointStruct(
                    id=doc.id,
                    vector=doc.embedding,
                    payload=doc.to_payload()
                )
                for doc in documents
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
            return [doc.id for doc in documents]
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            raise

    def get_document(self, document_id: str) -> Optional[MetadataDocument]:
        """Retrieve a document by ID."""
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[document_id],
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                return None
                
            point = points[0]
            return MetadataDocument.from_payload(
                point.payload,
                embedding=point.vector
            )
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            return None

    def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
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

    def search(
        self,
        query: Union[str, SearchQuery],
        limit: int = 10,
        offset: int = 0,
        **kwargs
    ) -> List[SearchResult]:
        """Search for similar documents."""
        try:
            # Handle different query types
            if isinstance(query, str):
                search_query = SearchQuery(query=query)
            else:
                search_query = query
            
            # Create filter if needed
            search_filter = None
            if search_query.filter or search_query.type_filter:
                filter_conditions = []
                
                # Add metadata filter if provided
                if search_query.filter:
                    for key, value in search_query.filter.items():
                        if key.startswith('metadata.'):
                            filter_conditions.append(
                                models.FieldCondition(
                                    key=key,
                                    match=models.MatchValue(value=value)
                                )
                            )
                        elif key == 'project_id':
                            filter_conditions.append(
                                models.FieldCondition(
                                    key="source.project_id",
                                    match=models.MatchValue(value=value)
                                )
                            )
                        else:
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
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=search_query.embedding,
                query_filter=search_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                **kwargs
            )
            
            # Convert to SearchResult objects
            results = []
            for hit in search_results:
                document = MetadataDocument.model_validate(hit.payload)
                result = SearchResult(
                    document=document,
                    score=hit.score
                )
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def update_metadata(
        self,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for a document."""
        try:
            # Get existing document
            document = self.get_document(document_id)
            if not document:
                return False
            
            # Update metadata
            document.metadata.update(metadata)
            
            # Store updated document
            self.store_document(document)
            return True
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            return False 