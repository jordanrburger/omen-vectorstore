"""
Validation-specific implementation of the QdrantIndexer.
This version is optimized for validation testing with simplified functionality.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Union
import asyncio

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import PointStruct

from app.vectorizer import MetadataVectorizer

class ValidationIndexer:
    """Validation-specific implementation of metadata indexing."""

    def __init__(self, client: QdrantClient, vectorizer: MetadataVectorizer):
        """Initialize ValidationIndexer with client and vectorizer."""
        self.client = client
        self.vectorizer = vectorizer
        self.vector_size = 1536  # OpenAI embedding size

    async def create_collection(self, collection_name: str) -> None:
        """Create a new collection if it doesn't exist."""
        try:
            collection_info = self.client.get_collection(collection_name)
            if collection_info is None:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=0,  # Force immediate indexing
                        memmap_threshold=0,  # Use memory mapping from the start
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=16,  # Number of edges per node in the index graph
                        ef_construct=100,  # Size of the dynamic candidate list
                        full_scan_threshold=10000,  # When to switch to full scan
                    )
                )
        except UnexpectedResponse:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                    memmap_threshold=0,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                )
            )

    async def index_metadata(self, metadata: Union[List[Dict], Dict], collection_name: str) -> None:
        """Index metadata into the specified collection."""
        if not metadata:
            return

        # Convert single metadata dict to list
        if isinstance(metadata, dict):
            metadata = [metadata]

        # Process metadata in batches
        batch_size = 100
        for i in range(0, len(metadata), batch_size):
            batch = metadata[i:i + batch_size]
            
            # Vectorize batch
            vectors = self.vectorizer.vectorize_batch(batch)
            
            # Prepare points for indexing
            points = []
            for item, vector in zip(batch, vectors):
                point_id = str(uuid.uuid4())
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "metadata": item,
                            "text": self.vectorizer._prepare_text(item)
                        }
                    )
                )
            
            # Index batch
            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )

    async def search(
        self,
        query: str,
        collection_name: str,
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """Search for metadata using semantic similarity."""
        # Vectorize query
        query_vector = self.vectorizer.vectorize({"description": query})
        
        # Set up search parameters for better performance
        search_params = models.SearchParams(
            hnsw_ef=128,  # Size of the dynamic candidate list for search
            exact=False  # Use approximate search for better performance
        )
        
        # Search in Qdrant using query_points
        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,  # Use query parameter instead of query_vector
            limit=limit,
            query_filter=None,  # No filtering in validation tests
            search_params=search_params,
            score_threshold=score_threshold
        )
        
        # Format results
        results = []
        for point in search_result.points:
            result = {
                "score": point.score,
                "metadata": point.payload["metadata"],
                "text": point.payload["text"]
            }
            results.append(result)
        
        return results

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
        except Exception as e:
            logging.error(f"Error deleting collection {collection_name}: {e}")
            raise 