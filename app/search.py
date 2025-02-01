import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

def search_metadata(query: str, embedding_provider, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for metadata using semantic search.
    """
    client = QdrantClient(host="localhost", port=55000)
    
    # Generate embedding for the query
    query_embedding = embedding_provider.embed(query)
    
    # Search in Qdrant
    search_result = client.search(
        collection_name="keboola_metadata",
        query_vector=query_embedding,
        limit=limit
    )
    
    # Format results
    results = []
    for hit in search_result:
        results.append({
            "score": hit.score,
            "metadata": hit.payload
        })
    
    return results

def find_related_transformations(transformation_id: str, embedding_provider, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Find transformations related to a given transformation.
    """
    client = QdrantClient(host="localhost", port=55000)
    
    # First get the transformation we want to find related items for
    search_result = client.scroll(
        collection_name="keboola_metadata",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="id",
                    match=MatchValue(value=transformation_id)
                )
            ]
        ),
        limit=1
    )
    
    if not search_result[0]:
        logger.warning(f"Transformation {transformation_id} not found")
        return []
        
    base_transformation = search_result[0][0]
    
    # Now search for related transformations using the embedding
    related_result = client.search(
        collection_name="keboola_metadata",
        query_vector=base_transformation.vector,
        limit=limit + 1  # Add 1 to account for the input transformation
    )
    
    # Format results, excluding the input transformation
    results = []
    for hit in related_result:
        if hit.id != base_transformation.id:
            results.append({
                "score": hit.score,
                "metadata": hit.payload
            })
            if len(results) >= limit:
                break
    
    return results
