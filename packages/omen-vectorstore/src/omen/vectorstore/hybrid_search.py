"""
Hybrid search module combining vector search with ontology-based semantic search.
"""
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

from omen.core import get_logger
from omen.vectorstore.search import VectorSearch
from omen.vectorstore.models import SearchQuery, SearchResult, MetadataDocument, MetadataType
from omen.ontology.rdf_store import RDFStore
from omen.ontology.manager import OntologyManager

logger = get_logger(__name__)


class HybridSearch:
    """Hybrid search engine combining vector search with ontology-based semantic search."""

    def __init__(
        self,
        vector_search: Optional[VectorSearch] = None,
        ontology_manager: Optional[OntologyManager] = None,
        vector_weight: float = 0.7,
        semantic_weight: float = 0.3,
    ):
        """Initialize the hybrid search engine.
        
        Args:
            vector_search: Vector search engine instance
            ontology_manager: Ontology manager instance
            vector_weight: Weight of vector search results in combined ranking (0-1)
            semantic_weight: Weight of semantic search results in combined ranking (0-1)
        """
        self.vector_search = vector_search or VectorSearch()
        self.ontology_manager = ontology_manager or OntologyManager()
        
        # Ensure weights sum to 1.0
        total_weight = vector_weight + semantic_weight
        self.vector_weight = vector_weight / total_weight
        self.semantic_weight = semantic_weight / total_weight
        
        # Cache for entity-to-document mapping
        self._entity_document_cache = {}
    
    def search(
        self,
        query: Union[str, SearchQuery],
        limit: int = 10,
        offset: int = 0,
        type_filter: Optional[List[MetadataType]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        vector_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        include_related: bool = False,
        max_related_depth: int = 1,
    ) -> List[SearchResult]:
        """Perform hybrid search using both vector similarity and semantic knowledge graph.
        
        Args:
            query: The search query text or SearchQuery object
            limit: Maximum number of results to return
            offset: Offset for pagination
            type_filter: Filter results by metadata type
            metadata_filter: Filter results by metadata fields
            vector_weight: Weight of vector search results (0-1, overrides instance setting)
            semantic_weight: Weight of semantic search results (0-1, overrides instance setting)
            include_related: Whether to include semantically related documents in results
            max_related_depth: Maximum depth for finding related entities
            
        Returns:
            List of search results with combined scores
        """
        # Handle search query
        query_text = query if isinstance(query, str) else query.query
        
        # Set weights for this search
        v_weight = vector_weight if vector_weight is not None else self.vector_weight
        s_weight = semantic_weight if semantic_weight is not None else self.semantic_weight
        
        # Normalize weights
        total_weight = v_weight + s_weight
        v_weight = v_weight / total_weight
        s_weight = s_weight / total_weight
        
        # Perform vector search
        vector_results = self.vector_search.search(
            query=query,
            limit=limit * 2,  # Get more results for better hybrid ranking
            offset=offset,
            type_filter=type_filter,
            metadata_filter=metadata_filter,
        )
        
        # Extract entity types from type_filter
        entity_types = None
        if type_filter:
            entity_types = [t.value.lower() for t in type_filter]
        
        # Perform semantic search on the ontology
        semantic_results = self.ontology_manager.triple_store.semantic_search(
            query=query_text,
            limit=limit * 2,  # Get more results for better hybrid ranking
            entity_types=entity_types,
        )
        
        # Combine results
        combined_results = self._combine_results(
            vector_results=vector_results,
            semantic_results=semantic_results,
            vector_weight=v_weight,
            semantic_weight=s_weight,
            limit=limit,
        )
        
        # Include related entities if requested
        if include_related and combined_results:
            combined_results = self._include_related_entities(
                combined_results,
                max_depth=max_related_depth,
                limit=limit
            )
        
        return combined_results
    
    def _combine_results(
        self,
        vector_results: List[SearchResult],
        semantic_results: List[Dict[str, Any]],
        vector_weight: float,
        semantic_weight: float,
        limit: int,
    ) -> List[SearchResult]:
        """Combine vector and semantic search results with weighted scores.
        
        Args:
            vector_results: Results from vector search
            semantic_results: Results from semantic search
            vector_weight: Weight for vector results (0-1)
            semantic_weight: Weight for semantic results (0-1)
            limit: Maximum number of combined results to return
            
        Returns:
            Combined search results with adjusted scores
        """
        # Create a mapping of document ID to SearchResult
        document_map: Dict[str, SearchResult] = {}
        
        # Normalize vector scores (0-1)
        max_vector_score = max([r.score for r in vector_results]) if vector_results else 1.0
        
        # Add vector results to the map
        for result in vector_results:
            doc_id = result.document.id
            
            # Normalize score
            normalized_score = result.score / max_vector_score if max_vector_score > 0 else 0
            
            document_map[doc_id] = SearchResult(
                document=result.document,
                score=normalized_score * vector_weight,
                vector_score=normalized_score,
                semantic_score=0.0,
                related_entities=[],
            )
        
        # Normalize semantic scores (0-1)
        max_semantic_score = max([r["score"] for r in semantic_results]) if semantic_results else 1.0
        
        # Add semantic results to the map
        for result in semantic_results:
            entity_id = result["id"]
            
            # Skip if no corresponding document
            document = self._get_document_for_entity(entity_id)
            if not document:
                continue
                
            doc_id = document.id
            normalized_score = result["score"] / max_semantic_score if max_semantic_score > 0 else 0
            
            if doc_id in document_map:
                # Update existing entry
                document_map[doc_id].score += normalized_score * semantic_weight
                document_map[doc_id].semantic_score = normalized_score
            else:
                # Create new entry
                document_map[doc_id] = SearchResult(
                    document=document,
                    score=normalized_score * semantic_weight,
                    vector_score=0.0,
                    semantic_score=normalized_score,
                    related_entities=[],
                )
        
        # Convert map to list and sort by combined score
        combined_results = list(document_map.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results[:limit]
    
    def _include_related_entities(
        self,
        results: List[SearchResult],
        max_depth: int = 1,
        limit: int = 10,
    ) -> List[SearchResult]:
        """Include semantically related entities in search results.
        
        Args:
            results: Search results to enhance
            max_depth: Maximum relationship depth to traverse
            limit: Maximum number of final results to return
            
        Returns:
            Enhanced search results with related entities
        """
        enhanced_results = []
        
        for result in results:
            # Find corresponding entity in the ontology
            entity_id = self._get_entity_for_document(result.document)
            if not entity_id:
                enhanced_results.append(result)
                continue
            
            # Get related entities
            related_entities = self.ontology_manager.triple_store.get_related_entities(
                entity_id=entity_id,
                max_depth=max_depth,
            )
            
            # Add related entities to the result
            result.related_entities = related_entities
            enhanced_results.append(result)
        
        # Return limited results
        return enhanced_results[:limit]
    
    def _get_document_for_entity(self, entity_id: str) -> Optional[MetadataDocument]:
        """Find the corresponding document for an entity.
        
        This uses a simple heuristic where the entity ID matches the document ID,
        or is contained in document metadata.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Corresponding document or None if not found
        """
        # First check if we have a cached mapping
        if entity_id in self._entity_document_cache:
            return self._entity_document_cache[entity_id]
        
        # Try to find an exact match by ID
        # This would require implementing get_document_by_id in the VectorSearch class
        # For now, we use a simple workaround with a blank query
        
        # Create a query with a filter for the entity ID
        search_query = SearchQuery(
            query="",  # Empty query will rely on filters
            filter={"id": entity_id},
            limit=1,
        )
        
        # Perform vector search
        results = self.vector_search.search(query=search_query)
        
        if results:
            # Cache and return the document
            document = results[0].document
            self._entity_document_cache[entity_id] = document
            return document
        
        return None
    
    def _get_entity_for_document(self, document: MetadataDocument) -> Optional[str]:
        """Find the corresponding entity ID for a document.
        
        Args:
            document: The document
            
        Returns:
            Entity ID or None if not found
        """
        # Check if the document ID is an entity ID
        if self.ontology_manager.entity_exists(document.id):
            return document.id
        
        # Check metadata for entity ID
        if document.metadata and "entity_id" in document.metadata:
            entity_id = document.metadata["entity_id"]
            if self.ontology_manager.entity_exists(entity_id):
                return entity_id
        
        return None

    def entity_based_recommendation(
        self,
        entity_id: str,
        limit: int = 10,
        include_paths: bool = True,
    ) -> List[Dict[str, Any]]:
        """Find recommendations based on ontology relationships.
        
        This uses the knowledge graph to find semantically related entities.
        
        Args:
            entity_id: ID of the entity to find recommendations for
            limit: Maximum number of recommendations to return
            include_paths: Whether to include relationship paths in results
            
        Returns:
            List of recommended entities with relationship information
        """
        if not self.ontology_manager.entity_exists(entity_id):
            return []
            
        # Get related entities from the ontology
        related = self.ontology_manager.triple_store.get_related_entities(
            entity_id=entity_id,
            max_depth=2,
        )
        
        # Sort by path length (shorter path = stronger relationship)
        related.sort(key=lambda x: x["path_length"])
        
        # Format recommendations
        recommendations = []
        for rel in related[:limit]:
            target_id = rel["target_id"]
            
            # Find corresponding document
            document = self._get_document_for_entity(target_id)
            
            # Get target entity details
            target_entity = self.ontology_manager.get_entity(target_id)
            
            if target_entity:
                rec = {
                    "id": target_id,
                    "name": target_entity.name,
                    "type": target_entity.type.value,
                    "description": target_entity.description,
                    "relationship": rel["relationship"],
                    "relationship_path": rel["path"] if include_paths else None,
                    "score": 1.0 / rel["path_length"],  # Simple scoring based on path length
                    "document": document.to_dict() if document else None,
                }
                recommendations.append(rec)
        
        return recommendations

    def hybrid_recommendation(
        self,
        entity_id: str,
        limit: int = 10,
        vector_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Generate recommendations using both vector similarity and semantic relationships.
        
        Args:
            entity_id: ID of the entity to find recommendations for
            limit: Maximum number of recommendations to return
            vector_weight: Weight for vector-based recommendations (0-1)
            semantic_weight: Weight for semantic-based recommendations (0-1)
            
        Returns:
            List of hybrid recommendations
        """
        # Set weights for this recommendation
        v_weight = vector_weight if vector_weight is not None else self.vector_weight
        s_weight = semantic_weight if semantic_weight is not None else self.semantic_weight
        
        # Get the document for the entity
        document = self._get_document_for_entity(entity_id)
        if not document:
            # Fall back to entity-based recommendation if no document found
            return self.entity_based_recommendation(entity_id, limit)
        
        # Get vector-based similar documents
        vector_similar = self.vector_search.find_similar(
            document=document,
            limit=limit * 2,
        )
        
        # Get entity-based recommendations
        semantic_similar = self.entity_based_recommendation(
            entity_id=entity_id,
            limit=limit * 2,
            include_paths=True,
        )
        
        # Build result maps
        results = {}
        
        # Add vector results
        for result in vector_similar:
            doc_id = result.document.id
            results[doc_id] = {
                "id": doc_id,
                "document": result.document,
                "vector_score": result.score,
                "semantic_score": 0.0,
                "final_score": result.score * v_weight,
                "relationship_path": None,
            }
        
        # Add semantic results
        for result in semantic_similar:
            entity_id = result["id"]
            doc = self._get_document_for_entity(entity_id)
            
            if not doc:
                continue
                
            doc_id = doc.id
            sem_score = result["score"]
            
            if doc_id in results:
                # Update existing entry
                results[doc_id]["semantic_score"] = sem_score
                results[doc_id]["final_score"] += sem_score * s_weight
                results[doc_id]["relationship_path"] = result["relationship_path"]
            else:
                # Add new entry
                results[doc_id] = {
                    "id": doc_id,
                    "document": doc,
                    "vector_score": 0.0,
                    "semantic_score": sem_score,
                    "final_score": sem_score * s_weight,
                    "relationship_path": result["relationship_path"],
                }
        
        # Convert to list and sort by final score
        result_list = list(results.values())
        result_list.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Format final results
        formatted_results = []
        for result in result_list[:limit]:
            doc = result["document"]
            formatted_results.append({
                "id": doc.id,
                "title": doc.title,
                "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "type": doc.type.value if doc.type else None,
                "vector_score": result["vector_score"],
                "semantic_score": result["semantic_score"],
                "final_score": result["final_score"],
                "relationship_path": result["relationship_path"],
            })
        
        return formatted_results 