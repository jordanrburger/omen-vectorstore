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
        query: str,
        vector_weight: float = 0.7,
        semantic_weight: float = 0.3,
        limit: int = 10,
        filter_by_metadata_type: Optional[List[str]] = None,
        include_related_entities: bool = False,
        max_depth: int = 1,
        contextual_boost: bool = True, 
        use_query_expansion: bool = True
    ) -> List[SearchResult]:
        """Perform a hybrid search combining vector and semantic search.
        
        Args:
            query: Search query
            vector_weight: Weight for vector search results (0.0 to 1.0)
            semantic_weight: Weight for semantic search results (0.0 to 1.0)
            limit: Maximum number of results to return
            filter_by_metadata_type: Filter results by metadata type
            include_related_entities: Whether to include related entities from the ontology
            max_depth: Maximum depth for related entities search
            contextual_boost: Whether to boost scores based on relationship context
            use_query_expansion: Whether to expand the search query with semantic variants
            
        Returns:
            Hybrid search results
        """
        # Normalize weights
        total = vector_weight + semantic_weight
        if total == 0:
            total = 1.0
        vector_weight = vector_weight / total
        semantic_weight = semantic_weight / total
        
        # Expand query if requested
        expanded_query = query
        if use_query_expansion and semantic_weight > 0 and self.ontology_manager:
            try:
                expanded_query = self._expand_query(query)
                logger.info(f"Expanded query from '{query}' to '{expanded_query}'")
            except Exception as e:
                logger.error(f"Error expanding query: {e}")
        
        # Convert string types to MetadataType if needed
        type_filter = None
        if filter_by_metadata_type:
            from omen.vectorstore.models import MetadataType
            type_filter = []
            for type_str in filter_by_metadata_type:
                try:
                    type_filter.append(MetadataType(type_str))
                except ValueError:
                    logger.warning(f"Invalid metadata type: {type_str}")
        
        # Get vector search results
        vector_results = self.vector_search.search(
            query=expanded_query, 
            limit=limit * 2,
            type_filter=type_filter
        )
        
        # Convert to dict for efficient lookup
        vector_result_dict = {result.document.id: result for result in vector_results if result.document}
        
        # Initialize semantic results
        semantic_results = []
        semantic_result_dict = {}
        
        # Only perform semantic search if weight > 0 and we have an ontology
        if semantic_weight > 0 and self.ontology_manager:
            try:
                # Get semantic search results from the ontology
                semantic_hits = []  # Initialize as empty list first
                try:
                    semantic_hits = self.ontology_manager.triple_store.semantic_search(
                        query=query,
                        limit=limit * 2
                    ) or []
                except Exception as se:
                    logger.error(f"Error executing SPARQL query: {se}")
                    semantic_hits = []  # Use empty list on error
                
                if not semantic_hits:
                    logger.info("Semantic search returned no results, using vector search only")
                
                # Convert to SearchResult objects
                for hit in semantic_hits:
                    entity_id = hit.get("id")
                    entity_type = hit.get("type")
                    entity_score = hit.get("score", 0.0)
                    relationships = hit.get("relationships", [])
                    
                    if not entity_id:
                        continue
                    
                    # Find corresponding document in vector store
                    doc = self._get_document_for_entity(entity_id)
                    if not doc:
                        continue
                        
                    # Apply contextual boosting if enabled
                    if contextual_boost and relationships:
                        # Boost score based on relationship count and relevance
                        relationship_boost = min(0.3, 0.05 * len(relationships))
                        entity_score += relationship_boost
                        
                    # Create search result with semantic score
                    result = SearchResult(
                        document=doc,
                        score=entity_score,
                        vector_score=0.0,
                        semantic_score=entity_score,
                        relationships=relationships
                    )
                    
                    semantic_results.append(result)
                    semantic_result_dict[doc.id] = result
            except Exception as e:
                logger.error(f"Error in semantic search, falling back to vector search only: {e}")
                semantic_results = []
        
        # Combine results
        combined_results = {}
        
        # Add vector results with their scores
        for doc_id, result in vector_result_dict.items():
            combined_results[doc_id] = SearchResult(
                document=result.document,
                score=result.score * vector_weight,
                vector_score=result.score,
                semantic_score=0.0
            )
        
        # Add or update with semantic results
        for result in semantic_results:
            doc_id = result.document.id
            if doc_id in combined_results:
                # Document exists in vector results - update score
                existing = combined_results[doc_id]
                combined_results[doc_id] = SearchResult(
                    document=existing.document,
                    score=existing.vector_score * vector_weight + result.score * semantic_weight,
                    vector_score=existing.vector_score,
                    semantic_score=result.score,
                    relationships=result.relationships
                )
            else:
                # New document from semantic search
                combined_results[doc_id] = SearchResult(
                    document=result.document,
                    score=result.score * semantic_weight,
                    vector_score=0.0,
                    semantic_score=result.score,
                    relationships=result.relationships
                )
        
        # Convert dict to list and sort by score
        results = list(combined_results.values())
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Include related entities if requested
        if include_related_entities and self.ontology_manager:
            results = self._include_related_entities(
                results=results[:limit],
                max_depth=max_depth,
                limit=limit
            )
            
        # Return limited results
        return results[:limit]
    
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
        """Include related entities from the knowledge graph.
        
        Args:
            results: Search results to enhance
            max_depth: Maximum depth for related entities
            limit: Maximum number of final results
            
        Returns:
            Enhanced search results
        """
        if not results:
            return []
            
        try:
            # Process each search result
            for result in results:
                # Skip if no document
                if not result.document:
                    continue
                    
                doc_id = result.document.id
                
                # Get entity ID based on document type
                if hasattr(result.document, 'metadata') and result.document.metadata.get('entity_id'):
                    # If entity_id is directly available in metadata
                    entity_id = result.document.metadata.get('entity_id')
                elif hasattr(result.document.source, 'id') and hasattr(result.document.source, 'type'):
                    # Construct entity ID from source type and ID
                    entity_id = f"{result.document.source.type.value}-{result.document.source.id}"
                else:
                    # Skip if can't determine entity
                    continue
                
                # Check if entity exists in ontology
                if not self.ontology_manager.entity_exists(entity_id):
                    continue
                
                # Get related entities with path information
                try:
                    related = self.ontology_manager.get_related_entities(
                        entity_id=entity_id,
                        max_depth=max_depth
                    )
                    
                    # Skip if no related entities
                    if not related:
                        continue
                        
                    # Format related entities
                    related_entities = []
                    for rel in related:
                        try:
                            target_entity = self.ontology_manager.get_entity(rel["target_id"])
                            if not target_entity:
                                continue
                                
                            related_entities.append({
                                "target_id": rel["target_id"],
                                "target_name": target_entity.name,
                                "target_type": target_entity.type.value,
                                "relationship": rel["relationship"],
                                "direction": rel["direction"],
                                "path_length": rel["path_length"],
                                "path": rel.get("path", [])
                            })
                        except Exception as inner_e:
                            logger.error(f"Error formatting related entity {rel.get('target_id', 'unknown')}: {inner_e}")
                            continue
                    
                    # Add to result
                    result.related_entities = related_entities
                except Exception as e:
                    logger.error(f"Error getting related entities for {entity_id}: {e}")
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Error including related entities: {e}")
            return results  # Return original results on error
    
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

    def _expand_query(self, query: str) -> str:
        """Expand the search query with semantic variants.
        
        This method uses the ontology to find related terms and synonyms
        to expand the original query for better search results.
        
        Args:
            query: Original search query
            
        Returns:
            Expanded search query
        """
        if not self.ontology_manager:
            return query
            
        # Clean the query
        clean_query = query.lower().strip()
        
        # Check if we have any semantic expansion terms from the ontology
        try:
            # Use SPARQL to find related terms in the ontology
            expansion_terms = set()
            
            # Try to find entity names that match the query
            sparql_query = """
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                
                SELECT DISTINCT ?name
                WHERE {
                    ?entity rdfs:label ?name .
                    FILTER(CONTAINS(LCASE(?name), "%s"))
                }
                LIMIT 5
            """ % clean_query
            
            try:
                results = self.ontology_manager.triple_store.query_sparql(sparql_query) or []
                for result in results:
                    if "name" in result:
                        name = result["name"]
                        # Add significant terms from the name
                        for term in name.split():
                            if len(term) > 3 and term.lower() != clean_query:
                                expansion_terms.add(term.lower())
            except Exception as e:
                logger.warning(f"Error finding entity names for query expansion: {e}")
                
            # If we found any expansion terms, add them to the query
            if expansion_terms:
                expanded_query = clean_query + " " + " ".join(expansion_terms)
                return expanded_query
                
        except Exception as e:
            logger.error(f"Error during query expansion: {e}")
            
        return query 