"""
RDF triple store for ontology data with SPARQL query support.
"""
import os
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from pathlib import Path

import networkx as nx
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, NamespaceManager
from rdflib.plugins.sparql import prepareQuery

from omen.core import get_logger
from omen.core.config import settings
from omen.ontology.models import Entity, Relationship, Triple, EntityType, RelationshipType

logger = get_logger(__name__)

# Define Keboola-specific namespaces
KBC = Namespace("http://keboola.com/ontology#")
KBC_ENTITY = Namespace("http://keboola.com/ontology/entity#")
KBC_RELATIONSHIP = Namespace("http://keboola.com/ontology/relationship#")
KBC_PROPERTY = Namespace("http://keboola.com/ontology/property#")


class RDFStore:
    """RDF triple store for ontology data with SPARQL query support."""

    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        """Initialize RDF store with optional storage path.
        
        Args:
            storage_path: Path to RDF storage file
        """
        self.storage_path = Path(storage_path) if storage_path else \
            settings.ontology.storage_path / "ontology.ttl"
            
        # Create parent directories if they don't exist
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize graph with namespaces
        self.graph = Graph(store="Memory")
        self.namespace_manager = NamespaceManager(self.graph)
        self.namespace_manager.bind("kbc", KBC)
        self.namespace_manager.bind("kbc-entity", KBC_ENTITY)
        self.namespace_manager.bind("kbc-rel", KBC_RELATIONSHIP)
        self.namespace_manager.bind("kbc-prop", KBC_PROPERTY)
        self.namespace_manager.bind("rdfs", RDFS)
        self.namespace_manager.bind("owl", OWL)
        self.graph.namespace_manager = self.namespace_manager
        
        # Add basic ontology axioms
        self._add_ontology_axioms()
        
        # Try to load existing data if available
        self.load()

    def _add_ontology_axioms(self) -> None:
        """Add basic ontology axioms to the graph."""
        # Define entity class
        self.graph.add((KBC.Entity, RDF.type, OWL.Class))
        self.graph.add((KBC.Entity, RDFS.label, Literal("Entity")))
        
        # Define relationship class
        self.graph.add((KBC.Relationship, RDF.type, OWL.Class))
        self.graph.add((KBC.Relationship, RDFS.label, Literal("Relationship")))
        
        # Define entity types as subclasses
        for entity_type in EntityType:
            entity_class = KBC[entity_type.value.capitalize()]
            self.graph.add((entity_class, RDF.type, OWL.Class))
            self.graph.add((entity_class, RDFS.subClassOf, KBC.Entity))
            self.graph.add((entity_class, RDFS.label, Literal(entity_type.value)))
        
        # Define relationship types as subclasses
        for rel_type in RelationshipType:
            rel_class = KBC[rel_type.value]
            self.graph.add((rel_class, RDF.type, OWL.ObjectProperty))
            self.graph.add((rel_class, RDFS.subPropertyOf, KBC.relationship))
            self.graph.add((rel_class, RDFS.label, Literal(rel_type.value)))

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph.
        
        Args:
            entity: Entity to add
        """
        entity_uri = KBC_ENTITY[entity.id]
        entity_class = KBC[entity.type.value.capitalize()]
        
        # Add basic entity information
        self.graph.add((entity_uri, RDF.type, entity_class))
        self.graph.add((entity_uri, RDFS.label, Literal(entity.name)))
        
        if entity.description:
            self.graph.add((entity_uri, RDFS.comment, Literal(entity.description)))
        
        # Add creation and update timestamps
        self.graph.add((entity_uri, KBC.createdAt, Literal(entity.created_at.isoformat(), datatype=XSD.dateTime)))
        self.graph.add((entity_uri, KBC.updatedAt, Literal(entity.updated_at.isoformat(), datatype=XSD.dateTime)))
        
        # Add properties
        for prop_name, prop_value in entity.properties.items():
            prop_uri = KBC_PROPERTY[prop_name]
            
            # Handle different property value types
            if isinstance(prop_value, str):
                self.graph.add((entity_uri, prop_uri, Literal(prop_value)))
            elif isinstance(prop_value, int):
                self.graph.add((entity_uri, prop_uri, Literal(prop_value, datatype=XSD.integer)))
            elif isinstance(prop_value, float):
                self.graph.add((entity_uri, prop_uri, Literal(prop_value, datatype=XSD.float)))
            elif isinstance(prop_value, bool):
                self.graph.add((entity_uri, prop_uri, Literal(prop_value, datatype=XSD.boolean)))
            elif prop_value is None:
                continue
            else:
                # For complex types, store as JSON string
                import json
                self.graph.add((entity_uri, prop_uri, Literal(json.dumps(prop_value))))

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph.
        
        Args:
            relationship: Relationship to add
        """
        try:
            source_uri = KBC_ENTITY[relationship.source_id]
            target_uri = KBC_ENTITY[relationship.target_id]
            rel_type_uri = KBC[relationship.type.value]
            rel_uri = KBC_RELATIONSHIP[relationship.id]
            
            # Validate URIs and ensure they're properly formed
            for uri in [source_uri, target_uri, rel_type_uri, rel_uri]:
                if not isinstance(uri, URIRef):
                    logger.warning(f"Invalid URI reference: {uri}")
            
            # Verify that source and target entities are valid
            if not relationship.source_id or not relationship.target_id:
                logger.warning(f"Skipping relationship with empty source or target: {relationship.id}")
                return
                
            # Add the direct relationship between source and target
            self.graph.add((source_uri, rel_type_uri, target_uri))
            
            # Also create a relationship instance for more detailed information
            self.graph.add((rel_uri, RDF.type, KBC.Relationship))
            self.graph.add((rel_uri, KBC.sourceEntity, source_uri))
            self.graph.add((rel_uri, KBC.targetEntity, target_uri))
            self.graph.add((rel_uri, KBC.relationshipType, rel_type_uri))
            
            # Add creation and update timestamps
            if relationship.created_at:
                self.graph.add((rel_uri, KBC.createdAt, Literal(relationship.created_at.isoformat(), datatype=XSD.dateTime)))
            if relationship.updated_at:
                self.graph.add((rel_uri, KBC.updatedAt, Literal(relationship.updated_at.isoformat(), datatype=XSD.dateTime)))
            
            # Add properties
            for prop_name, prop_value in relationship.properties.items():
                if not prop_name:  # Skip empty property names
                    continue
                    
                prop_uri = KBC_PROPERTY[prop_name]
                
                # Handle different property value types (same as for entities)
                if isinstance(prop_value, str):
                    self.graph.add((rel_uri, prop_uri, Literal(prop_value)))
                elif isinstance(prop_value, int):
                    self.graph.add((rel_uri, prop_uri, Literal(prop_value, datatype=XSD.integer)))
                elif isinstance(prop_value, float):
                    self.graph.add((rel_uri, prop_uri, Literal(prop_value, datatype=XSD.float)))
                elif isinstance(prop_value, bool):
                    self.graph.add((rel_uri, prop_uri, Literal(prop_value, datatype=XSD.boolean)))
                elif prop_value is None:
                    continue
                else:
                    # For complex types, store as JSON string
                    import json
                    self.graph.add((rel_uri, prop_uri, Literal(json.dumps(prop_value))))
        except Exception as e:
            logger.error(f"Error adding relationship to RDF store: {e}")
            # Continue without failing the whole process

    def add_triple(self, triple: Triple) -> None:
        """Add a triple to the graph.
        
        Args:
            triple: Triple to add
        """
        subject_uri = KBC_ENTITY[triple.subject]
        predicate_uri = KBC[triple.predicate]
        object_uri = KBC_ENTITY[triple.object]
        
        self.graph.add((subject_uri, predicate_uri, object_uri))

    def get_triples(self) -> List[Triple]:
        """Get all triples from the graph.
        
        Returns:
            List of Triple objects
        """
        triples = []
        
        # Query for direct relationships between entities
        for s, p, o in self.graph.triples((None, None, None)):
            # Skip metadata triples
            if not (isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef)):
                continue
                
            # Skip if not entity-to-entity relationship
            if not (str(s).startswith(str(KBC_ENTITY)) and str(o).startswith(str(KBC_ENTITY))):
                continue
                
            # Skip if predicate is not a relationship type
            if not any(str(p) == str(KBC[rel.value]) for rel in RelationshipType):
                continue
                
            # Extract IDs from URIs
            subject_id = str(s).split('#')[-1]
            predicate = str(p).split('#')[-1]
            object_id = str(o).split('#')[-1]
            
            triple = Triple(
                subject=subject_id,
                predicate=predicate,
                object=object_id
            )
            triples.append(triple)
            
        return triples

    def query_sparql(self, query_string: str) -> List[Dict[str, Any]]:
        """Execute a SPARQL query on the graph.
        
        Args:
            query_string: SPARQL query string
            
        Returns:
            List of result dictionaries
        """
        try:
            # Prepare query with namespace bindings
            query = prepareQuery(
                query_string,
                initNs={
                    "kbc": KBC,
                    "kbc-entity": KBC_ENTITY,
                    "kbc-rel": KBC_RELATIONSHIP,
                    "kbc-prop": KBC_PROPERTY,
                    "rdf": RDF,
                    "rdfs": RDFS,
                }
            )
            
            # Execute query
            results = list(self.graph.query(query))
            
            # Convert results to dictionaries
            result_list = []
            var_names = [str(var) for var in query.algebra.vars]
            
            for result in results:
                result_dict = {}
                for i, var in enumerate(var_names):
                    value = result[i]
                    
                    # Convert URIRef to string
                    if isinstance(value, URIRef):
                        result_dict[var] = str(value)
                    # Convert Literal to python value
                    elif isinstance(value, Literal):
                        result_dict[var] = value.value
                    # Convert BNode to string
                    elif isinstance(value, BNode):
                        result_dict[var] = f"_:{value}"
                    # None or other
                    else:
                        result_dict[var] = value
                        
                result_list.append(result_dict)
                
            return result_list
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            raise

    def to_networkx(self) -> nx.DiGraph:
        """Convert RDF graph to NetworkX directed graph.
        
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add entities as nodes
        entities_query = """
        SELECT ?entity ?label ?type
        WHERE {
            ?entity rdf:type ?class .
            ?class rdfs:subClassOf kbc:Entity .
            OPTIONAL { ?entity rdfs:label ?label }
            BIND(STRAFTER(STR(?class), "#") AS ?type)
        }
        """
        
        for result in self.query_sparql(entities_query):
            entity_id = result["entity"].split('#')[-1]
            entity_label = result.get("label", entity_id)
            entity_type = result.get("type", "Unknown")
            
            G.add_node(entity_id, label=entity_label, type=entity_type)
        
        # Add relationships as edges
        relationship_query = """
        SELECT ?source ?target ?rel_type
        WHERE {
            ?source ?rel ?target .
            ?rel rdfs:subPropertyOf kbc:relationship .
            BIND(STRAFTER(STR(?rel), "#") AS ?rel_type)
        }
        """
        
        for result in self.query_sparql(relationship_query):
            source_id = result["source"].split('#')[-1]
            target_id = result["target"].split('#')[-1]
            rel_type = result.get("rel_type", "Unknown")
            
            G.add_edge(source_id, target_id, type=rel_type)
            
        return G

    def serialize(self, format: str = "turtle") -> str:
        """Serialize the graph to a string.
        
        Args:
            format: Serialization format (turtle, xml, json-ld, etc.)
            
        Returns:
            Serialized graph as string
        """
        return self.graph.serialize(format=format)

    def deserialize(self, data: str, format: str = "turtle") -> None:
        """Deserialize a string into the graph.
        
        Args:
            data: Serialized graph data
            format: Serialization format
        """
        self.graph.parse(data=data, format=format)

    def save(self, format: str = "turtle") -> None:
        """Save the graph to storage_path.
        
        Args:
            format: Serialization format
        """
        try:
            # Create parent directories if they don't exist
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize and save
            serialized = self.serialize(format=format)
            with open(self.storage_path, "w") as f:
                f.write(serialized)
                
            logger.info(f"Saved RDF graph to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving RDF graph: {e}")
            raise

    def load(self) -> bool:
        """Load the graph from storage_path if it exists.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.storage_path.exists():
            logger.info(f"RDF storage path {self.storage_path} does not exist, using empty graph")
            return False
            
        try:
            # Clear existing graph
            self.graph = Graph(store="Memory")
            self.graph.namespace_manager = self.namespace_manager
            
            # Load from file
            self.graph.parse(source=str(self.storage_path), format="turtle")
            
            # Re-add basic ontology axioms to ensure they exist
            self._add_ontology_axioms()
            
            logger.info(f"Loaded RDF graph from {self.storage_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading RDF graph: {e}")
            return False

    def clear(self) -> None:
        """Clear the graph."""
        self.graph = Graph(store="Memory")
        self.graph.namespace_manager = self.namespace_manager
        self._add_ontology_axioms()

    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get all entities of a specified type.
        
        Args:
            entity_type: Type of entity to retrieve (e.g., "table", "column")
            
        Returns:
            List of entity dictionaries
        """
        entity_class = KBC[entity_type.capitalize()]
        query = f"""
            SELECT ?entity ?name ?desc
            WHERE {{
                ?entity rdf:type {entity_class.n3()} .
                ?entity rdfs:label ?name .
                OPTIONAL {{ ?entity rdfs:comment ?desc }}
            }}
        """
        
        results = []
        for row in self.query_sparql(query):
            entity_id = str(row.get('entity')).split('#')[-1]
            results.append({
                'id': entity_id,
                'name': str(row.get('name')),
                'description': str(row.get('desc')) if 'desc' in row else None,
                'type': entity_type
            })
        
        return results

    def get_entity_properties(self, entity_id: str) -> Dict[str, Any]:
        """Get all properties for a specific entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Dictionary of property names and values
        """
        entity_uri = KBC_ENTITY[entity_id]
        properties = {}
        
        for _, p, o in self.graph.triples((entity_uri, None, None)):
            # Skip rdf:type and rdfs:label/comment
            if p in (RDF.type, RDFS.label, RDFS.comment):
                continue
                
            # Extract property name
            prop_name = str(p).split('#')[-1]
            
            # Extract value
            if isinstance(o, Literal):
                properties[prop_name] = o.value
            elif isinstance(o, URIRef):
                properties[prop_name] = str(o).split('#')[-1]
            else:
                properties[prop_name] = str(o)
                
        return properties

    def get_related_entities(self, entity_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get entities related to the specified entity up to a certain path depth.
        
        Args:
            entity_id: ID of the entity
            max_depth: Maximum path length to explore
            
        Returns:
            List of related entity dictionaries with relationship information
        """
        entity_uri = KBC_ENTITY[entity_id]
        results = []
        visited = set()
        
        def explore(node, depth=0, path=None):
            if depth >= max_depth or node in visited:
                return
                
            visited.add(node)
            path = path or []
            
            # Explore outgoing relationships
            for s, p, o in self.graph.triples((node, None, None)):
                # Skip if not entity-to-entity relationship
                if not isinstance(o, URIRef) or not str(o).startswith(str(KBC_ENTITY)):
                    continue
                    
                # Skip rdf:type and other metadata
                if p in (RDF.type, RDFS.label, RDFS.comment):
                    continue
                    
                # Get relation and entity information
                rel_type = str(p).split('#')[-1]
                target_id = str(o).split('#')[-1]
                
                # Get entity name
                target_name = None
                for _, _, name in self.graph.triples((o, RDFS.label, None)):
                    target_name = str(name)
                    break
                    
                # Add to results
                results.append({
                    'source_id': entity_id,
                    'target_id': target_id,
                    'relationship': rel_type,
                    'direction': 'outgoing',
                    'path_length': depth + 1,
                    'path': path + [{'id': target_id, 'name': target_name, 'rel': rel_type}]
                })
                
                # Recurse
                explore(o, depth + 1, path + [{'id': target_id, 'name': target_name, 'rel': rel_type}])
            
            # Explore incoming relationships
            for s, p, o in self.graph.triples((None, None, node)):
                # Skip if not entity-to-entity relationship
                if not isinstance(s, URIRef) or not str(s).startswith(str(KBC_ENTITY)):
                    continue
                    
                # Skip rdf:type and other metadata
                if p in (RDF.type, RDFS.label, RDFS.comment):
                    continue
                    
                # Get relation and entity information
                rel_type = str(p).split('#')[-1]
                source_id = str(s).split('#')[-1]
                
                # Get entity name
                source_name = None
                for _, _, name in self.graph.triples((s, RDFS.label, None)):
                    source_name = str(name)
                    break
                    
                # Add to results
                results.append({
                    'source_id': source_id,
                    'target_id': entity_id,
                    'relationship': rel_type,
                    'direction': 'incoming',
                    'path_length': depth + 1,
                    'path': path + [{'id': source_id, 'name': source_name, 'rel': rel_type}]
                })
                
                # Recurse
                explore(s, depth + 1, path + [{'id': source_id, 'name': source_name, 'rel': rel_type}])
        
        # Start exploration
        explore(entity_uri)
        
        return results

    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform an advanced semantic search within the RDF graph.
        
        This method uses SPARQL to search for entities that match the query semantically,
        taking into account entity names, types, properties, and relationship patterns.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        if len(self.graph) == 0:
            logger.info("No entities in graph for semantic search")
            return []
            
        try:
            # Clean and prepare the query
            clean_query = query.lower().strip()
            query_terms = [term for term in clean_query.split() if len(term) > 2]
            
            if not query_terms:
                # If no meaningful terms, use the original query
                query_terms = [clean_query]
            
            # Get the proper namespace bindings from the graph
            from rdflib import Namespace, Graph, URIRef, Literal
            from rdflib.namespace import RDF, RDFS, XSD
            
            # Define the namespace for our ontology entities
            ONT = Namespace("http://keboola.com/ontologies/metadata#")
            
            # Prepare the SPARQL query with advanced semantic search capabilities
            sparql_query = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ont: <http://keboola.com/ontologies/metadata#>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                
                SELECT DISTINCT ?entity ?name ?type ?desc (SUM(?matchScore) AS ?score) ?relCount
                WHERE {
                    # Get basic entity info
                    ?entity rdf:type ?entityClass .
                    ?entity rdfs:label ?name .
                    
                    # Only include actual entities
                    FILTER EXISTS { ?entityClass rdfs:subClassOf* ont:Entity }
                    
                    # Get entity type
                    ?entity ont:type ?type .
                    
                    # Optional description
                    OPTIONAL { ?entity rdfs:comment ?desc }
                    
                    # Count related entities (weighting for well-connected entities)
                    {
                        SELECT ?entity (COUNT(?related) AS ?relCount)
                        WHERE {
                            { ?entity ?anyRelation ?related }
                            UNION
                            { ?related ?anyRelation ?entity }
                            FILTER(?anyRelation != rdf:type)
                        }
                        GROUP BY ?entity
                    }
                    
                    # Calculate name match score - exact matches get higher weight
                    {
                        SELECT ?entity (SUM(?termScore) AS ?matchScore)
                        WHERE {
                            ?entity rdfs:label ?name .
                            
                            # Define scores for different match types
                            VALUES (?term ?termScore) {
            """
            
            # Add each query term with different match scores
            for term in query_terms:
                sparql_query += f"""
                                ("{term}" 3.0) # Full query term
                """
            
            sparql_query += """
                            }
                            
                            # Calculate scores based on match type
                            BIND(
                                IF(CONTAINS(LCASE(?name), ?term), ?termScore,
                                  IF(CONTAINS(LCASE(STR(?desc)), ?term), ?termScore * 0.5, 0)
                                ) AS ?score
                            )
                            
                            # Only include entities with at least one match
                            FILTER(?score > 0)
                        }
                        GROUP BY ?entity
                    }
                    
                    # Add relationship-based context scoring
                    UNION
                    {
                        # Find entities related to entities that match the query
                        ?matchedEntity rdfs:label ?matchedName .
                        ?matchedEntity ?relation ?entity .
                        ?entity rdfs:label ?name .
                        ?entity ont:type ?type .
                        
                        OPTIONAL { ?entity rdfs:comment ?desc }
                        
                        # Find matched entities first
                        FILTER(
                """
            
            # Add term filters for related entity search
            term_filters = []
            for term in query_terms:
                term_filters.append(f"CONTAINS(LCASE(?matchedName), \"{term}\")")
            
            sparql_query += " || ".join(term_filters)
            
            sparql_query += """
                        )
                        
                        # Exclude type relations
                        FILTER(?relation != rdf:type)
                        
                        # Score is lower for related entities
                        BIND(1.0 AS ?matchScore)
                    }
                }
                GROUP BY ?entity ?name ?type ?desc ?relCount
                ORDER BY DESC(?score) DESC(?relCount)
                LIMIT %d
            """ % limit
            
            # Initialize results list
            results = []
            
            try:
                # Execute the query
                qres = self.graph.query(sparql_query)
                
                # Process results
                for row in qres:
                    # Extract entity URI and properties
                    entity_uri = str(row.entity) if hasattr(row, 'entity') else None
                    if not entity_uri:
                        continue
                        
                    entity_id = entity_uri.split('#')[-1]
                    name = str(row.name) if hasattr(row, 'name') else None
                    entity_type = str(row.type).split('#')[-1] if hasattr(row, 'type') else "unknown"
                    description = str(row.desc) if hasattr(row, 'desc') else None
                    score = float(row.score) if hasattr(row, 'score') else 0.5
                    rel_count = int(row.relCount) if hasattr(row, 'relCount') else 0
                    
                    # Add relationship count to score to boost well-connected entities
                    adjusted_score = score + (rel_count * 0.01)
                    
                    # Build entity info
                    entity_info = {
                        "id": entity_id,
                        "name": name,
                        "type": entity_type,
                        "score": adjusted_score
                    }
                    
                    if description:
                        entity_info["description"] = description
                    
                    # Get relationships for context
                    entity_info["relationships"] = self._get_direct_relationships(entity_id)
                    
                    results.append(entity_info)
                    
                # If we didn't get results (or very few), fall back to a simpler label search
                if not results:
                    simple_results: Dict[str, Dict[str, Any]] = {}
                    for term in query_terms:
                        fallback = f"""
                            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                            SELECT ?entity ?name WHERE {{
                                ?entity rdfs:label ?name .
                                FILTER(CONTAINS(LCASE(?name), \"{term}\"))
                            }} LIMIT {limit * 5}
                        """
                        try:
                            for row in self.graph.query(fallback):
                                ent = str(row[0]) if row[0] is not None else ""
                                name = str(row[1]) if row[1] is not None else ""
                                if not ent:
                                    continue
                                eid = ent.split('#')[-1]
                                hit = simple_results.setdefault(eid, {"id": eid, "name": name, "score": 1.0, "relationships": []})
                                hit["score"] += 1.0
                        except Exception as qerr:
                            logger.warning(f"Fallback SPARQL failed: {qerr}")
                            continue
                    results = sorted(simple_results.values(), key=lambda x: x["score"], reverse=True)[:limit]
                return results
            except Exception as e:
                logger.error(f"Error executing SPARQL query: {e}")
                return []
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _get_direct_relationships(self, entity_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get direct relationships for an entity.
        
        Args:
            entity_id: Entity ID
            limit: Maximum number of relationships to return
            
        Returns:
            List of relationship dictionaries
        """
        try:
            # Create entity URI
            from rdflib import URIRef, Namespace
            
            ONT = Namespace("http://keboola.com/ontologies/metadata#")
            entity_uri = ONT[entity_id]
            
            # Query for direct relationships
            sparql_query = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ont: <http://keboola.com/ontologies/metadata#>
                
                SELECT DISTINCT ?relation ?relLabel ?target ?targetName ?targetType
                WHERE {
                    {
                        # Outgoing relationships
                        <%s> ?relation ?target .
                        ?target rdfs:label ?targetName .
                        ?target ont:type ?targetType .
                        
                        # Get relation label if available
                        OPTIONAL { ?relation rdfs:label ?relLabel }
                        
                        # Exclude rdf:type relationships
                        FILTER(?relation != rdf:type)
                    }
                    UNION
                    {
                        # Incoming relationships
                        ?target ?relation <%s> .
                        ?target rdfs:label ?targetName .
                        ?target ont:type ?targetType .
                        
                        # Get relation label if available
                        OPTIONAL { ?relation rdfs:label ?relLabel }
                        
                        # Exclude rdf:type relationships
                        FILTER(?relation != rdf:type)
                    }
                }
                LIMIT %d
            """ % (entity_uri, entity_uri, limit)
            
            relationships = []
            try:
                qres = self.graph.query(sparql_query)
                
                for row in qres:
                    # Extract relationship info
                    rel_uri = str(row.relation) if hasattr(row, 'relation') else None
                    rel_label = str(row.relLabel) if hasattr(row, 'relLabel') else None
                    target_uri = str(row.target) if hasattr(row, 'target') else None
                    target_name = str(row.targetName) if hasattr(row, 'targetName') else None
                    target_type = str(row.targetType).split('#')[-1] if hasattr(row, 'targetType') else "unknown"
                    
                    if not rel_uri or not target_uri:
                        continue
                    
                    # Use URI fragment as relation type if no label
                    rel_type = rel_label or rel_uri.split('#')[-1]
                    
                    # Extract target ID from URI
                    target_id = target_uri.split('#')[-1]
                    
                    # Add relationship info
                    relationships.append({
                        "relation_type": rel_type,
                        "target_id": target_id,
                        "target_name": target_name,
                        "target_type": target_type
                    })
                
                return relationships
            except Exception as e:
                logger.error(f"Error getting direct relationships: {e}")
                return []
        except Exception as e:
            logger.error(f"Error building relationship query: {e}")
            return []

    def get_paths_between_entities(
        self, source_id: str, target_id: str, max_length: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """Find paths between two entities in the knowledge graph.
        
        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            max_length: Maximum path length to consider
            
        Returns:
            List of paths, where each path is a list of connection dictionaries
        """
        source_uri = KBC_ENTITY[source_id]
        target_uri = KBC_ENTITY[target_id]
        
        # Convert to NetworkX graph for pathfinding
        nx_graph = self.to_networkx()
        
        # Find all simple paths between the entities
        source_node = str(source_uri)
        target_node = str(target_uri)
        
        # Ensure nodes exist in the graph
        if source_node not in nx_graph or target_node not in nx_graph:
            return []
            
        # Find paths
        paths = []
        try:
            for path in nx.all_simple_paths(nx_graph, source_node, target_node, cutoff=max_length):
                formatted_path = []
                
                # Process each edge in the path
                for i in range(len(path) - 1):
                    s_uri = URIRef(path[i])
                    t_uri = URIRef(path[i + 1])
                    
                    # Find the relationship type
                    rel_type = None
                    for _, p, _ in self.graph.triples((s_uri, None, t_uri)):
                        rel_type = str(p).split('#')[-1]
                        break
                    
                    # Get entity names
                    s_name = None
                    for _, _, name in self.graph.triples((s_uri, RDFS.label, None)):
                        s_name = str(name)
                        break
                        
                    t_name = None
                    for _, _, name in self.graph.triples((t_uri, RDFS.label, None)):
                        t_name = str(name)
                        break
                    
                    # Add to formatted path
                    s_id = str(s_uri).split('#')[-1]
                    t_id = str(t_uri).split('#')[-1]
                    
                    formatted_path.append({
                        'source_id': s_id,
                        'source_name': s_name,
                        'target_id': t_id,
                        'target_name': t_name,
                        'relationship': rel_type
                    })
                
                paths.append(formatted_path)
        except nx.NetworkXNoPath:
            # No path exists
            pass
        
        return paths 