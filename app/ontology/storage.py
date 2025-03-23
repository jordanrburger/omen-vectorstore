"""
Storage module for RDF triples.

This module provides functionality for storing, retrieving, and
querying RDF triples in the ontology.
"""

import json
import logging
from typing import Dict, List, Set, Optional, Any, Iterator, Tuple
from pathlib import Path
import os

from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS
from rdflib.plugins.sparql import prepareQuery
from rdflib.namespace import XSD

from app.ontology.models import Triple, Entity, Relationship, EntityType, RelationshipType


logger = logging.getLogger(__name__)


# Define base namespace
KBL = Namespace("http://keboola.com/ontology#")


class TripleStore:
    """
    Storage for RDF triples with serialization and querying capabilities.
    
    This class manages the persistence of RDF triples and provides
    methods for SPARQL querying and graph manipulation.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the triple store.
        
        Args:
            storage_path: Optional path to store serialized RDF data
        """
        self.graph = Graph()
        self.graph.bind("kbl", KBL)
        self.storage_path = storage_path
        self._init_namespaces()
    
    def _init_namespaces(self):
        """Initialize standard namespaces."""
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("xsd", XSD)
    
    def add_triple(self, triple: Triple) -> None:
        """
        Add a triple to the store.
        
        Args:
            triple: The triple to add
        """
        subject = URIRef(f"{KBL}{triple.subject}")
        predicate = URIRef(f"{KBL}{triple.predicate}")
        
        # Handle different object types
        if triple.object.startswith("http"):
            obj = URIRef(triple.object)
        elif triple.object.isdigit():
            obj = Literal(int(triple.object))
        elif triple.object.replace(".", "", 1).isdigit():
            obj = Literal(float(triple.object))
        elif triple.object.lower() in ("true", "false"):
            obj = Literal(triple.object.lower() == "true")
        else:
            obj = URIRef(f"{KBL}{triple.object}")
        
        self.graph.add((subject, predicate, obj))
        
        # Add metadata as additional triples if present
        if triple.metadata:
            for key, value in triple.metadata.items():
                if value is not None:
                    meta_predicate = URIRef(f"{KBL}meta_{key}")
                    if isinstance(value, (int, float, bool)):
                        meta_obj = Literal(value)
                    else:
                        meta_obj = Literal(str(value))
                    self.graph.add((subject, meta_predicate, meta_obj))
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the store.
        
        Args:
            entity: The entity to add
        """
        entity_uri = URIRef(f"{KBL}{entity.id}")
        type_uri = URIRef(f"{KBL}{entity.type.value}")
        
        # Add basic entity triples
        self.graph.add((entity_uri, RDF.type, type_uri))
        self.graph.add((entity_uri, RDFS.label, Literal(entity.name)))
        
        # Add properties
        for key, value in entity.properties.items():
            prop_uri = URIRef(f"{KBL}prop_{key}")
            if isinstance(value, (int, float, bool)):
                obj = Literal(value)
            else:
                obj = Literal(str(value))
            self.graph.add((entity_uri, prop_uri, obj))
        
        # Add metadata
        for key, value in entity.metadata.items():
            if value is not None:
                meta_uri = URIRef(f"{KBL}meta_{key}")
                if isinstance(value, (int, float, bool)):
                    obj = Literal(value)
                else:
                    obj = Literal(str(value))
                self.graph.add((entity_uri, meta_uri, obj))
    
    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship to the store.
        
        Args:
            relationship: The relationship to add
        """
        triple = Triple.from_relationship(relationship)
        self.add_triple(triple)
        
        # Add metadata for the relationship
        rel_uri = URIRef(f"{KBL}{relationship.id}")
        source_uri = URIRef(f"{KBL}{relationship.source_id}")
        target_uri = URIRef(f"{KBL}{relationship.target_id}")
        rel_type_uri = URIRef(f"{KBL}{relationship.type.value}")
        
        self.graph.add((rel_uri, RDF.type, rel_type_uri))
        self.graph.add((rel_uri, URIRef(f"{KBL}hasSource"), source_uri))
        self.graph.add((rel_uri, URIRef(f"{KBL}hasTarget"), target_uri))
        
        # Add relationship properties
        for key, value in relationship.properties.items():
            prop_uri = URIRef(f"{KBL}prop_{key}")
            if isinstance(value, (int, float, bool)):
                obj = Literal(value)
            else:
                obj = Literal(str(value))
            self.graph.add((rel_uri, prop_uri, obj))
    
    def get_triples(self) -> List[Triple]:
        """
        Get all triples from the store.
        
        Returns:
            List of Triple objects
        """
        triples = []
        
        for s, p, o in self.graph:
            # Filter out RDF and RDFS statements
            if str(p).startswith(str(KBL)):
                subject = str(s).replace(str(KBL), "")
                predicate = str(p).replace(str(KBL), "")
                
                if isinstance(o, Literal):
                    obj = str(o)
                else:
                    obj = str(o).replace(str(KBL), "")
                
                triples.append(Triple(subject=subject, predicate=predicate, object=obj))
        
        return triples
    
    def query_sparql(self, query_string: str) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL query on the graph.
        
        Args:
            query_string: SPARQL query string
            
        Returns:
            List of results as dictionaries
        """
        try:
            query = prepareQuery(
                query_string,
                initNs={"kbl": KBL, "rdf": RDF, "rdfs": RDFS}
            )
            results = self.graph.query(query)
            
            output = []
            for row in results:
                result = {}
                for var in results.vars:
                    value = row[var]
                    if value:
                        if isinstance(value, URIRef):
                            # Convert URI to local name
                            if str(value).startswith(str(KBL)):
                                result[var] = str(value).replace(str(KBL), "")
                            else:
                                result[var] = str(value)
                        elif isinstance(value, Literal):
                            result[var] = value.toPython()
                        else:
                            result[var] = str(value)
                output.append(result)
            
            return output
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            return []
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Dict[str, Any]]:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: The entity type to filter by
            
        Returns:
            List of entity dictionaries
        """
        query = f"""
        SELECT ?id ?name
        WHERE {{
            ?entity rdf:type kbl:{entity_type.value} .
            ?entity rdfs:label ?name .
            BIND(STRAFTER(STR(?entity), "#") AS ?id)
        }}
        """
        return self.query_sparql(query)
    
    def get_relationships_by_type(self, rel_type: RelationshipType) -> List[Dict[str, Any]]:
        """
        Get all relationships of a specific type.
        
        Args:
            rel_type: The relationship type to filter by
            
        Returns:
            List of relationship dictionaries
        """
        query = f"""
        SELECT ?id ?source ?target
        WHERE {{
            ?rel rdf:type kbl:{rel_type.value} .
            ?rel kbl:hasSource ?source_uri .
            ?rel kbl:hasTarget ?target_uri .
            BIND(STRAFTER(STR(?rel), "#") AS ?id)
            BIND(STRAFTER(STR(?source_uri), "#") AS ?source)
            BIND(STRAFTER(STR(?target_uri), "#") AS ?target)
        }}
        """
        return self.query_sparql(query)
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the graph to a file.
        
        Args:
            path: Path to save the file, defaults to storage_path
        """
        save_path = path or self.storage_path
        if not save_path:
            logger.warning("No storage path specified, skipping save")
            return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save graph in Turtle format
        self.graph.serialize(destination=save_path, format="turtle")
        logger.info(f"Saved RDF graph to {save_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load the graph from a file.
        
        Args:
            path: Path to load the file from, defaults to storage_path
        """
        load_path = path or self.storage_path
        if not load_path or not os.path.exists(load_path):
            logger.warning(f"Storage file not found at {load_path}")
            return
        
        try:
            self.graph.parse(load_path, format="turtle")
            logger.info(f"Loaded RDF graph from {load_path}")
        except Exception as e:
            logger.error(f"Error loading RDF graph: {e}")
    
    def clear(self) -> None:
        """Clear the graph."""
        self.graph = Graph()
        self._init_namespaces()
    
    def get_entity_count(self) -> int:
        """
        Get the count of entities in the graph.
        
        Returns:
            Number of entities
        """
        query = """
        SELECT (COUNT(DISTINCT ?entity) as ?count)
        WHERE {
            ?entity rdf:type ?type .
            FILTER(STRSTARTS(STR(?type), STR(kbl:)))
        }
        """
        results = self.query_sparql(query)
        return results[0]['count'] if results else 0
    
    def get_relationship_count(self) -> int:
        """
        Get the count of relationships in the graph.
        
        Returns:
            Number of relationships
        """
        query = """
        SELECT (COUNT(DISTINCT ?rel) as ?count)
        WHERE {
            ?rel kbl:hasSource ?source .
            ?rel kbl:hasTarget ?target .
        }
        """
        results = self.query_sparql(query)
        return results[0]['count'] if results else 0
    
    def to_networkx(self):
        """
        Convert the RDF graph to a NetworkX graph.
        
        Returns:
            NetworkX graph
        """
        try:
            import networkx as nx
            
            G = nx.DiGraph()
            
            # Add nodes
            for s, p, o in self.graph.triples((None, RDF.type, None)):
                if str(o).startswith(str(KBL)):
                    entity_id = str(s).replace(str(KBL), "")
                    entity_type = str(o).replace(str(KBL), "")
                    
                    # Get label if available
                    name = None
                    for _, _, label in self.graph.triples((s, RDFS.label, None)):
                        name = str(label)
                        break
                    
                    G.add_node(entity_id, type=entity_type, name=name)
            
            # Add edges
            for s, p, o in self.graph:
                if (str(p).startswith(str(KBL)) and 
                    not str(p).startswith(str(KBL) + "meta_") and
                    not str(p).startswith(str(KBL) + "prop_") and
                    p != RDFS.label and p != RDF.type):
                    
                    source = str(s).replace(str(KBL), "")
                    relation = str(p).replace(str(KBL), "")
                    
                    if isinstance(o, URIRef) and str(o).startswith(str(KBL)):
                        target = str(o).replace(str(KBL), "")
                        G.add_edge(source, target, type=relation)
            
            return G
        except ImportError:
            logger.error("NetworkX not available, cannot convert graph")
            return None 