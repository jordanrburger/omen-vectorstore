"""
RDF triple store for ontology data with SPARQL query support.
"""
import os
from typing import Dict, List, Optional, Any, Union, Tuple
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
        source_uri = KBC_ENTITY[relationship.source_id]
        target_uri = KBC_ENTITY[relationship.target_id]
        rel_type_uri = KBC[relationship.type.value]
        rel_uri = KBC_RELATIONSHIP[relationship.id]
        
        # Add the direct relationship between source and target
        self.graph.add((source_uri, rel_type_uri, target_uri))
        
        # Also create a relationship instance for more detailed information
        self.graph.add((rel_uri, RDF.type, KBC.Relationship))
        self.graph.add((rel_uri, KBC.sourceEntity, source_uri))
        self.graph.add((rel_uri, KBC.targetEntity, target_uri))
        self.graph.add((rel_uri, KBC.relationshipType, rel_type_uri))
        
        # Add creation and update timestamps
        self.graph.add((rel_uri, KBC.createdAt, Literal(relationship.created_at.isoformat(), datatype=XSD.dateTime)))
        self.graph.add((rel_uri, KBC.updatedAt, Literal(relationship.updated_at.isoformat(), datatype=XSD.dateTime)))
        
        # Add properties
        for prop_name, prop_value in relationship.properties.items():
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