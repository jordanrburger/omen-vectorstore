"""RDF storage and querying functionality using rdflib."""

import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from rdflib import Graph, URIRef, Literal, BNode, RDF, RDFS, XSD
from rdflib.namespace import Namespace, NamespaceManager
from rdflib.plugins.stores.memory import Memory
from rdflib.query import ResultRow
from pathlib import Path

from app.ontology.models import Entity, EntityType, Relationship, RelationshipType
from app.ontology.manager import OntologyManager

logger = logging.getLogger(__name__)

# Define namespaces
KBC = Namespace("http://keboola.com/ontology/")
KBC_ENTITY = Namespace("http://keboola.com/ontology/entity/")
KBC_RELATIONSHIP = Namespace("http://keboola.com/ontology/relationship/")
KBC_PROPERTY = Namespace("http://keboola.com/ontology/property/")

class RDFStore:
    """RDF triple store for ontology data with SPARQL query support."""
    
    def __init__(self):
        """Initialize the RDF store with necessary namespaces."""
        self.graph = Graph(store=Memory())
        
        # Set up namespace manager
        self.namespace_manager = NamespaceManager(self.graph)
        self.namespace_manager.bind("kbc", KBC)
        self.namespace_manager.bind("kbc-entity", KBC_ENTITY)
        self.namespace_manager.bind("kbc-rel", KBC_RELATIONSHIP)
        self.namespace_manager.bind("kbc-prop", KBC_PROPERTY)
        self.graph.namespace_manager = self.namespace_manager
        
        # Add basic ontology axioms
        self._add_ontology_axioms()
    
    def _add_ontology_axioms(self):
        """Add basic ontology axioms to the graph."""
        # Define entity types
        for entity_type in EntityType:
            self.graph.add((KBC_ENTITY[entity_type.name], RDF.type, RDFS.Class))
        
        # Define relationship types
        for rel_type in RelationshipType:
            self.graph.add((KBC_RELATIONSHIP[rel_type.name], RDF.type, RDF.Property))
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the RDF store.
        
        Args:
            entity: Entity to add
        """
        entity_uri = KBC_ENTITY[entity.id]
        
        # Add entity type
        self.graph.add((entity_uri, RDF.type, KBC_ENTITY[entity.type.name]))
        
        # Add properties
        for key, value in entity.properties.items():
            if isinstance(value, (str, int, float, bool)):
                self.graph.add((entity_uri, KBC_PROPERTY[key], Literal(value)))
            elif isinstance(value, list):
                for item in value:
                    self.graph.add((entity_uri, KBC_PROPERTY[key], Literal(item)))
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the RDF store.
        
        Args:
            relationship: Relationship to add
        """
        rel_uri = KBC_RELATIONSHIP[relationship.id]
        source_uri = KBC_ENTITY[relationship.source_id]
        target_uri = KBC_ENTITY[relationship.target_id]
        
        # Add relationship type and properties
        self.graph.add((rel_uri, RDF.type, KBC_RELATIONSHIP[relationship.type.name]))
        self.graph.add((rel_uri, KBC_PROPERTY["source"], source_uri))
        self.graph.add((rel_uri, KBC_PROPERTY["target"], target_uri))
        
        # Add relationship properties
        for key, value in relationship.properties.items():
            if isinstance(value, (str, int, float, bool)):
                self.graph.add((rel_uri, KBC_PROPERTY[key], Literal(value)))
            elif isinstance(value, list):
                for item in value:
                    self.graph.add((rel_uri, KBC_PROPERTY[key], Literal(item)))
    
    def load_from_ontology_manager(self, ontology_manager: OntologyManager) -> None:
        """Load entities and relationships from an OntologyManager.
        
        Args:
            ontology_manager: OntologyManager to load from
        """
        # Clear existing data
        self.graph = Graph(store=Memory())
        
        # Reset namespace manager
        self.namespace_manager = NamespaceManager(self.graph)
        self.namespace_manager.bind("kbc", KBC)
        self.namespace_manager.bind("kbc-entity", KBC_ENTITY)
        self.namespace_manager.bind("kbc-rel", KBC_RELATIONSHIP)
        self.namespace_manager.bind("kbc-prop", KBC_PROPERTY)
        self.graph.namespace_manager = self.namespace_manager
        
        # Add basic ontology axioms
        self._add_ontology_axioms()
        
        # Add entities
        for entity in ontology_manager.entities.values():
            self.add_entity(entity)
        
        # Add relationships
        for relationship in ontology_manager.relationships.values():
            self.add_relationship(relationship)
    
    def query(self, sparql_query: str) -> List[ResultRow]:
        """Execute a SPARQL query against the RDF store.
        
        Args:
            sparql_query: SPARQL query string
            
        Returns:
            List of query results
        """
        try:
            # Add namespace prefixes to the query
            prefixes = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX kbc: <http://keboola.com/ontology/>
            PREFIX kbc-entity: <http://keboola.com/ontology/entity/>
            PREFIX kbc-rel: <http://keboola.com/ontology/relationship/>
            PREFIX kbc-prop: <http://keboola.com/ontology/property/>
            """
            
            query_with_prefixes = prefixes + sparql_query
            results = self.graph.query(query_with_prefixes)
            return list(results)
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            raise
    
    def find_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Find all entities of a specific type.
        
        Args:
            entity_type: Type of entities to find
            
        Returns:
            List of matching entities
        """
        query = f"""
        SELECT DISTINCT ?entity ?id ?type
        WHERE {{
            ?entity rdf:type kbc-entity:{entity_type.name} .
            ?entity kbc-prop:id ?id .
            ?entity kbc-prop:type ?type .
        }}
        """
        results = self.query(query)
        
        entities = []
        for row in results:
            try:
                entity = self._result_to_entity(row)
                if entity:
                    entities.append(entity)
            except Exception as e:
                logger.error(f"Error converting result to entity: {e}")
        
        return entities
    
    def find_relationships_by_type(self, relationship_type: RelationshipType) -> List[Relationship]:
        """Find all relationships of a specific type.
        
        Args:
            relationship_type: Type of relationships to find
            
        Returns:
            List of matching relationships
        """
        query = f"""
        SELECT DISTINCT ?rel ?id ?type ?source ?target
        WHERE {{
            ?rel rdf:type kbc-rel:{relationship_type.name} .
            ?rel kbc-prop:id ?id .
            ?rel kbc-prop:type ?type .
            ?rel kbc-prop:source ?source .
            ?rel kbc-prop:target ?target .
        }}
        """
        results = self.query(query)
        
        relationships = []
        for row in results:
            try:
                rel = self._result_to_relationship(row)
                if rel:
                    relationships.append(rel)
            except Exception as e:
                logger.error(f"Error converting result to relationship: {e}")
        
        return relationships
    
    def find_related_entities(self, entity_id: str) -> List[Tuple[Entity, Relationship]]:
        """Find all entities related to a given entity.
        
        Args:
            entity_id: ID of the entity to find relations for
            
        Returns:
            List of tuples containing (related_entity, relationship)
        """
        query = f"""
        SELECT DISTINCT ?rel ?rel_id ?rel_type ?target ?target_id ?target_type
        WHERE {{
            ?rel kbc-prop:source kbc-entity:{entity_id} .
            ?rel kbc-prop:id ?rel_id .
            ?rel kbc-prop:type ?rel_type .
            ?rel kbc-prop:target ?target .
            ?target kbc-prop:id ?target_id .
            ?target kbc-prop:type ?target_type .
        }}
        """
        results = self.query(query)
        return [(self._result_to_entity(row[3:]), self._result_to_relationship(row[:3])) 
                for row in results]
    
    def _result_to_entity(self, result_row: ResultRow) -> Entity:
        """Convert a SPARQL result row to an Entity object."""
        entity_id = str(result_row["id"].toPython() if hasattr(result_row["id"], "toPython") else result_row["id"])
        type_str = str(result_row["type"].toPython() if hasattr(result_row["type"], "toPython") else result_row["type"])
        
        # Extract type from URI
        type_name = type_str.split('/')[-1] if '/' in type_str else type_str
        try:
            entity_type = EntityType[type_name.upper()]
        except KeyError:
            # Default to UNKNOWN type if not found
            entity_type = EntityType.UNKNOWN
        
        # Get additional properties
        properties = {"id": entity_id, "type": entity_type.name}
        for key, value in result_row.items():
            if key not in ["id", "type"]:
                properties[key] = value.toPython() if hasattr(value, "toPython") else value
        
        return Entity(
            id=entity_id,
            type=entity_type,
            name=properties.get("name", entity_id),
            properties=properties
        )

    def _result_to_relationship(self, result_row: ResultRow) -> Relationship:
        """Convert a SPARQL result row to a Relationship object."""
        rel_id = str(result_row["id"].toPython() if hasattr(result_row["id"], "toPython") else result_row["id"])
        type_str = str(result_row["type"].toPython() if hasattr(result_row["type"], "toPython") else result_row["type"])
        
        # Extract type from URI
        type_name = type_str.split('/')[-1] if '/' in type_str else type_str
        try:
            rel_type = RelationshipType[type_name.upper()]
        except KeyError:
            # Default to UNKNOWN type if not found
            rel_type = RelationshipType.UNKNOWN
        
        # Extract source and target IDs from URIs
        source_uri = str(result_row["source"].toPython() if hasattr(result_row["source"], "toPython") else result_row["source"])
        target_uri = str(result_row["target"].toPython() if hasattr(result_row["target"], "toPython") else result_row["target"])
        
        source_id = source_uri.split('/')[-1] if '/' in source_uri else source_uri
        target_id = target_uri.split('/')[-1] if '/' in target_uri else target_uri
        
        return Relationship(
            id=rel_id,
            type=rel_type,
            source_id=source_id,
            target_id=target_id
        )
    
    def find_all_entities(self) -> List[Entity]:
        """Find all entities in the RDF store.
        
        Returns:
            List of all entities
        """
        query = """
        SELECT DISTINCT ?entity ?id ?type
        WHERE {
            ?entity rdf:type ?type .
            FILTER(STRSTARTS(STR(?type), STR(kbc-entity:)))
            ?entity kbc-prop:id ?id .
        }
        """
        results = self.query(query)
        
        entities = []
        for row in results:
            try:
                entity = self._result_to_entity(row)
                if entity:
                    entities.append(entity)
            except Exception as e:
                logger.error(f"Error converting result to entity: {e}")
        
        return entities
    
    def find_all_relationships(self) -> List[Relationship]:
        """Find all relationships in the RDF store.
        
        Returns:
            List of all relationships
        """
        query = """
        SELECT DISTINCT ?rel ?id ?type ?source ?target
        WHERE {
            ?rel rdf:type ?type .
            FILTER(STRSTARTS(STR(?type), STR(kbc-rel:)))
            ?rel kbc-prop:id ?id .
            ?rel kbc-prop:source ?source .
            ?rel kbc-prop:target ?target .
        }
        """
        results = self.query(query)
        
        relationships = []
        for row in results:
            try:
                rel = self._result_to_relationship(row)
                if rel:
                    relationships.append(rel)
            except Exception as e:
                logger.error(f"Error converting result to relationship: {e}")
        
        return relationships
    
    def serialize(self, format: str = "turtle") -> str:
        """Serialize the RDF graph to a string.
        
        Args:
            format: Serialization format (turtle, xml, n3, etc.)
            
        Returns:
            Serialized RDF string
        """
        return self.graph.serialize(format=format)
    
    def deserialize(self, data: str, format: str = "turtle") -> None:
        """Deserialize RDF data into the graph.
        
        Args:
            data: RDF data string
            format: Serialization format (turtle, xml, n3, etc.)
        """
        self.graph = Graph().parse(data=data, format=format) 