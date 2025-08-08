"""
RDFLib implementation of the ontology storage backend.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, NamespaceManager
from rdflib.plugins.sparql import prepareQuery

from omen.core import get_logger, load_settings
from omen.ontology.models import Entity, Relationship, Triple, EntityType, RelationshipType
from omen.storage.base import OntologyStoreBackend

logger = get_logger(__name__)
settings = load_settings()

# Define Keboola-specific namespaces
KBC = Namespace("http://keboola.com/ontology#")
KBC_ENTITY = Namespace("http://keboola.com/ontology/entity#")
KBC_RELATIONSHIP = Namespace("http://keboola.com/ontology/relationship#")
KBC_PROPERTY = Namespace("http://keboola.com/ontology/property#")


class RDFLibStore(OntologyStoreBackend):
    """RDFLib implementation of ontology storage backend."""

    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        """Initialize RDFLib store with optional storage path."""
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

    def store_entity(self, entity: Entity) -> str:
        """Store an entity in the ontology."""
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
                
        return entity.id

    def store_relationship(self, relationship: Relationship) -> str:
        """Store a relationship in the ontology."""
        try:
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
            if relationship.created_at:
                self.graph.add((rel_uri, KBC.createdAt, Literal(relationship.created_at.isoformat(), datatype=XSD.dateTime)))
            if relationship.updated_at:
                self.graph.add((rel_uri, KBC.updatedAt, Literal(relationship.updated_at.isoformat(), datatype=XSD.dateTime)))
            
            # Add properties
            for prop_name, prop_value in relationship.properties.items():
                if not prop_name:  # Skip empty property names
                    continue
                    
                prop_uri = KBC_PROPERTY[prop_name]
                
                # Handle different property value types
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
                    
            return relationship.id
        except Exception as e:
            logger.error(f"Error storing relationship: {e}")
            raise

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        try:
            entity_uri = KBC_ENTITY[entity_id]
            
            # Check if entity exists
            if (entity_uri, RDF.type, None) not in self.graph:
                return None
            
            # Get entity type
            entity_type = None
            for _, _, type_uri in self.graph.triples((entity_uri, RDF.type, None)):
                if type_uri != KBC.Entity and isinstance(type_uri, URIRef):
                    type_name = str(type_uri).split('#')[-1].lower()
                    entity_type = EntityType(type_name)
                    break
            
            if not entity_type:
                return None
            
            # Get basic properties
            name = None
            description = None
            properties = {}
            created_at = None
            updated_at = None
            
            for p, o in self.graph.predicate_objects(entity_uri):
                if p == RDFS.label:
                    name = str(o)
                elif p == RDFS.comment:
                    description = str(o)
                elif p == KBC.createdAt:
                    created_at = o.toPython()
                elif p == KBC.updatedAt:
                    updated_at = o.toPython()
                elif isinstance(p, URIRef) and str(p).startswith(str(KBC_PROPERTY)):
                    prop_name = str(p).split('#')[-1]
                    properties[prop_name] = o.toPython()
            
            return Entity(
                id=entity_id,
                type=entity_type,
                name=name or entity_id,
                description=description,
                properties=properties,
                created_at=created_at,
                updated_at=updated_at
            )
        except Exception as e:
            logger.error(f"Error retrieving entity: {e}")
            return None

    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Retrieve a relationship by ID."""
        try:
            rel_uri = KBC_RELATIONSHIP[relationship_id]
            
            # Check if relationship exists
            if (rel_uri, RDF.type, KBC.Relationship) not in self.graph:
                return None
            
            # Get source and target entities
            source_id = None
            target_id = None
            rel_type = None
            properties = {}
            created_at = None
            updated_at = None
            
            for p, o in self.graph.predicate_objects(rel_uri):
                if p == KBC.sourceEntity:
                    source_id = str(o).split('#')[-1]
                elif p == KBC.targetEntity:
                    target_id = str(o).split('#')[-1]
                elif p == KBC.relationshipType:
                    rel_type = RelationshipType(str(o).split('#')[-1])
                elif p == KBC.createdAt:
                    created_at = o.toPython()
                elif p == KBC.updatedAt:
                    updated_at = o.toPython()
                elif isinstance(p, URIRef) and str(p).startswith(str(KBC_PROPERTY)):
                    prop_name = str(p).split('#')[-1]
                    properties[prop_name] = o.toPython()
            
            if not all([source_id, target_id, rel_type]):
                return None
            
            return Relationship(
                id=relationship_id,
                type=rel_type,
                source_id=source_id,
                target_id=target_id,
                properties=properties,
                created_at=created_at,
                updated_at=updated_at
            )
        except Exception as e:
            logger.error(f"Error retrieving relationship: {e}")
            return None

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships."""
        try:
            entity_uri = KBC_ENTITY[entity_id]
            
            # Remove all triples where this entity is subject or object
            self.graph.remove((entity_uri, None, None))
            self.graph.remove((None, None, entity_uri))
            
            # Remove any relationships that reference this entity
            for s in self.graph.subjects(KBC.sourceEntity, entity_uri):
                self.graph.remove((s, None, None))
            for s in self.graph.subjects(KBC.targetEntity, entity_uri):
                self.graph.remove((s, None, None))
                
            return True
        except Exception as e:
            logger.error(f"Error deleting entity: {e}")
            return False

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        try:
            rel_uri = KBC_RELATIONSHIP[relationship_id]
            self.graph.remove((rel_uri, None, None))
            return True
        except Exception as e:
            logger.error(f"Error deleting relationship: {e}")
            return False

    def query(
        self,
        query: str,
        query_type: str = "sparql",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute a query against the ontology."""
        if query_type.lower() != "sparql":
            raise ValueError("Only SPARQL queries are supported by RDFLib backend")
            
        try:
            # Prepare query with namespace bindings
            prepared_query = prepareQuery(
                query,
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
            results = list(self.graph.query(prepared_query))
            
            # Convert results to dictionaries
            result_list = []
            var_names = [str(var) for var in prepared_query.algebra.vars]
            
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
            logger.error(f"Error executing query: {e}")
            raise

    def get_connected_entities(
        self,
        entity_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "both"
    ) -> List[Tuple[Entity, Relationship]]:
        """Get entities connected to the given entity."""
        try:
            entity_uri = KBC_ENTITY[entity_id]
            results = []
            
            # Build relationship type filter
            rel_type_filter = ""
            if relationship_types:
                rel_types = [f"kbc:{t}" for t in relationship_types]
                rel_type_filter = f"VALUES ?relType {{ {' '.join(rel_types)} }}"
            
            # Query for connected entities based on direction
            if direction in ["both", "out"]:
                outgoing_query = f"""
                    SELECT ?target ?rel
                    WHERE {{
                        <{entity_uri}> ?relType ?target .
                        ?rel kbc:sourceEntity <{entity_uri}> ;
                             kbc:targetEntity ?target ;
                             kbc:relationshipType ?relType .
                        {rel_type_filter}
                    }}
                """
                for result in self.query(outgoing_query):
                    target_id = str(result["target"]).split("#")[-1]
                    rel_id = str(result["rel"]).split("#")[-1]
                    target = self.get_entity(target_id)
                    rel = self.get_relationship(rel_id)
                    if target and rel:
                        results.append((target, rel))
            
            if direction in ["both", "in"]:
                incoming_query = f"""
                    SELECT ?source ?rel
                    WHERE {{
                        ?source ?relType <{entity_uri}> .
                        ?rel kbc:sourceEntity ?source ;
                             kbc:targetEntity <{entity_uri}> ;
                             kbc:relationshipType ?relType .
                        {rel_type_filter}
                    }}
                """
                for result in self.query(incoming_query):
                    source_id = str(result["source"]).split("#")[-1]
                    rel_id = str(result["rel"]).split("#")[-1]
                    source = self.get_entity(source_id)
                    rel = self.get_relationship(rel_id)
                    if source and rel:
                        results.append((source, rel))
            
            return results
        except Exception as e:
            logger.error(f"Error getting connected entities: {e}")
            return []

    def save(self) -> None:
        """Save the graph to storage_path."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.graph.serialize(destination=str(self.storage_path), format="turtle")
            logger.info(f"Saved RDF graph to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving RDF graph: {e}")
            raise

    def load(self) -> bool:
        """Load the graph from storage_path if it exists."""
        if not self.storage_path.exists():
            logger.info(f"RDF storage path {self.storage_path} does not exist, using empty graph")
            return False
            
        try:
            self.graph.parse(source=str(self.storage_path), format="turtle")
            logger.info(f"Loaded RDF graph from {self.storage_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading RDF graph: {e}")
            return False 