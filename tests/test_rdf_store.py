"""Tests for the RDF store functionality."""

import unittest
import tempfile
import os
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from app.ontology.rdf_store import RDFStore, KBC_ENTITY, KBC_RELATIONSHIP, KBC_PROPERTY
from app.ontology.models import Entity, EntityType, Relationship, RelationshipType
from app.ontology.manager import OntologyManager


class TestRDFStore(unittest.TestCase):
    """Test cases for the RDFStore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rdf_store = RDFStore()
        
        # Create test ontology manager
        self.ontology_manager = OntologyManager()
        
        # Add test entities
        self.table_entity = Entity(
            id="test_table",
            type=EntityType.TABLE,
            name="Test Table",
            properties={
                "description": "A test table",
                "row_count": 100
            }
        )
        self.ontology_manager.add_entity(self.table_entity)
        
        self.bucket_entity = Entity(
            id="test_bucket",
            type=EntityType.BUCKET,
            name="Test Bucket",
            properties={
                "description": "A test bucket"
            }
        )
        self.ontology_manager.add_entity(self.bucket_entity)
        
        # Add test relationship
        self.contains_relationship = Relationship(
            id="test_relationship",
            type=RelationshipType.CONTAINS,
            source_id=self.bucket_entity.id,
            target_id=self.table_entity.id,
            properties={
                "created_at": "2024-03-24T12:00:00Z"
            }
        )
        self.ontology_manager.add_relationship(self.contains_relationship)
    
    def test_initialization(self):
        """Test RDF store initialization."""
        self.assertIsInstance(self.rdf_store.graph, Graph)
        self.assertIsNotNone(self.rdf_store.namespace_manager)
        
        # Check that namespaces are bound
        self.assertIn("kbc", self.rdf_store.namespace_manager.namespaces())
        self.assertIn("kbc-entity", self.rdf_store.namespace_manager.namespaces())
        self.assertIn("kbc-rel", self.rdf_store.namespace_manager.namespaces())
        self.assertIn("kbc-prop", self.rdf_store.namespace_manager.namespaces())
        
        # Check that ontology axioms are added
        for entity_type in EntityType:
            self.assertTrue(
                (KBC_ENTITY[entity_type.name], RDF.type, RDFS.Class) in self.rdf_store.graph
            )
        
        for rel_type in RelationshipType:
            self.assertTrue(
                (KBC_RELATIONSHIP[rel_type.name], RDF.type, RDF.Property) in self.rdf_store.graph
            )
    
    def test_add_entity(self):
        """Test adding an entity to the RDF store."""
        self.rdf_store.add_entity(self.table_entity)
        
        # Query for the entity
        results = list(self.rdf_store.graph.triples((None, URIRef("http://keboola.com/ontology/property/name"), Literal("Test Table"))))
        self.assertEqual(len(results), 1)
    
    def test_add_relationship(self):
        """Test adding a relationship to the RDF store."""
        self.rdf_store.add_entity(self.bucket_entity)
        self.rdf_store.add_entity(self.table_entity)
        self.rdf_store.add_relationship(self.contains_relationship)
        
        # Query for the relationship
        results = list(self.rdf_store.graph.triples((None, URIRef("http://keboola.com/ontology/property/source"), None)))
        self.assertEqual(len(results), 1)
    
    def test_find_entities_by_type(self):
        """Test finding entities by type."""
        self.rdf_store.load_from_ontology_manager(self.ontology_manager)
        
        # Find tables
        tables = self.rdf_store.find_entities_by_type(EntityType.TABLE)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].id, "test_table")
        self.assertEqual(tables[0].type, EntityType.TABLE)
        self.assertEqual(tables[0].properties["name"], "Test Table")
        
        # Find buckets
        buckets = self.rdf_store.find_entities_by_type(EntityType.BUCKET)
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0].id, "test_bucket")
        self.assertEqual(buckets[0].type, EntityType.BUCKET)
        self.assertEqual(buckets[0].properties["name"], "Test Bucket")
    
    def test_find_relationships_by_type(self):
        """Test finding relationships by type."""
        self.rdf_store.load_from_ontology_manager(self.ontology_manager)
        
        # Find contains relationships
        contains_rels = self.rdf_store.find_relationships_by_type(RelationshipType.CONTAINS)
        self.assertEqual(len(contains_rels), 1)
        self.assertEqual(contains_rels[0].id, "test_relationship")
        self.assertEqual(contains_rels[0].type, RelationshipType.CONTAINS)
        self.assertEqual(contains_rels[0].source_id, "test_bucket")
        self.assertEqual(contains_rels[0].target_id, "test_table")
    
    def test_find_related_entities(self):
        """Test finding related entities."""
        self.rdf_store.load_from_ontology_manager(self.ontology_manager)
        
        # Find entities related to bucket
        related = self.rdf_store.find_related_entities("test_bucket")
        self.assertEqual(len(related), 1)
        related_entity, relationship = related[0]
        
        self.assertEqual(related_entity.id, "test_table")
        self.assertEqual(related_entity.type, EntityType.TABLE)
        self.assertEqual(relationship.id, "test_relationship")
        self.assertEqual(relationship.type, RelationshipType.CONTAINS)
    
    def test_load_from_ontology_manager(self):
        """Test loading data from ontology manager."""
        self.rdf_store.load_from_ontology_manager(self.ontology_manager)
        
        # Check entities
        self.assertTrue(
            (KBC_ENTITY["test_table"], RDF.type, KBC_ENTITY["TABLE"]) in self.rdf_store.graph
        )
        self.assertTrue(
            (KBC_ENTITY["test_bucket"], RDF.type, KBC_ENTITY["BUCKET"]) in self.rdf_store.graph
        )
        
        # Check relationship
        self.assertTrue(
            (KBC_RELATIONSHIP["test_relationship"], RDF.type, KBC_RELATIONSHIP["CONTAINS"]) in self.rdf_store.graph
        )
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        self.rdf_store.load_from_ontology_manager(self.ontology_manager)
        
        # Serialize to string
        serialized = self.rdf_store.serialize()
        self.assertIsInstance(serialized, str)
        
        # Create new store and load serialized data
        new_store = RDFStore()
        new_store.deserialize(serialized)
        
        # Verify data was loaded correctly
        self.assertTrue(
            (KBC_ENTITY["test_table"], RDF.type, KBC_ENTITY["TABLE"]) in new_store.graph
        )
        self.assertTrue(
            (KBC_RELATIONSHIP["test_relationship"], RDF.type, KBC_RELATIONSHIP["CONTAINS"]) in new_store.graph
        )
    
    def test_sparql_query(self):
        """Test SPARQL query execution."""
        self.rdf_store.load_from_ontology_manager(self.ontology_manager)
        
        # Execute SPARQL query
        query = """
        SELECT ?s ?p ?o
        WHERE {
            ?s ?p ?o .
        }
        """
        results = list(self.rdf_store.query(query))
        self.assertTrue(len(results) > 0)


if __name__ == "__main__":
    unittest.main() 