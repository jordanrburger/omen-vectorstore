import unittest
from datetime import datetime

from app.ontology.manager import OntologyManager
from app.ontology.models import Entity, EntityType, Relationship, RelationshipType
from app.ontology.schema import SchemaValidator


class TestOntologyManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.ontology = OntologyManager()

    def test_add_entity(self):
        """Test adding entities to the ontology."""
        # Create and add a valid entity
        entity = Entity(
            id="test_bucket",
            type=EntityType.BUCKET,
            name="Test Bucket",
            properties={
                "name": "Test Bucket",
                "description": "A test bucket"
            }
        )
        self.ontology.add_entity(entity)

        # Verify entity was added
        self.assertEqual(len(self.ontology.entities), 1)
        self.assertEqual(self.ontology.entities["test_bucket"], entity)

        # Test adding duplicate entity
        with self.assertRaises(ValueError):
            self.ontology.add_entity(entity)

    def test_add_relationship(self):
        """Test adding relationships to the ontology."""
        # Create and add source and target entities
        source = Entity(
            id="test_bucket",
            type=EntityType.BUCKET,
            name="Test Bucket",
            properties={"name": "Test Bucket"}
        )
        target = Entity(
            id="test_table",
            type=EntityType.TABLE,
            name="Test Table",
            properties={"name": "Test Table"}
        )
        self.ontology.add_entity(source)
        self.ontology.add_entity(target)

        # Create and add a valid relationship
        relationship = Relationship(
            id="test_rel",
            type=RelationshipType.CONTAINS,
            source_id="test_bucket",
            target_id="test_table",
            properties={
                "created_at": datetime.now().isoformat()
            }
        )
        self.ontology.add_relationship(relationship)

        # Verify relationship was added
        self.assertEqual(len(self.ontology.relationships), 1)
        self.assertEqual(self.ontology.relationships["test_rel"], relationship)

        # Test adding duplicate relationship
        with self.assertRaises(ValueError):
            self.ontology.add_relationship(relationship)

        # Test adding relationship with non-existent source
        invalid_rel = Relationship(
            id="invalid_rel",
            type=RelationshipType.CONTAINS,
            source_id="non_existent",
            target_id="test_table",
            properties={}
        )
        with self.assertRaises(ValueError):
            self.ontology.add_relationship(invalid_rel)

    def test_get_entity(self):
        """Test retrieving entities from the ontology."""
        # Create and add a test entity
        entity = Entity(
            id="test_bucket",
            type=EntityType.BUCKET,
            name="Test Bucket",
            properties={"name": "Test Bucket"}
        )
        self.ontology.add_entity(entity)

        # Test getting existing entity
        retrieved = self.ontology.get_entity("test_bucket")
        self.assertEqual(retrieved, entity)

        # Test getting non-existent entity
        with self.assertRaises(KeyError):
            self.ontology.get_entity("non_existent")

    def test_get_relationship(self):
        """Test retrieving relationships from the ontology."""
        # Create and add test entities and relationship
        source = Entity(
            id="test_bucket",
            type=EntityType.BUCKET,
            name="Test Bucket",
            properties={"name": "Test Bucket"}
        )
        target = Entity(
            id="test_table",
            type=EntityType.TABLE,
            name="Test Table",
            properties={"name": "Test Table"}
        )
        self.ontology.add_entity(source)
        self.ontology.add_entity(target)

        relationship = Relationship(
            id="test_rel",
            type=RelationshipType.CONTAINS,
            source_id="test_bucket",
            target_id="test_table",
            properties={}
        )
        self.ontology.add_relationship(relationship)

        # Test getting existing relationship
        retrieved = self.ontology.get_relationship("test_rel")
        self.assertEqual(retrieved, relationship)

        # Test getting non-existent relationship
        with self.assertRaises(KeyError):
            self.ontology.get_relationship("non_existent")

    def test_get_entities_by_type(self):
        """Test retrieving entities by type."""
        # Create and add test entities of different types
        bucket = Entity(
            id="test_bucket",
            type=EntityType.BUCKET,
            name="Test Bucket",
            properties={"name": "Test Bucket"}
        )
        table = Entity(
            id="test_table",
            type=EntityType.TABLE,
            name="Test Table",
            properties={"name": "Test Table"}
        )
        self.ontology.add_entity(bucket)
        self.ontology.add_entity(table)

        # Test getting entities by type
        buckets = self.ontology.get_entities_by_type(EntityType.BUCKET)
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0], bucket)

        tables = self.ontology.get_entities_by_type(EntityType.TABLE)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0], table)

        # Test getting entities of non-existent type
        empty = self.ontology.get_entities_by_type(EntityType.CONFIGURATION)
        self.assertEqual(len(empty), 0)

    def test_get_relationships_by_type(self):
        """Test retrieving relationships by type."""
        # Create and add test entities and relationships
        source = Entity(
            id="test_bucket",
            type=EntityType.BUCKET,
            name="Test Bucket",
            properties={"name": "Test Bucket"}
        )
        target = Entity(
            id="test_table",
            type=EntityType.TABLE,
            name="Test Table",
            properties={"name": "Test Table"}
        )
        self.ontology.add_entity(source)
        self.ontology.add_entity(target)

        contains_rel = Relationship(
            id="contains_rel",
            type=RelationshipType.CONTAINS,
            source_id="test_bucket",
            target_id="test_table",
            properties={}
        )
        self.ontology.add_relationship(contains_rel)

        # Test getting relationships by type
        contains = self.ontology.get_relationships_by_type(RelationshipType.CONTAINS)
        self.assertEqual(len(contains), 1)
        self.assertEqual(contains[0], contains_rel)

        # Test getting relationships of non-existent type
        empty = self.ontology.get_relationships_by_type(RelationshipType.BELONGS_TO)
        self.assertEqual(len(empty), 0)

    def test_get_related_entities(self):
        """Test retrieving related entities."""
        # Create and add test entities
        bucket = Entity(
            id="test_bucket",
            type=EntityType.BUCKET,
            name="Test Bucket",
            properties={"name": "Test Bucket"}
        )
        table1 = Entity(
            id="test_table1",
            type=EntityType.TABLE,
            name="Test Table 1",
            properties={"name": "Test Table 1"}
        )
        table2 = Entity(
            id="test_table2",
            type=EntityType.TABLE,
            name="Test Table 2",
            properties={"name": "Test Table 2"}
        )
        self.ontology.add_entity(bucket)
        self.ontology.add_entity(table1)
        self.ontology.add_entity(table2)

        # Add relationships
        rel1 = Relationship(
            id="rel1",
            type=RelationshipType.CONTAINS,
            source_id="test_bucket",
            target_id="test_table1",
            properties={}
        )
        rel2 = Relationship(
            id="rel2",
            type=RelationshipType.CONTAINS,
            source_id="test_bucket",
            target_id="test_table2",
            properties={}
        )
        self.ontology.add_relationship(rel1)
        self.ontology.add_relationship(rel2)

        # Test getting related entities
        related = self.ontology.get_related_entities("test_bucket")
        self.assertEqual(len(related), 2)
        self.assertIn(table1, related)
        self.assertIn(table2, related)

        # Test getting related entities with relationship type filter
        related = self.ontology.get_related_entities(
            "test_bucket",
            relationship_type=RelationshipType.CONTAINS
        )
        self.assertEqual(len(related), 2)
        self.assertIn(table1, related)
        self.assertIn(table2, related)

        # Test getting related entities with entity type filter
        related = self.ontology.get_related_entities(
            "test_bucket",
            entity_type=EntityType.TABLE
        )
        self.assertEqual(len(related), 2)
        self.assertIn(table1, related)
        self.assertIn(table2, related)

        # Test getting related entities with both filters
        related = self.ontology.get_related_entities(
            "test_bucket",
            relationship_type=RelationshipType.CONTAINS,
            entity_type=EntityType.TABLE
        )
        self.assertEqual(len(related), 2)
        self.assertIn(table1, related)
        self.assertIn(table2, related)

        # Test getting related entities for non-existent entity
        with self.assertRaises(KeyError):
            self.ontology.get_related_entities("non_existent")

    def test_remove_entity(self):
        """Test removing entities from the ontology."""
        # Create and add test entities and relationship
        source = Entity(
            id="test_bucket",
            type=EntityType.BUCKET,
            name="Test Bucket",
            properties={"name": "Test Bucket"}
        )
        target = Entity(
            id="test_table",
            type=EntityType.TABLE,
            name="Test Table",
            properties={"name": "Test Table"}
        )
        self.ontology.add_entity(source)
        self.ontology.add_entity(target)

        relationship = Relationship(
            id="test_rel",
            type=RelationshipType.CONTAINS,
            source_id="test_bucket",
            target_id="test_table",
            properties={}
        )
        self.ontology.add_relationship(relationship)

        # Test removing entity with relationships
        self.ontology.remove_entity("test_bucket")
        self.assertEqual(len(self.ontology.entities), 1)
        self.assertEqual(len(self.ontology.relationships), 0)

        # Test removing non-existent entity
        with self.assertRaises(KeyError):
            self.ontology.remove_entity("non_existent")

    def test_remove_relationship(self):
        """Test removing relationships from the ontology."""
        # Create and add test entities and relationship
        source = Entity(
            id="test_bucket",
            type=EntityType.BUCKET,
            name="Test Bucket",
            properties={"name": "Test Bucket"}
        )
        target = Entity(
            id="test_table",
            type=EntityType.TABLE,
            name="Test Table",
            properties={"name": "Test Table"}
        )
        self.ontology.add_entity(source)
        self.ontology.add_entity(target)

        relationship = Relationship(
            id="test_rel",
            type=RelationshipType.CONTAINS,
            source_id="test_bucket",
            target_id="test_table",
            properties={}
        )
        self.ontology.add_relationship(relationship)

        # Test removing existing relationship
        self.ontology.remove_relationship("test_rel")
        self.assertEqual(len(self.ontology.relationships), 0)

        # Test removing non-existent relationship
        with self.assertRaises(KeyError):
            self.ontology.remove_relationship("non_existent")


if __name__ == "__main__":
    unittest.main() 