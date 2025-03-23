"""Tests for the ontology builder functionality."""

import unittest
from unittest.mock import Mock, patch
import json
import uuid

from app.ontology.models import Entity, Relationship, Triple, EntityType, RelationshipType
from app.ontology.builder import OntologyBuilder
from app.ontology.manager import OntologyManager
from app.ontology.schema import SchemaValidator
from app.ontology.schema_definition import default_schema
from app.llm_client import LLMClient


class TestOntologyBuilder(unittest.TestCase):
    """Test cases for the OntologyBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock LLM client
        self.mock_llm_client = Mock(spec=LLMClient)
        
        # Create ontology manager
        self.ontology_manager = OntologyManager()
        
        # Create schema validator
        self.schema_validator = SchemaValidator(default_schema)
        
        # Create test fixture data
        self.table_metadata = {
            "id": "123",
            "name": "customers",
            "bucket": {
                "id": "456",
                "name": "in.c-sales",
                "stage": "in"
            },
            "columns": [
                {"name": "id", "type": "INTEGER", "nullable": False},
                {"name": "email", "type": "STRING", "nullable": True},
                {"name": "created_at", "type": "TIMESTAMP", "nullable": False}
            ],
            "primary_key": ["id"],
            "row_count": 1000,
            "data_size_bytes": 50000
        }
        
        self.transformation_metadata = {
            "id": "789",
            "name": "Process Customers",
            "type": "python",
            "description": "Processes customer data",
            "blocks": [
                {
                    "id": "block1",
                    "name": "Main Block",
                    "code": "print('Processing customers')\n# Load the data\ncustomers = load_table('in.c-sales.customers')"
                }
            ],
            "inputs": [
                {"source": "in.c-sales.customers", "destination": "customers"}
            ],
            "outputs": [
                {"source": "processed_customers", "destination": "out.c-processed.customers"}
            ]
        }
        
        # Create mock entity extraction response
        self.mock_entity_extraction_response = [
            {
                "id": "bucket_456",
                "type": "BUCKET",
                "properties": {
                    "name": "in.c-sales",
                    "stage": "in"
                }
            },
            {
                "id": "table_123",
                "type": "TABLE",
                "properties": {
                    "name": "customers",
                    "row_count": 1000,
                    "data_size_bytes": 50000,
                    "primary_key": ["id"]
                }
            },
            {
                "id": "column_id",
                "type": "COLUMN",
                "properties": {
                    "name": "id",
                    "data_type": "INTEGER",
                    "nullable": False
                }
            },
            {
                "id": "column_email",
                "type": "COLUMN",
                "properties": {
                    "name": "email",
                    "data_type": "STRING",
                    "nullable": True
                }
            },
            {
                "id": "column_created_at",
                "type": "COLUMN",
                "properties": {
                    "name": "created_at",
                    "data_type": "TIMESTAMP",
                    "nullable": False
                }
            }
        ]
        
        # Create mock relationship detection response
        self.mock_relationship_detection_response = [
            {
                "id": "rel_1",
                "type": "CONTAINS",
                "source_id": "bucket_456",
                "target_id": "table_123",
                "properties": {}
            },
            {
                "id": "rel_2",
                "type": "HAS_COLUMN",
                "source_id": "table_123",
                "target_id": "column_id",
                "properties": {"position": 0}
            },
            {
                "id": "rel_3",
                "type": "HAS_COLUMN",
                "source_id": "table_123",
                "target_id": "column_email",
                "properties": {"position": 1}
            },
            {
                "id": "rel_4",
                "type": "HAS_COLUMN",
                "source_id": "table_123",
                "target_id": "column_created_at",
                "properties": {"position": 2}
            }
        ]
        
        # Configure mock LLM client to return entity extraction response
        def mock_generate_side_effect(system_prompt, user_prompt, expected_format):
            if "identify entities from the provided metadata" in system_prompt:
                return self.mock_entity_extraction_response
            elif "identify relationships between the provided entities" in system_prompt:
                return self.mock_relationship_detection_response
            else:
                return {}
        
        self.mock_llm_client.generate.side_effect = mock_generate_side_effect
        
        # Create ontology builder
        self.builder = OntologyBuilder(
            llm_client=self.mock_llm_client,
            ontology_manager=self.ontology_manager,
            schema_validator=self.schema_validator,
            batch_size=10,
            max_workers=1,  # Use 1 worker for predictable test behavior
            max_retries=1
        )
    
    def test_extract_entities(self):
        """Test extracting entities from metadata."""
        # Extract entities from table metadata
        entities = self.builder.extract_entities(self.table_metadata)
        
        # Verify that the expected entities were extracted
        self.assertEqual(len(entities), 5)
        
        # Check types of extracted entities
        entity_types = [entity.type for entity in entities]
        self.assertIn(EntityType.BUCKET, entity_types)
        self.assertIn(EntityType.TABLE, entity_types)
        self.assertIn(EntityType.COLUMN, entity_types)
        
        # Find table entity
        table_entity = next((e for e in entities if e.type == EntityType.TABLE), None)
        self.assertIsNotNone(table_entity)
        self.assertEqual(table_entity.properties.get("name"), "customers")
        
        # Verify LLM client was called correctly
        self.mock_llm_client.generate.assert_called_once()
        args, kwargs = self.mock_llm_client.generate.call_args
        self.assertEqual(kwargs["expected_format"], "json")
    
    def test_detect_relationships(self):
        """Test detecting relationships between entities."""
        # Create test entities from the mock entity extraction response
        entities = []
        for entity_data in self.mock_entity_extraction_response:
            entity = Entity(
                id=entity_data["id"],
                type=EntityType(entity_data["type"]),
                properties=entity_data["properties"]
            )
            entities.append(entity)
        
        # Detect relationships between the entities
        relationships = self.builder.detect_relationships(entities)
        
        # Verify that the expected relationships were detected
        self.assertEqual(len(relationships), 4)
        
        # Check types of detected relationships
        relationship_types = [rel.type for rel in relationships]
        self.assertIn(RelationshipType.CONTAINS, relationship_types)
        self.assertIn(RelationshipType.HAS_COLUMN, relationship_types)
        
        # Check source and target of relationships
        contains_rel = next((r for r in relationships if r.type == RelationshipType.CONTAINS), None)
        self.assertIsNotNone(contains_rel)
        self.assertEqual(contains_rel.source_id, "bucket_456")
        self.assertEqual(contains_rel.target_id, "table_123")
        
        # Verify LLM client was called correctly
        self.mock_llm_client.generate.assert_called_once()
        args, kwargs = self.mock_llm_client.generate.call_args
        self.assertEqual(kwargs["expected_format"], "json")
    
    def test_build_ontology(self):
        """Test building an ontology from metadata."""
        # Build ontology from metadata
        ontology = self.builder.build_ontology([self.table_metadata, self.transformation_metadata])
        
        # Verify that all entities, relationships, and triples were added to the ontology
        entities = ontology.get_all_entities()
        relationships = ontology.get_all_relationships()
        triples = ontology.get_all_triples()
        
        # Just check counts, as the exact entities/relationships depend on the mock responses
        self.assertGreater(len(entities), 0)
        self.assertGreater(len(relationships), 0)
        self.assertGreater(len(triples), 0)
        
        # Check if we can retrieve entities by type
        bucket_entities = ontology.get_entities_by_type(EntityType.BUCKET)
        table_entities = ontology.get_entities_by_type(EntityType.TABLE)
        column_entities = ontology.get_entities_by_type(EntityType.COLUMN)
        
        self.assertGreater(len(bucket_entities), 0)
        self.assertGreater(len(table_entities), 0)
        self.assertGreater(len(column_entities), 0)
    
    def test_update_ontology_from_metadata(self):
        """Test updating an ontology with new metadata."""
        # First, create a small ontology with just the table metadata
        self.builder.build_ontology([self.table_metadata])
        
        # Get the initial counts
        initial_entity_count = len(self.ontology_manager.get_all_entities())
        initial_relationship_count = len(self.ontology_manager.get_all_relationships())
        initial_triple_count = len(self.ontology_manager.get_all_triples())
        
        # Clear the mock to reset call counts
        self.mock_llm_client.reset_mock()
        
        # Update the ontology with the transformation metadata
        new_triples = self.builder.update_ontology_from_metadata(self.transformation_metadata)
        
        # Get the updated counts
        updated_entity_count = len(self.ontology_manager.get_all_entities())
        updated_relationship_count = len(self.ontology_manager.get_all_relationships())
        updated_triple_count = len(self.ontology_manager.get_all_triples())
        
        # Verify that new entities, relationships, and triples were added
        self.assertGreater(updated_entity_count, initial_entity_count)
        self.assertGreater(updated_relationship_count, initial_relationship_count)
        self.assertGreater(updated_triple_count, initial_triple_count)
        
        # Verify that the new triples were returned
        self.assertEqual(len(new_triples), updated_triple_count - initial_triple_count)
    
    def test_deduplicate_relationships(self):
        """Test deduplicating relationships."""
        # Create some relationships
        rel1 = Relationship(id="rel1", type=RelationshipType.CONTAINS, source_id="source1", target_id="target1")
        rel2 = Relationship(id="rel2", type=RelationshipType.HAS_COLUMN, source_id="source2", target_id="target2")
        rel3 = Relationship(id="rel3", type=RelationshipType.CONTAINS, source_id="source1", target_id="target1")  # Duplicate of rel1
        rel4 = Relationship(id="rel4", type=RelationshipType.INPUTS_FROM, source_id="source3", target_id="target3")
        
        existing_relationships = [rel1, rel2]
        new_relationships = [rel3, rel4]
        
        # Deduplicate
        result = self.builder._deduplicate_relationships(existing_relationships, new_relationships)
        
        # Check the result
        self.assertEqual(len(result), 3)  # rel1, rel2, rel4 (rel3 is deduplicated)
        result_ids = [rel.id for rel in result]
        self.assertIn("rel1", result_ids)
        self.assertIn("rel2", result_ids)
        self.assertIn("rel4", result_ids)
        self.assertNotIn("rel3", result_ids)
    
    @patch("concurrent.futures.ThreadPoolExecutor")
    def test_extract_entities_batch(self, mock_executor):
        """Test extracting entities in batches."""
        # Prepare the mock executor with a fake context manager
        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        
        # Create multiple metadata items
        metadata_items = [self.table_metadata, self.transformation_metadata]
        
        # Configure submission of tasks
        def submit_side_effect(func, metadata):
            future = Mock()
            if metadata == self.table_metadata:
                future.result.return_value = self.builder.extract_entities(metadata)
            else:
                future.result.return_value = []  # Empty list for transformation metadata
            return future
        
        mock_executor_instance.submit.side_effect = submit_side_effect
        
        # Mock as_completed to return our futures
        with patch("app.ontology.builder.as_completed") as mock_as_completed:
            # Create futures for each metadata item
            futures = []
            for metadata in metadata_items:
                future = Mock()
                future.result.return_value = self.builder.extract_entities(metadata) if metadata == self.table_metadata else []
                futures.append(future)
            
            mock_as_completed.return_value = futures
            
            # Extract entities in batch
            entities = self.builder.extract_entities_batch(metadata_items)
            
            # Verify that entities were extracted and combined
            self.assertEqual(len(entities), 5)  # All entities from table_metadata
    
    def test_build_triples_from_relationships(self):
        """Test building triples from relationships."""
        # Create relationships
        relationships = []
        for rel_data in self.mock_relationship_detection_response:
            relationship = Relationship(
                id=rel_data["id"],
                type=RelationshipType(rel_data["type"]),
                source_id=rel_data["source_id"],
                target_id=rel_data["target_id"],
                properties=rel_data["properties"]
            )
            relationships.append(relationship)
        
        # Build triples
        triples = self.builder._build_triples_from_relationships(relationships)
        
        # Verify that triples were created correctly
        self.assertEqual(len(triples), len(relationships))
        
        for i, triple in enumerate(triples):
            rel = relationships[i]
            self.assertEqual(triple.subject, rel.source_id)
            self.assertEqual(triple.predicate, rel.type.value)
            self.assertEqual(triple.object, rel.target_id)
            self.assertEqual(triple.metadata, rel.properties)


if __name__ == "__main__":
    unittest.main() 