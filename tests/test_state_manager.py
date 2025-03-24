"""Tests for the state manager functionality."""

import unittest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from app.state_manager import StateManager
from app.ontology.manager import OntologyManager
from app.ontology.action_graph import ActionGraph, Action, ActionType
from app.ontology.models import Entity, EntityType, Relationship, RelationshipType
from app.ontology.rdf_store import RDFStore


class TestStateManager(unittest.TestCase):
    """Test cases for the StateManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test state files
        self.temp_dir = tempfile.mkdtemp()
        self.state_manager = StateManager(state_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory and its contents
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test state manager initialization."""
        # Check that state directory was created
        self.assertTrue(os.path.exists(self.temp_dir))
        
        # Check that state files don't exist yet
        self.assertFalse(os.path.exists(self.state_manager.extraction_state_file))
        self.assertFalse(os.path.exists(self.state_manager.metadata_file))
        self.assertFalse(os.path.exists(self.state_manager.ontology_state_file))
        self.assertFalse(os.path.exists(self.state_manager.action_graph_state_file))
        self.assertFalse(os.path.exists(self.state_manager.rdf_state_file))
    
    def test_extraction_state(self):
        """Test extraction state management."""
        # Test saving state
        test_state = {
            "last_run": "2024-03-24T12:00:00Z",
            "bucket_hashes": {"bucket1": "hash1"},
            "table_hashes": {"table1": "hash2"}
        }
        self.state_manager.save_extraction_state(test_state)
        
        # Check that file was created
        self.assertTrue(os.path.exists(self.state_manager.extraction_state_file))
        
        # Test loading state
        loaded_state = self.state_manager.load_extraction_state()
        self.assertEqual(loaded_state, test_state)
    
    def test_metadata(self):
        """Test metadata management."""
        # Test saving metadata
        test_metadata = {
            "buckets": [{"id": "bucket1"}],
            "tables": {"bucket1": [{"id": "table1"}]},
            "configurations": [{"id": "config1"}]
        }
        self.state_manager.save_metadata(test_metadata)
        
        # Check that file was created
        self.assertTrue(os.path.exists(self.state_manager.metadata_file))
        
        # Test loading metadata
        loaded_metadata = self.state_manager.load_metadata()
        self.assertEqual(loaded_metadata, test_metadata)
    
    def test_ontology_state(self):
        """Test ontology state management."""
        # Create test ontology
        ontology_manager = OntologyManager()
        
        # Add test entity
        entity = Entity(
            id="test_entity",
            type=EntityType.TABLE,
            name="Test Table",
            properties={"description": "Test Description"}
        )
        ontology_manager.add_entity(entity)
        
        # Add test relationship
        relationship = Relationship(
            id="test_rel",
            type=RelationshipType.CONTAINS,
            source_id="bucket1",
            target_id="test_entity",
            properties={"created_at": "2024-03-24T12:00:00Z"}
        )
        ontology_manager.add_relationship(relationship)
        
        # Test saving ontology state
        self.state_manager.save_ontology_state(ontology_manager)
        
        # Check that file was created
        self.assertTrue(os.path.exists(self.state_manager.ontology_state_file))
        
        # Test loading ontology state
        loaded_ontology = self.state_manager.load_ontology_state()
        self.assertIsNotNone(loaded_ontology)
        self.assertEqual(len(loaded_ontology.entities), 1)
        self.assertEqual(len(loaded_ontology.relationships), 1)
        
        # Verify entity properties
        loaded_entity = loaded_ontology.get_entity("test_entity")
        self.assertEqual(loaded_entity.id, "test_entity")
        self.assertEqual(loaded_entity.type, EntityType.TABLE)
        self.assertEqual(loaded_entity.name, "Test Table")
        self.assertEqual(loaded_entity.properties["description"], "Test Description")
        
        # Verify relationship properties
        loaded_rel = loaded_ontology.get_relationship("test_rel")
        self.assertEqual(loaded_rel.id, "test_rel")
        self.assertEqual(loaded_rel.type, RelationshipType.CONTAINS)
        self.assertEqual(loaded_rel.source_id, "bucket1")
        self.assertEqual(loaded_rel.target_id, "test_entity")
    
    def test_action_graph_state(self):
        """Test action graph state management."""
        # Create test action graph
        action_graph = ActionGraph()
        
        # Add test action
        action = Action(
            id="test_action",
            type=ActionType.DATA_TRANSFORM,
            name="Test Transform",
            description="Test transformation",
            source_entities=["table1"],
            target_entities=["table2"],
            parameters={"type": "python"},
            metadata={"created_at": "2024-03-24T12:00:00Z"}
        )
        action_graph.add_action(action)
        
        # Test saving action graph state
        self.state_manager.save_action_graph_state(action_graph)
        
        # Check that file was created
        self.assertTrue(os.path.exists(self.state_manager.action_graph_state_file))
        
        # Test loading action graph state
        loaded_graph = self.state_manager.load_action_graph_state()
        self.assertIsNotNone(loaded_graph)
        self.assertEqual(len(loaded_graph.actions), 1)
        
        # Verify action properties
        loaded_action = loaded_graph.get_action("test_action")
        self.assertEqual(loaded_action.id, "test_action")
        self.assertEqual(loaded_action.type, ActionType.DATA_TRANSFORM)
        self.assertEqual(loaded_action.name, "Test Transform")
        self.assertEqual(loaded_action.source_entities, ["table1"])
        self.assertEqual(loaded_action.target_entities, ["table2"])
    
    def test_rdf_state(self):
        """Test RDF state management."""
        # Create test RDF store
        rdf_store = RDFStore()
        
        # Create test ontology manager
        ontology_manager = OntologyManager()
        
        # Add test entity
        entity = Entity(
            id="test_entity",
            type=EntityType.TABLE,
            name="Test Table",
            properties={"description": "Test Description"}
        )
        ontology_manager.add_entity(entity)
        
        # Add test relationship
        relationship = Relationship(
            id="test_rel",
            type=RelationshipType.CONTAINS,
            source_id="bucket1",
            target_id="test_entity",
            properties={"created_at": "2024-03-24T12:00:00Z"}
        )
        ontology_manager.add_relationship(relationship)
        
        # Load data into RDF store
        rdf_store.load_from_ontology_manager(ontology_manager)
        
        # Test saving RDF state
        self.state_manager.save_rdf_state(rdf_store)
        
        # Check that file was created
        self.assertTrue(os.path.exists(self.state_manager.rdf_state_file))
        
        # Test loading RDF state
        loaded_store = self.state_manager.load_rdf_state()
        self.assertIsNotNone(loaded_store)
        
        # Verify data was preserved
        tables = loaded_store.find_entities_by_type(EntityType.TABLE)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].id, "test_entity")
        self.assertEqual(tables[0].type, EntityType.TABLE)
        self.assertEqual(tables[0].name, "Test Table")
        self.assertEqual(tables[0].properties["description"], "Test Description")
        
        contains_rels = loaded_store.find_relationships_by_type(RelationshipType.CONTAINS)
        self.assertEqual(len(contains_rels), 1)
        self.assertEqual(contains_rels[0].id, "test_rel")
        self.assertEqual(contains_rels[0].type, RelationshipType.CONTAINS)
        self.assertEqual(contains_rels[0].source_id, "bucket1")
        self.assertEqual(contains_rels[0].target_id, "test_entity")
    
    def test_compute_hash(self):
        """Test hash computation."""
        # Test with simple dictionary
        test_data = {"key1": "value1", "key2": "value2"}
        hash1 = self.state_manager.compute_hash(test_data)
        
        # Test with same data in different order
        test_data2 = {"key2": "value2", "key1": "value1"}
        hash2 = self.state_manager.compute_hash(test_data2)
        
        # Hashes should be the same
        self.assertEqual(hash1, hash2)
        
        # Test with different data
        test_data3 = {"key1": "value1", "key2": "value3"}
        hash3 = self.state_manager.compute_hash(test_data3)
        
        # Hash should be different
        self.assertNotEqual(hash1, hash3)
    
    def test_error_handling(self):
        """Test error handling in state manager."""
        # Test loading non-existent files
        self.assertEqual(self.state_manager.load_extraction_state(), {})
        self.assertIsNone(self.state_manager.load_metadata())
        self.assertIsNone(self.state_manager.load_ontology_state())
        self.assertIsNone(self.state_manager.load_action_graph_state())
        self.assertIsNone(self.state_manager.load_rdf_state())
        
        # Test saving with invalid data
        with self.assertLogs(level='ERROR'):
            self.state_manager.save_metadata(None)
            self.state_manager.save_ontology_state(None)
            self.state_manager.save_action_graph_state(None)
            self.state_manager.save_rdf_state(None)
        
        # Test loading corrupted files
        with open(self.state_manager.metadata_file, 'w') as f:
            f.write('invalid json')
        
        with self.assertLogs(level='ERROR'):
            self.assertIsNone(self.state_manager.load_metadata())


if __name__ == "__main__":
    unittest.main() 