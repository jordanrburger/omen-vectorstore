"""Tests for the action graph generator functionality."""

import unittest
from unittest.mock import Mock, patch
import uuid
import networkx as nx
from datetime import datetime

from app.ontology.models import Entity, Relationship, Triple, EntityType, RelationshipType
from app.ontology.manager import OntologyManager
from app.ontology.schema import SchemaValidator
from app.ontology.schema_definition import default_schema
from app.ontology.action_graph import Action, ActionType, ActionGraph, ActionGraphBuilder
from app.llm_client import LLMClient


class TestAction(unittest.TestCase):
    """Test cases for the Action class."""
    
    def test_initialization(self):
        """Test initializing an action."""
        # Create an action
        action = Action(
            id="action_1",
            type=ActionType.DATA_TRANSFORM,
            name="Transform Customer Data",
            description="Transforms customer data for analysis",
            source_entities=["table_1", "table_2"],
            target_entities=["table_3"],
            parameters={"transformation_type": "python"},
            metadata={"created_at": "2023-01-01T00:00:00"}
        )
        
        # Check attributes
        self.assertEqual(action.id, "action_1")
        self.assertEqual(action.type, ActionType.DATA_TRANSFORM)
        self.assertEqual(action.name, "Transform Customer Data")
        self.assertEqual(action.description, "Transforms customer data for analysis")
        self.assertEqual(action.source_entities, ["table_1", "table_2"])
        self.assertEqual(action.target_entities, ["table_3"])
        self.assertEqual(action.parameters, {"transformation_type": "python"})
        self.assertEqual(action.metadata, {"created_at": "2023-01-01T00:00:00"})
    
    def test_auto_id_generation(self):
        """Test that an ID is automatically generated if not provided."""
        action = Action(
            id="",
            type=ActionType.DATA_TRANSFORM,
            name="Test Action",
            description="",
            source_entities=[],
            target_entities=[],
            parameters={},
            metadata={}
        )
        
        self.assertIsNotNone(action.id)
        self.assertNotEqual(action.id, "")
    
    def test_string_to_list_conversion(self):
        """Test that string inputs for entities are converted to lists."""
        action = Action(
            id="action_1",
            type=ActionType.DATA_TRANSFORM,
            name="Test Action",
            description="",
            source_entities="table_1",  # String instead of list
            target_entities="table_2",  # String instead of list
            parameters={},
            metadata={}
        )
        
        self.assertIsInstance(action.source_entities, list)
        self.assertIsInstance(action.target_entities, list)
        self.assertEqual(action.source_entities, ["table_1"])
        self.assertEqual(action.target_entities, ["table_2"])
    
    def test_to_dict(self):
        """Test converting an action to a dictionary."""
        action = Action(
            id="action_1",
            type=ActionType.DATA_TRANSFORM,
            name="Test Action",
            description="Test Description",
            source_entities=["table_1"],
            target_entities=["table_2"],
            parameters={"param1": "value1"},
            metadata={"meta1": "value1"}
        )
        
        action_dict = action.to_dict()
        
        self.assertEqual(action_dict["id"], "action_1")
        self.assertEqual(action_dict["type"], "DATA_TRANSFORM")
        self.assertEqual(action_dict["name"], "Test Action")
        self.assertEqual(action_dict["description"], "Test Description")
        self.assertEqual(action_dict["source_entities"], ["table_1"])
        self.assertEqual(action_dict["target_entities"], ["table_2"])
        self.assertEqual(action_dict["parameters"], {"param1": "value1"})
        self.assertEqual(action_dict["metadata"], {"meta1": "value1"})
    
    def test_from_dict(self):
        """Test creating an action from a dictionary."""
        action_dict = {
            "id": "action_1",
            "type": "DATA_TRANSFORM",
            "name": "Test Action",
            "description": "Test Description",
            "source_entities": ["table_1"],
            "target_entities": ["table_2"],
            "parameters": {"param1": "value1"},
            "metadata": {"meta1": "value1"}
        }
        
        action = Action.from_dict(action_dict)
        
        self.assertEqual(action.id, "action_1")
        self.assertEqual(action.type, ActionType.DATA_TRANSFORM)
        self.assertEqual(action.name, "Test Action")
        self.assertEqual(action.description, "Test Description")
        self.assertEqual(action.source_entities, ["table_1"])
        self.assertEqual(action.target_entities, ["table_2"])
        self.assertEqual(action.parameters, {"param1": "value1"})
        self.assertEqual(action.metadata, {"meta1": "value1"})


class TestActionGraph(unittest.TestCase):
    """Test cases for the ActionGraph class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ontology = OntologyManager()
        self.action_graph = ActionGraph()

        # Create test entities
        self.bucket = Entity(
            id="test_bucket",
            type=EntityType.BUCKET,
            name="Test Bucket",
            properties={"name": "Test Bucket"}
        )
        self.table1 = Entity(
            id="test_table1",
            type=EntityType.TABLE,
            name="Test Table 1",
            properties={"name": "Test Table 1"}
        )
        self.table2 = Entity(
            id="test_table2",
            type=EntityType.TABLE,
            name="Test Table 2",
            properties={"name": "Test Table 2"}
        )
        self.config = Entity(
            id="test_config",
            type=EntityType.CONFIGURATION,
            name="Test Config",
            properties={"name": "Test Config"}
        )

        # Add entities to ontology
        self.ontology.add_entity(self.bucket)
        self.ontology.add_entity(self.table1)
        self.ontology.add_entity(self.table2)
        self.ontology.add_entity(self.config)

        # Add relationships
        self.contains_rel1 = Relationship(
            id="contains_rel1",
            type=RelationshipType.CONTAINS,
            source_id="test_bucket",
            target_id="test_table1",
            properties={}
        )
        self.contains_rel2 = Relationship(
            id="contains_rel2",
            type=RelationshipType.CONTAINS,
            source_id="test_bucket",
            target_id="test_table2",
            properties={}
        )
        self.belongs_rel = Relationship(
            id="belongs_rel",
            type=RelationshipType.BELONGS_TO,
            source_id="test_table1",
            target_id="test_config",
            properties={}
        )

        self.ontology.add_relationship(self.contains_rel1)
        self.ontology.add_relationship(self.contains_rel2)
        self.ontology.add_relationship(self.belongs_rel)
    
    def test_add_action(self):
        """Test adding actions to the graph."""
        # Create and add a valid action
        action = Action(
            id="test_action",
            type=ActionType.LOAD_DATA,
            source_id="test_table1",
            target_id="test_table2",
            properties={
                "created_at": datetime.now().isoformat(),
                "description": "Load data from table1 to table2"
            }
        )
        self.action_graph.add_action(action)

        # Verify action was added
        self.assertEqual(len(self.action_graph.actions), 1)
        self.assertEqual(self.action_graph.actions["test_action"], action)

        # Test adding duplicate action
        with self.assertRaises(ValueError):
            self.action_graph.add_action(action)
    
    def test_get_action(self):
        """Test retrieving actions from the graph."""
        # Create and add a test action
        action = Action(
            id="test_action",
            type=ActionType.LOAD_DATA,
            source_id="test_table1",
            target_id="test_table2",
            properties={
                "description": "Load data from table1 to table2"
            }
        )
        self.action_graph.add_action(action)

        # Test getting existing action
        retrieved = self.action_graph.get_action("test_action")
        self.assertEqual(retrieved, action)

        # Test getting non-existent action
        with self.assertRaises(KeyError):
            self.action_graph.get_action("non_existent")
    
    def test_get_actions_by_type(self):
        """Test retrieving actions by type."""
        # Create and add test actions of different types
        load_action = Action(
            id="load_action",
            type=ActionType.LOAD_DATA,
            source_id="test_table1",
            target_id="test_table2",
            properties={"description": "Load data"}
        )
        transform_action = Action(
            id="transform_action",
            type=ActionType.TRANSFORM_DATA,
            source_id="test_table2",
            target_id="test_table2",
            properties={"description": "Transform data"}
        )
        self.action_graph.add_action(load_action)
        self.action_graph.add_action(transform_action)

        # Test getting actions by type
        load_actions = self.action_graph.get_actions_by_type(ActionType.LOAD_DATA)
        self.assertEqual(len(load_actions), 1)
        self.assertEqual(load_actions[0], load_action)

        transform_actions = self.action_graph.get_actions_by_type(ActionType.TRANSFORM_DATA)
        self.assertEqual(len(transform_actions), 1)
        self.assertEqual(transform_actions[0], transform_action)

        # Test getting actions of non-existent type
        empty = self.action_graph.get_actions_by_type(ActionType.VALIDATE_DATA)
        self.assertEqual(len(empty), 0)
    
    def test_get_actions_for_entity(self):
        """Test retrieving actions for a specific entity."""
        # Create and add test actions
        action1 = Action(
            id="action1",
            type=ActionType.LOAD_DATA,
            source_id="test_table1",
            target_id="test_table2",
            properties={"description": "Load data"}
        )
        action2 = Action(
            id="action2",
            type=ActionType.TRANSFORM_DATA,
            source_id="test_table2",
            target_id="test_table2",
            properties={"description": "Transform data"}
        )
        self.action_graph.add_action(action1)
        self.action_graph.add_action(action2)

        # Test getting actions for source entity
        source_actions = self.action_graph.get_actions_for_entity("test_table1", as_source=True)
        self.assertEqual(len(source_actions), 1)
        self.assertEqual(source_actions[0], action1)

        # Test getting actions for target entity
        target_actions = self.action_graph.get_actions_for_entity("test_table2", as_target=True)
        self.assertEqual(len(target_actions), 2)
        self.assertIn(action1, target_actions)
        self.assertIn(action2, target_actions)

        # Test getting actions for non-existent entity
        with self.assertRaises(KeyError):
            self.action_graph.get_actions_for_entity("non_existent")
    
    def test_remove_action(self):
        """Test removing actions from the graph."""
        # Create and add a test action
        action = Action(
            id="test_action",
            type=ActionType.LOAD_DATA,
            source_id="test_table1",
            target_id="test_table2",
            properties={"description": "Load data"}
        )
        self.action_graph.add_action(action)

        # Test removing existing action
        self.action_graph.remove_action("test_action")
        self.assertEqual(len(self.action_graph.actions), 0)

        # Test removing non-existent action
        with self.assertRaises(KeyError):
            self.action_graph.remove_action("non_existent")
    
    def test_get_action_chain(self):
        """Test retrieving action chains between entities."""
        # Create and add test actions forming a chain
        action1 = Action(
            id="action1",
            type=ActionType.LOAD_DATA,
            source_id="test_table1",
            target_id="test_table2",
            properties={"description": "Load data"}
        )
        action2 = Action(
            id="action2",
            type=ActionType.TRANSFORM_DATA,
            source_id="test_table2",
            target_id="test_config",
            properties={"description": "Transform data"}
        )
        self.action_graph.add_action(action1)
        self.action_graph.add_action(action2)

        # Test getting action chain
        chain = self.action_graph.get_action_chain("test_table1", "test_config")
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0], action1)
        self.assertEqual(chain[1], action2)

        # Test getting chain for non-connected entities
        empty_chain = self.action_graph.get_action_chain("test_bucket", "test_config")
        self.assertEqual(len(empty_chain), 0)

        # Test getting chain for non-existent entity
        with self.assertRaises(KeyError):
            self.action_graph.get_action_chain("non_existent", "test_config")


class TestActionGraphBuilder(unittest.TestCase):
    """Test cases for the ActionGraphBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create an ontology manager
        self.ontology_manager = OntologyManager()
        
        # Create schema validator
        self.schema_validator = SchemaValidator(default_schema)
        
        # Create mock LLM client
        self.mock_llm_client = Mock(spec=LLMClient)
        
        # Create entities
        self.bucket_entity = Entity(
            id="bucket_1",
            type=EntityType.BUCKET,
            properties={"name": "in.c-main"}
        )
        
        self.table_entity = Entity(
            id="table_1",
            type=EntityType.TABLE,
            properties={"name": "customers", "row_count": 1000}
        )
        
        self.column_entity = Entity(
            id="column_1",
            type=EntityType.COLUMN,
            properties={"name": "email", "data_type": "string"}
        )
        
        self.transformation_entity = Entity(
            id="transformation_1",
            type=EntityType.TRANSFORMATION,
            properties={
                "name": "Process Customers",
                "type": "python",
                "description": "Processes customer data"
            }
        )
        
        self.component_entity = Entity(
            id="component_1",
            type=EntityType.COMPONENT,
            properties={"name": "keboola.python-transformation", "type": "transformation"}
        )
        
        self.extractor_component = Entity(
            id="component_2",
            type=EntityType.COMPONENT,
            properties={"name": "keboola.ex-db-mysql", "type": "extractor"}
        )
        
        self.extractor_config = Entity(
            id="config_1",
            type=EntityType.CONFIGURATION,
            properties={"name": "MySQL Extractor", "description": "Extracts data from MySQL"}
        )
        
        # Add entities to the ontology manager
        self.ontology_manager.add_entity(self.bucket_entity)
        self.ontology_manager.add_entity(self.table_entity)
        self.ontology_manager.add_entity(self.column_entity)
        self.ontology_manager.add_entity(self.transformation_entity)
        self.ontology_manager.add_entity(self.component_entity)
        self.ontology_manager.add_entity(self.extractor_component)
        self.ontology_manager.add_entity(self.extractor_config)
        
        # Create relationships
        self.contains_rel = Relationship(
            id="rel_1",
            type=RelationshipType.CONTAINS,
            source_id="bucket_1",
            target_id="table_1",
            properties={}
        )
        
        self.has_column_rel = Relationship(
            id="rel_2",
            type=RelationshipType.HAS_COLUMN,
            source_id="table_1",
            target_id="column_1",
            properties={}
        )
        
        self.belongs_to_rel = Relationship(
            id="rel_3",
            type=RelationshipType.BELONGS_TO,
            source_id="transformation_1",
            target_id="component_1",
            properties={}
        )
        
        self.inputs_from_rel = Relationship(
            id="rel_4",
            type=RelationshipType.INPUTS_FROM,
            source_id="transformation_1",
            target_id="table_1",
            properties={}
        )
        
        self.belongs_to_extractor_rel = Relationship(
            id="rel_5",
            type=RelationshipType.BELONGS_TO,
            source_id="config_1",
            target_id="component_2",
            properties={}
        )
        
        self.outputs_to_rel = Relationship(
            id="rel_6",
            type=RelationshipType.OUTPUTS_TO,
            source_id="config_1",
            target_id="table_1",
            properties={}
        )
        
        # Add relationships to the ontology manager
        self.ontology_manager.add_relationship(self.contains_rel)
        self.ontology_manager.add_relationship(self.has_column_rel)
        self.ontology_manager.add_relationship(self.belongs_to_rel)
        self.ontology_manager.add_relationship(self.inputs_from_rel)
        self.ontology_manager.add_relationship(self.belongs_to_extractor_rel)
        self.ontology_manager.add_relationship(self.outputs_to_rel)
        
        # Create action graph builder
        self.builder = ActionGraphBuilder(
            ontology_manager=self.ontology_manager,
            llm_client=self.mock_llm_client,
            schema_validator=self.schema_validator
        )
    
    def test_build_action_graph(self):
        """Test building an action graph from the ontology."""
        # Build the action graph
        action_graph = self.builder.build_action_graph()
        
        # Check that actions were created
        actions = action_graph.get_all_actions()
        
        # We should have at least two actions:
        # 1. A DATA_TRANSFORM action for the transformation
        # 2. A DATA_LOAD action for the extractor configuration
        self.assertGreaterEqual(len(actions), 2)
        
        # Check that we have the right types of actions
        action_types = [action.type for action in actions.values()]
        self.assertIn(ActionType.DATA_TRANSFORM, action_types)
        self.assertIn(ActionType.DATA_LOAD, action_types)
        
        # Find the transform action
        transform_action = None
        for action in actions.values():
            if action.type == ActionType.DATA_TRANSFORM:
                transform_action = action
                break
        
        self.assertIsNotNone(transform_action)
        self.assertEqual(transform_action.name, "Process Customers")
        self.assertEqual(transform_action.source_entities, ["table_1"])
        
        # Find the data load action
        load_action = None
        for action in actions.values():
            if action.type == ActionType.DATA_LOAD:
                load_action = action
                break
        
        self.assertIsNotNone(load_action)
        self.assertEqual(load_action.target_entities, ["table_1"])
    
    def test_get_transformation_inputs(self):
        """Test getting inputs for a transformation."""
        inputs = self.builder._get_transformation_inputs("transformation_1")
        self.assertEqual(inputs, ["table_1"])
    
    def test_get_transformation_outputs(self):
        """Test getting outputs for a transformation."""
        # Add an OUTPUTS_TO relationship
        outputs_rel = Relationship(
            id="rel_7",
            type=RelationshipType.OUTPUTS_TO,
            source_id="transformation_1",
            target_id="table_2",
            properties={}
        )
        
        # Add output table entity
        output_table_entity = Entity(
            id="table_2",
            type=EntityType.TABLE,
            properties={"name": "processed_customers"}
        )
        
        self.ontology_manager.add_entity(output_table_entity)
        self.ontology_manager.add_relationship(outputs_rel)
        
        outputs = self.builder._get_transformation_outputs("transformation_1")
        self.assertEqual(outputs, ["table_2"])
    
    def test_get_component_for_config(self):
        """Test getting the component for a configuration."""
        component_id = self.builder._get_component_for_config("config_1")
        self.assertEqual(component_id, "component_2")
    
    def test_get_config_outputs(self):
        """Test getting outputs for a configuration."""
        outputs = self.builder._get_config_outputs("config_1")
        self.assertEqual(outputs, ["table_1"])


if __name__ == "__main__":
    unittest.main() 