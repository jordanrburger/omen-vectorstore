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
        # Create actions
        self.action1 = Action(
            id="action_1",
            type=ActionType.DATA_LOAD,
            name="Load Customers",
            description="Load customer data from source",
            source_entities=[],
            target_entities=["table_customers"],
            parameters={},
            metadata={}
        )
        
        self.action2 = Action(
            id="action_2",
            type=ActionType.DATA_TRANSFORM,
            name="Transform Customers",
            description="Transform customer data",
            source_entities=["table_customers"],
            target_entities=["table_customers_transformed"],
            parameters={},
            metadata={}
        )
        
        self.action3 = Action(
            id="action_3",
            type=ActionType.DATA_WRITE,
            name="Write Transformed Customers",
            description="Write transformed customer data to destination",
            source_entities=["table_customers_transformed"],
            target_entities=[],
            parameters={},
            metadata={}
        )
        
        # Create action graph
        self.graph = ActionGraph()
        self.graph.add_action(self.action1)
        self.graph.add_action(self.action2)
        self.graph.add_action(self.action3)
    
    def test_add_action(self):
        """Test adding an action to the graph."""
        # Create a new graph
        graph = ActionGraph()
        
        # Add an action
        graph.add_action(self.action1)
        
        # Check that the action was added
        self.assertIn(self.action1.id, graph.actions)
        self.assertIn(self.action1.id, graph.graph.nodes())
        
        # Check that entity nodes were added
        for entity_id in self.action1.target_entities:
            self.assertIn(entity_id, graph.graph.nodes())
            self.assertIn(entity_id, graph.entity_to_actions)
            self.assertIn(self.action1.id, graph.entity_to_actions[entity_id])
    
    def test_get_action(self):
        """Test getting an action by ID."""
        # Get an existing action
        action = self.graph.get_action("action_1")
        self.assertEqual(action, self.action1)
        
        # Get a non-existent action
        action = self.graph.get_action("non_existent")
        self.assertIsNone(action)
    
    def test_get_all_actions(self):
        """Test getting all actions in the graph."""
        actions = self.graph.get_all_actions()
        
        self.assertEqual(len(actions), 3)
        self.assertIn("action_1", actions)
        self.assertIn("action_2", actions)
        self.assertIn("action_3", actions)
    
    def test_get_actions_for_entity(self):
        """Test getting actions for a specific entity."""
        # Get actions for an entity with one action
        actions = self.graph.get_actions_for_entity("table_customers")
        self.assertEqual(len(actions), 2)  # action1 (output) and action2 (input)
        
        # Get actions for a non-existent entity
        actions = self.graph.get_actions_for_entity("non_existent")
        self.assertEqual(len(actions), 0)
    
    def test_get_upstream_actions(self):
        """Test getting upstream actions."""
        # action1 is upstream of action2
        upstream_actions = self.graph.get_upstream_actions("action_2")
        self.assertEqual(len(upstream_actions), 1)
        self.assertEqual(upstream_actions[0], self.action1)
        
        # action2 is upstream of action3
        upstream_actions = self.graph.get_upstream_actions("action_3")
        self.assertEqual(len(upstream_actions), 1)
        self.assertEqual(upstream_actions[0], self.action2)
        
        # action1 has no upstream actions
        upstream_actions = self.graph.get_upstream_actions("action_1")
        self.assertEqual(len(upstream_actions), 0)
    
    def test_get_downstream_actions(self):
        """Test getting downstream actions."""
        # action2 is downstream of action1
        downstream_actions = self.graph.get_downstream_actions("action_1")
        self.assertEqual(len(downstream_actions), 1)
        self.assertEqual(downstream_actions[0], self.action2)
        
        # action3 is downstream of action2
        downstream_actions = self.graph.get_downstream_actions("action_2")
        self.assertEqual(len(downstream_actions), 1)
        self.assertEqual(downstream_actions[0], self.action3)
        
        # action3 has no downstream actions
        downstream_actions = self.graph.get_downstream_actions("action_3")
        self.assertEqual(len(downstream_actions), 0)
    
    def test_get_execution_order(self):
        """Test getting the execution order of actions."""
        execution_order = self.graph.get_execution_order()
        
        # Check that we have all actions
        self.assertEqual(len(execution_order), 3)
        
        # Check that the order is correct
        # action1 should come before action2, and action2 should come before action3
        action1_index = execution_order.index(self.action1)
        action2_index = execution_order.index(self.action2)
        action3_index = execution_order.index(self.action3)
        
        self.assertLess(action1_index, action2_index)
        self.assertLess(action2_index, action3_index)
    
    def test_to_dict(self):
        """Test converting the action graph to a dictionary."""
        graph_dict = self.graph.to_dict()
        
        self.assertIn("actions", graph_dict)
        self.assertIn("entity_to_actions", graph_dict)
        
        self.assertEqual(len(graph_dict["actions"]), 3)
        self.assertIn("action_1", graph_dict["actions"])
        self.assertIn("action_2", graph_dict["actions"])
        self.assertIn("action_3", graph_dict["actions"])
        
        self.assertIn("table_customers", graph_dict["entity_to_actions"])
        self.assertIn("table_customers_transformed", graph_dict["entity_to_actions"])
    
    def test_from_dict(self):
        """Test creating an action graph from a dictionary."""
        graph_dict = self.graph.to_dict()
        
        # Create a new graph from the dictionary
        new_graph = ActionGraph.from_dict(graph_dict)
        
        # Check that the new graph has the same actions
        self.assertEqual(len(new_graph.actions), 3)
        self.assertIn("action_1", new_graph.actions)
        self.assertIn("action_2", new_graph.actions)
        self.assertIn("action_3", new_graph.actions)
        
        # Check that the actions have the same properties
        self.assertEqual(new_graph.actions["action_1"].name, "Load Customers")
        self.assertEqual(new_graph.actions["action_2"].name, "Transform Customers")
        self.assertEqual(new_graph.actions["action_3"].name, "Write Transformed Customers")


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