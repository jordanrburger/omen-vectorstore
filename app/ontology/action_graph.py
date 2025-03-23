"""
Module for generating and managing action graphs from Keboola ontology.

The action graph represents data flow and transformations in a Keboola project,
with actions representing specific operations like data loading, transformation,
or writing. This module provides classes for creating and managing action graphs.
"""

import logging
import uuid
import json
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

from app.ontology.models import Entity, Relationship, EntityType, RelationshipType
from app.ontology.manager import OntologyManager
from app.ontology.schema import SchemaValidator

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions in the action graph."""
    DATA_LOAD = auto()
    DATA_TRANSFORM = auto()
    DATA_WRITE = auto()
    ORCHESTRATION = auto()
    CUSTOM = auto()


class Action:
    """Represents a data action in the action graph."""
    
    def __init__(
        self,
        id: str,
        type: ActionType,
        name: str,
        description: str,
        source_entities: Union[List[str], str],
        target_entities: Union[List[str], str],
        parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ):
        """Initialize an action.
        
        Args:
            id: Unique identifier for the action
            type: Type of action
            name: Name of the action
            description: Description of the action
            source_entities: List of source entity IDs or a single source entity ID
            target_entities: List of target entity IDs or a single target entity ID
            parameters: Optional parameters for the action
            metadata: Optional metadata for the action
        """
        # Generate a UUID if no ID is provided
        self.id = id if id else str(uuid.uuid4())
        self.type = type
        self.name = name
        self.description = description
        
        # Convert source_entities to list if it's a string
        if isinstance(source_entities, str):
            self.source_entities = [source_entities]
        else:
            self.source_entities = source_entities
        
        # Convert target_entities to list if it's a string
        if isinstance(target_entities, str):
            self.target_entities = [target_entities]
        else:
            self.target_entities = target_entities
        
        self.parameters = parameters or {}
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the action to a dictionary.
        
        Returns:
            Dictionary representation of the action
        """
        return {
            "id": self.id,
            "type": self.type.name,
            "name": self.name,
            "description": self.description,
            "source_entities": self.source_entities,
            "target_entities": self.target_entities,
            "parameters": self.parameters,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        """Create an action from a dictionary.
        
        Args:
            data: Dictionary with action data
            
        Returns:
            New Action instance
        """
        action_type = ActionType[data["type"]]
        
        return cls(
            id=data["id"],
            type=action_type,
            name=data["name"],
            description=data["description"],
            source_entities=data["source_entities"],
            target_entities=data["target_entities"],
            parameters=data.get("parameters", {}),
            metadata=data.get("metadata", {})
        )


class ActionGraph:
    """Graph of data actions and their relationships."""
    
    def __init__(self):
        """Initialize an empty action graph."""
        self.actions: Dict[str, Action] = {}
        self.graph = nx.DiGraph()
        self.entity_to_actions: Dict[str, List[str]] = defaultdict(list)
    
    def add_action(self, action: Action) -> None:
        """Add an action to the graph.
        
        Args:
            action: Action to add
        """
        # Add the action to the actions dictionary
        self.actions[action.id] = action
        
        # Add the action node to the graph
        self.graph.add_node(action.id, type="action", name=action.name)
        
        # Add entity nodes and edges
        for entity_id in action.source_entities:
            if entity_id:
                if entity_id not in self.graph:
                    self.graph.add_node(entity_id, type="entity")
                
                self.graph.add_edge(entity_id, action.id)
                self.entity_to_actions[entity_id].append(action.id)
        
        for entity_id in action.target_entities:
            if entity_id:
                if entity_id not in self.graph:
                    self.graph.add_node(entity_id, type="entity")
                
                self.graph.add_edge(action.id, entity_id)
                self.entity_to_actions[entity_id].append(action.id)
    
    def get_action(self, action_id: str) -> Optional[Action]:
        """Get an action by ID.
        
        Args:
            action_id: ID of the action to get
            
        Returns:
            Action if found, None otherwise
        """
        return self.actions.get(action_id)
    
    def get_all_actions(self) -> Dict[str, Action]:
        """Get all actions in the graph.
        
        Returns:
            Dictionary of action IDs to actions
        """
        return self.actions
    
    def get_actions_for_entity(self, entity_id: str) -> List[Action]:
        """Get all actions that use a specific entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            List of actions that use the entity
        """
        action_ids = self.entity_to_actions.get(entity_id, [])
        return [self.actions[action_id] for action_id in action_ids]
    
    def get_upstream_actions(self, action_id: str) -> List[Action]:
        """Get actions that are upstream of the given action.
        
        Args:
            action_id: ID of the action
            
        Returns:
            List of upstream actions
        """
        action = self.get_action(action_id)
        if not action:
            return []
        
        upstream_actions = []
        
        for entity_id in action.source_entities:
            for upstream_action_id in self.entity_to_actions.get(entity_id, []):
                upstream_action = self.get_action(upstream_action_id)
                
                # Skip if the action is the same or if it's not a producer of the entity
                if (upstream_action and upstream_action.id != action_id and 
                    entity_id in upstream_action.target_entities):
                    upstream_actions.append(upstream_action)
        
        return upstream_actions
    
    def get_downstream_actions(self, action_id: str) -> List[Action]:
        """Get actions that are downstream of the given action.
        
        Args:
            action_id: ID of the action
            
        Returns:
            List of downstream actions
        """
        action = self.get_action(action_id)
        if not action:
            return []
        
        downstream_actions = []
        
        for entity_id in action.target_entities:
            for downstream_action_id in self.entity_to_actions.get(entity_id, []):
                downstream_action = self.get_action(downstream_action_id)
                
                # Skip if the action is the same or if it's not a consumer of the entity
                if (downstream_action and downstream_action.id != action_id and 
                    entity_id in downstream_action.source_entities):
                    downstream_actions.append(downstream_action)
        
        return downstream_actions
    
    def get_execution_order(self) -> List[Action]:
        """Get a topological ordering of actions for execution.
        
        Returns:
            List of actions in execution order
        """
        # Create a graph with only action nodes
        action_graph = nx.DiGraph()
        
        for action_id in self.actions:
            action_graph.add_node(action_id)
        
        # Add edges between actions based on entity dependencies
        for action_id, action in self.actions.items():
            downstream_actions = self.get_downstream_actions(action_id)
            
            for downstream_action in downstream_actions:
                action_graph.add_edge(action_id, downstream_action.id)
        
        # Get topological sort
        try:
            ordered_action_ids = list(nx.topological_sort(action_graph))
            return [self.actions[action_id] for action_id in ordered_action_ids]
        except nx.NetworkXUnfeasible:
            logger.warning("Action graph contains cycles, cannot determine exact execution order")
            return list(self.actions.values())
    
    def visualize(self, output_file: str = None) -> None:
        """Visualize the action graph.
        
        Args:
            output_file: Path to save the visualization to. If None, display interactively.
        """
        plt.figure(figsize=(12, 8))
        
        # Create position layout
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw action nodes
        action_nodes = [node for node, attrs in self.graph.nodes(data=True) 
                       if attrs.get("type") == "action"]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=action_nodes, 
                              node_color='lightblue', node_size=700, alpha=0.8)
        
        # Draw entity nodes
        entity_nodes = [node for node, attrs in self.graph.nodes(data=True) 
                       if attrs.get("type") == "entity"]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=entity_nodes, 
                              node_color='lightgreen', node_size=500, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5, arrows=True)
        
        # Draw labels
        labels = {}
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "action":
                action = self.actions[node]
                labels[node] = f"{action.name}\n({action.type.name})"
            else:
                # Use the last part of the entity ID for readability
                labels[node] = node.split('_')[-1]
        
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("Action Graph")
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the action graph to a dictionary.
        
        Returns:
            Dictionary representation of the action graph
        """
        return {
            "actions": {action_id: action.to_dict() for action_id, action in self.actions.items()},
            "entity_to_actions": self.entity_to_actions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionGraph':
        """Create an action graph from a dictionary.
        
        Args:
            data: Dictionary with action graph data
            
        Returns:
            New ActionGraph instance
        """
        graph = cls()
        
        # Add actions
        for action_data in data["actions"].values():
            action = Action.from_dict(action_data)
            graph.add_action(action)
        
        return graph


class ActionGraphBuilder:
    """Builds action graphs from Keboola ontology."""
    
    def __init__(
        self, 
        ontology_manager: OntologyManager,
        schema_validator: Optional[SchemaValidator] = None,
        llm_client = None
    ):
        """Initialize an action graph builder.
        
        Args:
            ontology_manager: Manager for the ontology
            schema_validator: Optional validator for the ontology schema
            llm_client: Optional client for language model inferencing
        """
        self.ontology_manager = ontology_manager
        self.schema_validator = schema_validator
        self.llm_client = llm_client
    
    def build_action_graph(self) -> ActionGraph:
        """Build an action graph from the ontology.
        
        Returns:
            Action graph representing the data flow
        """
        logger.info("Building action graph from ontology")
        action_graph = ActionGraph()
        
        # Process transformations to create DATA_TRANSFORM actions
        self._process_transformations(action_graph)
        
        # Process configurations to create DATA_LOAD and DATA_WRITE actions
        self._process_configurations(action_graph)
        
        # Process orchestrations to create ORCHESTRATION actions
        self._process_orchestrations(action_graph)
        
        # Infer additional actions using LLM if available
        if self.llm_client:
            self._infer_additional_actions(action_graph)
        
        logger.info(f"Built action graph with {len(action_graph.actions)} actions")
        return action_graph
    
    def _process_transformations(self, action_graph: ActionGraph) -> None:
        """Process transformation entities to create DATA_TRANSFORM actions.
        
        Args:
            action_graph: Action graph to add actions to
        """
        transformation_entities = self.ontology_manager.get_entities_by_type(EntityType.TRANSFORMATION)
        
        for transformation_entity in transformation_entities:
            component_id = self._get_component_for_transformation(transformation_entity.id)
            
            if not component_id:
                logger.warning(f"No component found for transformation {transformation_entity.id}")
                continue
            
            component_entity = self.ontology_manager.get_entity(component_id)
            if not component_entity:
                logger.warning(f"Component entity {component_id} not found")
                continue
            
            # Get inputs and outputs for the transformation
            source_entities = self._get_transformation_inputs(transformation_entity.id)
            target_entities = self._get_transformation_outputs(transformation_entity.id)
            
            transform_action = Action(
                id=f"transform_{transformation_entity.id}",
                type=ActionType.DATA_TRANSFORM,
                name=transformation_entity.properties.get("name", f"Transformation {transformation_entity.id}"),
                description=transformation_entity.properties.get("description", ""),
                source_entities=source_entities,
                target_entities=target_entities,
                parameters={
                    "transformation_type": transformation_entity.properties.get("type", "unknown"),
                    "component": component_entity.properties.get("name", "unknown")
                },
                metadata={
                    "transformation_id": transformation_entity.id,
                    "component_id": component_id
                }
            )
            
            action_graph.add_action(transform_action)
    
    def _process_configurations(self, action_graph: ActionGraph) -> None:
        """Process configuration entities to create DATA_LOAD and DATA_WRITE actions.
        
        Args:
            action_graph: Action graph to add actions to
        """
        config_entities = self.ontology_manager.get_entities_by_type(EntityType.CONFIGURATION)
        
        for config_entity in config_entities:
            component_id = self._get_component_for_config(config_entity.id)
            
            if not component_id:
                logger.warning(f"No component found for configuration {config_entity.id}")
                continue
            
            component_entity = self.ontology_manager.get_entity(component_id)
            if not component_entity:
                logger.warning(f"Component entity {component_id} not found")
                continue
            
            component_type = component_entity.properties.get("type", "unknown")
            
            # Handle extractors (DATA_LOAD)
            if component_type == "extractor":
                outputs = self._get_config_outputs(config_entity.id)
                
                if outputs:
                    load_action = Action(
                        id=f"load_{config_entity.id}",
                        type=ActionType.DATA_LOAD,
                        name=config_entity.properties.get("name", f"Load {config_entity.id}"),
                        description=config_entity.properties.get("description", ""),
                        source_entities=[],  # External source, no entity in our graph
                        target_entities=outputs,
                        parameters={
                            "component": component_entity.properties.get("name", "unknown")
                        },
                        metadata={
                            "config_id": config_entity.id,
                            "component_id": component_id
                        }
                    )
                    
                    action_graph.add_action(load_action)
            
            # Handle writers (DATA_WRITE)
            elif component_type == "writer":
                inputs = self._get_config_inputs(config_entity.id)
                
                if inputs:
                    write_action = Action(
                        id=f"write_{config_entity.id}",
                        type=ActionType.DATA_WRITE,
                        name=config_entity.properties.get("name", f"Write {config_entity.id}"),
                        description=config_entity.properties.get("description", ""),
                        source_entities=inputs,
                        target_entities=[],  # External target, no entity in our graph
                        parameters={
                            "component": component_entity.properties.get("name", "unknown")
                        },
                        metadata={
                            "config_id": config_entity.id,
                            "component_id": component_id
                        }
                    )
                    
                    action_graph.add_action(write_action)
    
    def _process_orchestrations(self, action_graph: ActionGraph) -> None:
        """Process orchestration entities to create ORCHESTRATION actions.
        
        Args:
            action_graph: Action graph to add actions to
        """
        orchestration_entities = self.ontology_manager.get_entities_by_type(EntityType.ORCHESTRATION)
        
        for orchestration_entity in orchestration_entities:
            # Get tasks associated with the orchestration
            orchestration_tasks = self._get_orchestration_tasks(orchestration_entity.id)
            
            if not orchestration_tasks:
                logger.warning(f"No tasks found for orchestration {orchestration_entity.id}")
                continue
            
            # Create a list of related actions
            related_actions = []
            
            for task_id in orchestration_tasks:
                task_entity = self.ontology_manager.get_entity(task_id)
                
                if not task_entity:
                    logger.warning(f"Task entity {task_id} not found")
                    continue
                
                # Get the configuration associated with the task
                config_id = self._get_config_for_task(task_id)
                
                if not config_id:
                    logger.warning(f"No configuration found for task {task_id}")
                    continue
                
                # Find actions related to this configuration
                for action in action_graph.actions.values():
                    if action.metadata.get("config_id") == config_id:
                        related_actions.append(action.id)
            
            if related_actions:
                orchestration_action = Action(
                    id=f"orchestration_{orchestration_entity.id}",
                    type=ActionType.ORCHESTRATION,
                    name=orchestration_entity.properties.get("name", f"Orchestration {orchestration_entity.id}"),
                    description=orchestration_entity.properties.get("description", ""),
                    source_entities=[],  # Orchestrations don't directly use entities
                    target_entities=[],  # Orchestrations don't directly produce entities
                    parameters={
                        "related_actions": related_actions
                    },
                    metadata={
                        "orchestration_id": orchestration_entity.id,
                        "tasks": orchestration_tasks
                    }
                )
                
                action_graph.add_action(orchestration_action)
    
    def _infer_additional_actions(self, action_graph: ActionGraph) -> None:
        """Infer additional actions using language model.
        
        Args:
            action_graph: Action graph to add inferred actions to
        """
        if not self.llm_client:
            logger.warning("No LLM client available for inferring additional actions")
            return
        
        try:
            logger.info("Inferring additional actions using LLM")
            
            # Get all entities and their properties
            entities_data = {}
            for entity in self.ontology_manager.get_all_entities():
                entities_data[entity.id] = {
                    "type": entity.type.name,
                    "properties": entity.properties
                }
            
            # Get all relationships
            relationships_data = []
            for relationship in self.ontology_manager.get_all_relationships():
                relationships_data.append({
                    "type": relationship.type.name,
                    "source_id": relationship.source_id,
                    "target_id": relationship.target_id
                })
            
            # Get existing actions
            actions_data = [action.to_dict() for action in action_graph.actions.values()]
            
            # Prepare prompt for LLM
            prompt = {
                "entities": entities_data,
                "relationships": relationships_data,
                "existing_actions": actions_data,
                "missing_action_types": [
                    {"name": "DATA_LOAD", "description": "Load data from an external source"},
                    {"name": "DATA_TRANSFORM", "description": "Transform data within the system"},
                    {"name": "DATA_WRITE", "description": "Write data to an external destination"}
                ]
            }
            
            # Call LLM to infer additional actions
            inferred_actions = self.llm_client.generate(
                system_prompt="You are an AI assistant that analyzes data flows in a Keboola project. "
                              "Based on the provided entities, relationships, and existing actions, "
                              "identify missing actions that should be part of the data flow.",
                user_prompt=json.dumps(prompt),
                response_format="json"
            )
            
            # Add inferred actions to the graph
            if isinstance(inferred_actions, dict) and "actions" in inferred_actions:
                for action_data in inferred_actions["actions"]:
                    try:
                        action_type = ActionType[action_data.get("type", "CUSTOM")]
                        
                        action = Action(
                            id=action_data.get("id", f"inferred_{str(uuid.uuid4())}"),
                            type=action_type,
                            name=action_data.get("name", "Inferred Action"),
                            description=action_data.get("description", "Action inferred by LLM"),
                            source_entities=action_data.get("source_entities", []),
                            target_entities=action_data.get("target_entities", []),
                            parameters=action_data.get("parameters", {}),
                            metadata={
                                "inferred": True,
                                "confidence": action_data.get("confidence", 0.5)
                            }
                        )
                        
                        action_graph.add_action(action)
                        logger.info(f"Added inferred action: {action.name}")
                    
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Error processing inferred action: {e}")
        
        except Exception as e:
            logger.error(f"Error inferring additional actions: {e}")
    
    def _get_component_for_transformation(self, transformation_id: str) -> Optional[str]:
        """Get the component ID for a transformation.
        
        Args:
            transformation_id: ID of the transformation
            
        Returns:
            Component ID or None if not found
        """
        # Find BELONGS_TO relationships where the transformation is the source
        belongs_to_rels = self.ontology_manager.get_relationships_by_source(
            transformation_id, RelationshipType.BELONGS_TO
        )
        
        for rel in belongs_to_rels:
            target_entity = self.ontology_manager.get_entity(rel.target_id)
            
            if target_entity and target_entity.type == EntityType.COMPONENT:
                return target_entity.id
        
        return None
    
    def _get_component_for_config(self, config_id: str) -> Optional[str]:
        """Get the component ID for a configuration.
        
        Args:
            config_id: ID of the configuration
            
        Returns:
            Component ID or None if not found
        """
        # Find BELONGS_TO relationships where the configuration is the source
        belongs_to_rels = self.ontology_manager.get_relationships_by_source(
            config_id, RelationshipType.BELONGS_TO
        )
        
        for rel in belongs_to_rels:
            target_entity = self.ontology_manager.get_entity(rel.target_id)
            
            if target_entity and target_entity.type == EntityType.COMPONENT:
                return target_entity.id
        
        return None
    
    def _get_transformation_inputs(self, transformation_id: str) -> List[str]:
        """Get input entities for a transformation.
        
        Args:
            transformation_id: ID of the transformation
            
        Returns:
            List of input entity IDs
        """
        # Find INPUTS_FROM relationships where the transformation is the source
        inputs_from_rels = self.ontology_manager.get_relationships_by_source(
            transformation_id, RelationshipType.INPUTS_FROM
        )
        
        return [rel.target_id for rel in inputs_from_rels]
    
    def _get_transformation_outputs(self, transformation_id: str) -> List[str]:
        """Get output entities for a transformation.
        
        Args:
            transformation_id: ID of the transformation
            
        Returns:
            List of output entity IDs
        """
        # Find OUTPUTS_TO relationships where the transformation is the source
        outputs_to_rels = self.ontology_manager.get_relationships_by_source(
            transformation_id, RelationshipType.OUTPUTS_TO
        )
        
        return [rel.target_id for rel in outputs_to_rels]
    
    def _get_config_inputs(self, config_id: str) -> List[str]:
        """Get input entities for a configuration.
        
        Args:
            config_id: ID of the configuration
            
        Returns:
            List of input entity IDs
        """
        # Find INPUTS_FROM relationships where the configuration is the source
        inputs_from_rels = self.ontology_manager.get_relationships_by_source(
            config_id, RelationshipType.INPUTS_FROM
        )
        
        return [rel.target_id for rel in inputs_from_rels]
    
    def _get_config_outputs(self, config_id: str) -> List[str]:
        """Get output entities for a configuration.
        
        Args:
            config_id: ID of the configuration
            
        Returns:
            List of output entity IDs
        """
        # Find OUTPUTS_TO relationships where the configuration is the source
        outputs_to_rels = self.ontology_manager.get_relationships_by_source(
            config_id, RelationshipType.OUTPUTS_TO
        )
        
        return [rel.target_id for rel in outputs_to_rels]
    
    def _get_orchestration_tasks(self, orchestration_id: str) -> List[str]:
        """Get tasks for an orchestration.
        
        Args:
            orchestration_id: ID of the orchestration
            
        Returns:
            List of task entity IDs
        """
        # Find CONTAINS relationships where the orchestration is the source
        contains_rels = self.ontology_manager.get_relationships_by_source(
            orchestration_id, RelationshipType.CONTAINS
        )
        
        task_ids = []
        
        for rel in contains_rels:
            target_entity = self.ontology_manager.get_entity(rel.target_id)
            
            if target_entity and target_entity.type == EntityType.TASK:
                task_ids.append(target_entity.id)
        
        return task_ids
    
    def _get_config_for_task(self, task_id: str) -> Optional[str]:
        """Get the configuration ID for a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Configuration ID or None if not found
        """
        # Find BELONGS_TO relationships where the task is the source
        belongs_to_rels = self.ontology_manager.get_relationships_by_source(
            task_id, RelationshipType.BELONGS_TO
        )
        
        for rel in belongs_to_rels:
            target_entity = self.ontology_manager.get_entity(rel.target_id)
            
            if target_entity and target_entity.type == EntityType.CONFIGURATION:
                return target_entity.id
        
        return None 