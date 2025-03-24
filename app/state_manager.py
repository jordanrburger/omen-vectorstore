"""
State manager for handling metadata and extraction state persistence.
"""

import logging
import os
import json
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path

from app.ontology.manager import OntologyManager
from app.ontology.action_graph import ActionGraph
from app.ontology.rdf_store import RDFStore

logger = logging.getLogger(__name__)


class StateManager:
    """Manages state and metadata persistence for the application."""

    def __init__(self, state_dir: Optional[str] = None):
        """Initialize the state manager.
        
        Args:
            state_dir: Directory for storing state files
        """
        self.state_dir = state_dir or os.path.join(os.getcwd(), "state")
        os.makedirs(self.state_dir, exist_ok=True)
        
        # Define state file paths
        self.extraction_state_file = os.path.join(self.state_dir, "extraction_state.json")
        self.metadata_file = os.path.join(self.state_dir, "metadata.json")
        self.ontology_state_file = os.path.join(self.state_dir, "ontology_state.json")
        self.action_graph_state_file = os.path.join(self.state_dir, "action_graph_state.json")
        self.rdf_state_file = os.path.join(self.state_dir, "rdf_state.ttl")
    
    def load_extraction_state(self) -> Dict[str, Any]:
        """Load the extraction state from disk.
        
        Returns:
            Dictionary containing extraction state
        """
        try:
            if os.path.exists(self.extraction_state_file):
                with open(self.extraction_state_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading extraction state: {e}")
            return {}
    
    def save_extraction_state(self, state: Dict[str, Any]) -> None:
        """Save the extraction state to disk.
        
        Args:
            state: Dictionary containing extraction state
        """
        try:
            with open(self.extraction_state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving extraction state: {e}")
    
    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load metadata from disk.
        
        Returns:
            Dictionary containing metadata or None if not found
        """
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return None
    
    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata to disk.
        
        Args:
            metadata: Dictionary containing metadata
        """
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def load_ontology_state(self) -> Optional[OntologyManager]:
        """Load ontology state from disk.
        
        Returns:
            OntologyManager instance or None if not found
        """
        try:
            if os.path.exists(self.ontology_state_file):
                with open(self.ontology_state_file, "r") as f:
                    state_data = json.load(f)
                    ontology_manager = OntologyManager()
                    ontology_manager.from_dict(state_data)
                    return ontology_manager
            return None
        except Exception as e:
            logger.error(f"Error loading ontology state: {e}")
            return None
    
    def save_ontology_state(self, ontology_manager: OntologyManager) -> None:
        """Save ontology state to disk.
        
        Args:
            ontology_manager: OntologyManager instance to save
        """
        try:
            state_data = ontology_manager.to_dict()
            with open(self.ontology_state_file, "w") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving ontology state: {e}")
    
    def load_action_graph_state(self) -> Optional[ActionGraph]:
        """Load action graph state from disk.
        
        Returns:
            ActionGraph instance or None if not found
        """
        try:
            if os.path.exists(self.action_graph_state_file):
                with open(self.action_graph_state_file, "r") as f:
                    state_data = json.load(f)
                    action_graph = ActionGraph()
                    action_graph.from_dict(state_data)
                    return action_graph
            return None
        except Exception as e:
            logger.error(f"Error loading action graph state: {e}")
            return None
    
    def save_action_graph_state(self, action_graph: ActionGraph) -> None:
        """Save action graph state to disk.
        
        Args:
            action_graph: ActionGraph instance to save
        """
        try:
            state_data = action_graph.to_dict()
            with open(self.action_graph_state_file, "w") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving action graph state: {e}")
    
    def load_rdf_state(self) -> Optional[RDFStore]:
        """Load RDF state from disk.
        
        Returns:
            RDFStore instance or None if not found
        """
        try:
            if os.path.exists(self.rdf_state_file):
                with open(self.rdf_state_file, "r") as f:
                    rdf_data = f.read()
                    rdf_store = RDFStore()
                    rdf_store.deserialize(rdf_data, format="turtle")
                    return rdf_store
            return None
        except Exception as e:
            logger.error(f"Error loading RDF state: {e}")
            return None
    
    def save_rdf_state(self, rdf_store: RDFStore) -> None:
        """Save RDF state to disk.
        
        Args:
            rdf_store: RDFStore instance to save
        """
        try:
            rdf_data = rdf_store.serialize(format="turtle")
            with open(self.rdf_state_file, "w") as f:
                f.write(rdf_data)
        except Exception as e:
            logger.error(f"Error saving RDF state: {e}")
    
    def compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute a hash for a dictionary of data.
        
        Args:
            data: Dictionary to compute hash for
            
        Returns:
            String containing the computed hash
        """
        # Convert dictionary to sorted JSON string for consistent hashing
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
