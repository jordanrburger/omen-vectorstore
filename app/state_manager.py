import hashlib
import json
import logging
import os
from typing import Dict, Optional


class StateManager:
    """Manages state persistence for the application."""

    def __init__(self, state_file: str = ".omen_state"):
        """Initialize the state manager with a state file path."""
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load state from file or return empty state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Error loading state from {self.state_file}: {e}")
                return {}
        return {}

    def _save_state(self) -> None:
        """Save current state to file."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving state to {self.state_file}: {e}")

    def load_extraction_state(self) -> Dict:
        """Load the extraction state."""
        return self.state.get("extraction_state", {})

    def save_extraction_state(self, state: Dict) -> None:
        """Save the extraction state."""
        self.state["extraction_state"] = state
        self._save_state()

    def load_metadata(self) -> Optional[Dict]:
        """Load the previously extracted metadata."""
        return self.state.get("metadata")

    def save_metadata(self, metadata: Dict) -> None:
        """Save the extracted metadata."""
        self.state["metadata"] = metadata
        self._save_state()

    def get_metadata_hash(
        self,
        metadata_type: str,
        metadata_id: str,
    ) -> Optional[str]:
        """Get stored hash for a metadata item."""
        return self.state.get(metadata_type, {}).get(metadata_id)

    def set_metadata_hash(
        self,
        metadata_type: str,
        metadata_id: str,
        hash_value: str,
    ) -> None:
        """Store hash for a metadata item."""
        if metadata_type not in self.state:
            self.state[metadata_type] = {}
        self.state[metadata_type][metadata_id] = hash_value
        self._save_state()

    def compute_hash(self, data: Dict) -> str:
        """Compute a stable hash for a dictionary."""
        # Sort the dictionary to ensure stable hashing
        sorted_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(sorted_str.encode()).hexdigest()

    def has_changed(
        self,
        metadata_type: str,
        metadata_id: str,
        current_data: Dict,
    ) -> bool:
        """Check if metadata has changed since last indexing."""
        stored_hash = self.get_metadata_hash(metadata_type, metadata_id)
        if stored_hash is None:
            return True
            
        current_hash = self.compute_hash(current_data)
        return stored_hash != current_hash

    def update_metadata_hash(
        self,
        metadata_type: str,
        metadata_id: str,
        data: Dict,
    ) -> None:
        """Update stored hash for metadata after indexing."""
        hash_value = self.compute_hash(data)
        self.set_metadata_hash(metadata_type, metadata_id, hash_value)
