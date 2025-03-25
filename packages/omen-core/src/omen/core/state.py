"""
State management for the OMEN platform.
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omen.core.config import settings
from omen.core.logging import get_logger

logger = get_logger(__name__)


class StateManager:
    """Manages application state, including persistence to disk."""

    def __init__(self, state_file: Optional[Union[str, Path]] = None):
        """Initialize the state manager.
        
        Args:
            state_file: Path to the state file. If None, uses the default from settings.
        """
        self.state_file = Path(state_file) if state_file else settings.state_file
        self.state: Dict[str, Any] = {
            "last_updated": None,
            "indexed_items": {},
            "version": "0.1.0",
        }
        self._load_state()

    def _load_state(self) -> None:
        """Load state from disk if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    loaded_state = json.load(f)
                    self.state.update(loaded_state)
                logger.info(f"Loaded state from {self.state_file}")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        else:
            logger.info(f"No state file found at {self.state_file}, using default state")

    def save_state(self) -> None:
        """Save current state to disk."""
        # Ensure parent directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Update last_updated timestamp
        self.state["last_updated"] = time.time()
        
        try:
            # Temporary file to ensure atomic writes
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(self.state, f, indent=2)
            
            # Rename to target file (atomic on most file systems)
            temp_file.replace(self.state_file)
            logger.info(f"Saved state to {self.state_file}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the state."""
        return self.state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the state."""
        self.state[key] = value
    
    def update(self, key: str, value: Dict[str, Any]) -> None:
        """Update a dictionary in the state."""
        if key not in self.state or not isinstance(self.state[key], dict):
            self.state[key] = {}
        self.state[key].update(value)
    
    def add_to_list(self, key: str, value: Any) -> None:
        """Add a value to a list in the state."""
        if key not in self.state or not isinstance(self.state[key], list):
            self.state[key] = []
        self.state[key].append(value)
    
    def remove_from_list(self, key: str, value: Any) -> None:
        """Remove a value from a list in the state."""
        if key in self.state and isinstance(self.state[key], list):
            if value in self.state[key]:
                self.state[key].remove(value)
    
    def mark_indexed(self, item_type: str, item_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark an item as indexed."""
        if "indexed_items" not in self.state:
            self.state["indexed_items"] = {}
        
        if item_type not in self.state["indexed_items"]:
            self.state["indexed_items"][item_type] = {}
        
        self.state["indexed_items"][item_type][item_id] = {
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
    
    def is_indexed(self, item_type: str, item_id: str) -> bool:
        """Check if an item is indexed."""
        if "indexed_items" not in self.state:
            return False
        
        if item_type not in self.state["indexed_items"]:
            return False
        
        return item_id in self.state["indexed_items"][item_type]
    
    def get_indexed_items(self, item_type: Optional[str] = None) -> Dict[str, Any]:
        """Get all indexed items, optionally filtered by type."""
        if "indexed_items" not in self.state:
            return {}
        
        if item_type:
            return self.state["indexed_items"].get(item_type, {})
        
        return self.state["indexed_items"]
    
    def clear_indexed_items(self, item_type: Optional[str] = None) -> None:
        """Clear indexed items, optionally filtered by type."""
        if "indexed_items" not in self.state:
            return
        
        if item_type:
            self.state["indexed_items"][item_type] = {}
        else:
            self.state["indexed_items"] = {}


# Default state manager instance
state_manager = StateManager() 