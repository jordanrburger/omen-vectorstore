"""
State management for the OMEN platform.

This module provides state management functionality, including saving and loading
state from files, and managing indexed items.
"""
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from omen.core.config import settings
from omen.core.logging import get_logger

logger = logging.getLogger(__name__)


class StateManager:
    """Manager for application state."""

    def __init__(self, state_file: Optional[Path] = None):
        """Initialize the state manager.
        
        Args:
            state_file: Path to state file
        """
        self.state_file = state_file or Path("state/state.json")
        self.state: Dict[str, Any] = {}
        self.indexed_items: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    self.state = data.get("state", {})
                    self.indexed_items = data.get("indexed_items", {})
                logger.info(f"Loaded state from {self.state_file}")
            except json.JSONDecodeError as e:
                logger.error(f"Error loading state: {e}")
                self.state = {}
                self.indexed_items = {}
        else:
            logger.info(f"No state file found at {self.state_file}, using default state")
            self.state = {}
            self.indexed_items = {}

    def save_state(self) -> None:
        """Save state to file."""
        # Create backup if file exists
        if self.state_file.exists():
            backup_file = self.state_file.with_suffix(".json.bak")
            shutil.copy2(self.state_file, backup_file)
            logger.info(f"Created backup at {backup_file}")

        # Save new state
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(
                {
                    "state": self.state,
                    "indexed_items": self.indexed_items
                },
                f,
                indent=2
            )
        logger.info(f"Saved state to {self.state_file}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from state.
        
        Args:
            key: Key to get
            default: Default value if key not found
            
        Returns:
            Value from state
        """
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in state.
        
        Args:
            key: Key to set
            value: Value to set
        """
        self.state[key] = value
        self.save_state()

    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively update a dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._deep_update(d[k], v)
            else:
                d[k] = v
        return d

    def update(self, key: str, value: Any) -> None:
        """Update a value in state.
        
        Args:
            key: Key to update
            value: Value to update with
        """
        if key in self.state:
            if isinstance(self.state[key], dict) and isinstance(value, dict):
                self.state[key] = self._deep_update(self.state[key].copy(), value)
            else:
                self.state[key] = value
        else:
            self.state[key] = value
        self.save_state()

    def delete(self, key: str) -> None:
        """Delete a key from state.
        
        Args:
            key: Key to delete
        """
        if key in self.state:
            del self.state[key]
            self.save_state()

    def clear(self) -> None:
        """Clear all state."""
        self.state = {}
        self.indexed_items = {}
        self.save_state()

    def mark_indexed(self, item_type: str, item_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark an item as indexed.
        
        Args:
            item_type: Type of item
            item_id: ID of item
            metadata: Optional metadata about the item
        """
        if item_type not in self.indexed_items:
            self.indexed_items[item_type] = {}
        self.indexed_items[item_type][item_id] = metadata or {}
        self.save_state()

    def is_indexed(self, item_type: str, item_id: str) -> bool:
        """Check if an item is indexed.
        
        Args:
            item_type: Type of item
            item_id: ID of item
            
        Returns:
            True if item is indexed
        """
        return (
            item_type in self.indexed_items
            and item_id in self.indexed_items[item_type]
        )

    def get_indexed_items(self, item_type: str) -> Dict[str, Dict[str, Any]]:
        """Get all indexed items of a type.
        
        Args:
            item_type: Type of item
            
        Returns:
            Dictionary of indexed items
        """
        return self.indexed_items.get(item_type, {})

    def remove_indexed(self, item_type: str, item_id: str) -> None:
        """Remove an indexed item.
        
        Args:
            item_type: Type of item
            item_id: ID of item
        """
        if item_type in self.indexed_items and item_id in self.indexed_items[item_type]:
            del self.indexed_items[item_type][item_id]
            self.save_state()

    def get_all_indexed_types(self) -> Set[str]:
        """Get all indexed item types.
        
        Returns:
            Set of indexed item types
        """
        return set(self.indexed_items.keys())


# Default state manager instance
state_manager = StateManager() 