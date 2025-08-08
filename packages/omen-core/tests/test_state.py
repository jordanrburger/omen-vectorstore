"""
Tests for state management.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any

import pytest

from omen.core.state import StateManager

@pytest.fixture
def test_state_file(tmp_path: Path) -> Path:
    """Create a test state file."""
    return tmp_path / "state.json"

@pytest.fixture
def test_state_manager(test_state_file: Path) -> StateManager:
    """Create a test state manager."""
    return StateManager(state_file=test_state_file)

def test_state_manager_creation(test_state_file: Path):
    """Test state manager creation."""
    state_manager = StateManager(state_file=test_state_file)
    assert state_manager.state_file == test_state_file
    assert isinstance(state_manager.state, dict)
    assert isinstance(state_manager.indexed_items, dict)

def test_state_save_load(test_state_manager: StateManager):
    """Test state save and load."""
    # Set some test data
    test_state_manager.set("test_key", "test_value")
    test_state_manager.mark_indexed("test_type", "test_id", {"metadata": "test"})
    
    # Save state
    test_state_manager.save_state()
    
    # Create new manager and load state
    new_manager = StateManager(state_file=test_state_manager.state_file)
    assert new_manager.get("test_key") == "test_value"
    assert new_manager.is_indexed("test_type", "test_id")
    assert new_manager.get_indexed_items("test_type")["test_id"] == {"metadata": "test"}

def test_state_update(test_state_manager: StateManager):
    """Test state update."""
    test_state_manager.set("test_key", {"key1": "value1"})
    test_state_manager.update("test_key", {"key2": "value2"})
    assert test_state_manager.get("test_key") == {"key1": "value1", "key2": "value2"}

def test_state_clear(test_state_manager: StateManager):
    """Test state clear."""
    test_state_manager.set("test_key", "test_value")
    test_state_manager.mark_indexed("test_type", "test_id")
    test_state_manager.clear()
    assert test_state_manager.get("test_key") is None
    assert not test_state_manager.is_indexed("test_type", "test_id")

def test_state_backup(test_state_manager: StateManager):
    """Test state backup."""
    test_state_manager.set("test_key", "test_value")
    test_state_manager.mark_indexed("test_type", "test_id")
    
    backup_file = test_state_manager.state_file.with_suffix(".json.bak")
    test_state_manager.save_state()
    assert backup_file.exists()
    
    with open(backup_file, "r") as f:
        backup_data = json.load(f)
        assert backup_data["state"]["test_key"] == "test_value"
        assert "test_id" in backup_data["indexed_items"]["test_type"]

def test_state_restore(test_state_manager: StateManager):
    """Test state restore."""
    # Set initial state
    test_state_manager.set("initial_key", "initial_value")
    test_state_manager.save_state()
    
    # Create backup
    backup_file = test_state_manager.state_file.with_suffix(".json.bak")
    test_state_manager.save_state()
    
    # Modify state
    test_state_manager.set("initial_key", "modified_value")
    
    # Restore from backup
    shutil.copy2(backup_file, test_state_manager.state_file)
    test_state_manager._load_state()
    assert test_state_manager.get("initial_key") == "initial_value"

def test_state_invalid_json(test_state_file: Path):
    """Test handling of invalid JSON."""
    with open(test_state_file, "w") as f:
        f.write("invalid json")
    
    state_manager = StateManager(state_file=test_state_file)
    assert isinstance(state_manager.state, dict)
    assert isinstance(state_manager.indexed_items, dict)

def test_state_nested_update(test_state_manager: StateManager):
    """Test nested state update."""
    test_state_manager.set("nested_dict", {"level1": {"level2": "value"}})
    test_state_manager.update("nested_dict", {"level1": {"level3": "new_value"}})
    assert test_state_manager.get("nested_dict") == {
        "level1": {
            "level2": "value",
            "level3": "new_value"
        }
    } 