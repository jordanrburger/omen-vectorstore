import os
import json
import pytest
from unittest.mock import mock_open, patch
from app.state_manager import StateManager

@pytest.fixture
def state_manager():
    """Create a state manager instance for testing."""
    manager = StateManager(state_file=".test_state")
    yield manager
    # Cleanup
    if os.path.exists(".test_state"):
        os.remove(".test_state")

def test_init_creates_empty_state():
    """Test that initializing StateManager creates empty state when file doesn't exist."""
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False
        manager = StateManager()
        assert manager.state == {}

def test_load_state_from_file():
    """Test loading state from an existing file."""
    test_state = {"test": "data"}
    mock_file = mock_open(read_data=json.dumps(test_state))
    
    with patch("os.path.exists") as mock_exists, \
         patch("builtins.open", mock_file):
        mock_exists.return_value = True
        manager = StateManager()
        assert manager.state == test_state

def test_load_state_handles_invalid_json():
    """Test handling of invalid JSON in state file."""
    with patch("os.path.exists") as mock_exists, \
         patch("builtins.open", mock_open(read_data="invalid json")):
        mock_exists.return_value = True
        manager = StateManager()
        assert manager.state == {}

def test_save_state():
    """Test saving state to file."""
    manager = StateManager()
    test_state = {"test": "data"}
    manager.state = test_state
    
    mock_file = mock_open()
    with patch("builtins.open", mock_file):
        manager._save_state()
    
    mock_file.assert_called_once_with(manager.state_file, "w")
    handle = mock_file()
    # Combine all write calls to get the full written string
    written_data = "".join(call.args[0] for call in handle.write.call_args_list)
    # Verify the written data can be parsed back to the original state
    assert json.loads(written_data) == test_state

def test_load_extraction_state():
    """Test loading extraction state."""
    manager = StateManager()
    test_state = {"extraction_state": {"test": "data"}}
    manager.state = test_state
    assert manager.load_extraction_state() == {"test": "data"}

def test_save_extraction_state():
    """Test saving extraction state."""
    manager = StateManager()
    test_state = {"test": "data"}
    
    with patch.object(manager, "_save_state") as mock_save:
        manager.save_extraction_state(test_state)
        assert manager.state["extraction_state"] == test_state
        mock_save.assert_called_once()

def test_load_metadata():
    """Test loading metadata."""
    manager = StateManager()
    test_metadata = {"test": "metadata"}
    manager.state = {"metadata": test_metadata}
    assert manager.load_metadata() == test_metadata

def test_save_metadata():
    """Test saving metadata."""
    manager = StateManager()
    test_metadata = {"test": "metadata"}
    
    with patch.object(manager, "_save_state") as mock_save:
        manager.save_metadata(test_metadata)
        assert manager.state["metadata"] == test_metadata
        mock_save.assert_called_once()

def test_get_metadata_hash():
    """Test getting metadata hash."""
    manager = StateManager()
    manager.state = {
        "test_type": {
            "test_id": "test_hash"
        }
    }
    assert manager.get_metadata_hash("test_type", "test_id") == "test_hash"
    assert manager.get_metadata_hash("nonexistent", "id") is None

def test_set_metadata_hash():
    """Test setting metadata hash."""
    manager = StateManager()
    
    with patch.object(manager, "_save_state") as mock_save:
        manager.set_metadata_hash("test_type", "test_id", "test_hash")
        assert manager.state["test_type"]["test_id"] == "test_hash"
        mock_save.assert_called_once()

def test_compute_hash():
    """Test computing hash from dictionary."""
    manager = StateManager()
    test_data = {"test": "data"}
    hash1 = manager.compute_hash(test_data)
    
    # Test hash is stable
    hash2 = manager.compute_hash(test_data)
    assert hash1 == hash2
    
    # Test hash changes with different data
    different_data = {"different": "data"}
    different_hash = manager.compute_hash(different_data)
    assert hash1 != different_hash

def test_has_changed():
    """Test checking if metadata has changed."""
    manager = StateManager()
    test_data = {"test": "data"}
    test_hash = manager.compute_hash(test_data)
    
    # Test when no previous hash exists
    manager.state = {}  # Ensure empty state
    assert manager.has_changed("test_type", "test_id", test_data) is True
    
    # Test when hash matches
    manager.state = {"test_type": {"test_id": test_hash}}
    assert manager.has_changed("test_type", "test_id", test_data) is False
    
    # Test when hash differs
    different_data = {"different": "data"}
    assert manager.has_changed("test_type", "test_id", different_data) is True

def test_update_metadata_hash():
    """Test updating metadata hash."""
    manager = StateManager()
    test_data = {"test": "data"}
    
    with patch.object(manager, "set_metadata_hash") as mock_set:
        manager.update_metadata_hash("test_type", "test_id", test_data)
        mock_set.assert_called_once_with(
            "test_type",
            "test_id",
            manager.compute_hash(test_data)
        ) 