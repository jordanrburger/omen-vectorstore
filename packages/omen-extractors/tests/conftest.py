"""
Shared test fixtures and configuration for the omen-extractors package.
"""

import os
import pytest
from pathlib import Path

@pytest.fixture(autouse=True)
def mock_state_dir(tmp_path):
    """Mock the state directory for all tests."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OMEN_STATE_DIR", str(state_dir))
        yield state_dir

@pytest.fixture
def sample_state_file(mock_state_dir):
    """Create a sample state file for testing."""
    state_file = mock_state_dir / "state.json"
    state_file.write_text('{"last_run": "2024-03-26T00:00:00Z", "processed_tables": ["table1"], "processed_buckets": ["bucket1"]}')
    return state_file

@pytest.fixture
def mock_requests():
    """Mock the requests module for all tests."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("requests.get", lambda *args, **kwargs: None)
        yield 