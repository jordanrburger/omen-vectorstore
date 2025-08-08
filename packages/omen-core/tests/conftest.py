"""
Test configuration and fixtures for omen-core package.
"""

import os
import pytest
from pathlib import Path
from typing import Generator

from omen.core.config import AppSettings, load_settings
from omen.core.state import StateManager

@pytest.fixture
def test_env() -> Generator[None, None, None]:
    """Set up test environment variables."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    os.environ["QDRANT_HOST"] = "localhost"
    os.environ["QDRANT_PORT"] = "6333"
    os.environ["KEBOOLA_URL"] = "https://test.keboola.com"
    os.environ["KEBOOLA_TOKEN"] = "test-token"
    yield
    # Clean up
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("QDRANT_HOST", None)
    os.environ.pop("QDRANT_PORT", None)
    os.environ.pop("KEBOOLA_URL", None)
    os.environ.pop("KEBOOLA_TOKEN", None)

@pytest.fixture
def test_config(test_env: None) -> AppSettings:
    """Create a test configuration."""
    return load_settings()

@pytest.fixture
def test_state_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary state directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    yield state_dir
    # Clean up
    if state_dir.exists():
        for file in state_dir.glob("*"):
            file.unlink()
        state_dir.rmdir()

@pytest.fixture
def test_state_manager(test_state_dir: Path) -> StateManager:
    """Create a test state manager."""
    return StateManager(state_dir=test_state_dir) 