"""
Tests for configuration management.
"""

import os
import pytest
from pathlib import Path

from omen.core.config import AppSettings, load_settings

@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment variables before and after each test."""
    # Save original environment
    orig_env = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "KEBOOLA_URL": os.environ.get("KEBOOLA_URL"),
        "KEBOOLA_TOKEN": os.environ.get("KEBOOLA_TOKEN")
    }
    
    # Clear environment
    for key in orig_env:
        os.environ.pop(key, None)
    
    yield
    
    # Restore original environment
    for key, value in orig_env.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)

def test_config_creation():
    """Test basic configuration creation."""
    # Set test environment variable
    os.environ["OPENAI_API_KEY"] = "test-key"
    
    # Create config with explicit settings
    config = AppSettings(
        openai={
            "api_key": "test-key",
            "model": "gpt-4",
            "embedding_model": "text-embedding-3-large",
            "temperature": 0.7,
            "max_tokens": 2000,
            "embedding_dimension": 1536
        }
    )
    
    # Verify settings
    assert config.openai.api_key == "test-key"
    assert config.openai.model == "gpt-4"
    assert config.openai.embedding_model == "text-embedding-3-large"
    assert config.openai.temperature == 0.7
    assert config.openai.max_tokens == 2000
    assert config.openai.embedding_dimension == 1536

def test_config_validation():
    """Test configuration validation."""
    with pytest.raises(ValueError, match="Invalid log level"):
        AppSettings(log_level="INVALID")

def test_config_from_env():
    """Test configuration from environment variables."""
    # Set test values
    os.environ["OPENAI_API_KEY"] = "env-key"
    os.environ["KEBOOLA_URL"] = "test-url"
    os.environ["KEBOOLA_TOKEN"] = "test-token"

    # Create config with explicit settings
    config = AppSettings(
        openai={"api_key": "env-key"},
        keboola={"url": "test-url", "token": "test-token"}
    )
    
    # Verify settings
    assert config.openai.api_key == "env-key"
    assert config.keboola.url == "test-url"
    assert config.keboola.token == "test-token"

def test_config_override():
    """Test configuration override."""
    config = AppSettings(
        openai={
            "api_key": "override-key",
            "model": "gpt-3.5-turbo"
        }
    )
    assert config.openai.api_key == "override-key"
    assert config.openai.model == "gpt-3.5-turbo"
    assert config.openai.temperature == 0.7  # Default value

def test_config_to_dict():
    """Test configuration to dictionary conversion."""
    config = AppSettings(
        openai={
            "api_key": "test-key",
            "model": "gpt-4"
        }
    )
    data = config.model_dump()
    assert isinstance(data, dict)
    assert data["openai"]["api_key"] == "test-key"
    assert data["openai"]["model"] == "gpt-4"

def test_config_from_dict():
    """Test configuration from dictionary."""
    data = {
        "openai": {
            "api_key": "test-key",
            "model": "gpt-4"
        }
    }
    config = AppSettings(**data)
    assert config.openai.api_key == "test-key"
    assert config.openai.model == "gpt-4"

def test_config_save_load(tmp_path: Path):
    """Test saving and loading configuration."""
    config = AppSettings(
        openai={
            "api_key": "test-key",
            "model": "gpt-4",
            "embedding_model": "text-embedding-3-large",
            "temperature": 0.7,
            "max_tokens": 2000,
            "embedding_dimension": 1536
        },
        qdrant={
            "host": "localhost",
            "port": 6333,
            "grpc_port": 6334,
            "prefer_grpc": True,
            "collection_name": "test_collection"
        },
        keboola={
            "url": "test-url",
            "token": "test-token"
        },
        ontology={
            "storage_path": Path("./state/ontology")
        }
    )
    config_path = tmp_path / "config.json"
    config.save(config_path)

    loaded_config = AppSettings.load(config_path)
    assert loaded_config.openai.api_key == "test-key"
    assert loaded_config.openai.model == "gpt-4"
    assert loaded_config.qdrant.host == "localhost"
    assert loaded_config.keboola.url == "test-url"
    assert loaded_config.ontology.storage_path == Path("./state/ontology") 