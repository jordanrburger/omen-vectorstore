import pytest
from unittest.mock import patch
import os
from app.config import Config, load_config


@pytest.fixture(autouse=True)
def mock_dotenv():
    """Mock load_dotenv to prevent loading real environment variables."""
    with patch('app.config.load_dotenv') as mock:
        yield mock


class TestConfig:
    """Test configuration handling."""

    def test_config_default_values(self):
        """Test that Config has correct default values."""
        config = Config()
        assert config.keboola_token == ""
        assert config.keboola_api_url == "https://connection.keboola.com"
        assert config.qdrant_host == "localhost"
        assert config.qdrant_port == 6333
        assert config.qdrant_collection == "keboola_metadata"
        assert config.openai_api_key is None
        assert config.embedding_model == "text-embedding-ada-002"
        assert config.device == "cpu"

    def test_config_from_env(self, monkeypatch):
        """Test that Config.from_env loads values from environment variables."""
        monkeypatch.setenv("KEBOOLA_TOKEN", "test-token")
        monkeypatch.setenv("KEBOOLA_API_URL", "https://test.keboola.com")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        config = Config.from_env()
        assert config.keboola_token == "test-token"
        assert config.keboola_api_url == "https://test.keboola.com"
        assert config.openai_api_key == "test-key"

    def test_config_from_env_defaults(self, monkeypatch):
        """Test that Config.from_env uses defaults when env vars are missing."""
        monkeypatch.delenv("KEBOOLA_TOKEN", raising=False)
        monkeypatch.delenv("KEBOOLA_API_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        config = Config.from_env()
        assert config.keboola_token == ""
        assert config.keboola_api_url == "https://connection.keboola.com"
        assert config.openai_api_key is None

    def test_config_to_dict(self):
        """Test that Config.to_dict returns correct dictionary."""
        config = Config(
            keboola_token="test-token",
            keboola_api_url="https://test.keboola.com",
            openai_api_key="test-key"
        )
        config_dict = config.to_dict()
        assert config_dict["keboola_token"] == "test-token"
        assert config_dict["keboola_api_url"] == "https://test.keboola.com"
        assert config_dict["openai_api_key"] == "test-key"

    def test_load_config(self, monkeypatch):
        """Test that load_config loads values from environment variables."""
        monkeypatch.setenv("KEBOOLA_TOKEN", "test-token")
        monkeypatch.setenv("KEBOOLA_API_URL", "https://test.keboola.com")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        config = load_config()
        assert config["KEBOOLA_TOKEN"] == "test-token"
        assert config["KEBOOLA_API_URL"] == "https://test.keboola.com"
        assert config["OPENAI_API_KEY"] == "test-key"

    def test_load_config_missing_required(self, monkeypatch):
        """Test that load_config raises error when required config is missing."""
        monkeypatch.delenv("KEBOOLA_TOKEN", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="KEBOOLA_TOKEN environment variable is required"):
            load_config()

    def test_load_config_defaults(self, monkeypatch):
        """Test that load_config uses defaults when optional config is missing."""
        monkeypatch.setenv("KEBOOLA_TOKEN", "test-token")
        monkeypatch.delenv("KEBOOLA_API_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        config = load_config()
        assert config["KEBOOLA_TOKEN"] == "test-token"
        assert config["KEBOOLA_API_URL"] == "https://connection.keboola.com"
        assert config["OPENAI_API_KEY"] is None 