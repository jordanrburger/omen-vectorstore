import os
import pytest
from app.config import Config, load_config


@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment variables before each test."""
    original_env = dict(os.environ)
    os.environ.clear()
    yield
    os.environ.clear()
    os.environ.update(original_env)


class TestConfig:
    def test_config_default_values(self):
        """Test that Config uses correct default values."""
        config = Config(storage_api_token="test-token")
        assert config.storage_api_token == "test-token"
        assert config.storage_api_url == "https://connection.keboola.com/v2/storage"
        assert config.qdrant_host == "localhost"
        assert config.qdrant_port == 55000
        assert config.qdrant_collection == "keboola_metadata"
        assert config.openai_api_key is None
        assert config.embedding_model == "text-embedding-ada-002"
        assert config.device == "cpu"

    def test_config_from_env(self):
        """Test that Config.from_env correctly loads environment variables."""
        os.environ.update({
            "KEBOOLA_TOKEN": "test-token",
            "KEBOOLA_API_URL": "https://test.keboola.com",
            "QDRANT_HOST": "test-host",
            "QDRANT_PORT": "5555",
            "QDRANT_COLLECTION": "test-collection",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "test-model",
            "DEVICE": "cuda",
        })
        config = Config.from_env()
        assert config.storage_api_token == "test-token"
        assert config.storage_api_url == "https://test.keboola.com"
        assert config.qdrant_host == "test-host"
        assert config.qdrant_port == 5555
        assert config.qdrant_collection == "test-collection"
        assert config.openai_api_key == "test-key"
        assert config.embedding_model == "test-model"
        assert config.device == "cuda"

    def test_config_from_env_defaults(self):
        """Test that Config.from_env uses defaults when env vars are missing."""
        config = Config.from_env()
        assert config.storage_api_token == ""
        assert config.storage_api_url == "https://connection.keboola.com/v2/storage"
        assert config.qdrant_host == "localhost"
        assert config.qdrant_port == 55000
        assert config.qdrant_collection == "keboola_metadata"
        assert config.openai_api_key is None
        assert config.embedding_model == "text-embedding-ada-002"
        assert config.device == "cpu"

    def test_config_to_dict(self):
        """Test that Config.to_dict correctly converts config to dictionary."""
        config = Config(
            storage_api_token="test-token",
            storage_api_url="test-url",
            qdrant_host="test-host",
            qdrant_port=5555,
            qdrant_collection="test-collection",
            openai_api_key="test-key",
            embedding_model="test-model",
            device="cuda",
        )
        config_dict = config.to_dict()
        assert config_dict == {
            "storage_api_token": "test-token",
            "storage_api_url": "test-url",
            "qdrant_host": "test-host",
            "qdrant_port": 5555,
            "qdrant_collection": "test-collection",
            "openai_api_key": "test-key",
            "embedding_model": "test-model",
            "device": "cuda",
        }

    def test_load_config(self):
        """Test that load_config correctly loads all configuration."""
        os.environ.update({
            "KEBOOLA_TOKEN": "test-token",
            "KEBOOLA_API_URL": "https://test.keboola.com",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "test-model",
            "SENTENCE_TRANSFORMER_MODEL": "test-transformer",
            "QDRANT_HOST": "test-host",
            "QDRANT_PORT": "5555",
            "QDRANT_COLLECTION": "test-collection",
        })
        config = load_config()
        assert config["KEBOOLA_TOKEN"] == "test-token"
        assert config["KEBOOLA_API_URL"] == "https://test.keboola.com"
        assert config["OPENAI_API_KEY"] == "test-key"
        assert config["OPENAI_MODEL"] == "test-model"
        assert config["SENTENCE_TRANSFORMER_MODEL"] == "test-transformer"
        assert config["QDRANT_HOST"] == "test-host"
        assert config["QDRANT_PORT"] == 5555
        assert config["QDRANT_COLLECTION"] == "test-collection"

    def test_load_config_missing_required(self):
        """Test that load_config raises error when required config is missing."""
        with pytest.raises(ValueError, match="KEBOOLA_TOKEN environment variable is required"):
            load_config()

    def test_load_config_defaults(self):
        """Test that load_config uses defaults when optional config is missing."""
        os.environ["KEBOOLA_TOKEN"] = "test-token"
        config = load_config()
        assert config["KEBOOLA_TOKEN"] == "test-token"
        assert config["KEBOOLA_API_URL"] == "https://connection.keboola.com"
        assert config["OPENAI_API_KEY"] is None
        assert config["OPENAI_MODEL"] == "text-embedding-ada-002"
        assert config["SENTENCE_TRANSFORMER_MODEL"] == "all-MiniLM-L6-v2"
        assert config["QDRANT_HOST"] == "localhost"
        assert config["QDRANT_PORT"] == 55000
        assert config["QDRANT_COLLECTION"] == "keboola_metadata" 