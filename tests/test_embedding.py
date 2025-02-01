import pytest
from unittest.mock import Mock, patch

from app.embedding import get_embedding_provider
from app.vectorizer import OpenAIProvider, SentenceTransformerProvider

@pytest.fixture
def mock_config():
    return {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL": "text-embedding-3-small",
        "SENTENCE_TRANSFORMER_MODEL": "all-MiniLM-L6-v2"
    }

@patch('app.embedding.OpenAIProvider')
def test_get_embedding_provider_openai(mock_openai_provider, mock_config):
    """Test OpenAI provider creation when API key is present."""
    provider = get_embedding_provider(mock_config)
    
    mock_openai_provider.assert_called_once_with(
        api_key="test-key",
        model="text-embedding-3-small"
    )

@patch('app.embedding.OpenAIProvider')
def test_get_embedding_provider_openai_default_model(mock_openai_provider):
    """Test OpenAI provider creation with default model."""
    config = {"OPENAI_API_KEY": "test-key"}
    provider = get_embedding_provider(config)
    
    mock_openai_provider.assert_called_once_with(
        api_key="test-key",
        model="text-embedding-ada-002"  # Default model
    )

@patch('app.embedding.SentenceTransformerProvider')
def test_get_embedding_provider_sentence_transformer(mock_st_provider):
    """Test SentenceTransformer provider creation when no OpenAI key."""
    config = {
        "SENTENCE_TRANSFORMER_MODEL": "all-MiniLM-L6-v2"
    }
    provider = get_embedding_provider(config)
    
    mock_st_provider.assert_called_once_with(
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )

@patch('app.embedding.SentenceTransformerProvider')
def test_get_embedding_provider_sentence_transformer_default_model(mock_st_provider):
    """Test SentenceTransformer provider creation with default model."""
    config = {}  # Empty config
    provider = get_embedding_provider(config)
    
    mock_st_provider.assert_called_once_with(
        model_name="all-MiniLM-L6-v2",  # Default model
        device="cpu"
    )

@patch('app.embedding.logger')
@patch('app.embedding.OpenAIProvider')
def test_get_embedding_provider_logs_openai(mock_openai_provider, mock_logger, mock_config):
    """Test that provider selection is logged for OpenAI."""
    provider = get_embedding_provider(mock_config)
    
    mock_logger.info.assert_called_once_with("Using OpenAI embedding provider")

@patch('app.embedding.logger')
@patch('app.embedding.SentenceTransformerProvider')
def test_get_embedding_provider_logs_sentence_transformer(mock_st_provider, mock_logger):
    """Test that provider selection is logged for SentenceTransformer."""
    config = {}
    provider = get_embedding_provider(config)
    
    mock_logger.info.assert_called_once_with("Using SentenceTransformer embedding provider") 