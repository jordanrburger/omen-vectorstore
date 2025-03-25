"""Tests for LLM client."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import tenacity

from omen.core.config import AppSettings
from omen.core.llm import LLMClient


@pytest.fixture
def mock_openai(monkeypatch):
    """Mock OpenAI API."""
    mock_chat = MagicMock()
    mock_embedding = MagicMock()
    monkeypatch.setattr("openai.ChatCompletion", mock_chat)
    monkeypatch.setattr("openai.Embedding", mock_embedding)
    return mock_chat, mock_embedding


@pytest.fixture
def test_config():
    """Test configuration."""
    return AppSettings(
        openai={
            "api_key": "test-key",
            "model": "gpt-4",
            "embedding_model": "text-embedding-3-large",
            "temperature": 0.7,
            "max_tokens": 2000,
            "embedding_dimension": 1536
        }
    )


def test_llm_client_creation(test_config):
    """Test LLM client creation."""
    client = LLMClient(api_key="test-key")
    assert client.provider == "openai"
    assert client.model == "gpt-4"
    assert client.temperature == 0.7
    assert client.max_tokens == 2000
    assert client.api_key == "test-key"


def test_llm_client_openai_generation(mock_openai):
    """Test OpenAI text generation."""
    mock_chat, _ = mock_openai
    mock_chat.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test response"))]
    )

    client = LLMClient(api_key="test-key")
    response = client.generate("Test prompt")
    assert response == "Test response"

    mock_chat.create.assert_called_once()
    call_args = mock_chat.create.call_args[1]
    assert call_args["messages"][0]["content"] == "Test prompt"


def test_llm_client_anthropic_generation():
    """Test Anthropic text generation."""
    client = LLMClient(provider="anthropic", api_key="test-key")
    with pytest.raises(tenacity.RetryError):
        client.generate("Test prompt")


def test_llm_client_embedding(mock_openai):
    """Test embedding generation."""
    _, mock_embedding = mock_openai
    mock_embedding.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
    )

    client = LLMClient(api_key="test-key")
    embedding = client.generate_embedding("Test text")
    assert embedding == [0.1, 0.2, 0.3]

    mock_embedding.create.assert_called_once()
    call_args = mock_embedding.create.call_args[1]
    assert call_args["input"] == "Test text"


def test_llm_client_structured_generation(mock_openai):
    """Test structured data generation."""
    mock_chat, _ = mock_openai
    mock_chat.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"key": "value"}'))]
    )

    client = LLMClient(api_key="test-key")
    result = client.generate_structured("Test prompt", {"key": "str"})
    assert result == {"key": "value"}

    mock_chat.create.assert_called_once()
    call_args = mock_chat.create.call_args[1]
    assert "Generate a JSON object" in call_args["messages"][0]["content"]


def test_llm_client_invalid_provider():
    """Test invalid provider handling."""
    with pytest.raises(ValueError, match="Invalid provider"):
        LLMClient(provider="invalid", api_key="test-key")


def test_llm_client_missing_api_key():
    """Test missing API key handling."""
    # Clear environment variable
    os.environ.pop("OPENAI_API_KEY", None)
    with pytest.raises(ValueError, match="API key not provided"):
        LLMClient(provider="openai")


def test_llm_client_retry(mock_openai):
    """Test retry functionality."""
    mock_chat, _ = mock_openai
    mock_chat.create.side_effect = [
        Exception("API error"),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Test response"))])
    ]

    client = LLMClient(api_key="test-key")
    response = client.generate("Test prompt")
    assert response == "Test response"
    assert mock_chat.create.call_count == 2


def test_llm_client_system_prompt(mock_openai):
    """Test system prompt handling."""
    mock_chat, _ = mock_openai
    mock_chat.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test response"))]
    )

    client = LLMClient(api_key="test-key")
    response = client.generate("Test prompt", system_prompt="System prompt")
    assert response == "Test response"

    mock_chat.create.assert_called_once()
    call_args = mock_chat.create.call_args[1]
    assert len(call_args["messages"]) == 2
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][0]["content"] == "System prompt"
    assert call_args["messages"][1]["role"] == "user"
    assert call_args["messages"][1]["content"] == "Test prompt" 