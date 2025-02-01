import pytest
from unittest.mock import Mock, patch
import argparse
from typing import Dict, List

from app.main import (
    get_embedding_provider,
    extract_metadata,
    search_metadata,
    index_command,
    search_command,
    main
)
from app.config import Config
from app.vectorizer import OpenAIProvider, SentenceTransformerProvider
from app.keboola_client import KeboolaClient

@pytest.fixture
def mock_config():
    config = Mock(spec=Config)
    config.keboola_api_url = "https://connection.keboola.com"
    config.keboola_token = "test-token"
    config.openai_api_key = "test-key"
    config.embedding_model = "text-embedding-3-small"
    config.qdrant_host = "localhost"
    config.qdrant_port = 6333
    config.qdrant_collection = "test-collection"
    config.device = "cpu"
    return config

@pytest.fixture
def mock_keboola_client():
    client = Mock()
    client.list_buckets.return_value = [
        {"id": "in.c-test", "name": "Test Bucket"}
    ]
    client.list_tables.return_value = {
        "in.c-test": [
            {"id": "in.c-test.customers", "name": "customers"}
        ]
    }
    client.get_table_details.return_value = {
        "id": "in.c-test.customers",
        "name": "customers",
        "columns": ["id", "name", "email"]
    }
    client.list_configurations.return_value = [
        {
            "id": "123",
            "name": "Test Config",
            "component": "keboola.python-transformation"
        }
    ]
    return client

@pytest.fixture
def mock_indexer():
    indexer = Mock()
    indexer.search_metadata.return_value = [
        {
            'score': 0.95,
            'metadata_type': 'transformation',
            'metadata': {
                'id': '123',
                'name': 'Test Transformation',
                'description': 'A test transformation',
                'type': 'transformation',
                'component': 'keboola.python-transformation'
            }
        }
    ]
    return indexer

@patch('app.main.OpenAIProvider')
def test_get_embedding_provider_openai(mock_openai_provider):
    """Test OpenAI provider creation when API key is present."""
    config = Mock(
        openai_api_key="test-key",
        embedding_model="text-embedding-3-small"
    )
    
    provider = get_embedding_provider(config)
    
    mock_openai_provider.assert_called_once_with(
        api_key="test-key",
        model="text-embedding-3-small"
    )

@patch('app.main.SentenceTransformerProvider')
def test_get_embedding_provider_sentence_transformer(mock_st_provider):
    """Test SentenceTransformer provider creation when no OpenAI key."""
    config = Mock(
        openai_api_key=None,
        embedding_model="all-MiniLM-L6-v2",
        device="cpu"
    )
    
    provider = get_embedding_provider(config)
    
    mock_st_provider.assert_called_once_with(
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )

def test_extract_metadata(mock_keboola_client):
    """Test metadata extraction from Keboola."""
    metadata = extract_metadata(mock_keboola_client)
    
    assert "buckets" in metadata
    assert "tables" in metadata
    assert "table_details" in metadata
    assert "configurations" in metadata
    
    assert len(metadata["buckets"]) == 1
    assert len(metadata["tables"]["in.c-test"]) == 1
    assert len(metadata["configurations"]) == 1
    
    mock_keboola_client.list_buckets.assert_called_once()
    mock_keboola_client.list_tables.assert_called_once()
    mock_keboola_client.list_configurations.assert_called_once()

def test_extract_metadata_handles_errors(mock_keboola_client):
    """Test metadata extraction handles errors gracefully."""
    mock_keboola_client.get_table_details.side_effect = Exception("API Error")
    
    metadata = extract_metadata(mock_keboola_client)
    assert metadata is not None
    assert "table_details" in metadata
    assert len(metadata["table_details"]) == 0

def test_search_metadata(mock_indexer):
    """Test metadata search functionality."""
    embedding_provider = Mock()
    embedding_provider.get_embedding.return_value = [0.1] * 1536
    
    results = search_metadata(
        query="test query",
        indexer=mock_indexer,
        embedding_provider=embedding_provider,
        metadata_type="transformations",
        limit=3
    )
    
    assert len(results) == 1
    assert results[0]["score"] == 0.95
    assert results[0]["metadata"]["name"] == "Test Transformation"
    
    mock_indexer.search_metadata.assert_called_once()

@patch('app.main.QdrantIndexer')
@patch('app.main.KeboolaClient')
@patch('app.main.StateManager')
def test_index_command(mock_state_manager, mock_client, mock_indexer, mock_config):
    """Test the index command."""
    mock_client_instance = Mock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.list_buckets.return_value = []
    mock_client_instance.list_tables.return_value = {}
    mock_client_instance.list_configurations.return_value = []
    
    mock_indexer_instance = Mock()
    mock_indexer.return_value = mock_indexer_instance
    
    index_command(mock_config)
    
    mock_client.assert_called_once()
    mock_indexer.assert_called_once()
    mock_indexer_instance.index_metadata.assert_called_once()

@patch('app.main.QdrantIndexer')
@patch('app.main.get_embedding_provider')
def test_search_command(mock_get_provider, mock_indexer, mock_config, capsys):
    """Test the search command with result printing."""
    mock_provider = Mock()
    mock_get_provider.return_value = mock_provider
    
    mock_indexer_instance = Mock()
    mock_indexer.return_value = mock_indexer_instance
    mock_indexer_instance.search_metadata.return_value = [
        {
            'score': 0.95,
            'metadata_type': 'transformation',
            'metadata': {
                'id': '123',
                'name': 'Test Transformation',
                'description': 'A test transformation',
                'type': 'transformation',
                'component': 'keboola.python-transformation'
            }
        }
    ]
    
    search_command(mock_config, "test query", "transformations", 3)
    
    captured = capsys.readouterr()
    assert "Result 1:" in captured.out
    assert "Score: 0.950" in captured.out
    assert "Name: Test Transformation" in captured.out
    assert "Type: transformation" in captured.out

@patch('argparse.ArgumentParser.parse_args')
@patch('app.main.Config')
def test_main_index_command(mock_config, mock_args):
    """Test main function with index command."""
    mock_args.return_value = Mock(command="index")
    mock_config.from_env.return_value = Mock()
    
    with patch('app.main.index_command') as mock_index:
        main()
        mock_index.assert_called_once()

@patch('argparse.ArgumentParser.parse_args')
@patch('app.main.Config')
def test_main_search_command(mock_config, mock_args):
    """Test main function with search command."""
    mock_args.return_value = Mock(
        command="search",
        query="test query",
        type=None,
        limit=10
    )
    mock_config.from_env.return_value = Mock()
    
    with patch('app.main.search_command') as mock_search:
        main()
        mock_search.assert_called_once_with(
            mock_config.from_env.return_value,
            "test query",
            None,
            10
        )

@patch('argparse.ArgumentParser.parse_args')
@patch('app.main.Config')
def test_main_no_command(mock_config, mock_args, capsys):
    """Test main function with no command."""
    mock_args.return_value = Mock(command=None)
    mock_config.from_env.return_value = Mock()
    
    main()
    captured = capsys.readouterr()
    assert "usage:" in captured.out 

@patch('app.main.KeboolaClient')
def test_client_initialization(mock_client, mock_config):
    """Test client initialization with correct URL."""
    client = KeboolaClient(
        api_url=mock_config.keboola_api_url,
        token=mock_config.keboola_token
    )
    assert client.url == "https://connection.keboola.com/v2/storage"
    assert client.token == "test-token" 