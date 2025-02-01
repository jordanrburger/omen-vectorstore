import pytest
from unittest.mock import Mock, patch
from qdrant_client.http.models import ScoredPoint, Filter, FieldCondition, MatchValue
from app.search import search_metadata, find_related_transformations

@pytest.fixture
def mock_qdrant_client():
    with patch('app.search.QdrantClient') as mock_client:
        client_instance = Mock()
        mock_client.return_value = client_instance
        
        # Setup search results
        client_instance.search.return_value = [
            ScoredPoint(
                id=1,
                version=1,
                score=0.95,
                payload={
                    'id': 'tr1',
                    'name': 'Test Transformation',
                    'type': 'transformation',
                    'description': 'A test transformation'
                },
                vector=None
            ),
            ScoredPoint(
                id=2,
                version=1,
                score=0.85,
                payload={
                    'id': 'tr2',
                    'name': 'Another Transformation',
                    'type': 'transformation',
                    'description': 'Another test transformation'
                },
                vector=None
            )
        ]
        
        # Setup scroll results
        client_instance.scroll.return_value = ([
            ScoredPoint(
                id=1,
                version=1,
                score=1.0,
                payload={
                    'id': 'tr1',
                    'name': 'Test Transformation',
                    'type': 'transformation'
                },
                vector=[0.1] * 1536
            )
        ], None)
        
        yield client_instance

@pytest.fixture
def mock_embedding_provider():
    provider = Mock()
    provider.embed.return_value = [0.1] * 1536
    return provider

def test_search_metadata_basic(mock_qdrant_client, mock_embedding_provider):
    """Test basic metadata search functionality."""
    results = search_metadata(
        query="test query",
        embedding_provider=mock_embedding_provider,
        limit=2
    )
    
    assert len(results) == 2
    assert results[0]["score"] == 0.95
    assert results[0]["metadata"]["name"] == "Test Transformation"
    assert results[1]["score"] == 0.85
    assert results[1]["metadata"]["name"] == "Another Transformation"
    
    mock_embedding_provider.embed.assert_called_once_with("test query")
    mock_qdrant_client.search.assert_called_once()

def test_search_metadata_empty_results(mock_qdrant_client, mock_embedding_provider):
    """Test metadata search with no results."""
    mock_qdrant_client.search.return_value = []
    
    results = search_metadata(
        query="nonexistent query",
        embedding_provider=mock_embedding_provider
    )
    
    assert len(results) == 0
    mock_embedding_provider.embed.assert_called_once_with("nonexistent query")
    mock_qdrant_client.search.assert_called_once()

def test_find_related_transformations_basic(mock_qdrant_client, mock_embedding_provider):
    """Test finding related transformations."""
    results = find_related_transformations(
        transformation_id="tr1",
        embedding_provider=mock_embedding_provider,
        limit=2
    )
    
    assert len(results) == 1  # One result because we exclude the input transformation
    assert results[0]["score"] == 0.85
    assert results[0]["metadata"]["name"] == "Another Transformation"
    
    # Verify the scroll filter was correct
    mock_qdrant_client.scroll.assert_called_once()
    call_args = mock_qdrant_client.scroll.call_args[1]
    assert isinstance(call_args["scroll_filter"], Filter)
    assert len(call_args["scroll_filter"].must) == 1
    assert isinstance(call_args["scroll_filter"].must[0], FieldCondition)
    assert call_args["scroll_filter"].must[0].key == "id"
    assert call_args["scroll_filter"].must[0].match.value == "tr1"

def test_find_related_transformations_not_found(mock_qdrant_client, mock_embedding_provider):
    """Test finding related transformations when base transformation not found."""
    mock_qdrant_client.scroll.return_value = ([], None)
    
    results = find_related_transformations(
        transformation_id="nonexistent",
        embedding_provider=mock_embedding_provider
    )
    
    assert len(results) == 0
    mock_qdrant_client.scroll.assert_called_once()
    mock_qdrant_client.search.assert_not_called()

def test_find_related_transformations_no_related(mock_qdrant_client, mock_embedding_provider):
    """Test finding related transformations when no related ones exist."""
    # Setup scroll to return the base transformation
    mock_qdrant_client.scroll.return_value = ([
        ScoredPoint(
            id=1,
            version=1,
            score=1.0,
            payload={'id': 'tr1', 'name': 'Test'},
            vector=[0.1] * 1536
        )
    ], None)
    
    # Setup search to return only the base transformation
    mock_qdrant_client.search.return_value = [
        ScoredPoint(
            id=1,
            version=1,
            score=1.0,
            payload={'id': 'tr1', 'name': 'Test'},
            vector=None
        )
    ]
    
    results = find_related_transformations(
        transformation_id="tr1",
        embedding_provider=mock_embedding_provider
    )
    
    assert len(results) == 0
    mock_qdrant_client.scroll.assert_called_once()
    mock_qdrant_client.search.assert_called_once()

def test_search_metadata_with_limit(mock_qdrant_client, mock_embedding_provider):
    """Test metadata search respects the limit parameter."""
    # Override the default search results for this test
    mock_qdrant_client.search.return_value = [
        ScoredPoint(
            id=1,
            version=1,
            score=0.95,
            payload={
                'id': 'tr1',
                'name': 'Test Transformation',
                'type': 'transformation',
                'description': 'A test transformation'
            },
            vector=None
        )
    ]
    
    results = search_metadata(
        query="test query",
        embedding_provider=mock_embedding_provider,
        limit=1
    )
    
    assert len(results) == 1
    assert results[0]["score"] == 0.95
    
    mock_qdrant_client.search.assert_called_once_with(
        collection_name="keboola_metadata",
        query_vector=[0.1] * 1536,
        limit=1
    ) 