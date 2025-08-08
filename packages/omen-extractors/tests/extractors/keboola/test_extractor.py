"""
Tests for the Keboola Storage API metadata extractor.
"""

import os
import json
from datetime import datetime
from unittest.mock import Mock, patch, mock_open
import pytest
import uuid

from omen.extractors.keboola.extractor import KeboolaExtractor
from omen.vectorstore.models import MetadataType

# Sample test data
SAMPLE_BUCKET = {
    "id": "in.c-test",
    "name": "test-bucket",
    "stage": "in",
    "description": "Test bucket",
    "created": "2024-03-26T00:00:00Z",
    "lastChangeDate": "2024-03-26T00:00:00Z",
    "uri": "keboola://test-bucket",
}

SAMPLE_BUCKET_DETAIL = {
    "id": "in.c-test",
    "name": "test-bucket",
    "stage": "in",
    "description": "Test bucket",
    "created": "2024-03-26T00:00:00Z",
    "lastChangeDate": "2024-03-26T00:00:00Z",
    "uri": "keboola://test-bucket",
    "attributes": {"key": "value"},
    "backend": "snowflake",
    "sharing": {"type": "private"},
}

SAMPLE_TABLE = {
    "id": "in.c-test.test-table",
    "name": "test-table",
    "bucket": {
        "id": "in.c-test",
        "name": "test-bucket",
        "stage": "in",
        "uri": "keboola://test-bucket",
    },
    "created": "2024-03-26T00:00:00Z",
    "lastImportDate": "2024-03-26T00:00:00Z",
    "lastChangeDate": "2024-03-26T00:00:00Z",
    "uri": "keboola://test-bucket/test-table",
}

SAMPLE_TABLE_DETAIL = {
    "id": "in.c-test.test-table",
    "name": "test-table",
    "bucket": {
        "id": "in.c-test",
        "name": "test-bucket",
        "stage": "in",
        "uri": "keboola://test-bucket",
    },
    "created": "2024-03-26T00:00:00Z",
    "lastImportDate": "2024-03-26T00:00:00Z",
    "lastChangeDate": "2024-03-26T00:00:00Z",
    "uri": "keboola://test-bucket/test-table",
    "primaryKey": ["id"],
    "rowsCount": 1000,
    "dataSizeBytes": 1024,
    "columnMetadata": {
        "id": {
            "type": "VARCHAR",
            "nullable": False,
            "length": 255,
            "default": None,
            "format": None,
            "description": "Primary key",
            "basetype": "string",
        },
        "name": {
            "type": "VARCHAR",
            "nullable": True,
            "length": 255,
            "default": None,
            "format": None,
            "description": "Name field",
            "basetype": "string",
        },
    },
    "attributes": {"key": "value"},
    "definition": {
        "backend": "snowflake",
        "type": "table",
    },
    "dataStorage": {
        "backend": "snowflake",
    },
}

@pytest.fixture
def mock_response():
    """Create a mock response object."""
    mock = Mock()
    mock.raise_for_status = Mock()
    return mock

@pytest.fixture
def extractor():
    """Create a KeboolaExtractor instance with test token."""
    return KeboolaExtractor(token="test-token", url="https://test.keboola.com")

def test_init():
    """Test extractor initialization."""
    extractor = KeboolaExtractor(token="test-token")
    assert extractor.token == "test-token"
    assert extractor.url == "https://connection.keboola.com"
    assert extractor.headers == {"X-StorageApi-Token": "test-token"}

def test_init_with_custom_url():
    """Test extractor initialization with custom URL."""
    extractor = KeboolaExtractor(token="test-token", url="https://test.keboola.com")
    assert extractor.url == "https://test.keboola.com"

def test_init_without_kbcstorage():
    """Test extractor initialization without kbcstorage package."""
    with patch("omen.extractors.keboola.extractor.Client", None):
        with pytest.raises(ImportError):
            KeboolaExtractor(token="test-token")

def test_load_state_no_file(extractor):
    """Test loading state when file doesn't exist."""
    with patch("os.path.exists", return_value=False):
        state = extractor._load_state()
        assert state == {
            "last_run": None,
            "processed_tables": set(),
            "processed_buckets": set(),
        }

def test_load_state_with_file(extractor):
    """Test loading state from existing file."""
    mock_state = {
        "last_run": "2024-03-26T00:00:00Z",
        "processed_tables": ["table1", "table2"],
        "processed_buckets": ["bucket1", "bucket2"],
    }
    mock_json = json.dumps(mock_state)
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=mock_json)):
        state = extractor._load_state()
        assert state["last_run"] == mock_state["last_run"]
        assert state["processed_tables"] == set(mock_state["processed_tables"])
        assert state["processed_buckets"] == set(mock_state["processed_buckets"])

def test_save_state(extractor):
    """Test saving state to file."""
    state = {
        "last_run": "2024-03-26T00:00:00Z",
        "processed_tables": {"table1", "table2"},
        "processed_buckets": {"bucket1", "bucket2"},
    }
    mock_file = mock_open()
    with patch("builtins.open", mock_file):
        extractor._save_state(state)
        mock_file.assert_called_once()
        
        # Combine all write calls into a single string
        write_calls = mock_file().write.call_args_list
        written_json = "".join(call[0][0] for call in write_calls)
        
        # Parse and verify the saved data
        saved_data = json.loads(written_json)
        assert saved_data["last_run"] == state["last_run"]
        assert set(saved_data["processed_tables"]) == state["processed_tables"]
        assert set(saved_data["processed_buckets"]) == state["processed_buckets"]

def test_extract_column_metadata_dict(extractor):
    """Test extracting column metadata from dictionary format."""
    column_metadata = {
        "id": {
            "type": "VARCHAR",
            "nullable": False,
            "description": "Primary key",
        },
    }
    columns = extractor._extract_column_metadata(column_metadata)
    assert len(columns) == 1
    assert columns[0]["name"] == "id"
    assert columns[0]["type"] == "VARCHAR"
    assert columns[0]["nullable"] is False
    assert columns[0]["description"] == "Primary key"

def test_extract_column_metadata_list(extractor):
    """Test extracting column metadata from list format."""
    column_metadata = [
        {
            "name": "id",
            "type": "VARCHAR",
            "nullable": False,
            "description": "Primary key",
        },
    ]
    columns = extractor._extract_column_metadata(column_metadata)
    assert len(columns) == 1
    assert columns[0]["name"] == "id"
    assert columns[0]["type"] == "VARCHAR"
    assert columns[0]["nullable"] is False
    assert columns[0]["description"] == "Primary key"

def test_extract_buckets(extractor, mock_response):
    """Test extracting bucket metadata."""
    mock_response.json.side_effect = [
        [SAMPLE_BUCKET],  # List of buckets
        SAMPLE_BUCKET_DETAIL,  # Bucket detail
    ]
    with patch("requests.get", return_value=mock_response):
        buckets = extractor.extract_buckets(set())
        assert len(buckets) == 1
        bucket = buckets[0]
        assert bucket.source.type == MetadataType.BUCKET
        assert bucket.source.id == SAMPLE_BUCKET["id"]
        assert bucket.source.url == SAMPLE_BUCKET["uri"]
        assert "name" in bucket.content
        assert "stage" in bucket.content
        assert "description" in bucket.content

def test_extract_tables(extractor, mock_response):
    """Test extracting table metadata."""
    mock_response.json.side_effect = [
        [SAMPLE_TABLE],  # List of tables
        SAMPLE_TABLE_DETAIL,  # Table detail
    ]
    with patch("requests.get", return_value=mock_response):
        tables = extractor.extract_tables(set())
        assert len(tables) == 1
        table = tables[0]
        assert table.source.type == MetadataType.TABLE
        assert table.source.id == SAMPLE_TABLE["id"]
        assert table.source.url == SAMPLE_TABLE["uri"]
        assert "name" in table.content
        assert "bucket" in table.content
        assert "columns" in table.content
        assert "rowsCount" in table.content

def test_extract_tables_list_response(extractor, mock_response):
    """Test extracting table metadata when detail response is a list."""
    mock_response.json.side_effect = [
        [SAMPLE_TABLE],  # List of tables
        [SAMPLE_TABLE_DETAIL],  # Table detail as list
    ]
    with patch("requests.get", return_value=mock_response):
        tables = extractor.extract_tables(set())
        assert len(tables) == 1
        table = tables[0]
        assert table.source.type == MetadataType.TABLE
        assert table.source.id == SAMPLE_TABLE["id"]

def test_extract_tables_empty_detail(extractor, mock_response):
    """Test handling empty table detail response."""
    mock_response.json.side_effect = [
        [SAMPLE_TABLE],  # List of tables
        [],  # Empty detail response
    ]
    with patch("requests.get", return_value=mock_response):
        tables = extractor.extract_tables(set())
        assert len(tables) == 0

def test_extract_incremental(extractor, mock_response):
    """Test incremental extraction."""
    # Mock state with processed items
    state = {
        "last_run": "2024-03-26T00:00:00Z",
        "processed_tables": {SAMPLE_TABLE["id"]},
        "processed_buckets": {SAMPLE_BUCKET["id"]},
    }
    
    # Mock responses
    mock_response.json.side_effect = [
        [SAMPLE_BUCKET],  # List of buckets
        SAMPLE_BUCKET_DETAIL,  # Bucket detail
        [SAMPLE_TABLE],  # List of tables
        SAMPLE_TABLE_DETAIL,  # Table detail
    ]
    
    with patch("requests.get", return_value=mock_response), \
         patch.object(extractor, "_load_state", return_value=state):
        documents = extractor.extract(incremental=True)
        assert len(documents) == 0  # No new documents since all are processed

def test_extract_full(extractor, mock_response):
    """Test full extraction."""
    # Mock responses
    mock_response.json.side_effect = [
        [SAMPLE_BUCKET],  # List of buckets
        SAMPLE_BUCKET_DETAIL,  # Bucket detail
        [SAMPLE_TABLE],  # List of tables
        SAMPLE_TABLE_DETAIL,  # Table detail
    ]
    
    with patch("requests.get", return_value=mock_response):
        documents = extractor.extract(incremental=False)
        assert len(documents) == 2  # One bucket and one table

def test_error_handling(extractor, mock_response):
    """Test error handling during extraction."""
    mock_response.raise_for_status.side_effect = Exception("API Error")
    
    with patch("requests.get", return_value=mock_response):
        buckets = extractor.extract_buckets(set())
        assert len(buckets) == 0  # Should handle error gracefully 