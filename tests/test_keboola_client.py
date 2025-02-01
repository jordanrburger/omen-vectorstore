import unittest
from unittest.mock import MagicMock, patch
import pytest
import logging
import requests

from app.keboola_client import KeboolaClient
from app.state_manager import StateManager


class TestKeboolaClient(unittest.TestCase):

    @patch("app.keboola_client.Client")
    def test_list_buckets_success(self, MockClient):
        # Setup mock for the client
        mock_instance = MagicMock()
        mock_instance.buckets.list.return_value = [
            {"id": "bucket1"},
            {"id": "bucket2"},
        ]
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        buckets = list(client.list_buckets_paginated())
        self.assertEqual(len(buckets), 2)
        self.assertEqual(buckets[0]["id"], "bucket1")
        self.assertEqual(buckets[1]["id"], "bucket2")
        mock_instance.buckets.list.assert_called_once()

    @patch("app.keboola_client.Client")
    def test_list_buckets_failure(self, MockClient):
        # Setup mock to raise exception
        mock_instance = MagicMock()
        mock_instance.buckets.list.side_effect = Exception("Error fetching buckets")
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        buckets = list(client.list_buckets_paginated())
        self.assertEqual(len(buckets), 0)
        mock_instance.buckets.list.assert_called_once()

    @patch("app.keboola_client.Client")
    def test_list_tables_success(self, MockClient):
        # Setup mock for the client
        mock_instance = MagicMock()
        mock_instance.buckets.list_tables.return_value = [
            {"id": "table1"},
            {"id": "table2"},
        ]
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        tables = list(client.list_tables_paginated("bucket1"))
        self.assertEqual(len(tables), 2)
        self.assertEqual(tables[0]["id"], "table1")
        self.assertEqual(tables[1]["id"], "table2")
        mock_instance.buckets.list_tables.assert_called_once_with("bucket1")

    @patch("app.keboola_client.Client")
    def test_list_tables_failure(self, MockClient):
        # Setup mock to raise exception
        mock_instance = MagicMock()
        mock_instance.buckets.list_tables.side_effect = Exception(
            "Error fetching tables"
        )
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        tables = list(client.list_tables_paginated("bucket1"))
        self.assertEqual(len(tables), 0)
        mock_instance.buckets.list_tables.assert_called_once_with("bucket1")

    @patch("app.keboola_client.Client")
    def test_get_table_details_success(self, MockClient):
        # Setup mock for the client
        mock_instance = MagicMock()
        mock_instance.tables.detail.return_value = {
            "id": "table1",
            "name": "Test Table",
        }
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        details = client.get_table_details("table1")
        self.assertEqual(details["id"], "table1")
        self.assertEqual(details["name"], "Test Table")
        mock_instance.tables.detail.assert_called_once_with("table1")

    @patch("app.keboola_client.Client")
    def test_get_table_details_failure(self, MockClient):
        # Setup mock to raise exception
        mock_instance = MagicMock()
        mock_instance.tables.detail.side_effect = Exception(
            "Error fetching table details"
        )
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        details = client.get_table_details("table1")
        self.assertIsNone(details)
        mock_instance.tables.detail.assert_called_once_with("table1")


@pytest.fixture
def mock_kbcstorage_client():
    """Create a mock Keboola Storage API client."""
    client = MagicMock()
    client.buckets = MagicMock()
    client.tables = MagicMock()
    client.components = MagicMock()
    client.transformations = MagicMock()
    return client

@pytest.fixture
def mock_state_manager():
    """Create a mock state manager."""
    manager = MagicMock()
    manager.load_extraction_state.return_value = {}
    manager.load_metadata.return_value = None
    manager.compute_hash.return_value = "test_hash"
    return manager

@pytest.fixture
def mock_session():
    """Create a mock session for testing."""
    session = MagicMock()
    session.get = MagicMock()
    return session

@pytest.fixture
def client(mock_kbcstorage_client, mock_state_manager):
    """Create a client with mock dependencies for testing."""
    client = KeboolaClient(
        api_url="https://connection.keboola.com/v2/storage/",
        token="test-token",
        state_manager=mock_state_manager
    )
    client.client = mock_kbcstorage_client
    return client

@pytest.fixture
def client_with_mock_session(mock_session):
    """Create a client with a mock session."""
    client = KeboolaClient("https://connection.keboola.com", "token")
    client.session = mock_session
    return client

def test_init(client):
    """Test client initialization."""
    assert client.token == "test-token"
    assert client.url == "https://connection.keboola.com/v2/storage"

def test_extract_metadata_full(client, mock_state_manager):
    """Test full metadata extraction."""
    # Mock bucket listing
    client.list_buckets_paginated = MagicMock(return_value=[{"id": "bucket1"}])
    client.list_tables_paginated = MagicMock(return_value=[{"id": "table1"}])
    client.get_table_details = MagicMock(return_value={"columns": []})
    client.client.components.list.return_value = []

    metadata = client.extract_metadata(force_full=True)
    assert "buckets" in metadata
    assert len(metadata["buckets"]) == 1
    assert metadata["buckets"][0]["id"] == "bucket1"

def test_extract_metadata_incremental(client, mock_state_manager):
    """Test incremental metadata extraction."""
    # Mock state manager
    previous_state = {
        "bucket_hashes": {"bucket1": "test_hash"},
        "table_hashes": {},
        "config_hashes": {},
        "config_row_hashes": {},
        "column_hashes": {}
    }
    previous_metadata = {
        "buckets": [{"id": "bucket1"}],
        "tables": {},
        "table_details": {},
        "configurations": {},
        "config_rows": {},
        "columns": {}
    }
    mock_state_manager.load_extraction_state.return_value = previous_state
    mock_state_manager.load_metadata.return_value = previous_metadata

    # Mock bucket listing
    client.list_buckets_paginated = MagicMock(return_value=[{"id": "bucket1"}])
    client.client.components.list.return_value = []

    metadata = client.extract_metadata()
    assert "buckets" in metadata
    assert len(metadata["buckets"]) == 1
    assert metadata["buckets"][0]["id"] == "bucket1"

def test_list_buckets_paginated(client):
    """Test paginated bucket listing."""
    buckets = [{"id": "bucket1"}, {"id": "bucket2"}]
    client.client.buckets.list.return_value = buckets

    result = list(client.list_buckets_paginated())
    assert len(result) == 2
    assert result == buckets

def test_list_tables_paginated(client):
    """Test paginated table listing."""
    tables = [{"id": "table1"}, {"id": "table2"}]
    client.client.buckets.list_tables.return_value = tables

    result = list(client.list_tables_paginated("bucket1"))
    assert len(result) == 2
    assert result == tables

def test_list_configurations_paginated(client):
    """Test paginated configuration listing."""
    configs = [{"id": "config1"}, {"id": "config2"}]
    client.client.components.list_configs.return_value = configs

    result = list(client.list_configurations_paginated("component1"))
    assert len(result) == 2
    assert result == configs

def test_list_config_rows_paginated(client):
    """Test paginated configuration row listing."""
    rows = [{"id": "row1"}, {"id": "row2"}]
    client.client.components.list_config_rows.return_value = rows

    result = list(client.list_config_rows_paginated("component1", "config1"))
    assert len(result) == 2
    assert result == rows

def test_get_table_details(client):
    """Test getting table details."""
    details = {"id": "table1", "columns": []}
    client.client.tables.detail.return_value = details

    result = client.get_table_details("table1")
    assert result == details
    client.client.tables.detail.assert_called_once_with("table1")

@patch('app.keboola_client.logging')
def test_list_buckets_error_handling(mock_logging, client):
    """Test error handling in bucket listing."""
    error = Exception("API Error")
    client.client.buckets.list.side_effect = error

    list(client.list_buckets_paginated())
    mock_logging.error.assert_called_once_with("Error fetching buckets: %s", error)

@patch('app.keboola_client.logging')
def test_list_tables_error_handling(mock_logging, client):
    """Test error handling in table listing."""
    error = Exception("API Error")
    client.client.buckets.list_tables.side_effect = error

    list(client.list_tables_paginated("bucket1"))
    mock_logging.error.assert_called_once_with(
        "Error fetching tables for bucket bucket1: API Error"
    )

@patch('app.keboola_client.logging')
def test_list_config_rows_error_handling(mock_logging, client):
    """Test error handling in configuration row listing."""
    error = Exception("API Error")
    client.client.components.list_config_rows.side_effect = error

    list(client.list_config_rows_paginated("component1", "config1"))
    mock_logging.error.assert_called_once_with(
        "Error fetching configuration rows for config config1: API Error"
    )

@patch('app.keboola_client.logging')
def test_get_table_details_error_handling(mock_logging, client):
    """Test error handling in table details fetching."""
    error = Exception("API Error")
    client.client.tables.detail.side_effect = error

    result = client.get_table_details("table1")
    assert result is None
    mock_logging.error.assert_called_once_with(
        "Error fetching table details for table %s: %s", "table1", error
    )

def test_get_transformation_details(client):
    """Test fetching transformation details."""
    mock_transformation = {"id": "trans1", "name": "Test Transformation"}
    client.client.transformations.get.return_value = mock_transformation

    details = client.get_transformation_details("trans1")
    assert details == mock_transformation
    client.client.transformations.get.assert_called_once_with("trans1")

def test_get_all_transformations(client):
    """Test fetching all transformations."""
    mock_transformations = [
        {"id": "trans1", "name": "Test 1"},
        {"id": "trans2", "name": "Test 2"}
    ]
    mock_details = {"id": "trans1", "name": "Test 1", "blocks": []}

    client.client.transformations.list.return_value = mock_transformations
    client.client.transformations.get.return_value = mock_details

    result = client.get_all_transformations()
    assert "transformations" in result
    assert len(result["transformations"]) == 2
    assert all(t_id in result["transformations"] for t_id in ["trans1", "trans2"])

def test_extract_metadata_components_error(client):
    """Test handling of component extraction errors."""
    # Mock client to raise an error when listing components
    client.client.components.list = MagicMock(side_effect=Exception("Component API Error"))
    
    # Mock state manager
    client.state_manager.load_extraction_state.return_value = {}
    client.state_manager.load_metadata.return_value = None
    
    # Mock successful bucket and table operations
    client.list_buckets_paginated = MagicMock(return_value=[])
    
    metadata = client.extract_metadata()
    assert "configurations" in metadata
    assert len(metadata["configurations"]) == 0

def test_extract_metadata_with_state(client):
    """Test metadata extraction with existing state."""
    # Mock state manager with existing state
    previous_state = {
        "bucket_hashes": {"bucket1": "hash1"},
        "table_hashes": {"table1": "hash1"},
        "config_hashes": {},
        "config_row_hashes": {},
        "column_hashes": {}
    }
    previous_metadata = {
        "buckets": [{"id": "bucket1"}],
        "tables": {"bucket1": [{"id": "table1"}]},
        "table_details": {"table1": {"columns": []}},
        "configurations": {},
        "config_rows": {},
        "columns": {"bucket1": {"table1": []}}
    }
    
    client.state_manager.load_extraction_state.return_value = previous_state
    client.state_manager.load_metadata.return_value = previous_metadata
    client.state_manager.compute_hash.return_value = "hash1"
    
    # Mock bucket and table listing
    client.list_buckets_paginated = MagicMock(return_value=[{"id": "bucket1"}])
    client.list_tables_paginated = MagicMock(return_value=[{"id": "table1"}])
    client.get_table_details = MagicMock(return_value={"columns": []})
    client.client.components.list.return_value = []
    
    metadata = client.extract_metadata()
    assert "buckets" in metadata
    assert len(metadata["buckets"]) == 1
    assert metadata["buckets"][0]["id"] == "bucket1"
    assert "tables" in metadata
    assert "bucket1" in metadata["tables"]
    assert len(metadata["tables"]["bucket1"]) == 1

def test_extract_metadata_with_changed_table(client):
    """Test metadata extraction when a table has changed."""
    # Mock state manager with existing state
    previous_state = {
        "bucket_hashes": {"bucket1": "hash1"},
        "table_hashes": {"table1": "old_hash"},  # Different hash to trigger change
        "config_hashes": {},
        "config_row_hashes": {},
        "column_hashes": {}
    }
    previous_metadata = {
        "buckets": [{"id": "bucket1"}],
        "tables": {"bucket1": [{"id": "table1", "old": True}]},
        "table_details": {"table1": {"columns": []}},
        "configurations": {},
        "config_rows": {},
        "columns": {"bucket1": {"table1": []}}
    }

    client.state_manager.load_extraction_state.return_value = previous_state
    client.state_manager.load_metadata.return_value = previous_metadata
    client.state_manager.compute_hash.side_effect = ["hash1", "old_hash"]  # Same hash for bucket and table

    # Mock bucket and table listing
    mock_table = {"id": "table1", "old": True}  # Use the same table data as in previous_metadata
    client.list_buckets_paginated = MagicMock(return_value=[{"id": "bucket1"}])
    client.list_tables_paginated = MagicMock(return_value=[mock_table])
    client.get_table_details = MagicMock(return_value={"columns": [{"name": "col1"}]})
    client.client.components.list.return_value = []

    metadata = client.extract_metadata()
    assert "buckets" in metadata
    assert len(metadata["buckets"]) == 1
    assert metadata["buckets"][0]["id"] == "bucket1"
    assert "tables" in metadata
    assert "bucket1" in metadata["tables"]
    assert len(metadata["tables"]["bucket1"]) == 1
    assert metadata["tables"]["bucket1"][0] == mock_table  # Compare with the mock table object

def test_extract_metadata_with_configurations(client):
    """Test metadata extraction with configurations and rows."""
    # Mock state manager
    client.state_manager.load_extraction_state.return_value = {}
    client.state_manager.load_metadata.return_value = None
    client.state_manager.compute_hash.return_value = "hash1"
    
    # Mock bucket and table listing (empty for simplicity)
    client.list_buckets_paginated = MagicMock(return_value=[])
    
    # Mock components and configurations
    mock_component = {"id": "component1"}
    mock_config = {"id": "config1", "name": "Test Config"}
    mock_row = {"id": "row1", "name": "Test Row"}
    
    client.client.components.list.return_value = [mock_component]
    client.list_configurations_paginated = MagicMock(return_value=[mock_config])
    client.list_config_rows_paginated = MagicMock(return_value=[mock_row])
    
    metadata = client.extract_metadata()
    assert "configurations" in metadata
    assert "component1" in metadata["configurations"]
    assert len(metadata["configurations"]["component1"]) == 1
    assert metadata["configurations"]["component1"][0] == mock_config
    assert "config_rows" in metadata
    assert "config1" in metadata["config_rows"]
    assert len(metadata["config_rows"]["config1"]) == 1
    assert metadata["config_rows"]["config1"][0] == mock_row

def test_extract_metadata_with_config_row_error(client):
    """Test metadata extraction when config row fetching fails."""
    # Mock state manager
    client.state_manager.load_extraction_state.return_value = {}
    client.state_manager.load_metadata.return_value = None
    client.state_manager.compute_hash.return_value = "hash1"
    
    # Mock bucket and table listing (empty for simplicity)
    client.list_buckets_paginated = MagicMock(return_value=[])
    
    # Mock components and configurations
    mock_component = {"id": "component1"}
    mock_config = {"id": "config1", "name": "Test Config"}
    
    client.client.components.list.return_value = [mock_component]
    client.list_configurations_paginated = MagicMock(return_value=[mock_config])
    client.list_config_rows_paginated = MagicMock(side_effect=Exception("Row error"))
    
    metadata = client.extract_metadata()
    assert "configurations" in metadata
    assert "component1" in metadata["configurations"]
    assert len(metadata["configurations"]["component1"]) == 1
    assert "config_rows" in metadata
    assert "config1" in metadata["config_rows"]
    assert len(metadata["config_rows"]["config1"]) == 0  # Empty due to error

def test_extract_metadata_with_config_error(client):
    """Test metadata extraction when configuration fetching fails."""
    # Mock state manager
    client.state_manager.load_extraction_state.return_value = {}
    client.state_manager.load_metadata.return_value = None
    client.state_manager.compute_hash.return_value = "hash1"
    
    # Mock bucket and table listing (empty for simplicity)
    client.list_buckets_paginated = MagicMock(return_value=[])
    
    # Mock components and configurations
    mock_component = {"id": "component1"}
    
    client.client.components.list.return_value = [mock_component]
    client.list_configurations_paginated = MagicMock(side_effect=Exception("Config error"))
    
    metadata = client.extract_metadata()
    assert "configurations" in metadata
    assert "component1" in metadata["configurations"]
    assert len(metadata["configurations"]["component1"]) == 0  # Empty due to error

def test_extract_metadata_reuse_config(client):
    """Test that configuration metadata is reused when unchanged."""
    # Mock state manager with existing state
    previous_state = {
        "bucket_hashes": {},
        "table_hashes": {},
        "config_hashes": {"config1": "hash1"},
        "config_row_hashes": {},
        "column_hashes": {}
    }
    previous_metadata = {
        "buckets": [],
        "tables": {},
        "table_details": {},
        "configurations": {
            "component1": [{"id": "config1", "name": "Previous Config"}]
        },
        "config_rows": {
            "config1": [{"id": "row1", "name": "Previous Row"}]
        },
        "columns": {}
    }

    client.state_manager.load_extraction_state.return_value = previous_state
    client.state_manager.load_metadata.return_value = previous_metadata
    client.state_manager.compute_hash.side_effect = ["hash1"]  # Same hash for config

    # Mock bucket and table listing (empty for simplicity)
    client.list_buckets_paginated = MagicMock(return_value=[])

    # Mock components and configurations
    mock_component = {"id": "component1"}
    mock_config = {"id": "config1", "name": "Previous Config"}  # Use the same config data as in previous_metadata

    client.client.components.list.return_value = [mock_component]
    client.list_configurations_paginated = MagicMock(return_value=[mock_config])

    metadata = client.extract_metadata()
    assert "configurations" in metadata
    assert "component1" in metadata["configurations"]
    assert len(metadata["configurations"]["component1"]) == 1
    assert metadata["configurations"]["component1"][0]["name"] == "Previous Config"  # Verify reuse

@patch('app.keboola_client.logging')
def test_list_configurations_error_handling(mock_logging, client):
    """Test error handling in list_configurations."""
    error = Exception("API Error")
    client.client.components.list_configs.side_effect = error

    list(client.list_configurations_paginated("component1"))
    mock_logging.error.assert_called_once_with(
        "Error fetching configurations for component component1: API Error"
    )

def test_get_transformation_details_with_phases(client):
    """Test fetching transformation details with multiple phases."""
    mock_response = {
        "id": "123",
        "phases": [
            {
                "id": 1,
                "code": "SELECT * FROM table1",
                "name": "Phase 1"
            },
            {
                "id": 2,
                "code": "SELECT * FROM table2",
                "name": "Phase 2"
            }
        ]
    }
    client.client.transformations.get = MagicMock(return_value=mock_response)

    result = client.get_transformation_details("123")
    assert result == mock_response
    client.client.transformations.get.assert_called_once_with("123")

def test_get_transformation_details_error(client):
    """Test error handling in get_transformation_details."""
    client.client.transformations.get = MagicMock(side_effect=Exception("API Error"))

    with pytest.raises(Exception, match="API Error"):
        client.get_transformation_details("123")

def test_get_all_transformations_with_multiple_components(client):
    """Test fetching transformations from multiple components."""
    mock_transformations = [
        {"id": "config1", "name": "Python Transform"},
        {"id": "config2", "name": "SQL Transform"}
    ]
    mock_details = {
        "config1": {"id": "config1", "phases": [{"code": "print('hello')"}]},
        "config2": {"id": "config2", "phases": [{"code": "SELECT 1"}]}
    }

    client.client.transformations.list = MagicMock(return_value=mock_transformations)
    client.client.transformations.get = MagicMock(side_effect=lambda x: mock_details[x])

    result = client.get_all_transformations()
    assert "transformations" in result
    assert len(result["transformations"]) == 2
    assert result["transformations"]["config1"]["phases"][0]["code"] == "print('hello')"
    assert result["transformations"]["config2"]["phases"][0]["code"] == "SELECT 1"

def test_get_all_transformations_with_errors(client):
    """Test error handling in get_all_transformations."""
    mock_transformations = [{"id": "config1"}]
    client.client.transformations.list = MagicMock(return_value=mock_transformations)
    client.client.transformations.get = MagicMock(side_effect=Exception("API Error"))

    result = client.get_all_transformations()
    assert "transformations" in result
    assert len(result["transformations"]) == 0

def test_get_all_transformations_empty(client):
    """Test handling of empty transformation list."""
    client.client.transformations.list = MagicMock(return_value=[])

    result = client.get_all_transformations()
    assert "transformations" in result
    assert len(result["transformations"]) == 0

if __name__ == "__main__":
    unittest.main()
