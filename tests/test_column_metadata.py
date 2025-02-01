import unittest
from unittest.mock import MagicMock, patch

from app.keboola_client import KeboolaClient
from app.state_manager import StateManager
from app.indexer import QdrantIndexer


class TestColumnMetadata(unittest.TestCase):
    def setUp(self):
        self.mock_table_detail = {
            "id": "table1",
            "name": "Test Table",
            "columns": [
                {
                    "name": "id",
                    "type": "INTEGER",
                    "basetype": "INTEGER",
                    "description": "Primary key",
                },
                {
                    "name": "name",
                    "type": "VARCHAR",
                    "basetype": "STRING",
                    "description": "Customer name",
                },
                {
                    "name": "created_at",
                    "type": "TIMESTAMP",
                    "basetype": "TIMESTAMP",
                    "description": "Record creation time",
                },
            ],
            "metadata": {
                "KBC.name": "Test Table",
                "KBC.description": "Table containing test data",
            }
        }

    @patch("app.keboola_client.Client")
    def test_get_column_metadata(self, MockClient):
        # Setup mock for the client
        mock_instance = MagicMock()
        mock_instance.tables.detail.return_value = self.mock_table_detail
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        details = client.get_table_details("table1")

        # Verify column metadata is present
        self.assertIn("columns", details)
        columns = details["columns"]
        self.assertEqual(len(columns), 3)

        # Verify first column details
        first_column = columns[0]
        self.assertEqual(first_column["name"], "id")
        self.assertEqual(first_column["type"], "INTEGER")
        self.assertEqual(first_column["description"], "Primary key")

    @patch("app.keboola_client.Client")
    def test_extract_metadata_includes_columns(self, MockClient):
        # Setup mock for the client
        mock_instance = MagicMock()
        mock_instance.buckets.list.return_value = [{"id": "bucket1"}]
        mock_instance.buckets.list_tables.return_value = [{"id": "table1"}]
        mock_instance.tables.detail.return_value = self.mock_table_detail
        MockClient.return_value = mock_instance

        # Setup mock state manager
        mock_state_manager = MagicMock(spec=StateManager)
        mock_state_manager._load_state.return_value = {}
        mock_state_manager.get_metadata_hash.return_value = None
        mock_state_manager.compute_hash.return_value = "test_hash"
        mock_state_manager.has_changed.return_value = True

        client = KeboolaClient("http://fake.api", "token", state_manager=mock_state_manager)
        metadata = client.extract_metadata(force_full=True)

        # Verify table details include column metadata
        self.assertIn("table_details", metadata)
        self.assertIn("table1", metadata["table_details"])
        table_details = metadata["table_details"]["table1"]
        self.assertIn("columns", table_details)

        # Verify column metadata is correctly extracted
        columns = table_details["columns"]
        self.assertEqual(len(columns), 3)
        self.assertEqual(
            columns[1],
            {
                "name": "name",
                "type": "VARCHAR",
                "basetype": "STRING",
                "description": "Customer name",
                "table_id": "table1",
                "bucket_id": "bucket1"
            }
        )

    @patch("app.indexer.QdrantClient")
    def test_index_column_metadata(self, MockQdrantClient):
        from app.indexer import QdrantIndexer
        
        # Setup mock embedding provider
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [
            [0.1, 0.2],  # id column
            [0.3, 0.4],  # name column
            [0.5, 0.6],  # created_at column
        ]

        # Setup mock Qdrant client
        mock_qdrant = MagicMock()
        MockQdrantClient.return_value = mock_qdrant

        # Create indexer and index metadata
        indexer = QdrantIndexer()
        metadata = {
            "table_details": {
                "table1": self.mock_table_detail
            }
        }
        indexer.index_metadata(metadata, mock_embedding_provider)

        # Verify embeddings were created for columns
        expected_texts = [
            "Name: id | Type: INTEGER | Description: Primary key",
            "Name: name | Type: VARCHAR | Description: Customer name",
            "Name: created_at | Type: TIMESTAMP | Description: Record creation time",
        ]
        mock_embedding_provider.embed.assert_called_with(expected_texts)

        # Verify points were created with correct metadata
        points = mock_qdrant.upsert.call_args[1]["points"]
        self.assertEqual(len(points), 3)
        
        # Verify first column point
        first_point = points[0]
        self.assertEqual(first_point.payload["metadata_type"], "columns")
        self.assertEqual(first_point.payload["table_id"], "table1")
        self.assertEqual(first_point.payload["raw_metadata"]["name"], "id")
        self.assertEqual(first_point.payload["raw_metadata"]["type"], "INTEGER")
        self.assertEqual(first_point.payload["raw_metadata"]["description"], "Primary key")

    def test_search_column_metadata(self):
        """Test searching for column metadata."""
        # Setup mock embedding provider
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2]]  # Mock query embedding

        # Setup mock Qdrant client
        mock_qdrant = MagicMock()
        mock_qdrant.search.return_value = [
            MagicMock(
                score=0.95,
                payload={
                    "metadata_type": "columns",
                    "raw_metadata": {
                        "name": "customer_id",
                        "type": "INTEGER",
                        "description": "Primary key for customer table",
                    },
                    "table_id": "in.c-main.customers",
                },
            ),
            MagicMock(
                score=0.85,
                payload={
                    "metadata_type": "columns",
                    "raw_metadata": {
                        "name": "user_id",
                        "type": "INTEGER",
                        "description": "Foreign key to users table",
                    },
                    "table_id": "in.c-main.orders",
                },
            ),
        ]

        # Create indexer with mock client
        with patch("app.indexer.QdrantClient", return_value=mock_qdrant):
            indexer = QdrantIndexer()
            results = indexer.search_metadata(
                "find customer ID columns",
                mock_embedding_provider,
                metadata_type="columns",
            )

        # Verify search was called with correct parameters
        mock_qdrant.search.assert_called_once()
        search_args = mock_qdrant.search.call_args[1]
        self.assertEqual(search_args["collection_name"], "keboola_metadata")
        self.assertEqual(search_args["query_vector"], [0.1, 0.2])
        self.assertEqual(search_args["limit"], 10)

        # Verify filter was set correctly for columns
        filter_condition = search_args["query_filter"].must[0]
        self.assertEqual(filter_condition.key, "metadata_type")
        self.assertEqual(filter_condition.match.value, "columns")

        # Verify results are formatted correctly
        self.assertEqual(len(results), 2)
        first_result = results[0]
        self.assertEqual(first_result["score"], 0.95)
        self.assertEqual(first_result["metadata_type"], "columns")
        self.assertEqual(first_result["metadata"]["name"], "customer_id")
        self.assertEqual(first_result["metadata"]["type"], "INTEGER")

        # Verify table relationship is preserved
        self.assertEqual(
            mock_qdrant.search.call_args[1]["query_filter"].must[0].match.value,
            "columns",
        )

    def test_find_table_columns(self):
        """Test finding all columns for a specific table."""
        # Setup mock Qdrant client
        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = (
            [
                MagicMock(
                    payload={
                        "metadata_type": "columns",
                        "raw_metadata": {
                            "name": "id",
                            "type": "INTEGER",
                            "description": "Primary key",
                        },
                        "table_id": "in.c-main.customers",
                    }
                ),
                MagicMock(
                    payload={
                        "metadata_type": "columns",
                        "raw_metadata": {
                            "name": "email",
                            "type": "VARCHAR",
                            "description": "Customer email",
                        },
                        "table_id": "in.c-main.customers",
                    }
                ),
            ],
            None,  # No next page token
        )

        # Create indexer with mock client
        with patch("app.indexer.QdrantClient", return_value=mock_qdrant):
            indexer = QdrantIndexer()
            results = indexer.find_table_columns("in.c-main.customers")

        # Verify scroll was called with correct filter
        mock_qdrant.scroll.assert_called_once()
        scroll_args = mock_qdrant.scroll.call_args[1]
        filter_conditions = scroll_args["scroll_filter"].must
        self.assertEqual(len(filter_conditions), 2)
        self.assertEqual(filter_conditions[0].key, "metadata_type")
        self.assertEqual(filter_conditions[0].match.value, "columns")
        self.assertEqual(filter_conditions[1].key, "table_id")
        self.assertEqual(filter_conditions[1].match.value, "in.c-main.customers")

        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["metadata"]["name"], "id")
        self.assertEqual(results[1]["metadata"]["name"], "email")
        self.assertEqual(results[0]["table_id"], "in.c-main.customers")

    def test_find_table_columns_with_query(self):
        """Test finding columns in a table filtered by a search query."""
        # Setup mock embedding provider
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2]]

        # Setup mock Qdrant client
        mock_qdrant = MagicMock()
        mock_qdrant.search.return_value = [
            MagicMock(
                score=0.95,
                payload={
                    "metadata_type": "columns",
                    "raw_metadata": {
                        "name": "email",
                        "type": "VARCHAR",
                        "description": "Customer email address",
                    },
                    "table_id": "in.c-main.customers",
                }
            )
        ]

        # Create indexer with mock client
        with patch("app.indexer.QdrantClient", return_value=mock_qdrant):
            indexer = QdrantIndexer()
            results = indexer.find_table_columns(
                "in.c-main.customers",
                query="email columns",
                embedding_provider=mock_embedding_provider,
            )

        # Verify search was called with correct parameters
        mock_qdrant.search.assert_called_once()
        search_args = mock_qdrant.search.call_args[1]
        filter_conditions = search_args["query_filter"].must
        self.assertEqual(len(filter_conditions), 2)
        self.assertEqual(filter_conditions[0].key, "metadata_type")
        self.assertEqual(filter_conditions[0].match.value, "columns")
        self.assertEqual(filter_conditions[1].key, "table_id")
        self.assertEqual(filter_conditions[1].match.value, "in.c-main.customers")

        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["name"], "email")
        self.assertEqual(results[0]["table_id"], "in.c-main.customers")

    def test_find_similar_columns(self):
        """Test finding similar columns across tables."""
        # Setup mock Qdrant client for initial column lookup
        mock_qdrant = MagicMock()
        mock_vector = [0.1, 0.2]
        mock_qdrant.scroll.return_value = (
            [
                MagicMock(
                    vector=mock_vector,
                    payload={
                        "metadata_type": "columns",
                        "raw_metadata": {
                            "name": "email",
                            "type": "VARCHAR",
                            "description": "Customer email",
                        },
                        "table_id": "in.c-main.customers",
                    }
                )
            ],
            None,
        )
    
        # Setup mock embedding provider
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [mock_vector]
    
        # Setup mock search results
        mock_qdrant.search.return_value = [
            MagicMock(
                score=0.95,
                payload={
                    "metadata_type": "columns",
                    "raw_metadata": {
                        "name": "user_email",
                        "type": "VARCHAR",
                        "description": "User email address",
                    },
                    "table_id": "in.c-main.users",
                }
            ),
            MagicMock(
                score=0.85,
                payload={
                    "metadata_type": "columns",
                    "raw_metadata": {
                        "name": "contact_email",
                        "type": "VARCHAR",
                        "description": "Contact email address",
                    },
                    "table_id": "in.c-main.contacts",
                }
            ),
        ]
    
        # Create indexer with mock client
        with patch("app.indexer.QdrantClient", return_value=mock_qdrant):
            indexer = QdrantIndexer()
            results = indexer.find_similar_columns(
                "email",
                "in.c-main.customers",
                mock_embedding_provider,
            )
    
        # Verify search was called with correct parameters
        mock_qdrant.search.assert_called_once()
        search_args = mock_qdrant.search.call_args[1]
        self.assertEqual(search_args["collection_name"], "keboola_metadata")
        self.assertEqual(search_args["query_vector"], mock_vector) 