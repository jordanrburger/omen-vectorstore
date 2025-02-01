import unittest
from unittest.mock import MagicMock, patch

from app.keboola_client import KeboolaClient
from app.indexer import QdrantIndexer


class TestColumnStatistics(unittest.TestCase):
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
                    "statistics": {
                        "min": "1",
                        "max": "1000",
                        "avg": "500.5",
                        "median": "500",
                        "unique_count": "1000",
                        "null_count": "0",
                    },
                },
                {
                    "name": "amount",
                    "type": "NUMERIC",
                    "basetype": "NUMERIC",
                    "description": "Transaction amount",
                    "statistics": {
                        "min": "10.50",
                        "max": "999.99",
                        "avg": "245.75",
                        "median": "199.99",
                        "unique_count": "856",
                        "null_count": "12",
                    },
                },
                {
                    "name": "category",
                    "type": "VARCHAR",
                    "basetype": "STRING",
                    "description": "Transaction category",
                    "statistics": {
                        "unique_count": "15",
                        "null_count": "5",
                        "most_common": ["food", "transport", "entertainment"],
                    },
                },
            ],
            "metadata": {
                "KBC.name": "Test Table",
                "KBC.description": "Table containing test data",
            }
        }

    @patch("app.keboola_client.Client")
    def test_extract_column_statistics(self, MockClient):
        """Test that column statistics are properly extracted."""
        # Setup mock for the client
        mock_instance = MagicMock()
        mock_instance.tables.detail.return_value = self.mock_table_detail
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        details = client.get_table_details("table1")

        # Verify column statistics are present
        self.assertIn("columns", details)
        columns = details["columns"]
        self.assertEqual(len(columns), 3)

        # Verify numeric column statistics
        amount_column = next(col for col in columns if col["name"] == "amount")
        self.assertIn("statistics", amount_column)
        stats = amount_column["statistics"]
        self.assertEqual(stats["min"], "10.50")
        self.assertEqual(stats["max"], "999.99")
        self.assertEqual(stats["avg"], "245.75")
        self.assertEqual(stats["unique_count"], "856")
        self.assertEqual(stats["null_count"], "12")

        # Verify string column statistics
        category_column = next(col for col in columns if col["name"] == "category")
        self.assertIn("statistics", category_column)
        stats = category_column["statistics"]
        self.assertEqual(stats["unique_count"], "15")
        self.assertEqual(stats["null_count"], "5")
        self.assertIn("most_common", stats)
        self.assertEqual(len(stats["most_common"]), 3)

    @patch("app.indexer.QdrantClient")
    def test_index_column_statistics(self, MockQdrantClient):
        """Test that column statistics are properly indexed."""
        # Setup mock embedding provider
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [
            [0.1, 0.2],  # id column
            [0.3, 0.4],  # amount column
            [0.5, 0.6],  # category column
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

        # Verify embeddings were created with statistics included
        expected_texts = [
            "Name: id | Type: INTEGER | Description: Primary key | Statistics: min=1, max=1000, avg=500.5, unique=1000",
            "Name: amount | Type: NUMERIC | Description: Transaction amount | Statistics: min=10.50, max=999.99, avg=245.75, unique=856",
            "Name: category | Type: VARCHAR | Description: Transaction category | Statistics: unique=15, most_common=food,transport,entertainment",
        ]
        mock_embedding_provider.embed.assert_called_with(expected_texts)

        # Verify points were created with statistics in payload
        points = mock_qdrant.upsert.call_args[1]["points"]
        self.assertEqual(len(points), 3)
        
        # Verify numeric column point
        amount_point = next(p for p in points if p.payload["raw_metadata"]["name"] == "amount")
        self.assertEqual(amount_point.payload["metadata_type"], "columns")
        self.assertEqual(amount_point.payload["table_id"], "table1")
        self.assertIn("statistics", amount_point.payload["raw_metadata"])
        stats = amount_point.payload["raw_metadata"]["statistics"]
        self.assertEqual(stats["min"], "10.50")
        self.assertEqual(stats["max"], "999.99")
        self.assertEqual(stats["avg"], "245.75")

        # Verify string column point
        category_point = next(p for p in points if p.payload["raw_metadata"]["name"] == "category")
        self.assertIn("statistics", category_point.payload["raw_metadata"])
        stats = category_point.payload["raw_metadata"]["statistics"]
        self.assertEqual(stats["unique_count"], "15")
        self.assertIn("most_common", stats)

    def test_search_by_statistics(self):
        """Test searching columns by their statistics."""
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
                        "name": "amount",
                        "type": "NUMERIC",
                        "description": "Transaction amount",
                        "statistics": {
                            "min": "10.50",
                            "max": "999.99",
                            "avg": "245.75",
                            "unique_count": "856",
                            "null_count": "12",
                        },
                    },
                    "table_id": "in.c-main.transactions",
                },
            ),
        ]

        # Create indexer with mock client
        with patch("app.indexer.QdrantClient", return_value=mock_qdrant):
            indexer = QdrantIndexer()
            results = indexer.search_metadata(
                "find numeric columns with average above 200",
                mock_embedding_provider,
                metadata_type="columns",
            )

        # Verify search results
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["metadata"]["name"], "amount")
        self.assertEqual(result["metadata"]["statistics"]["avg"], "245.75") 