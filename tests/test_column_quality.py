import unittest
from unittest.mock import MagicMock, patch

from app.keboola_client import KeboolaClient
from app.indexer import QdrantIndexer


class TestColumnQuality(unittest.TestCase):
    def setUp(self):
        self.mock_table_detail = {
            "id": "table1",
            "name": "Test Table",
            "columns": [
                {
                    "name": "email",
                    "type": "VARCHAR",
                    "basetype": "STRING",
                    "description": "Customer email address",
                    "quality_metrics": {
                        "completeness": 0.98,  # 98% of values are non-null
                        "validity": 0.95,      # 95% match email pattern
                        "format_pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
                        "common_issues": ["invalid_format", "missing_domain"],
                        "last_validated": "2024-03-20T10:00:00Z",
                    },
                },
                {
                    "name": "age",
                    "type": "INTEGER",
                    "basetype": "INTEGER",
                    "description": "Customer age",
                    "quality_metrics": {
                        "completeness": 0.85,  # 85% of values are non-null
                        "validity": 0.99,      # 99% are valid integers
                        "range_check": {
                            "min_valid": 18,
                            "max_valid": 120,
                            "out_of_range_count": 25,
                        },
                        "common_issues": ["out_of_range", "missing_values"],
                        "last_validated": "2024-03-20T10:00:00Z",
                    },
                },
                {
                    "name": "phone",
                    "type": "VARCHAR",
                    "basetype": "STRING",
                    "description": "Customer phone number",
                    "quality_metrics": {
                        "completeness": 0.75,  # 75% of values are non-null
                        "validity": 0.80,      # 80% match phone pattern
                        "format_pattern": "^\\+?[1-9]\\d{1,14}$",
                        "standardization": {
                            "standardized_count": 850,
                            "total_count": 1000,
                            "standard": "E.164",
                        },
                        "common_issues": ["invalid_format", "missing_country_code"],
                        "last_validated": "2024-03-20T10:00:00Z",
                    },
                },
            ],
            "metadata": {
                "KBC.name": "Test Table",
                "KBC.description": "Table containing customer data",
            }
        }

    @patch("app.keboola_client.Client")
    def test_extract_quality_metrics(self, MockClient):
        """Test that column quality metrics are properly extracted."""
        # Setup mock for the client
        mock_instance = MagicMock()
        mock_instance.tables.detail.return_value = self.mock_table_detail
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        details = client.get_table_details("table1")

        # Verify column quality metrics are present
        self.assertIn("columns", details)
        columns = details["columns"]
        self.assertEqual(len(columns), 3)

        # Verify string column with pattern validation
        email_column = next(col for col in columns if col["name"] == "email")
        self.assertIn("quality_metrics", email_column)
        metrics = email_column["quality_metrics"]
        self.assertEqual(metrics["completeness"], 0.98)
        self.assertEqual(metrics["validity"], 0.95)
        self.assertIn("format_pattern", metrics)
        self.assertIn("common_issues", metrics)

        # Verify numeric column with range validation
        age_column = next(col for col in columns if col["name"] == "age")
        self.assertIn("quality_metrics", age_column)
        metrics = age_column["quality_metrics"]
        self.assertEqual(metrics["completeness"], 0.85)
        self.assertEqual(metrics["validity"], 0.99)
        self.assertIn("range_check", metrics)
        self.assertEqual(metrics["range_check"]["min_valid"], 18)
        self.assertEqual(metrics["range_check"]["max_valid"], 120)

    @patch("app.indexer.QdrantClient")
    def test_index_quality_metrics(self, MockQdrantClient):
        """Test that column quality metrics are properly indexed."""
        # Setup mock embedding provider
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [
            [0.1] * 1536,  # email column
            [0.2] * 1536,  # age column
            [0.3] * 1536,  # phone column
        ]

        # Setup mock Qdrant client
        mock_qdrant = MagicMock()
        MockQdrantClient.return_value = mock_qdrant

        # Create indexer and index metadata
        indexer = QdrantIndexer()
        metadata = {
            "table_details": {
                "table1": {
                    "columns": [
                        {
                            "name": "email",
                            "type": "VARCHAR",
                            "description": "Customer email address",
                            "quality_metrics": {
                                "completeness": 0.98,
                                "validity": 0.95,
                                "standardization": {"standard": "RFC 5322"},
                                "common_issues": ["invalid_format", "missing_domain"]
                            }
                        },
                        {
                            "name": "age",
                            "type": "INTEGER",
                            "description": "Customer age",
                            "quality_metrics": {
                                "completeness": 0.85,
                                "validity": 0.99,
                                "range_check": {
                                    "min_valid": 18,
                                    "max_valid": 120
                                },
                                "common_issues": ["out_of_range", "missing_values"]
                            }
                        },
                        {
                            "name": "phone",
                            "type": "VARCHAR",
                            "description": "Customer phone number",
                            "quality_metrics": {
                                "completeness": 0.75,
                                "validity": 0.80,
                                "standardization": {"standard": "E.164"},
                                "common_issues": ["invalid_format", "missing_country_code"]
                            }
                        }
                    ]
                }
            }
        }
        indexer.index_metadata(metadata, mock_embedding_provider)

        # Verify embeddings were created with quality metrics included
        expected_texts = [
            "Name: email | Type: VARCHAR | Description: Customer email address | Quality: 98% complete, 95% valid, Standard: RFC 5322, Issues: invalid_format, missing_domain",
            "Name: age | Type: INTEGER | Description: Customer age | Quality: 85% complete, 99% valid, Valid range: 18-120, Issues: out_of_range, missing_values",
            "Name: phone | Type: VARCHAR | Description: Customer phone number | Quality: 75% complete, 80% valid, Standard: E.164, Issues: invalid_format, missing_country_code",
        ]
        mock_embedding_provider.embed.assert_called_with(expected_texts)

        # Verify points were created with quality metrics in payload
        points = mock_qdrant.upsert.call_args[1]["points"]
        self.assertEqual(len(points), 3)
        
        # Verify string column with pattern validation
        email_point = next(p for p in points if p.payload["raw_metadata"]["name"] == "email")
        self.assertIn("quality_metrics", email_point.payload["raw_metadata"])
        metrics = email_point.payload["raw_metadata"]["quality_metrics"]
        self.assertEqual(metrics["completeness"], 0.98)
        self.assertEqual(metrics["validity"], 0.95)

        # Verify numeric column with range validation
        age_point = next(p for p in points if p.payload["raw_metadata"]["name"] == "age")
        self.assertIn("quality_metrics", age_point.payload["raw_metadata"])
        metrics = age_point.payload["raw_metadata"]["quality_metrics"]
        self.assertEqual(metrics["completeness"], 0.85)
        self.assertEqual(metrics["validity"], 0.99)
        self.assertIn("range_check", metrics)

    def test_search_by_quality_metrics(self):
        """Test searching columns by their quality metrics."""
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
                        "name": "phone",
                        "type": "VARCHAR",
                        "description": "Customer phone number",
                        "quality_metrics": {
                            "completeness": 0.75,
                            "validity": 0.80,
                            "format_pattern": "^\\+?[1-9]\\d{1,14}$",
                            "standardization": {
                                "standardized_count": 850,
                                "total_count": 1000,
                                "standard": "E.164",
                            },
                            "common_issues": ["invalid_format", "missing_country_code"],
                        },
                    },
                    "table_id": "in.c-main.customers",
                },
            ),
        ]

        # Create indexer with mock client
        with patch("app.indexer.QdrantClient", return_value=mock_qdrant):
            indexer = QdrantIndexer()
            results = indexer.search_metadata(
                "find columns with data quality issues",
                mock_embedding_provider,
                metadata_type="columns",
            )

        # Verify search results
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["metadata"]["name"], "phone")
        self.assertEqual(result["metadata"]["quality_metrics"]["completeness"], 0.75)
        self.assertEqual(result["metadata"]["quality_metrics"]["validity"], 0.80)
        self.assertIn("missing_country_code", result["metadata"]["quality_metrics"]["common_issues"]) 