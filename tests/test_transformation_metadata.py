import unittest
from unittest.mock import MagicMock, patch

from app.keboola_client import KeboolaClient
from app.indexer import QdrantIndexer


class TestTransformationMetadata(unittest.TestCase):
    def setUp(self):
        self.mock_transformation = {
            "id": "12345",
            "name": "Customer Data Processing",
            "description": "Processes and enriches customer data",
            "type": "python",
            "blocks": [
                {
                    "name": "Load Data",
                    "code": """
                        import pandas as pd
                        
                        # Load customer data
                        customers = pd.read_csv('/data/in/tables/customers.csv')
                        orders = pd.read_csv('/data/in/tables/orders.csv')
                    """,
                    "inputs": [
                        {"source": "in.c-main.customers", "destination": "customers.csv"},
                        {"source": "in.c-main.orders", "destination": "orders.csv"},
                    ],
                },
                {
                    "name": "Process Data",
                    "code": """
                        # Merge customer and order data
                        customer_orders = pd.merge(
                            customers,
                            orders,
                            on='customer_id',
                            how='left'
                        )
                        
                        # Calculate customer metrics
                        metrics = customer_orders.groupby('customer_id').agg({
                            'order_id': 'count',
                            'total_amount': 'sum',
                            'order_date': 'max'
                        })
                    """,
                },
                {
                    "name": "Save Results",
                    "code": """
                        # Save processed data
                        metrics.to_csv('/data/out/tables/customer_metrics.csv')
                    """,
                    "outputs": [
                        {"source": "customer_metrics.csv", "destination": "out.c-processed.customer_metrics"},
                    ],
                },
            ],
            "dependencies": {
                "requires": ["in.c-main.customers", "in.c-main.orders"],
                "produces": ["out.c-processed.customer_metrics"],
            },
            "runtime": {
                "backend": "docker",
                "image": "python:3.9",
                "memory": "4g",
                "cpu_units": 2,
            },
            "schedule": {
                "frequency": "hourly",
                "start_time": "2024-03-20T00:00:00Z",
            },
            "metadata": {
                "KBC.createdBy.user.email": "john.doe@example.com",
                "KBC.updatedBy.user.email": "jane.smith@example.com",
                "KBC.lastUpdated": "2024-03-20T10:00:00Z",
            }
        }

    @patch("app.keboola_client.Client")
    def test_extract_transformation_metadata(self, MockClient):
        """Test that transformation metadata is properly extracted."""
        # Setup mock for the client
        mock_instance = MagicMock()
        mock_instance.transformations.get.return_value = self.mock_transformation
        MockClient.return_value = mock_instance

        client = KeboolaClient("http://fake.api", "token")
        transformation = client.get_transformation_details("12345")

        # Verify basic transformation metadata
        self.assertEqual(transformation["id"], "12345")
        self.assertEqual(transformation["name"], "Customer Data Processing")
        self.assertEqual(transformation["type"], "python")

        # Verify code blocks
        self.assertEqual(len(transformation["blocks"]), 3)
        load_block = transformation["blocks"][0]
        self.assertEqual(load_block["name"], "Load Data")
        self.assertEqual(len(load_block["inputs"]), 2)

        # Verify dependencies
        self.assertEqual(len(transformation["dependencies"]["requires"]), 2)
        self.assertEqual(len(transformation["dependencies"]["produces"]), 1)

        # Verify runtime configuration
        self.assertEqual(transformation["runtime"]["backend"], "docker")
        self.assertEqual(transformation["runtime"]["memory"], "4g")

    @patch("app.indexer.QdrantClient")
    def test_index_transformation_metadata(self, MockQdrantClient):
        """Test that transformation metadata is properly indexed."""
        # Setup mock embedding provider
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [
            [0.1, 0.2],  # transformation
            [0.3, 0.4],  # load block
            [0.5, 0.6],  # process block
            [0.7, 0.8],  # save block
        ]

        # Setup mock Qdrant client
        mock_qdrant = MagicMock()
        MockQdrantClient.return_value = mock_qdrant

        # Create indexer and index metadata
        indexer = QdrantIndexer()
        metadata = {
            "transformations": {
                "12345": self.mock_transformation
            }
        }
        indexer.index_metadata(metadata, mock_embedding_provider)

        # Verify embeddings were created
        mock_embedding_provider.embed.assert_called_once()
        args = mock_embedding_provider.embed.call_args[0][0]
        
        # Verify essential components in transformation text
        assert len(args) == 4  # One transformation + three blocks
        assert "Name: Customer Data Processing" in args[0]
        assert "Type: python" in args[0]
        assert "Description: Processes and enriches customer data" in args[0]
        assert "Dependencies: Requires" in args[0]
        assert "in.c-main.customers" in args[0]
        assert "in.c-main.orders" in args[0]
        assert "Produces" in args[0]
        assert "out.c-processed.customer_metrics" in args[0]
        assert "Runtime:" in args[0]
        
        # Verify block texts
        assert "Block: Load Data" in args[1]
        assert "read_csv" in args[1]
        
        assert "Block: Process Data" in args[2]
        assert "merge" in args[2]
        assert "groupby" in args[2]
        assert "aggregate" in args[2]
        
        assert "Block: Save Results" in args[3]
        assert "to_csv" in args[3]

    def test_search_transformations(self):
        """Test searching transformations and their blocks."""
        # Setup mock embedding provider
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2]]

        # Setup mock Qdrant client
        mock_qdrant = MagicMock()
        mock_qdrant.search.return_value = [
            MagicMock(
                score=0.95,
                payload={
                    "metadata_type": "transformation_blocks",
                    "raw_metadata": {
                        "name": "Process Data",
                        "code": "# Merge customer and order data...",
                    },
                    "transformation_id": "12345",
                },
            ),
        ]

        # Create indexer with mock client
        with patch("app.indexer.QdrantClient", return_value=mock_qdrant):
            indexer = QdrantIndexer()
            results = indexer.search_metadata(
                "find transformations that merge customer data",
                mock_embedding_provider,
                metadata_type="transformation_blocks",
            )

        # Verify search results
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["metadata_type"], "transformation_blocks")
        self.assertEqual(result["metadata"]["name"], "Process Data")
        self.assertEqual(result["transformation_id"], "12345")

    def test_find_related_transformations(self):
        """Test finding transformations related to a specific table."""
        # Setup mock embedding provider
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2]]

        # Setup mock Qdrant client
        mock_qdrant = MagicMock()
        mock_qdrant.search.return_value = [
            MagicMock(
                score=0.95,
                payload={
                    "metadata_type": "transformations",
                    "raw_metadata": self.mock_transformation,
                },
            ),
        ]

        # Create indexer with mock client
        with patch("app.indexer.QdrantClient", return_value=mock_qdrant):
            indexer = QdrantIndexer()
            results = indexer.find_related_transformations(
                "in.c-main.customers",
                mock_embedding_provider,
            )

        # Verify results
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["metadata"]["name"], "Customer Data Processing")
        self.assertIn("in.c-main.customers", result["metadata"]["dependencies"]["requires"]) 