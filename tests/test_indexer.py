import unittest
from unittest.mock import MagicMock, patch
import uuid
import pytest
import logging

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import ScoredPoint, Filter, FieldCondition, MatchValue

from app.indexer import QdrantIndexer
from app.batch_processor import BatchConfig


class TestQdrantIndexer(unittest.TestCase):
    def setUp(self):
        self.tenant_id = "test_tenant"
        self.collection_name = f"{self.tenant_id}_metadata"

    def test_ensure_collection_skips_existing(self):
        mock_qdrant_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = self.collection_name
        mock_qdrant_client.get_collections.return_value = MagicMock(
            collections=[mock_collection]
        )

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            QdrantIndexer(tenant_id=self.tenant_id)
            mock_qdrant_client.create_collection.assert_not_called()

    def test_ensure_collection_creates_new(self):
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collection.return_value = None

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            QdrantIndexer(tenant_id=self.tenant_id)
            mock_qdrant_client.create_collection.assert_called_once()
            # Verify tenant metadata was included
            call_args = mock_qdrant_client.create_collection.call_args[1]
            self.assertEqual(call_args["collection_name"], self.collection_name)
            self.assertEqual(call_args["metadata"]["tenant_id"], self.tenant_id)
            self.assertIn("created_at", call_args["metadata"])

    def test_delete_tenant_collection(self):
        mock_qdrant_client = MagicMock()
        
        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            indexer.delete_tenant_collection()
            mock_qdrant_client.delete_collection.assert_called_once_with(self.collection_name)

    def test_list_tenant_collections(self):
        mock_qdrant_client = MagicMock()
        mock_collections = MagicMock()
        mock_collection1 = MagicMock()
        mock_collection1.name = "tenant1_metadata"
        mock_collection1.metadata = {
            "tenant_id": "tenant1",
            "created_at": "2024-02-03T10:00:00"
        }
        mock_collection1.vectors_count = 100

        mock_collection2 = MagicMock()
        mock_collection2.name = "tenant2_metadata"
        mock_collection2.metadata = {
            "tenant_id": "tenant2",
            "created_at": "2024-02-03T11:00:00"
        }
        mock_collection2.vectors_count = 200

        mock_collections.collections = [mock_collection1, mock_collection2]
        mock_qdrant_client.get_collections.return_value = mock_collections

        collections = QdrantIndexer.list_tenant_collections(mock_qdrant_client)
        self.assertEqual(len(collections), 2)
        self.assertEqual(collections[0]["tenant_id"], "tenant1")
        self.assertEqual(collections[1]["tenant_id"], "tenant2")
        self.assertEqual(collections[0]["vectors_count"], 100)
        self.assertEqual(collections[1]["vectors_count"], 200)

    def test_index_metadata_processes_list(self):
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            metadata = {
                "buckets": [
                    {"id": "bucket1"},
                    {"id": "bucket2"},
                ]
            }
            indexer.index_metadata(metadata, mock_embedding_provider)

            assert mock_embedding_provider.embed.called
            assert mock_qdrant_client.upsert.called
            # Verify correct collection name is used
            call_args = mock_qdrant_client.upsert.call_args[1]
            self.assertEqual(call_args["collection_name"], self.collection_name)

    def test_index_metadata_processes_dict(self):
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            metadata = {
                "tables": {
                    "bucket1": [
                        {"id": "table1"},
                        {"id": "table2"},
                    ]
                }
            }
            indexer.index_metadata(metadata, mock_embedding_provider)

            assert mock_embedding_provider.embed.called
            assert mock_qdrant_client.upsert.called

    def test_prepare_text_for_embedding(self):
        mock_qdrant_client = MagicMock()
    
        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
    
            # Test table metadata
            table_item = {
                "name": "test",
                "description": "test desc",
                "type": "test type",
                "tags": ["tag1", "tag2"],
            }
            text = indexer._prepare_text_for_embedding(table_item, "tables")
            expected_text = "Name: test | Description: test desc"
            self.assertEqual(text, expected_text)

            # Test linked table metadata
            linked_table_item = {
                "name": "linked_test",
                "description": "linked desc",
                "isLinked": True,
                "sourceTable": {
                    "id": "src_123",
                    "project": {"id": "proj_456"}
                },
                "sourceTableDetails": {
                    "description": "source desc",
                    "isAccessible": True
                }
            }
            text = indexer._prepare_text_for_embedding(linked_table_item, "tables")
            expected_text = "Name: linked_test | Description: linked desc | Type: Linked Table | Source Table: src_123 | Source Project: proj_456 | Source Description: source desc"
            self.assertEqual(text, expected_text)

    def test_generate_point_id(self):
        mock_qdrant_client = MagicMock()
    
        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            item = {"id": "test_id"}
            point_id = indexer._generate_point_id("test_type", item)
    
            # Verify the point ID is a string and follows the expected format
            assert isinstance(point_id, str)
            # UUID format validation
            try:
                uuid.UUID(point_id)
                assert True
            except ValueError:
                assert False, "Point ID is not a valid UUID"

    def test_search_metadata(self):
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1] * 1536]  # Query embedding

        mock_result = ScoredPoint(
            id=1,
            version=1,
            score=0.9,
            payload={
                "metadata_type": "test_type",
                "raw_metadata": {"id": "test"},
            },
            vector=[0.1] * 1536,
        )
        mock_qdrant_client.search.return_value = [mock_result]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            results = indexer.search_metadata(
                "test query",
                mock_embedding_provider,
            )

            assert len(results) == 1
            assert results[0]["score"] == 0.9
            assert results[0]["metadata_type"] == "test_type"
            assert results[0]["metadata"]["id"] == "test"

    def test_prepare_transformation_text(self):
        mock_qdrant_client = MagicMock()

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            
            # Test transformation metadata
            transformation = {
                "name": "Customer Data Processing",
                "type": "python",
                "description": "Processes customer data",
                "dependencies": {
                    "requires": ["in.c-main.customers"],
                    "produces": ["out.c-processed.results"]
                },
                "runtime": {
                    "type": "docker",
                    "memory": "4g"
                }
            }
            text = indexer._prepare_transformation_text(transformation)
            
            # Verify essential components
            assert "Name: Customer Data Processing" in text
            assert "Type: python" in text
            assert "Description: Processes customer data" in text
            assert "Dependencies: Requires in.c-main.customers; Produces out.c-processed.results" in text
            assert "Runtime: docker, 4g memory" in text  # Accept both type and memory

    def test_prepare_block_text(self):
        mock_qdrant_client = MagicMock()

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            
            # Test block with inputs, outputs and code
            block = {
                "name": "Process Data",
                "inputs": [{"file": "customers.csv", "source": "in.c-main.customers"}],
                "outputs": [{"file": "results.csv", "destination": "out.c-processed.results"}],
                "code": """
                # Load and process customer data
                import pandas as pd
                df = pd.read_csv('customers.csv')
                df = df.merge(orders, on='customer_id')
                df.groupby('customer_id').agg({'amount': 'sum'})
                df.to_csv('results.csv')
                """
            }
            text = indexer._prepare_block_text(block)
            
            # Check each component separately to make debugging easier
            assert "Block: Process Data" in text
            assert "Inputs: customers.csv from in.c-main.customers" in text
            assert "Outputs: results.csv to out.c-processed.results" in text
            assert "Code: Load and process customer data" in text
            assert "read_csv" in text
            assert "merge" in text
            assert "groupby" in text
            assert "aggregate" in text
            assert "to_csv" in text

    def test_prepare_text_with_quality_metrics(self):
        mock_qdrant_client = MagicMock()

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            
            # Test column with quality metrics
            column = {
                "name": "email",
                "type": "VARCHAR",
                "description": "Customer email",
                "quality_metrics": {
                    "completeness": 0.98,
                    "validity": 0.95,
                    "standardization": {"standard": "RFC 5322"},
                    "common_issues": ["invalid_format", "missing_domain"],
                    "range_check": {
                        "min_valid": 5,
                        "max_valid": 255
                    }
                }
            }
            text = indexer._prepare_text_for_embedding(column, "columns")
            self.assertEqual(
                text,
                "Name: email | Type: VARCHAR | Description: Customer email | Quality: 98% complete, 95% valid, Valid range: 5-255, Standard: RFC 5322, Issues: invalid_format, missing_domain"
            )

    def test_batch_processing(self):
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        # Create mock embeddings with correct 1536 dimensions
        mock_embedding_provider.embed.return_value = [
            [0.1] * 1536,  # First item
            [0.2] * 1536,  # Second item
            [0.3] * 1536,  # Third item
        ]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            metadata = {
                "buckets": [
                    {"id": "bucket1", "name": "First Bucket"},
                    {"id": "bucket2", "name": "Second Bucket"},
                    {"id": "bucket3", "name": "Third Bucket"}
                ]
            }
            indexer.index_metadata(metadata, mock_embedding_provider)

            # Verify that batching was used correctly
            assert mock_embedding_provider.embed.call_count == 1
            assert mock_qdrant_client.upsert.call_count == 1

            # Verify batch
            batch_call = mock_qdrant_client.upsert.call_args_list[0]
            assert len(batch_call[1]["points"]) == 3
            
            # Verify vector dimensions
            for point in batch_call[1]["points"]:
                assert len(point.vector) == 1536

    def test_search_metadata_with_filters(self):
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2]]

        mock_result = ScoredPoint(
            id=1,
            version=1,
            score=0.9,
            payload={
                "metadata_type": "columns",
                "raw_metadata": {"name": "email", "type": "VARCHAR"},
            },
            vector=[0.1, 0.2],
        )
        mock_qdrant_client.search.return_value = [mock_result]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            results = indexer.search_metadata(
                "find email columns",
                mock_embedding_provider,
                metadata_type="columns",
                limit=5
            )

            # Verify search was called with correct parameters
            mock_qdrant_client.search.assert_called_once()
            search_args = mock_qdrant_client.search.call_args[1]
            self.assertEqual(search_args["collection_name"], self.collection_name)
            self.assertEqual(search_args["limit"], 5)
            self.assertEqual(
                search_args["query_filter"],
                Filter(
                    must=[
                        FieldCondition(
                            key="metadata_type",
                            match=MatchValue(value="columns")
                        )
                    ]
                )
            )

    def test_index_metadata_error_handling(self):
        """Test error handling in metadata processing."""
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.side_effect = Exception("Embedding error")

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            metadata = {
                "buckets": [{"id": "bucket1"}],
                "invalid_type": [{"id": "test"}]  # Invalid metadata type
            }
            
            # Test embedding error
            with pytest.raises(Exception) as exc_info:
                indexer.index_metadata(metadata, mock_embedding_provider)
            assert "Embedding error" in str(exc_info.value)

            # Test invalid metadata type
            mock_embedding_provider.embed.side_effect = None
            mock_embedding_provider.embed.return_value = [[0.1, 0.2]]
            indexer.index_metadata(metadata, mock_embedding_provider)
            # Should skip invalid metadata type without error

    def test_prepare_text_error_handling(self):
        """Test error handling in text preparation."""
        mock_qdrant_client = MagicMock()

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)

            # Test with empty metadata
            empty_metadata = {}
            text = indexer._prepare_text_for_embedding(empty_metadata, "unknown_type")
            assert text == "{}"

            # Test with metadata missing required fields
            incomplete_metadata = {"id": "test"}
            text = indexer._prepare_text_for_embedding(incomplete_metadata, "columns")
            assert "Name:" not in text

    def test_prepare_block_text_with_code_analysis(self):
        """Test block text preparation with code analysis."""
        mock_qdrant_client = MagicMock()

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            
            # Test block with complex code
            block = {
                "name": "Data Processing",
                "code": """
                # Import and clean data
                import pandas as pd
                import numpy as np

                # Load data
                df = pd.read_csv('input.csv')

                # Clean and transform
                df = df.fillna(0)
                df = df.drop_duplicates()
                df = df.rename(columns={'old': 'new'})
                df = df.sort_values('column')

                # Aggregate results
                result = df.groupby('category').agg({
                    'value': 'sum',
                    'count': 'count'
                })

                # Save output
                result.to_csv('output.csv')
                """
            }
            text = indexer._prepare_block_text(block)
            
            # Verify essential components
            assert "Block: Data Processing" in text
            assert "Import and clean data" in text
            assert "read_csv" in text
            assert "aggregate" in text
            assert "to_csv" in text

    def test_prepare_transformation_text_with_phases(self):
        """Test transformation text preparation with phases."""
        mock_qdrant_client = MagicMock()

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            
            # Test transformation with phases
            transformation = {
                "name": "ETL Pipeline",
                "type": "python",
                "description": "Complex ETL process",
                "phases": [
                    {
                        "name": "Extract",
                        "code": "# Extract data\ndf = pd.read_csv('source.csv')"
                    },
                    {
                        "name": "Transform",
                        "code": "# Transform data\ndf = df.groupby('id').sum()"
                    },
                    {
                        "name": "Load",
                        "code": "# Load data\ndf.to_csv('target.csv')"
                    }
                ],
                "dependencies": {
                    "requires": ["in.c-main.source"],
                    "produces": ["out.c-main.target"]
                }
            }
            text = indexer._prepare_transformation_text(transformation)
            
            # Verify essential components are included
            assert "Name: ETL Pipeline" in text
            assert "Type: python" in text
            assert "Description: Complex ETL process" in text
            assert "Dependencies: Requires in.c-main.source; Produces out.c-main.target" in text

    def test_batch_processing_with_retries(self):
        """Test batch processing with retries on failure."""
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            metadata = {
                "buckets": [
                    {"id": "bucket1", "name": "Test Bucket"}
                ]
            }
            
            # Configure client to fail twice then succeed
            fail_count = [0]
            def upsert_with_retries(*args, **kwargs):
                if fail_count[0] < 2:
                    fail_count[0] += 1
                    raise UnexpectedResponse(
                        status_code=500,
                        reason_phrase="Internal Server Error",
                        content=b"Test error",
                        headers={"Content-Type": "text/plain"},
                    )
            
            mock_qdrant_client.upsert.side_effect = upsert_with_retries
            
            # Configure retry settings
            indexer.batch_processor.config = BatchConfig(
                batch_size=1,
                max_retries=3,
                initial_retry_delay=0.01
            )
            
            # Index metadata
            indexer.index_metadata(metadata, mock_embedding_provider)
            
            # Verify retry attempts
            assert fail_count[0] == 2
            assert mock_qdrant_client.upsert.call_count == 3

    def test_batch_processing_memory_optimization(self):
        """Test memory optimization in batch processing."""
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            metadata = {
                "buckets": [{"id": f"bucket{i}", "name": f"Test Bucket {i}"} for i in range(100)]
            }
            
            # Configure small batch size
            indexer.batch_processor.config = BatchConfig(batch_size=10)
            
            # Index metadata
            indexer.index_metadata(metadata, mock_embedding_provider)
            
            # Verify batching
            assert mock_embedding_provider.embed.call_count == 10  # 100 items / 10 batch_size
            assert mock_qdrant_client.upsert.call_count == 10

    def test_batch_processing_progress_tracking(self):
        """Test progress tracking during batch processing."""
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [
            [0.1] * 1536,  # First item
            [0.2] * 1536,  # Second item
            [0.3] * 1536,  # Third item
        ]

        with patch("app.indexer.QdrantClient", return_value=mock_qdrant_client), \
             patch("app.batch_processor.tqdm") as mock_tqdm:
            mock_progress = MagicMock()
            mock_tqdm.return_value.__enter__.return_value = mock_progress
            
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            metadata = {
                "buckets": [{"id": f"bucket{i}"} for i in range(5)]
            }
            
            indexer.index_metadata(metadata, mock_embedding_provider)
            
            # Verify progress updates
            assert mock_progress.update.call_count == 1  # 5 items / 10 batch_size = 1 batch
            # Verify client calls
            assert mock_qdrant_client.upsert.called

    def test_prepare_text_formatting(self):
        """Test text preparation for different metadata types."""
        mock_qdrant_client = MagicMock()

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            
            # Test column text preparation
            column = {
                "name": "test_column",
                "type": "string",
                "description": "Test description",
                "statistics": {"unique_count": 100},
                "quality": {"completeness": 0.95}
            }
            column_text = indexer._prepare_column_text(column)
            assert "Name: test_column" in column_text
            assert "Type: string" in column_text
            assert "Description: Test description" in column_text
            assert "Statistics: unique=100" in column_text

            # Test transformation text preparation
            transformation = {
                "name": "test_transform",
                "type": "python",
                "description": "Test transformation",
                "dependencies": {
                    "requires": ["in.c-main.source"],
                    "produces": ["out.c-main.target"]
                }
            }
            transform_text = indexer._prepare_transformation_text(transformation)
            assert "Name: test_transform" in transform_text
            assert "Type: python" in transform_text
            assert "Description: Test transformation" in transform_text
            assert "Dependencies: Requires in.c-main.source; Produces out.c-main.target" in transform_text

    def test_custom_collection_name(self):
        mock_qdrant_client = MagicMock()
        custom_collection = "custom_collection"
        
        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id, collection_name=custom_collection)
            self.assertEqual(indexer.collection_name, custom_collection)

    def test_tenant_isolation(self):
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2]]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            # Create two tenants
            tenant1_indexer = QdrantIndexer(tenant_id="tenant1")
            tenant2_indexer = QdrantIndexer(tenant_id="tenant2")

            # Index data for each tenant
            tenant1_data = {"buckets": [{"id": "tenant1_bucket"}]}
            tenant2_data = {"buckets": [{"id": "tenant2_bucket"}]}

            tenant1_indexer.index_metadata(tenant1_data, mock_embedding_provider)
            tenant2_indexer.index_metadata(tenant2_data, mock_embedding_provider)

            # Verify each tenant's data was indexed to their respective collections
            tenant1_calls = [
                call for call in mock_qdrant_client.upsert.call_args_list 
                if call[1]["collection_name"] == "tenant1_metadata"
            ]
            tenant2_calls = [
                call for call in mock_qdrant_client.upsert.call_args_list 
                if call[1]["collection_name"] == "tenant2_metadata"
            ]

            self.assertEqual(len(tenant1_calls), 1)
            self.assertEqual(len(tenant2_calls), 1)

    def test_tenant_collection_deletion(self):
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.delete_collection.return_value = True

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            result = indexer.delete_tenant_collection()
            self.assertTrue(result)
            mock_qdrant_client.delete_collection.assert_called_once_with(self.collection_name)

    def test_tenant_collection_deletion_error(self):
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.delete_collection.side_effect = UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"Collection not found",
            headers={"Content-Type": "text/plain"},
        )

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            with self.assertRaises(UnexpectedResponse):
                indexer.delete_tenant_collection()

    def test_tenant_metadata_in_points(self):
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2]]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer(tenant_id=self.tenant_id)
            metadata = {"buckets": [{"id": "test_bucket"}]}
            indexer.index_metadata(metadata, mock_embedding_provider)

            # Verify tenant_id is included in point payload
            upsert_calls = mock_qdrant_client.upsert.call_args_list
            for call in upsert_calls:
                points = call[1]["points"]
                for point in points:
                    self.assertEqual(point.payload.get("tenant_id"), self.tenant_id)
