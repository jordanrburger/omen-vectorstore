import unittest
from unittest.mock import MagicMock, patch
import uuid
import pytest

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import ScoredPoint, Filter, FieldCondition, MatchValue

from app.indexer import QdrantIndexer


class TestQdrantIndexer(unittest.TestCase):
    def test_ensure_collection_skips_existing(self):
        mock_qdrant_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "keboola_metadata"
        mock_qdrant_client.get_collections.return_value = MagicMock(
            collections=[mock_collection]
        )

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            QdrantIndexer()
            mock_qdrant_client.create_collection.assert_not_called()

    def test_ensure_collection_creates_new(self):
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            QdrantIndexer()
            mock_qdrant_client.create_collection.assert_called_once()

    def test_ensure_collection_handles_storage_error(self):
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])
        mock_qdrant_client.create_collection.side_effect = UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal Server Error",
            content=b"No space left on device",
            headers={"Content-Type": "text/plain"},
        )

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            try:
                QdrantIndexer()
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "Not enough disk space" in str(e)

    def test_index_metadata_processes_list(self):
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer()
            metadata = {
                "buckets": [
                    {"id": "bucket1"},
                    {"id": "bucket2"},
                ]
            }
            indexer.index_metadata(metadata, mock_embedding_provider)

            assert mock_embedding_provider.embed.called
            assert mock_qdrant_client.upsert.called

    def test_index_metadata_processes_dict(self):
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer()
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
            indexer = QdrantIndexer()
            
            # Test table metadata
            table_item = {
                "name": "test",
                "description": "test desc",
                "type": "test type",
                "tags": ["tag1", "tag2"],
            }
            text = indexer._prepare_text_for_embedding(table_item, "tables")
            self.assertEqual(
                text,
                "Name: test | Description: test desc | Type: test type | Tags: tag1, tag2"
            )
            
            # Test column metadata
            column_item = {
                "name": "test_column",
                "type": "INTEGER",
                "description": "test column desc",
            }
            text = indexer._prepare_text_for_embedding(column_item, "columns")
            self.assertEqual(
                text,
                "Name: test_column | Type: INTEGER | Description: test column desc"
            )

    def test_generate_point_id(self):
        mock_qdrant_client = MagicMock()
    
        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer()
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
        mock_embedding_provider.embed.return_value = [[0.1, 0.2]]

        mock_result = ScoredPoint(
            id=1,
            version=1,
            score=0.9,
            payload={
                "metadata_type": "test_type",
                "raw_metadata": {"id": "test"},
            },
            vector=[0.1, 0.2],
        )
        mock_qdrant_client.search.return_value = [mock_result]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer()
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
            indexer = QdrantIndexer()
            
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
            self.assertEqual(
                text,
                "Name: Customer Data Processing | Type: python | Description: Processes customer data | Dependencies: Requires in.c-main.customers; Produces out.c-processed.results | Runtime: 4g memory"
            )

    def test_prepare_block_text(self):
        mock_qdrant_client = MagicMock()

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer()
            
            # Test block with inputs, outputs and code
            block = {
                "name": "Process Data",
                "inputs": [{"source": "in.c-main.customers", "destination": "customers.csv"}],
                "outputs": [{"source": "results.csv", "destination": "out.c-processed.results"}],
                "code": """
                # Load and process customer data
                import pandas as pd
                df = pd.read_csv('customers.csv')
                df = df.merge(orders, on='customer_id')
                df.groupby('customer_id').agg({'amount': 'sum'})
                df.to_csv('results.csv')
                """
            }
            text = indexer._prepare_block_text(block, "trans_123")
            self.assertEqual(
                text,
                "Block: Process Data | Inputs: customers.csv from in.c-main.customers | Outputs: results.csv to out.c-processed.results | Code: Load and process customer data, read_csv, to_csv, merge, groupby, aggregate"
            )

    def test_prepare_text_with_quality_metrics(self):
        mock_qdrant_client = MagicMock()

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer()
            
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
        mock_embedding_provider.embed.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer()
            metadata = {
                "buckets": [
                    {"id": "bucket1", "name": "First Bucket"},
                    {"id": "bucket2", "name": "Second Bucket"},
                    {"id": "bucket3", "name": "Third Bucket"}
                ]
            }
            indexer.index_metadata(metadata, mock_embedding_provider, batch_size=2)

            # Verify that batching was used correctly
            self.assertEqual(mock_embedding_provider.embed.call_count, 2)
            self.assertEqual(mock_qdrant_client.upsert.call_count, 2)

            # Verify first batch
            first_batch_call = mock_qdrant_client.upsert.call_args_list[0]
            self.assertEqual(len(first_batch_call[1]["points"]), 2)

            # Verify second batch
            second_batch_call = mock_qdrant_client.upsert.call_args_list[1]
            self.assertEqual(len(second_batch_call[1]["points"]), 1)

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
            indexer = QdrantIndexer()
            results = indexer.search_metadata(
                "find email columns",
                mock_embedding_provider,
                metadata_type="columns",
                limit=5
            )

            # Verify search was called with correct parameters
            mock_qdrant_client.search.assert_called_once()
            search_args = mock_qdrant_client.search.call_args[1]
            self.assertEqual(search_args["collection_name"], "keboola_metadata")
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
            indexer = QdrantIndexer()
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
            indexer = QdrantIndexer()

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
            indexer = QdrantIndexer()
            
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
            text = indexer._prepare_block_text(block, "trans_123")
            
            # Verify essential operations are included
            assert "Block: Data Processing" in text
            assert "Code: Import and clean data" in text
            assert "read_csv" in text
            assert "to_csv" in text
            assert "groupby" in text
            assert "aggregate" in text

    def test_prepare_transformation_text_with_phases(self):
        """Test transformation text preparation with phases."""
        mock_qdrant_client = MagicMock()

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer()
            
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

    def test_index_items_retry_logic(self):
        """Test retry logic in _index_items when storage errors occur."""
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

        # Mock storage error on first attempt
        mock_qdrant_client.upsert.side_effect = [
            UnexpectedResponse(
                status_code=500,
                reason_phrase="Internal Server Error",
                content=b"No space left on device",
                headers={"Content-Type": "text/plain"},
            ),
            None  # Success on second attempt with smaller batch
        ]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer()
            items = [{"id": "item1"}, {"id": "item2"}]
            indexer._index_items(items, "test_type", mock_embedding_provider, batch_size=2)

            # Verify retry with smaller batch
            assert mock_qdrant_client.upsert.call_count == 2
            first_call = mock_qdrant_client.upsert.call_args_list[0]
            second_call = mock_qdrant_client.upsert.call_args_list[1]
            assert len(first_call[1]["points"]) == 2
            assert len(second_call[1]["points"]) == 1

    def test_index_transformation_metadata(self):
        """Test indexing of transformation metadata."""
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer()
            metadata = {
                "transformations": {
                    "trans1": {
                        "name": "ETL Process",
                        "type": "python",
                        "description": "Data transformation",
                        "blocks": [
                            {
                                "name": "Extract",
                                "code": "df = pd.read_csv('input.csv')"
                            },
                            {
                                "name": "Load",
                                "code": "df.to_csv('output.csv')"
                            }
                        ]
                    }
                }
            }
            indexer._index_transformation_metadata(metadata, mock_embedding_provider)

            # Verify embeddings were generated for transformation and blocks
            mock_embedding_provider.embed.assert_called_once()
            texts = mock_embedding_provider.embed.call_args[0][0]
            assert len(texts) == 3  # 1 transformation + 2 blocks

            # Verify points were created and upserted
            mock_qdrant_client.upsert.assert_called_once()
            points = mock_qdrant_client.upsert.call_args[1]["points"]
            assert len(points) == 3
            assert points[0].payload["metadata_type"] == "transformations"
            assert points[1].payload["metadata_type"] == "transformation_blocks"
            assert points[2].payload["metadata_type"] == "transformation_blocks"

    def test_index_items_error_handling(self):
        """Test error handling in _index_items."""
        mock_qdrant_client = MagicMock()
        mock_embedding_provider = MagicMock()
        
        # Test embedding error
        mock_embedding_provider.embed.side_effect = Exception("Embedding failed")
        
        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer()
            items = [{"id": "item1"}, {"id": "item2"}]
            
            with pytest.raises(Exception) as exc_info:
                indexer._index_items(items, "test_type", mock_embedding_provider, batch_size=2)
            assert "Embedding failed" in str(exc_info.value)
            
            # Test upsert error
            mock_embedding_provider.embed.side_effect = None
            mock_embedding_provider.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
            mock_qdrant_client.upsert.side_effect = Exception("Upsert failed")
            
            with pytest.raises(Exception) as exc_info:
                indexer._index_items(items, "test_type", mock_embedding_provider, batch_size=2)
            assert "Upsert failed" in str(exc_info.value)
