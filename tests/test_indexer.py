from unittest.mock import MagicMock, patch

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import ScoredPoint

from app.indexer import QdrantIndexer


class TestQdrantIndexer:
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
            item = {
                "name": "test",
                "description": "test desc",
                "type": "test type",
                "tags": ["tag1", "tag2"],
            }
            text = indexer._prepare_text_for_embedding(item)

            assert "Name: test" in text
            assert "Description: test desc" in text
            assert "Type: test type" in text
            assert "Tags: tag1, tag2" in text

    def test_generate_point_id(self):
        mock_qdrant_client = MagicMock()

        with patch(
            "app.indexer.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            indexer = QdrantIndexer()
            item = {"id": "test_id"}
            point_id = indexer._generate_point_id("test_type", item)

            assert isinstance(point_id, int)
            assert point_id > 0

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
