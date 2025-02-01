from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.indexer import QdrantIndexer
from app.vectorizer import SentenceTransformerProvider


@pytest.fixture
def mock_embedding_provider():
    provider = MagicMock(spec=SentenceTransformerProvider)
    provider.embed.return_value = [[0.1, 0.2, 0.3]]  # Mock embedding
    return provider


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock(spec=QdrantClient)
    # Mock collection operations
    client.get_collections.return_value = MagicMock(collections=[])
    client.create_collection.return_value = None
    client.create_payload_index.return_value = None
    # Mock search operations
    client.search.return_value = [
        models.ScoredPoint(
            id=1,
            version=1,
            score=0.9,
            payload={
                "metadata_type": "tables",
                "raw_metadata": {
                    "name": "test_table",
                    "description": "test description",
                },
            },
        )
    ]
    return client


@pytest.fixture
def indexer(mock_qdrant_client):
    indexer = QdrantIndexer()
    indexer.client = mock_qdrant_client
    return indexer
