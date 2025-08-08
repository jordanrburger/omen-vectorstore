"""
Tests for Qdrant vector store backend.
"""

import pytest
from omen.storage.vector.qdrant import QdrantVectorStore
from tests.storage.test_base import BaseVectorStoreTest


class TestQdrantVectorStore(BaseVectorStoreTest):
    """Test Qdrant vector store implementation."""
    
    backend_class = QdrantVectorStore

    @pytest.fixture(autouse=True)
    def setup_collection(self, store):
        """Setup test collection and cleanup after test."""
        # Collection is created in store initialization
        yield
        # Cleanup
        store.client.delete_collection(collection_name=store.collection_name)

    def test_collection_creation(self, store):
        """Test collection is created with correct settings."""
        collections = store.client.get_collections().collections
        collection_names = [c.name for c in collections]
        assert store.collection_name in collection_names

        # Get collection info
        collection_info = store.client.get_collection(store.collection_name)
        assert collection_info.config.params.vectors.size == store.vector_size
        assert collection_info.config.params.vectors.distance == "Cosine"

    def test_payload_indexes(self, store):
        """Test payload indexes are created."""
        collection_info = store.client.get_collection(store.collection_name)
        indexes = collection_info.config.params.indexed
        assert any(idx.field_name == "source.type" for idx in indexes)
        assert any(idx.field_name == "source.project_id" for idx in indexes) 