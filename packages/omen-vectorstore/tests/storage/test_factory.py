"""
Tests for storage factory.
"""

import pytest

from omen.storage.factory import StorageFactory
from omen.storage.base import VectorStoreBackend, OntologyStoreBackend, UnifiedStorageBackend
from omen.storage.vector.qdrant import QdrantVectorStore
from omen.storage.ontology.rdflib import RDFLibStore


class TestStorageFactory:
    """Test storage factory functionality."""

    def test_register_vector_backend(self):
        """Test registering a new vector store backend."""
        class TestVectorStore(VectorStoreBackend):
            def store_document(self, document): pass
            def store_documents(self, documents): pass
            def get_document(self, document_id): pass
            def delete_document(self, document_id): pass
            def search(self, query, limit=10, offset=0, **kwargs): pass
            def update_metadata(self, document_id, metadata): pass

        StorageFactory.register_vector_backend("test", TestVectorStore)
        assert "test" in StorageFactory._vector_backends
        assert StorageFactory._vector_backends["test"] == TestVectorStore

    def test_register_ontology_backend(self):
        """Test registering a new ontology store backend."""
        class TestOntologyStore(OntologyStoreBackend):
            def store_entity(self, entity): pass
            def store_relationship(self, relationship): pass
            def get_entity(self, entity_id): pass
            def get_relationship(self, relationship_id): pass
            def delete_entity(self, entity_id): pass
            def delete_relationship(self, relationship_id): pass
            def query(self, query, query_type="sparql", **kwargs): pass
            def get_connected_entities(self, entity_id, relationship_types=None, direction="both"): pass

        StorageFactory.register_ontology_backend("test", TestOntologyStore)
        assert "test" in StorageFactory._ontology_backends
        assert StorageFactory._ontology_backends["test"] == TestOntologyStore

    def test_register_unified_backend(self):
        """Test registering a new unified store backend."""
        class TestUnifiedStore(UnifiedStorageBackend):
            def store_document(self, document): pass
            def store_documents(self, documents): pass
            def get_document(self, document_id): pass
            def delete_document(self, document_id): pass
            def search(self, query, limit=10, offset=0, **kwargs): pass
            def update_metadata(self, document_id, metadata): pass
            def store_entity(self, entity): pass
            def store_relationship(self, relationship): pass
            def get_entity(self, entity_id): pass
            def get_relationship(self, relationship_id): pass
            def delete_entity(self, entity_id): pass
            def delete_relationship(self, relationship_id): pass
            def query(self, query, query_type="sparql", **kwargs): pass
            def get_connected_entities(self, entity_id, relationship_types=None, direction="both"): pass
            def semantic_graph_search(self, query, start_nodes=None, max_distance=2, limit=10): pass

        StorageFactory.register_unified_backend("test", TestUnifiedStore)
        assert "test" in StorageFactory._unified_backends
        assert StorageFactory._unified_backends["test"] == TestUnifiedStore

    def test_create_vector_store(self):
        """Test creating vector store instances."""
        # Test default backend
        store = StorageFactory.create_vector_store()
        assert isinstance(store, QdrantVectorStore)

        # Test with custom config
        store = StorageFactory.create_vector_store(
            backend_type="qdrant",
            collection_name="test_collection"
        )
        assert isinstance(store, QdrantVectorStore)
        assert store.collection_name == "test_collection"

        # Test invalid backend
        with pytest.raises(ValueError):
            StorageFactory.create_vector_store(backend_type="invalid")

    def test_create_ontology_store(self):
        """Test creating ontology store instances."""
        # Test default backend
        store = StorageFactory.create_ontology_store()
        assert isinstance(store, RDFLibStore)

        # Test with custom config
        store = StorageFactory.create_ontology_store(
            backend_type="rdflib",
            storage_path="test.ttl"
        )
        assert isinstance(store, RDFLibStore)
        assert str(store.storage_path).endswith("test.ttl")

        # Test invalid backend
        with pytest.raises(ValueError):
            StorageFactory.create_ontology_store(backend_type="invalid")

    def test_create_unified_store(self):
        """Test creating unified store instances."""
        # Currently no unified backends
        with pytest.raises(ValueError):
            StorageFactory.create_unified_store(backend_type="test")

    def test_get_available_backends(self):
        """Test getting information about available backends."""
        backends = StorageFactory.get_available_backends()
        
        assert "vector" in backends
        assert "qdrant" in backends["vector"]
        assert isinstance(backends["vector"]["qdrant"], str)
        
        assert "ontology" in backends
        assert "rdflib" in backends["ontology"]
        assert isinstance(backends["ontology"]["rdflib"], str)
        
        assert "unified" in backends
        assert len(backends["unified"]) == 0  # No unified backends yet 