"""
Tests for RDFLib ontology store backend.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from omen.storage.ontology.rdflib import RDFLibStore
from tests.storage.test_base import BaseOntologyStoreTest


class TestRDFLibStore(BaseOntologyStoreTest):
    """Test RDFLib store implementation."""
    
    backend_class = RDFLibStore

    @pytest.fixture(autouse=True)
    def setup_storage(self):
        """Setup temporary storage directory."""
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        storage_path = Path(temp_dir) / "test_ontology.ttl"
        
        # Initialize store with temp path
        store = RDFLibStore(storage_path=storage_path)
        yield store
        
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_storage_path_creation(self, setup_storage):
        """Test storage path is created."""
        store = setup_storage
        assert store.storage_path.parent.exists()
        assert str(store.storage_path).endswith(".ttl")

    def test_namespace_initialization(self, setup_storage):
        """Test namespaces are properly initialized."""
        store = setup_storage
        namespaces = dict(store.graph.namespaces())
        
        assert "kbc" in namespaces
        assert "kbc-entity" in namespaces
        assert "kbc-rel" in namespaces
        assert "kbc-prop" in namespaces
        assert "rdfs" in namespaces
        assert "owl" in namespaces

    def test_save_and_load(self, setup_storage, test_entity):
        """Test saving and loading the graph."""
        store = setup_storage
        
        # Store an entity and save
        store.store_entity(test_entity)
        store.save()
        
        # Create new store instance with same path
        new_store = RDFLibStore(storage_path=store.storage_path)
        
        # Verify entity was loaded
        loaded_entity = new_store.get_entity(test_entity.id)
        assert loaded_entity is not None
        assert loaded_entity.id == test_entity.id
        assert loaded_entity.name == test_entity.name

    def test_ontology_axioms(self, setup_storage):
        """Test basic ontology axioms are present."""
        store = setup_storage
        
        # Query for entity and relationship classes
        results = store.query("""
            SELECT ?class ?label
            WHERE {
                ?class rdf:type owl:Class .
                ?class rdfs:label ?label .
            }
        """)
        
        # Verify basic classes exist
        class_labels = [r["label"] for r in results]
        assert "Entity" in class_labels
        assert "Relationship" in class_labels
        
        # Verify entity types are defined
        results = store.query("""
            SELECT ?class
            WHERE {
                ?class rdfs:subClassOf kbc:Entity .
            }
        """)
        assert len(results) > 0  # Should have multiple entity types

        # Verify relationship types are defined
        results = store.query("""
            SELECT ?rel
            WHERE {
                ?rel rdfs:subPropertyOf kbc:relationship .
            }
        """)
        assert len(results) > 0  # Should have multiple relationship types 