"""
Base test classes for storage backends.
"""

import pytest
from datetime import datetime
from typing import Type, Optional
from uuid import uuid4

from omen.storage.base import VectorStoreBackend, OntologyStoreBackend
from omen.vectorstore.models import MetadataDocument, SearchQuery, MetadataSource, MetadataType
from omen.ontology.models import Entity, Relationship, EntityType, RelationshipType


class BaseVectorStoreTest:
    """Base test class for vector store backends."""

    # Override in subclass
    backend_class: Optional[Type[VectorStoreBackend]] = None

    @pytest.fixture
    def store(self):
        """Create a test store instance."""
        if not self.backend_class:
            pytest.skip("No backend class specified")
        return self.backend_class()

    @pytest.fixture
    def test_document(self):
        """Create a test document."""
        return MetadataDocument(
            id=str(uuid4()),
            content="Test document content",
            embedding=[0.1] * 1536,  # Default OpenAI dimension
            source=MetadataSource(
                type=MetadataType.CUSTOM,
                id="test-1",
                project_id="test-project",
            ),
            metadata={
                "test_key": "test_value"
            }
        )

    def test_store_and_retrieve_document(self, store, test_document):
        """Test storing and retrieving a document."""
        # Store document
        doc_id = store.store_document(test_document)
        assert doc_id == test_document.id

        # Retrieve document
        retrieved = store.get_document(doc_id)
        assert retrieved is not None
        assert retrieved.id == test_document.id
        assert retrieved.content == test_document.content
        assert retrieved.metadata == test_document.metadata

    def test_store_multiple_documents(self, store, test_document):
        """Test storing multiple documents."""
        docs = [
            test_document,
            MetadataDocument(
                id=str(uuid4()),
                content="Another test document",
                embedding=[0.2] * 1536,
                source=MetadataSource(
                    type=MetadataType.CUSTOM,
                    id="test-2",
                    project_id="test-project",
                ),
            )
        ]
        
        # Store documents
        doc_ids = store.store_documents(docs)
        assert len(doc_ids) == 2
        assert doc_ids[0] == docs[0].id
        assert doc_ids[1] == docs[1].id

        # Retrieve and verify
        for doc in docs:
            retrieved = store.get_document(doc.id)
            assert retrieved is not None
            assert retrieved.id == doc.id
            assert retrieved.content == doc.content

    def test_delete_document(self, store, test_document):
        """Test deleting a document."""
        # Store and verify document exists
        doc_id = store.store_document(test_document)
        assert store.get_document(doc_id) is not None

        # Delete document
        success = store.delete_document(doc_id)
        assert success is True
        assert store.get_document(doc_id) is None

    def test_update_metadata(self, store, test_document):
        """Test updating document metadata."""
        # Store document
        doc_id = store.store_document(test_document)

        # Update metadata
        new_metadata = {"new_key": "new_value"}
        success = store.update_metadata(doc_id, new_metadata)
        assert success is True

        # Verify update
        retrieved = store.get_document(doc_id)
        assert retrieved is not None
        assert "new_key" in retrieved.metadata
        assert retrieved.metadata["new_key"] == "new_value"
        assert "test_key" in retrieved.metadata  # Original metadata should remain

    def test_search(self, store, test_document):
        """Test searching for documents."""
        # Store test document
        store.store_document(test_document)

        # Search with string query
        results = store.search("test document", limit=1)
        assert len(results) > 0
        assert results[0].document.id == test_document.id

        # Search with SearchQuery object
        query = SearchQuery(
            query="test document",
            filter={"test_key": "test_value"}
        )
        results = store.search(query, limit=1)
        assert len(results) > 0
        assert results[0].document.id == test_document.id


class BaseOntologyStoreTest:
    """Base test class for ontology store backends."""

    # Override in subclass
    backend_class: Optional[Type[OntologyStoreBackend]] = None

    @pytest.fixture
    def store(self):
        """Create a test store instance."""
        if not self.backend_class:
            pytest.skip("No backend class specified")
        return self.backend_class()

    @pytest.fixture
    def test_entity(self):
        """Create a test entity."""
        return Entity(
            id=str(uuid4()),
            type=EntityType.TABLE,
            name="Test Entity",
            description="Test entity description",
            properties={
                "test_prop": "test_value"
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

    @pytest.fixture
    def test_relationship(self, test_entity):
        """Create a test relationship."""
        target_entity = Entity(
            id=str(uuid4()),
            type=EntityType.COLUMN,
            name="Target Entity",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Store target entity first
        self.backend_class().store_entity(target_entity)
        
        return Relationship(
            id=str(uuid4()),
            type=RelationshipType.HAS_COLUMN,
            source_id=test_entity.id,
            target_id=target_entity.id,
            properties={
                "test_prop": "test_value"
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

    def test_store_and_retrieve_entity(self, store, test_entity):
        """Test storing and retrieving an entity."""
        # Store entity
        entity_id = store.store_entity(test_entity)
        assert entity_id == test_entity.id

        # Retrieve entity
        retrieved = store.get_entity(entity_id)
        assert retrieved is not None
        assert retrieved.id == test_entity.id
        assert retrieved.name == test_entity.name
        assert retrieved.type == test_entity.type
        assert retrieved.properties == test_entity.properties

    def test_store_and_retrieve_relationship(self, store, test_entity, test_relationship):
        """Test storing and retrieving a relationship."""
        # Store source entity
        store.store_entity(test_entity)

        # Store relationship
        rel_id = store.store_relationship(test_relationship)
        assert rel_id == test_relationship.id

        # Retrieve relationship
        retrieved = store.get_relationship(rel_id)
        assert retrieved is not None
        assert retrieved.id == test_relationship.id
        assert retrieved.type == test_relationship.type
        assert retrieved.source_id == test_relationship.source_id
        assert retrieved.target_id == test_relationship.target_id
        assert retrieved.properties == test_relationship.properties

    def test_delete_entity(self, store, test_entity):
        """Test deleting an entity."""
        # Store entity
        entity_id = store.store_entity(test_entity)
        assert store.get_entity(entity_id) is not None

        # Delete entity
        success = store.delete_entity(entity_id)
        assert success is True
        assert store.get_entity(entity_id) is None

    def test_delete_relationship(self, store, test_entity, test_relationship):
        """Test deleting a relationship."""
        # Store entities and relationship
        store.store_entity(test_entity)
        rel_id = store.store_relationship(test_relationship)
        assert store.get_relationship(rel_id) is not None

        # Delete relationship
        success = store.delete_relationship(rel_id)
        assert success is True
        assert store.get_relationship(rel_id) is None

    def test_get_connected_entities(self, store, test_entity, test_relationship):
        """Test getting connected entities."""
        # Store entities and relationship
        store.store_entity(test_entity)
        store.store_relationship(test_relationship)

        # Get connected entities
        connected = store.get_connected_entities(test_entity.id)
        assert len(connected) > 0
        assert connected[0][0].id == test_relationship.target_id

        # Test with relationship type filter
        connected = store.get_connected_entities(
            test_entity.id,
            relationship_types=[RelationshipType.HAS_COLUMN.value]
        )
        assert len(connected) > 0

        # Test with direction
        outgoing = store.get_connected_entities(test_entity.id, direction="out")
        assert len(outgoing) > 0
        incoming = store.get_connected_entities(test_entity.id, direction="in")
        assert len(incoming) == 0

    def test_query(self, store, test_entity):
        """Test querying the ontology."""
        # Store test entity
        store.store_entity(test_entity)

        # Execute a simple SPARQL query
        results = store.query("""
            SELECT ?entity ?name
            WHERE {
                ?entity rdf:type ?type .
                ?entity rdfs:label ?name .
            }
        """)
        assert len(results) > 0
        assert any(r.get("name") == test_entity.name for r in results) 