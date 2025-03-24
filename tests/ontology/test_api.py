"""
Tests for ontology API endpoints.
"""

import unittest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from app.ontology.api import router
from app.ontology.models import Entity, Relationship
from app.ontology.schema import EntityType, RelationshipType
from app.ontology.rdf_store import RDFStore
from app.ontology.action_graph import ActionGraph
from app.ontology.llm_utils import (
    SemanticMatcher,
    NLQueryConverter,
    OntologyExplainer,
    OntologyValidator
)

class TestOntologyAPI(unittest.TestCase):
    """Test ontology API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(router)
        
        # Mock dependencies
        self.mock_rdf_store = Mock(spec=RDFStore)
        self.mock_action_graph = Mock(spec=ActionGraph)
        self.mock_semantic_matcher = Mock(spec=SemanticMatcher)
        self.mock_nl_converter = Mock(spec=NLQueryConverter)
        self.mock_explainer = Mock(spec=OntologyExplainer)
        self.mock_validator = Mock(spec=OntologyValidator)
        
        # Test data
        self.test_entity = Entity(
            name="test_table",
            type=EntityType.TABLE,
            properties={"description": "Test table"}
        )
        self.test_relationship = Relationship(
            type=RelationshipType.CONTAINS,
            source_id="source_id",
            target_id="target_id",
            properties={"description": "Contains relationship"}
        )
        self.test_action = {
            "id": "action_id",
            "type": "LOAD_DATA",
            "source_id": "source_id",
            "target_id": "target_id",
            "description": "Load data",
            "properties": {}
        }
    
    def test_list_entities(self):
        """Test listing entities endpoint."""
        self.mock_rdf_store.find_all_entities.return_value = [self.test_entity]
        
        response = self.client.get("/ontology/entities")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["name"], "test_table")
    
    def test_get_entity(self):
        """Test getting a specific entity endpoint."""
        self.mock_rdf_store.find_entity_by_id.return_value = self.test_entity
        
        response = self.client.get("/ontology/entities/test_id")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "test_table")
    
    def test_get_entity_not_found(self):
        """Test getting a non-existent entity endpoint."""
        self.mock_rdf_store.find_entity_by_id.return_value = None
        
        response = self.client.get("/ontology/entities/nonexistent")
        
        self.assertEqual(response.status_code, 404)
    
    def test_list_relationships(self):
        """Test listing relationships endpoint."""
        self.mock_rdf_store.find_all_relationships.return_value = [self.test_relationship]
        
        response = self.client.get("/ontology/relationships")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["type"], "CONTAINS")
    
    def test_get_relationship(self):
        """Test getting a specific relationship endpoint."""
        self.mock_rdf_store.find_relationship_by_id.return_value = self.test_relationship
        self.mock_rdf_store.find_entity_by_id.side_effect = [
            Entity(name="source", type=EntityType.TABLE),
            Entity(name="target", type=EntityType.COLUMN)
        ]
        
        response = self.client.get("/ontology/relationships/test_id")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["type"], "CONTAINS")
    
    def test_list_actions(self):
        """Test listing actions endpoint."""
        self.mock_action_graph.get_all_actions.return_value = [self.test_action]
        
        response = self.client.get("/ontology/actions")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["type"], "LOAD_DATA")
    
    def test_get_action(self):
        """Test getting a specific action endpoint."""
        self.mock_action_graph.get_action.return_value = self.test_action
        
        response = self.client.get("/ontology/actions/test_id")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["type"], "LOAD_DATA")
    
    def test_get_action_chain(self):
        """Test getting action chain endpoint."""
        self.mock_action_graph.get_action_chain.return_value = [self.test_action]
        self.mock_rdf_store.find_entity_by_id.side_effect = [
            Entity(name="source", type=EntityType.TABLE),
            Entity(name="target", type=EntityType.COLUMN)
        ]
        
        response = self.client.get("/ontology/action-chains?source_id=source_id&target_id=target_id")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["type"], "LOAD_DATA")
    
    def test_semantic_search(self):
        """Test semantic search endpoint."""
        self.mock_rdf_store.find_all_entities.return_value = [self.test_entity]
        self.mock_semantic_matcher.find_similar_entities.return_value = [
            (self.test_entity, 0.8)
        ]
        
        response = self.client.post(
            "/ontology/search",
            json={"query": "test", "top_k": 5, "threshold": 0.7}
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["name"], "test_table")
    
    def test_natural_language_query(self):
        """Test natural language query endpoint."""
        self.mock_nl_converter.execute_nl_query.return_value = [{"result": "test"}]
        
        response = self.client.post(
            "/ontology/query",
            json={"query": "Find all tables"}
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["result"], "test")
    
    def test_explain_relationship(self):
        """Test relationship explanation endpoint."""
        self.mock_rdf_store.find_relationship_by_id.return_value = self.test_relationship
        self.mock_rdf_store.find_entity_by_id.side_effect = [
            Entity(name="source", type=EntityType.TABLE),
            Entity(name="target", type=EntityType.COLUMN)
        ]
        self.mock_explainer.explain_relationship.return_value = "Test explanation"
        
        response = self.client.get("/ontology/explain/relationship/test_id")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["explanation"], "Test explanation")
    
    def test_explain_action_chain(self):
        """Test action chain explanation endpoint."""
        self.mock_action_graph.get_action_chain.return_value = [self.test_action]
        self.mock_rdf_store.find_entity_by_id.side_effect = [
            Entity(name="source", type=EntityType.TABLE),
            Entity(name="target", type=EntityType.COLUMN)
        ]
        self.mock_explainer.explain_action_chain.return_value = "Test explanation"
        
        response = self.client.get(
            "/ontology/explain/action-chain?source_id=source_id&target_id=target_id"
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["explanation"], "Test explanation")
    
    def test_validate_entity(self):
        """Test entity validation endpoint."""
        self.mock_rdf_store.find_entity_by_id.return_value = self.test_entity
        self.mock_validator.validate_entity.return_value = []
        
        response = self.client.get("/ontology/validate/entity/test_id")
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["valid"])
    
    def test_validate_relationship(self):
        """Test relationship validation endpoint."""
        self.mock_rdf_store.find_relationship_by_id.return_value = self.test_relationship
        self.mock_rdf_store.find_entity_by_id.side_effect = [
            Entity(name="source", type=EntityType.TABLE),
            Entity(name="target", type=EntityType.COLUMN)
        ]
        self.mock_validator.validate_relationship.return_value = []
        
        response = self.client.get("/ontology/validate/relationship/test_id")
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["valid"]) 