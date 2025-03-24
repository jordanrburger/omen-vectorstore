"""
Tests for LLM-powered ontology utilities.
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from app.ontology.llm_utils import (
    SemanticMatcher,
    NLQueryConverter,
    OntologyExplainer,
    OntologyValidator
)
from app.ontology.models import Entity, Relationship
from app.ontology.schema import EntityType, RelationshipType
from app.ontology.rdf_store import RDFStore

class TestSemanticMatcher(unittest.TestCase):
    """Test semantic entity matching functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.matcher = SemanticMatcher()
        self.test_entities = [
            Entity(
                name="users_table",
                type=EntityType.TABLE,
                properties={"description": "User information"}
            ),
            Entity(
                name="orders_table",
                type=EntityType.TABLE,
                properties={"description": "Order information"}
            ),
            Entity(
                name="user_id",
                type=EntityType.COLUMN,
                properties={"description": "User identifier"}
            )
        ]
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_initialization(self, mock_transformer):
        """Test semantic matcher initialization."""
        matcher = SemanticMatcher(model_name="test-model")
        mock_transformer.assert_called_once_with("test-model")
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_embedding(self, mock_transformer):
        """Test embedding generation."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([1.0, 2.0, 3.0])
        mock_transformer.return_value = mock_model
        
        matcher = SemanticMatcher()
        embedding = matcher.get_embedding("test text")
        
        mock_model.encode.assert_called_once_with("test text")
        np.testing.assert_array_equal(embedding, np.array([1.0, 2.0, 3.0]))
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_find_similar_entities(self, mock_transformer):
        """Test finding similar entities."""
        # Mock embeddings
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.array([1.0, 0.0, 0.0]),  # query embedding
            np.array([1.0, 0.0, 0.0]),  # users_table embedding
            np.array([0.0, 1.0, 0.0]),  # orders_table embedding
            np.array([0.0, 0.0, 1.0])   # user_id embedding
        ]
        mock_transformer.return_value = mock_model
        
        matcher = SemanticMatcher()
        matches = matcher.find_similar_entities(
            "user data",
            self.test_entities,
            top_k=2,
            threshold=0.5
        )
        
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0].name, "users_table")
        self.assertGreater(matches[0][1], 0.5)

class TestNLQueryConverter(unittest.TestCase):
    """Test natural language to SPARQL conversion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_rdf_store = Mock(spec=RDFStore)
        self.converter = NLQueryConverter(self.mock_rdf_store)
    
    @patch('openai.ChatCompletion')
    def test_convert_to_sparql(self, mock_openai):
        """Test converting natural language to SPARQL."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="SELECT * WHERE { ?s ?p ?o }"))]
        mock_openai.create.return_value = mock_response
        
        query = "Find all tables"
        sparql = self.converter.convert_to_sparql(query)
        
        self.assertEqual(sparql, "SELECT * WHERE { ?s ?p ?o }")
        mock_openai.create.assert_called_once()
    
    def test_execute_nl_query(self):
        """Test executing a natural language query."""
        self.mock_rdf_store.query.return_value = [{"result": "test"}]
        
        results = self.converter.execute_nl_query("Find all tables")
        
        self.assertEqual(results, [{"result": "test"}])
        self.mock_rdf_store.query.assert_called_once()

class TestOntologyExplainer(unittest.TestCase):
    """Test ontology explanation generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_rdf_store = Mock(spec=RDFStore)
        self.explainer = OntologyExplainer(self.mock_rdf_store)
        
        self.source = Entity(
            name="users_table",
            type=EntityType.TABLE,
            properties={"description": "User information"}
        )
        self.target = Entity(
            name="user_id",
            type=EntityType.COLUMN,
            properties={"description": "User identifier"}
        )
        self.relationship = Relationship(
            type=RelationshipType.CONTAINS,
            properties={"description": "Table contains column"}
        )
    
    @patch('openai.ChatCompletion')
    def test_explain_relationship(self, mock_openai):
        """Test generating relationship explanations."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test explanation"))]
        mock_openai.create.return_value = mock_response
        
        explanation = self.explainer.explain_relationship(
            self.source,
            self.target,
            self.relationship
        )
        
        self.assertEqual(explanation, "Test explanation")
        mock_openai.create.assert_called_once()
    
    @patch('openai.ChatCompletion')
    def test_explain_action_chain(self, mock_openai):
        """Test generating action chain explanations."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test explanation"))]
        mock_openai.create.return_value = mock_response
        
        actions = [
            {"type": "LOAD_DATA", "description": "Load user data"},
            {"type": "TRANSFORM_DATA", "description": "Transform user data"}
        ]
        
        explanation = self.explainer.explain_action_chain(
            self.source,
            self.target,
            actions
        )
        
        self.assertEqual(explanation, "Test explanation")
        mock_openai.create.assert_called_once()

class TestOntologyValidator(unittest.TestCase):
    """Test ontology validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_rdf_store = Mock(spec=RDFStore)
        self.validator = OntologyValidator(self.mock_rdf_store)
        
        self.test_entity = Entity(
            name="test_table",
            type=EntityType.TABLE,
            properties={"description": "Test table"}
        )
        self.test_relationship = Relationship(
            type=RelationshipType.CONTAINS,
            properties={"description": "Contains relationship"}
        )
    
    def test_validate_entity(self):
        """Test entity validation."""
        issues = self.validator.validate_entity(self.test_entity)
        self.assertIsInstance(issues, list)
    
    def test_validate_relationship(self):
        """Test relationship validation."""
        source = Entity(name="source", type=EntityType.TABLE)
        target = Entity(name="target", type=EntityType.COLUMN)
        
        issues = self.validator.validate_relationship(
            self.test_relationship,
            source,
            target
        )
        self.assertIsInstance(issues, list)
    
    def test_validate_property_type(self):
        """Test property type validation."""
        result = self.validator._validate_property_type(
            "description",
            "test",
            EntityType.TABLE
        )
        self.assertTrue(result)
    
    def test_is_valid_relationship_type(self):
        """Test relationship type validation."""
        result = self.validator._is_valid_relationship_type(
            RelationshipType.CONTAINS,
            EntityType.TABLE,
            EntityType.COLUMN
        )
        self.assertTrue(result) 