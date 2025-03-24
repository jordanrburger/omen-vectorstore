import unittest
from unittest.mock import Mock, patch
import tempfile
import os
from rdflib import URIRef, Literal

from app.config import Config
from app.main import (
    get_embedding_provider,
    extract_metadata,
    search_metadata,
    build_ontology,
    build_action_graph,
    index_command,
    search_command,
    query_ontology_command
)
from app.ontology.manager import OntologyManager
from app.ontology.rdf_store import RDFStore
from app.ontology.action_graph import ActionGraph


class TestMain(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config(
            keboola_api_url="https://api.keboola.com",
            keboola_token="test_token",
            qdrant_collection="test_collection",
            embedding_model="test-model",
            device="cpu"
        )
        self.batch_config = Mock()
        self.batch_config.batch_size = 10
        self.batch_config.max_retries = 3
        self.batch_config.initial_retry_delay = 1.0

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch("app.main.OpenAIProvider")
    def test_get_embedding_provider_openai(self, mock_openai):
        self.config.openai_api_key = "test_key"
        provider = get_embedding_provider(self.config)
        mock_openai.assert_called_once_with(
            api_key="test_key",
            model="test-model"
        )
        self.assertEqual(provider, mock_openai.return_value)

    @patch("app.main.SentenceTransformerProvider")
    def test_get_embedding_provider_sentence_transformer(self, mock_st):
        self.config.openai_api_key = None
        provider = get_embedding_provider(self.config)
        mock_st.assert_called_once_with(
            model_name="test-model",
            device="cpu"
        )
        self.assertEqual(provider, mock_st.return_value)

    @patch("app.main.KeboolaClient")
    def test_extract_metadata(self, mock_client):
        # Mock client methods
        mock_client.return_value.list_buckets.return_value = [
            {"id": "bucket1", "name": "Test Bucket"}
        ]
        mock_client.return_value.list_tables.return_value = {
            "bucket1": [
                {"id": "table1", "name": "Test Table"}
            ]
        }
        mock_client.return_value.get_table_details.return_value = {
            "id": "table1",
            "columns": [
                {"name": "col1", "type": "string"}
            ]
        }
        mock_client.return_value.list_configurations.return_value = [
            {"id": "config1", "name": "Test Config"}
        ]

        metadata = extract_metadata(mock_client.return_value)

        self.assertEqual(len(metadata["buckets"]), 1)
        self.assertEqual(len(metadata["tables"]["bucket1"]), 1)
        self.assertEqual(len(metadata["table_details"]), 1)
        self.assertEqual(len(metadata["configurations"]), 1)

    @patch("app.main.QdrantIndexer")
    @patch("app.main.get_embedding_provider")
    def test_search_metadata(self, mock_get_provider, mock_indexer):
        mock_provider = Mock()
        mock_get_provider.return_value = mock_provider
        mock_indexer.return_value.search_metadata.return_value = [
            {
                "id": "result1",
                "score": 0.95,
                "metadata_type": "table",
                "name": "Test Table",
                "description": "Test Description"
            }
        ]

        results = search_metadata(
            query="test",
            indexer=mock_indexer.return_value,
            embedding_provider=mock_provider,
            metadata_type="table",
            component_type="extractor",
            table_id="table1",
            stage="in",
            limit=5
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "result1")
        mock_indexer.return_value.search_metadata.assert_called_once()

    @patch("app.main.KeboolaClient")
    def test_build_ontology(self, mock_client):
        # Mock client methods
        mock_client.return_value.extract_ontology_metadata.return_value = {
            "entities": [
                {
                    "id": "entity1",
                    "type": "Bucket",
                    "properties": {
                        "name": "Test Bucket",
                        "description": "Test Description"
                    }
                }
            ],
            "relationships": [
                {
                    "source_id": "entity1",
                    "target_id": "entity2",
                    "type": "CONTAINS"
                }
            ]
        }

        ontology_manager, rdf_store = build_ontology(mock_client.return_value)

        self.assertIsInstance(ontology_manager, OntologyManager)
        self.assertIsInstance(rdf_store, RDFStore)
        self.assertEqual(len(ontology_manager.entities), 1)
        self.assertEqual(len(ontology_manager.relationships), 1)

    @patch("app.main.OntologyManager")
    def test_build_action_graph(self, mock_ontology_manager):
        mock_manager = Mock()
        mock_ontology_manager.return_value = mock_manager
        mock_manager.entities = {"entity1": Mock()}
        mock_manager.relationships = {"rel1": Mock()}

        action_graph = build_action_graph(mock_manager)

        self.assertIsInstance(action_graph, ActionGraph)
        mock_manager.assert_called_once()

    @patch("app.main.KeboolaClient")
    @patch("app.main.QdrantIndexer")
    @patch("app.main.get_embedding_provider")
    @patch("app.main.build_ontology")
    @patch("app.main.build_action_graph")
    @patch("app.main.StateManager")
    def test_index_command(
        self,
        mock_state_manager,
        mock_build_action_graph,
        mock_build_ontology,
        mock_get_provider,
        mock_indexer,
        mock_client
    ):
        # Mock components
        mock_state_manager.return_value = Mock()
        mock_build_ontology.return_value = (Mock(), Mock())
        mock_build_action_graph.return_value = Mock()
        mock_get_provider.return_value = Mock()
        mock_indexer.return_value = Mock()
        mock_client.return_value = Mock()

        # Execute command
        index_command(self.config, self.batch_config)

        # Verify calls
        mock_client.return_value.extract_metadata.assert_called_once()
        mock_indexer.return_value.index_metadata.assert_called_once()
        mock_build_ontology.assert_called_once()
        mock_build_action_graph.assert_called_once()
        mock_state_manager.return_value.save_ontology_state.assert_called_once()
        mock_state_manager.return_value.save_rdf_state.assert_called_once()
        mock_state_manager.return_value.save_action_graph_state.assert_called_once()

    @patch("app.main.QdrantIndexer")
    @patch("app.main.get_embedding_provider")
    def test_search_command(self, mock_get_provider, mock_indexer):
        mock_provider = Mock()
        mock_get_provider.return_value = mock_provider
        mock_indexer.return_value.search_metadata.return_value = [
            {
                "id": "result1",
                "score": 0.95,
                "metadata_type": "table",
                "name": "Test Table",
                "description": "Test Description"
            }
        ]

        with patch("builtins.print") as mock_print:
            search_command(
                self.config,
                "test",
                metadata_type="table",
                component_type="extractor",
                table_id="table1",
                stage="in",
                limit=5
            )

            mock_print.assert_called()
            mock_indexer.return_value.search_metadata.assert_called_once()

    @patch("app.main.StateManager")
    def test_query_ontology_command(self, mock_state_manager):
        # Mock RDF store
        mock_rdf_store = Mock()
        mock_rdf_store.query.return_value = [
            Mock(vars=["var1", "var2"]),
            Mock(vars=["var1", "var2"])
        ]
        mock_rdf_store.query.return_value[0]["var1"] = URIRef("http://example.org/entity1")
        mock_rdf_store.query.return_value[0]["var2"] = Literal("Test Value")
        mock_state_manager.return_value.load_rdf_state.return_value = mock_rdf_store

        with patch("builtins.print") as mock_print:
            query_ontology_command(self.config, "SELECT ?var1 ?var2 WHERE { ?var1 ?var2 ?var3 }")

            mock_print.assert_called()
            mock_rdf_store.query.assert_called_once()

    @patch("app.main.StateManager")
    def test_query_ontology_command_no_store(self, mock_state_manager):
        mock_state_manager.return_value.load_rdf_state.return_value = None

        with patch("app.main.logging.error") as mock_error:
            query_ontology_command(self.config, "SELECT ?var1 ?var2 WHERE { ?var1 ?var2 ?var3 }")
            mock_error.assert_called_once_with("No RDF store found. Please run the index command first.")


if __name__ == "__main__":
    unittest.main() 