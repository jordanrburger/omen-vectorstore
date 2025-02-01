import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.vectorizer import OpenAIProvider, SentenceTransformerProvider


class TestSentenceTransformerProvider(unittest.TestCase):
    @patch(
        "app.vectorizer.SentenceTransformerProvider.__init__",
        return_value=None,
    )
    @patch("app.vectorizer.SentenceTransformerProvider.embed")
    def test_embed(self, mock_embed, mock_init):
        provider = SentenceTransformerProvider()
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        texts = ["test sentence"]
        embeddings = provider.embed(texts)
        self.assertEqual(embeddings, [[0.1, 0.2, 0.3]])
        mock_embed.assert_called_once_with(texts)

    def test_sentence_transformer_embed(self):
        provider = SentenceTransformerProvider()
        texts = ["This is a test", "Another test"]
        embeddings = provider.embed(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384  # Default model dimension
        assert isinstance(embeddings[0][0], float)

    def test_sentence_transformer_with_custom_model(self):
        """Test SentenceTransformer with a custom model."""
        with patch("app.vectorizer.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            mock_st.return_value = mock_model

            provider = SentenceTransformerProvider(
                model_name="all-mpnet-base-v2",
                device="cuda"
            )
            texts = ["This is a test", "Another test"]
            embeddings = provider.embed(texts)

            # Verify model initialization
            mock_st.assert_called_once_with("all-mpnet-base-v2", device="cuda")
            mock_model.encode.assert_called_once_with(
                texts,
                convert_to_tensor=False,
                normalize_embeddings=True
            )

            # Verify embeddings
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 3
            np.testing.assert_array_equal(
                embeddings,
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            )

    def test_sentence_transformer_initialization_error(self):
        """Test error handling during SentenceTransformer initialization."""
        with patch("app.vectorizer.SentenceTransformer") as mock_st:
            mock_st.side_effect = Exception("Model not found")
            
            with pytest.raises(Exception) as exc_info:
                provider = SentenceTransformerProvider(model_name="invalid-model")
            assert "Model not found" in str(exc_info.value)

    def test_sentence_transformer_device_error(self):
        """Test error handling when setting device fails."""
        with patch("app.vectorizer.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.side_effect = Exception("CUDA not available")
            mock_st.return_value = mock_model

            provider = SentenceTransformerProvider(device="cuda")
            texts = ["test text"]
            
            with pytest.raises(Exception) as exc_info:
                provider.embed(texts)
            
            assert "CUDA not available" in str(exc_info.value)


class TestOpenAIProvider(unittest.TestCase):
    @patch("app.vectorizer.OpenAIProvider.__init__", return_value=None)
    @patch("app.vectorizer.OpenAIProvider.embed")
    def test_embed(self, mock_embed, mock_init):
        provider = OpenAIProvider()
        mock_embed.return_value = [[0.4, 0.5, 0.6]]
        texts = ["another test"]
        embeddings = provider.embed(texts)
        self.assertEqual(embeddings, [[0.4, 0.5, 0.6]])
        mock_embed.assert_called_once_with(texts)

    @patch("app.vectorizer.OpenAI")
    def test_openai_embed_real_logic(self, mock_openai_cls):
        # Create mock embeddings response
        embeddings = [
            list(np.random.random(1536)),
            list(np.random.random(1536)),
        ]

        mock_response = MagicMock()
        mock_response.model = "text-embedding-3-small"
        mock_response.object = "list"
        mock_response.data = [
            MagicMock(
                object="embedding",
                embedding=embeddings[0],
                index=0,
            ),
            MagicMock(
                object="embedding",
                embedding=embeddings[1],
                index=1,
            ),
        ]
        mock_response.usage = {
            "prompt_tokens": 10,
            "total_tokens": 10,
        }

        # Setup mock client
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        # Test the provider
        provider = OpenAIProvider(api_key="test-key")
        texts = ["This is a test", "Another test"]
        result_embeddings = provider.embed(texts)

        # Verify results
        assert len(result_embeddings) == 2
        assert len(result_embeddings[0]) == 1536
        assert isinstance(result_embeddings[0][0], float)
        assert result_embeddings == embeddings

        # Verify API call
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=texts,
        )

    def test_openai_embed_with_custom_model(self):
        """Test OpenAI embeddings with a custom model."""
        # Create mock embeddings response
        embeddings = [
            list(np.random.random(1536)),
            list(np.random.random(1536)),
        ]

        mock_response = MagicMock()
        mock_response.model = "text-embedding-3-large"
        mock_response.object = "list"
        mock_response.data = [
            MagicMock(
                object="embedding",
                embedding=embeddings[0],
                index=0,
            ),
            MagicMock(
                object="embedding",
                embedding=embeddings[1],
                index=1,
            ),
        ]
        mock_response.usage = {
            "prompt_tokens": 15,
            "total_tokens": 15,
        }

        # Setup mock client
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch("app.vectorizer.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="test-key", model="text-embedding-3-large")
            texts = ["This is a test", "Another test"]
            result_embeddings = provider.embed(texts)

            # Verify results
            assert len(result_embeddings) == 2
            assert len(result_embeddings[0]) == 1536
            assert result_embeddings == embeddings

            # Verify API call with custom model
            mock_client.embeddings.create.assert_called_once_with(
                model="text-embedding-3-large",
                input=texts,
            )

    def test_openai_embed_error_handling(self):
        """Test error handling in OpenAI embeddings."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("API Error")

        with patch("app.vectorizer.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="test-key")
            with pytest.raises(Exception) as exc_info:
                provider.embed(["Test text"])
            assert "API Error" in str(exc_info.value)

    def test_openai_initialization_error(self):
        """Test error handling during OpenAI client initialization."""
        with patch("app.vectorizer.OpenAI") as mock_openai:
            mock_openai.side_effect = Exception("Invalid API key")
            
            with pytest.raises(Exception) as exc_info:
                provider = OpenAIProvider(api_key="invalid-key")
            assert "Invalid API key" in str(exc_info.value)


if __name__ == "__main__":
    unittest.main()
