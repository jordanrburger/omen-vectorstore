import unittest
from unittest.mock import MagicMock, patch

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
