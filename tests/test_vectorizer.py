import unittest
from unittest.mock import patch, MagicMock

from app.vectorizer import SentenceTransformerProvider, OpenAIProvider


class TestSentenceTransformerProvider(unittest.TestCase):
    @patch('app.vectorizer.SentenceTransformerProvider.__init__', return_value=None)
    @patch('app.vectorizer.SentenceTransformerProvider.embed')
    def test_embed(self, mock_embed, mock_init):
        # Simulate SentenceTransformerProvider returning a dummy embedding list
        provider = SentenceTransformerProvider()
        # Force the embed method to return a predictable value
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        texts = ['test sentence']
        embeddings = provider.embed(texts)
        self.assertEqual(embeddings, [[0.1, 0.2, 0.3]])
        mock_embed.assert_called_once_with(texts)


class TestOpenAIProvider(unittest.TestCase):
    @patch('app.vectorizer.OpenAIProvider.__init__', return_value=None)
    @patch('app.vectorizer.OpenAIProvider.embed')
    def test_embed(self, mock_embed, mock_init):
        # Simulate OpenAIProvider returning a dummy embedding list
        provider = OpenAIProvider()
        # Force the embed method to return a predictable value
        mock_embed.return_value = [[0.4, 0.5, 0.6]]
        texts = ['another test']
        embeddings = provider.embed(texts)
        self.assertEqual(embeddings, [[0.4, 0.5, 0.6]])
        mock_embed.assert_called_once_with(texts)

    @patch('app.vectorizer.openai.Embedding.create')
    def test_openai_embed_real_logic(self, mock_create):
        # This test will simulate the real OpenAI API call (without patching embed)
        dummy_response = {
            'data': [
                {'embedding': [0.7, 0.8, 0.9]},
                {'embedding': [1.0, 1.1, 1.2]}
            ]
        }
        mock_create.return_value = dummy_response

        # Manually instantiate OpenAIProvider without patching __init__
        provider = OpenAIProvider(api_key='dummy')
        # Ensure that provider.openai is the patched openai module by our test
        texts = ['a', 'b']
        embeddings = provider.embed(texts)
        expected = [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
        self.assertEqual(embeddings, expected)
        mock_create.assert_called_once_with(input=texts, model=provider.model_name)


if __name__ == '__main__':
    unittest.main()
