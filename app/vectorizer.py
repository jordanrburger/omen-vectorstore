import abc

class EmbeddingProvider(abc.ABC):
    @abc.abstractmethod
    def embed(self, texts):
        """Convert a list of texts to a list of embeddings."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        """Return embeddings for the input texts using SentenceTransformer."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()  # convert to list for consistency


class OpenAIProvider(EmbeddingProvider):
    def __init__(self, model_name='text-embedding-ada-002', api_key=None):
        import openai
        self.openai = openai
        self.model_name = model_name
        self.api_key = api_key
        if api_key:
            self.openai.api_key = api_key

    def embed(self, texts):
        """Return embeddings for the input texts using OpenAI's embedding API."""
        # Call OpenAI's embedding API
        response = self.openai.Embedding.create(input=texts, model=self.model_name)
        embeddings = [datum['embedding'] for datum in response['data']]
        return embeddings
