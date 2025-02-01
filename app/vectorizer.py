import logging
from abc import ABC, abstractmethod
from typing import List

from openai import OpenAI
from sentence_transformers import SentenceTransformer


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformer based embedding provider."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """Initialize the provider with a model."""
        self.model = SentenceTransformer(model_name, device=device)
        logging.info(
            f"Initialized SentenceTransformer with model {model_name} "
            f"on device {device}"
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the sentence transformer model."""
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=False,
                normalize_embeddings=True,
            )
            return embeddings.tolist()
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise


class OpenAIProvider(EmbeddingProvider):
    """OpenAI API based embedding provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ):
        """Initialize the provider with API key and model."""
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logging.info(f"Initialized OpenAI provider with model {model}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise
