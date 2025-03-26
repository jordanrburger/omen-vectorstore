"""
Embedding providers for vectorizing text.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from omen.core import get_logger
from omen.core.config import settings

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for the given texts."""
        pass

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Alias for embed method."""
        return self.embed(texts)


class SentenceTransformerProvider(EmbeddingProvider):
    """SentenceTransformer embedding provider."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the SentenceTransformer provider with model name."""
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized SentenceTransformer provider with model: {model_name}")

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings using SentenceTransformer."""
        if isinstance(texts, str):
            texts = [texts]
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the OpenAI provider with API key and model."""
        self.api_key = api_key or settings.openai.api_key
        self.model = model or settings.openai.embedding_model
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI provider with model: {self.model}")

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings using the OpenAI API."""
        if isinstance(texts, str):
            texts = [texts]
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Generated embeddings with shape: {len(embeddings)}x{len(embeddings[0])}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


def get_embedding_provider(provider_name: str = "openai") -> EmbeddingProvider:
    """Factory function to get an embedding provider."""
    if provider_name.lower() == "openai":
        return OpenAIProvider()
    elif provider_name.lower() == "sentence_transformer":
        return SentenceTransformerProvider()
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}") 