"""Text embedding providers."""
import os
from typing import List
import logging

from openai import OpenAI
from sentence_transformers import SentenceTransformer

class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Convert texts to vector embeddings."""
        raise NotImplementedError

class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        """Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logging.error(f"Error getting embeddings from OpenAI: {e}")
            raise

class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformer embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """Initialize Sentence Transformer provider.
        
        Args:
            model_name: Model name to use
            device: Device to use (cpu/cuda)
        """
        self.model = SentenceTransformer(model_name, device=device)
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from Sentence Transformer."""
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logging.error(f"Error getting embeddings from Sentence Transformer: {e}")
            raise
