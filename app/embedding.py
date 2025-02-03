"""Embedding providers for vector generation."""
import os
from typing import List, Union
import logging

from openai import OpenAI

class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Convert texts to vector embeddings."""
        raise NotImplementedError

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-ada-002."""
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        """Initialize OpenAI embedding provider.
        
        Args:
            model: OpenAI embedding model to use
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Convert texts to vector embeddings using OpenAI.
        
        Args:
            texts: List of texts to convert to embeddings
            
        Returns:
            List of embedding vectors
        """
        try:
            # Get embeddings from OpenAI
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            # Extract embedding vectors
            embeddings = [data.embedding for data in response.data]
            
            return embeddings
            
        except Exception as e:
            logging.error(f"Error getting embeddings from OpenAI: {str(e)}")
            raise
