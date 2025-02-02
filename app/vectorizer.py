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
    """OpenAI embedding provider."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """Initialize the OpenAI provider with API key and model."""
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logging.info("Initialized OpenAI provider with model: %s", model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the OpenAI API."""
        if isinstance(texts, str):
            texts = [texts]
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise


class MetadataVectorizer:
    """Handles vectorization of metadata documents."""

    def __init__(self, embedding_provider: EmbeddingProvider):
        """Initialize with an embedding provider."""
        self.embedding_provider = embedding_provider

    def _prepare_text(self, metadata: dict) -> str:
        """Prepare metadata for vectorization by converting to text."""
        parts = []
        
        # Add type and name if available
        if "type" in metadata:
            parts.append(f"Type: {metadata['type']}")
        if "name" in metadata:
            parts.append(f"Name: {metadata['name']}")
            
        # Add description if available
        if "description" in metadata:
            parts.append(f"Description: {metadata['description']}")
            
        # Add columns information if available
        if "columns" in metadata and isinstance(metadata["columns"], list):
            column_info = []
            for col in metadata["columns"]:
                if isinstance(col, dict):
                    col_name = col.get("name", "")
                    col_type = col.get("type", "")
                    if col_name and col_type:
                        column_info.append(f"{col_name} ({col_type})")
            if column_info:
                parts.append("Columns: " + ", ".join(column_info))
                
        # Add row count if available
        if "row_count" in metadata:
            parts.append(f"Row count: {metadata['row_count']}")
            
        # Add creation date if available
        if "created" in metadata:
            parts.append(f"Created: {metadata['created']}")
            
        # Combine all parts with newlines
        return "\n".join(parts)

    def vectorize(self, metadata: dict) -> List[float]:
        """Convert metadata to vector representation."""
        text = self._prepare_text(metadata)
        vectors = self.embedding_provider.embed([text])
        return vectors[0]  # Return the first (and only) vector

    def vectorize_batch(self, metadata_list: List[dict]) -> List[List[float]]:
        """Convert a batch of metadata to vector representations."""
        texts = [self._prepare_text(metadata) for metadata in metadata_list]
        return self.embedding_provider.embed(texts)
