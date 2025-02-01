import os
import logging
from typing import Dict, Optional
from pydantic import BaseModel


class Config(BaseModel):
    """Configuration model for the application."""
    storage_api_token: str
    storage_api_url: str = "https://connection.keboola.com/v2/storage"
    qdrant_host: str = "localhost"
    qdrant_port: int = 55000
    qdrant_collection: str = "keboola_metadata"
    openai_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-ada-002"
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            storage_api_token=os.getenv("KEBOOLA_TOKEN", ""),
            storage_api_url=os.getenv("KEBOOLA_API_URL", "https://connection.keboola.com/v2/storage"),
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "55000")),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "keboola_metadata"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            embedding_model=os.getenv("OPENAI_MODEL", "text-embedding-ada-002"),
            device=os.getenv("DEVICE", "cpu"),
        )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.model_dump()


def load_config() -> Dict:
    """Load configuration from environment variables."""
    if not os.getenv("KEBOOLA_TOKEN"):
        raise ValueError("KEBOOLA_TOKEN environment variable is required")

    return {
        "KEBOOLA_API_URL": os.getenv("KEBOOLA_API_URL", "https://connection.keboola.com"),
        "KEBOOLA_TOKEN": os.getenv("KEBOOLA_TOKEN"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "text-embedding-ada-002"),
        "SENTENCE_TRANSFORMER_MODEL": os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"),
        "QDRANT_HOST": os.getenv("QDRANT_HOST", "localhost"),
        "QDRANT_PORT": int(os.getenv("QDRANT_PORT", "55000")),
        "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "keboola_metadata"),
    }
