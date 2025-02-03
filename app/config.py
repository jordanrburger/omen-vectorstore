"""Configuration management."""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Application configuration."""
    
    # Keboola configuration
    keboola_api_url: str
    keboola_token: str
    
    # OpenAI configuration
    openai_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-ada-002"
    
    # Qdrant configuration
    qdrant_host: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "keboola_metadata"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            keboola_api_url=os.getenv("KEBOOLA_API_URL", ""),
            keboola_token=os.getenv("KEBOOLA_TOKEN", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            embedding_model=os.getenv("OPENAI_MODEL", "text-embedding-ada-002"),
            qdrant_host=os.getenv("QDRANT_HOST", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "keboola_metadata"),
        )
