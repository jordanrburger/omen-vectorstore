import os
import logging
from typing import Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv


def get_env_or_default(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable or default value."""
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    return value


class Config(BaseModel):
    """Configuration model for the application."""
    keboola_token: str = Field(default="")
    keboola_api_url: str = Field(default="https://connection.keboola.com")
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=55000)
    qdrant_collection: str = Field(default="keboola_metadata")
    openai_api_key: Optional[str] = Field(default=None)
    embedding_model: str = Field(default="text-embedding-ada-002")
    device: str = Field(default="cpu")

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        load_dotenv()  # Load .env file
        return cls(
            keboola_token=get_env_or_default("KEBOOLA_TOKEN", ""),
            keboola_api_url=get_env_or_default("KEBOOLA_API_URL", "https://connection.keboola.com"),
            qdrant_host=get_env_or_default("QDRANT_HOST", "localhost"),
            qdrant_port=int(get_env_or_default("QDRANT_PORT", "55000")),
            qdrant_collection=get_env_or_default("QDRANT_COLLECTION", "keboola_metadata"),
            openai_api_key=get_env_or_default("OPENAI_API_KEY"),
            embedding_model=get_env_or_default("OPENAI_MODEL", "text-embedding-ada-002"),
            device=get_env_or_default("DEVICE", "cpu"),
        )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.model_dump()


def load_config() -> Dict:
    """Load configuration from environment variables."""
    load_dotenv()  # Load .env file
    token = get_env_or_default("KEBOOLA_TOKEN")
    if not token:
        raise ValueError("KEBOOLA_TOKEN environment variable is required")

    return {
        "KEBOOLA_API_URL": get_env_or_default("KEBOOLA_API_URL", "https://connection.keboola.com"),
        "KEBOOLA_TOKEN": token,
        "OPENAI_API_KEY": get_env_or_default("OPENAI_API_KEY"),
        "OPENAI_MODEL": get_env_or_default("OPENAI_MODEL", "text-embedding-ada-002"),
        "SENTENCE_TRANSFORMER_MODEL": get_env_or_default("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"),
        "QDRANT_HOST": get_env_or_default("QDRANT_HOST", "localhost"),
        "QDRANT_PORT": int(get_env_or_default("QDRANT_PORT", "55000")),
        "QDRANT_COLLECTION": get_env_or_default("QDRANT_COLLECTION", "keboola_metadata"),
    }
