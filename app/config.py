import os
import logging
from pathlib import Path
from typing import Dict, Optional
from subprocess import run, PIPE

from pydantic import BaseModel

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    # Use shell to source the .env file
    env_content = env_path.read_text()
    env_vars = dict(
        line.split('=', 1)
        for line in env_content.splitlines()
        if line.strip() and not line.startswith('#')
    )
    os.environ.update(env_vars)
else:
    logging.warning(f".env file not found at {env_path}")


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
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logging.warning("OPENAI_API_KEY not found in environment variables")
            
        return cls(
            storage_api_token=os.getenv("KBC_STORAGE_TOKEN", ""),
            storage_api_url=os.getenv(
                "KBC_STORAGE_URL",
                "https://connection.keboola.com/v2/storage",
            ),
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "55000")),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "keboola_metadata"),
            openai_api_key=openai_api_key,
            embedding_model=os.getenv("OPENAI_MODEL", "text-embedding-ada-002"),
            device=os.getenv("DEVICE", "cpu"),
        )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.model_dump()


def load_config():
    """Load configuration from environment variables."""
    config = {
        # Keboola configuration
        "KEBOOLA_API_URL": os.getenv(
            "KEBOOLA_API_URL", "https://connection.keboola.com"
        ),
        "KEBOOLA_TOKEN": os.getenv("KEBOOLA_TOKEN"),
        # OpenAI configuration (optional)
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "text-embedding-ada-002"),
        # Sentence Transformer configuration (optional)
        "SENTENCE_TRANSFORMER_MODEL": os.getenv(
            "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
        ),
        # Qdrant configuration
        "QDRANT_HOST": os.getenv("QDRANT_HOST", "localhost"),
        "QDRANT_PORT": int(os.getenv("QDRANT_PORT", "55000")),
        "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "keboola_metadata"),
    }

    # Validate required configuration
    if not config["KEBOOLA_TOKEN"]:
        raise ValueError("KEBOOLA_TOKEN environment variable is required")

    return config
