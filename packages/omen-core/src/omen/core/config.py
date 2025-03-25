"""
Configuration management for the OMEN platform.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class OpenAISettings(BaseModel):
    """OpenAI API settings."""
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field("gpt-4", description="OpenAI model to use")
    embedding_model: str = Field("text-embedding-3-large", description="OpenAI embedding model to use")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: int = Field(2000, description="Maximum tokens to generate")
    embedding_dimension: int = Field(1536, description="Dimension of embeddings")


class QdrantSettings(BaseModel):
    """Qdrant settings."""
    host: str = Field("localhost", description="Qdrant host")
    port: int = Field(6333, description="Qdrant port")
    grpc_port: int = Field(6334, description="Qdrant gRPC port")
    prefer_grpc: bool = Field(True, description="Prefer gRPC over HTTP")
    collection_name: str = Field("keboola_metadata", description="Qdrant collection name")


class KeboolaSettings(BaseModel):
    """Keboola connection settings."""
    url: str = Field(..., description="Keboola Storage API URL")
    token: str = Field(..., description="Keboola Storage API token")


class OntologySettings(BaseModel):
    """Ontology settings."""
    storage_path: Path = Field(Path("./state/ontology"), description="Path to ontology storage")


class AppSettings(BaseModel):
    """OMEN application settings."""
    openai: OpenAISettings
    qdrant: QdrantSettings
    keboola: KeboolaSettings
    ontology: OntologySettings
    state_file: Path = Field(Path("./state/state.json"), description="Path to state file")
    log_level: str = Field("INFO", description="Log level")
    batch_size: int = Field(100, description="Batch size for processing")
    max_workers: int = Field(4, description="Maximum number of worker threads/processes")
    max_retries: int = Field(3, description="Maximum number of retries")


def load_settings(env_file: Optional[str] = None) -> AppSettings:
    """Load settings from environment variables and .env file."""
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    openai_settings = OpenAISettings(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
        embedding_dimension=int(os.getenv("OPENAI_EMBEDDING_DIMENSION", "1536")),
    )

    qdrant_settings = QdrantSettings(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
        prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "True").lower() == "true",
        collection_name=os.getenv("QDRANT_COLLECTION_NAME", "keboola_metadata"),
    )

    keboola_settings = KeboolaSettings(
        url=os.getenv("KEBOOLA_URL", ""),
        token=os.getenv("KEBOOLA_TOKEN", ""),
    )

    # Ensure state directories exist
    state_dir = Path(os.getenv("STATE_DIR", "./state"))
    state_dir.mkdir(exist_ok=True, parents=True)
    
    ontology_dir = Path(os.getenv("ONTOLOGY_DIR", "./state/ontology"))
    ontology_dir.mkdir(exist_ok=True, parents=True)

    ontology_settings = OntologySettings(
        storage_path=ontology_dir,
    )

    return AppSettings(
        openai=openai_settings,
        qdrant=qdrant_settings,
        keboola=keboola_settings,
        ontology=ontology_settings,
        state_file=state_dir / "state.json",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        batch_size=int(os.getenv("BATCH_SIZE", "100")),
        max_workers=int(os.getenv("MAX_WORKERS", "4")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
    )


# Default settings instance
settings = load_settings() 