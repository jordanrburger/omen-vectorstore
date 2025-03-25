"""
Configuration management for the OMEN platform.

This module provides configuration management functionality, including
loading settings from environment variables and configuration files.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

class OpenAISettings(BaseModel):
    """OpenAI API settings."""
    api_key: Optional[str] = None
    model: str = "gpt-4"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.7
    max_tokens: int = 2000
    embedding_dimension: int = 3072

    @model_validator(mode='before')
    @classmethod
    def load_env_vars(cls, values):
        """Load environment variables."""
        if not values.get('api_key'):
            values['api_key'] = os.getenv('OPENAI_API_KEY')
        return values

class QdrantSettings(BaseModel):
    """Qdrant vector store settings."""
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = True
    collection_name: str = "omen_collection"

class KeboolaSettings(BaseModel):
    """Keboola API settings."""
    url: Optional[str] = None
    token: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def load_env_vars(cls, values):
        """Load environment variables."""
        if not values.get('url'):
            values['url'] = os.getenv('KEBOOLA_URL')
        if not values.get('token'):
            values['token'] = os.getenv('KEBOOLA_TOKEN')
        return values

class OntologySettings(BaseModel):
    """Ontology storage settings."""
    storage_path: Path = Path("./state/ontology")

class AppSettings(BaseModel):
    """Main application settings."""
    openai: OpenAISettings = OpenAISettings()
    qdrant: QdrantSettings = QdrantSettings()
    keboola: KeboolaSettings = KeboolaSettings()
    ontology: OntologySettings = OntologySettings()
    state_file: Path = Path("state/state.json")
    log_level: str = "INFO"
    batch_size: int = 100
    max_workers: int = 4
    max_retries: int = 3

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

    def save(self, path: Path) -> None:
        """Save settings to a file.
        
        Args:
            path: Path to save settings to
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "AppSettings":
        """Load settings from a file.
        
        Args:
            path: Path to load settings from
            
        Returns:
            Loaded settings
        """
        if not path.exists():
            return cls()
        
        with open(path) as f:
            data = json.load(f)
            # Convert string paths back to Path objects
            if "state_file" in data:
                data["state_file"] = Path(data["state_file"])
            if "ontology" in data and "storage_path" in data["ontology"]:
                data["ontology"]["storage_path"] = Path(data["ontology"]["storage_path"])
            return cls.model_validate(data)

def load_settings() -> AppSettings:
    """Load settings from environment and config file.
    
    Returns:
        Loaded settings
    """
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
            # Convert string paths back to Path objects
            if "state_file" in data:
                data["state_file"] = Path(data["state_file"])
            if "ontology" in data and "storage_path" in data["ontology"]:
                data["ontology"]["storage_path"] = Path(data["ontology"]["storage_path"])
            return AppSettings.model_validate(data)
    return AppSettings.model_validate({})

# Default settings instance
settings = load_settings() 