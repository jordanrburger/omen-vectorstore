"""
Core functionality for the OMEN platform.
"""

from omen.core.batch import BatchProcessor, batch_processor
from omen.core.config import (
    AppSettings,
    KeboolaSettings,
    OpenAISettings,
    QdrantSettings,
    OntologySettings,
    load_settings,
    settings,
)
from omen.core.logging import configure_logging, get_logger
from omen.core.state import StateManager, state_manager

__all__ = [
    # Batch processing
    "BatchProcessor",
    "batch_processor",
    
    # Configuration
    "AppSettings",
    "KeboolaSettings",
    "OpenAISettings",
    "QdrantSettings",
    "OntologySettings",
    "load_settings",
    "settings",
    
    # Logging
    "configure_logging",
    "get_logger",
    
    # State management
    "StateManager",
    "state_manager",
]

# Configure logging by default
configure_logging()
