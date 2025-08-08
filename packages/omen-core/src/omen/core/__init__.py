"""
Core functionality for the OMEN platform.

This package provides the foundational components used across the OMEN platform,
including configuration management, logging, state management, and core utilities.
"""

import pkg_resources

from omen.core.config import AppSettings, load_settings
from omen.core.logging import configure_logging, get_logger
from omen.core.llm import LLMClient
from omen.core.state import StateManager
from omen.core.batch import BatchProcessor

__version__ = pkg_resources.get_distribution("omen-core").version

__all__ = [
    "AppSettings",
    "load_settings",
    "configure_logging",
    "get_logger",
    "LLMClient",
    "StateManager",
    "BatchProcessor",
]

# Configure logging by default
configure_logging()
