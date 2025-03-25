"""
Core functionality for the OMEN platform.

This package provides the foundational components used across the OMEN platform,
including configuration management, logging, state management, and core utilities.
"""

from omen.core.batch import BatchProcessor
from omen.core.config import Config
from omen.core.llm import LLMClient
from omen.core.logging import setup_logging
from omen.core.state import StateManager
from omen.core.utils import get_version

__all__ = [
    "BatchProcessor",
    "Config",
    "LLMClient",
    "setup_logging",
    "StateManager",
    "get_version",
]

# Configure logging by default
setup_logging()
