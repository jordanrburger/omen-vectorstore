"""
Logging configuration for the OMEN platform.
"""
import logging
import sys
from typing import Optional

from omen.core.config import settings


def configure_logging(level: Optional[str] = None) -> None:
    """Configure logging for the application."""
    log_level = level or settings.log_level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Set third-party loggers to a higher level to reduce noise
    for logger_name in ["httpx", "urllib3", "qdrant_client"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name) 