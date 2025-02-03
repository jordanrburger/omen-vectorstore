import os
import sys
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Python path: %s", sys.path)
logger.debug("Current directory: %s", os.getcwd())

try:
    from app.main import cli
    logger.debug("Successfully imported cli from app.main")
except Exception as e:
    logger.error("Failed to import cli from app.main: %s", e)
    raise

try:
    cli() 