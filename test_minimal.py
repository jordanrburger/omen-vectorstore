import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logger.debug("Starting minimal test")

try:
    logger.debug("Importing KeboolaClient")
    from app.keboola_client import KeboolaClient
    logger.debug("KeboolaClient imported successfully")
    
    logger.debug("Creating KeboolaClient instance")
    client = KeboolaClient(
        api_url=os.getenv("KEBOOLA_API_URL"),
        token=os.getenv("KEBOOLA_TOKEN")
    )
    logger.debug("KeboolaClient instance created")
    
    logger.debug("Extracting metadata")
    metadata = client.extract_metadata()
    logger.debug("Metadata extracted: %s", metadata)
    
except Exception as e:
    logger.error("Error occurred:", exc_info=True)
    raise 