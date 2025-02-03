import os
import sys
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def exception_hook(exctype, value, tb):
    """Custom exception hook to log uncaught exceptions."""
    logger.error("Uncaught exception:", exc_info=(exctype, value, tb))
    traceback.print_exception(exctype, value, tb)

sys.excepthook = exception_hook

# Import required modules
from app.keboola_client import KeboolaClient
from app.embedding import OpenAIEmbeddingProvider
from app.indexer import QdrantIndexer

def main():
    try:
        # Initialize clients
        logger.debug("Initializing Keboola client...")
        keboola_client = KeboolaClient(
            api_url=os.getenv("KEBOOLA_API_URL"),
            token=os.getenv("KEBOOLA_TOKEN")
        )
        logger.debug("Keboola client initialized")
        
        logger.debug("Initializing OpenAI embedding provider...")
        embedding_provider = OpenAIEmbeddingProvider()
        logger.debug("OpenAI embedding provider initialized")
        
        logger.debug("Initializing Qdrant indexer...")
        indexer = QdrantIndexer(tenant_id="1255")
        logger.debug("Qdrant indexer initialized")
        
        # Extract and index metadata
        logger.debug("Starting metadata extraction...")
        metadata = keboola_client.extract_metadata()
        logger.debug("Metadata extracted")
        
        logger.debug("Starting metadata indexing...")
        indexer.index_metadata(metadata, embedding_provider)
        logger.debug("Metadata indexing completed")
        
    except Exception as e:
        logger.error("Error in indexing process:", exc_info=True)
        raise

if __name__ == "__main__":
    main() 