import os
import logging
from app.keboola_client import KeboolaClient
from app.embedding import OpenAIEmbeddingProvider
from app.indexer import QdrantIndexer

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize Keboola client
        logger.debug("Initializing Keboola client...")
        keboola_client = KeboolaClient(
            api_url=os.getenv("KEBOOLA_API_URL"),
            token=os.getenv("KEBOOLA_TOKEN")
        )
        logger.debug("Keboola client initialized")
        
        # Initialize OpenAI embedding provider
        logger.debug("Initializing OpenAI embedding provider...")
        embedding_provider = OpenAIEmbeddingProvider()
        logger.debug("OpenAI embedding provider initialized")
        
        # Initialize Qdrant indexer
        logger.debug("Initializing Qdrant indexer...")
        indexer = QdrantIndexer(tenant_id="1255")
        logger.debug("Qdrant indexer initialized")
        
        # Try to extract metadata
        logger.debug("Extracting metadata...")
        metadata = keboola_client.extract_metadata()
        logger.debug("Metadata extracted: %s", metadata)
        
    except Exception as e:
        logger.error("Error in test:", exc_info=True)
        raise

if __name__ == "__main__":
    main() 