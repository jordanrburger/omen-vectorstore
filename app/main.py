import logging
from typing import Dict, List, Optional

from app.config import Config
from app.indexer import QdrantIndexer
from app.keboola_client import KeboolaClient
from app.state_manager import StateManager
from app.vectorizer import (
    EmbeddingProvider,
    OpenAIProvider,
    SentenceTransformerProvider,
)


def get_embedding_provider(config: Config) -> EmbeddingProvider:
    """Create an embedding provider based on configuration."""
    if config.openai_api_key:
        logging.info("Using OpenAI embedding provider")
        return OpenAIProvider(
            api_key=config.openai_api_key,
            model=config.embedding_model,
        )
    else:
        logging.info(
            f"Using SentenceTransformer provider with model {config.embedding_model}"
        )
        return SentenceTransformerProvider(
            model_name=config.embedding_model,
            device=config.device,
        )


def extract_metadata(client: KeboolaClient) -> Dict:
    """Extract metadata from Keboola Storage API."""
    logging.info("Starting metadata extraction")

    # Extract buckets and tables
    buckets = client.list_buckets()
    tables = client.list_tables()

    # Extract configurations
    configs = client.list_configurations(include_versions=True)

    metadata = {
        "buckets": buckets,
        "tables": tables,
        "configurations": configs,
    }

    logging.info(
        f"Extracted metadata: {len(buckets)} buckets, "
        f"{sum(len(t) for t in tables.values())} tables, "
        f"{len(configs)} configurations"
    )
    return metadata


def search_metadata(
    query: str,
    indexer: QdrantIndexer,
    embedding_provider: EmbeddingProvider,
    metadata_type: Optional[str] = None,
    limit: int = 10,
) -> List[Dict]:
    """Search metadata using semantic search."""
    logging.info(
        f"Searching for '{query}' " f"(type: {metadata_type or 'all'}, limit: {limit})"
    )

    try:
        results = indexer.search_metadata(
            query=query,
            embedding_provider=embedding_provider,
            metadata_type=metadata_type,
            limit=limit,
        )
        logging.info(f"Found {len(results)} results")
        return results

    except Exception as e:
        logging.error(f"Error searching metadata: {e}")
        raise


def main():
    """Main entry point for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config = Config.from_env()
    logging.info("Configuration loaded")

    # Initialize components
    state_manager = StateManager()
    client = KeboolaClient(
        api_url=config.storage_api_url,
        token=config.storage_api_token,
        state_manager=state_manager,
    )
    embedding_provider = get_embedding_provider(config)
    indexer = QdrantIndexer(
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection_name=config.qdrant_collection,
    )

    # Extract and index metadata
    metadata = extract_metadata(client)
    indexer.index_metadata(metadata, embedding_provider)
    logging.info("Metadata extraction and indexing completed")


if __name__ == "__main__":
    main()
