import logging
from typing import Dict, List, Optional

from app.config import Config
from app.indexer import QdrantIndexer
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


def search_metadata(
    query: str,
    indexer: QdrantIndexer,
    embedding_provider: EmbeddingProvider,
    metadata_type: Optional[str] = None,
    limit: int = 10,
) -> List[Dict]:
    """Search metadata using semantic search."""
    logging.info(
        f"Searching for '{query}' "
        f"(type: {metadata_type or 'all'}, limit: {limit})"
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
    embedding_provider = get_embedding_provider(config)
    indexer = QdrantIndexer(
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection_name=config.qdrant_collection,
    )

    # Search metadata
    query = input("Enter search query: ")
    metadata_type = input("Enter metadata type (or press Enter for all): ")
    if not metadata_type:
        metadata_type = None

    results = search_metadata(
        query=query,
        indexer=indexer,
        embedding_provider=embedding_provider,
        metadata_type=metadata_type,
    )

    # Print results
    print("\nSearch results:")
    print("-" * 50)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"Type: {result['metadata_type']}")
        metadata = result["metadata"]
        print(f"Name: {metadata.get('name', metadata.get('title', 'N/A'))}")
        if "description" in metadata:
            print(f"Description: {metadata['description']}")


if __name__ == "__main__":
    main()
