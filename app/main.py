import logging
from typing import Dict, List, Optional
import argparse

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

    # Extract table details with columns
    table_details = {}
    total_columns = 0
    for bucket_id, bucket_tables in tables.items():
        for table in bucket_tables:
            try:
                details = client.get_table_details(table["id"])
                if details:
                    table_details[table["id"]] = details
                    total_columns += len(details.get("columns", []))
            except Exception as e:
                logging.warning(f"Error fetching details for table {table['id']}: {e}")

    # Extract configurations
    configs = client.list_configurations(include_versions=True)

    metadata = {
        "buckets": buckets,
        "tables": tables,
        "table_details": table_details,
        "configurations": configs,
    }

    logging.info(
        f"Extracted metadata: {len(buckets)} buckets, "
        f"{sum(len(t) for t in tables.values())} tables with {total_columns} columns, "
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


def index_command(config: Config):
    """Command to extract and index metadata."""
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


def search_command(config: Config, query: str, metadata_type: Optional[str] = None, limit: int = 10):
    """Command to search indexed metadata."""
    embedding_provider = get_embedding_provider(config)
    indexer = QdrantIndexer(
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection_name=config.qdrant_collection,
    )

    results = search_metadata(
        query=query,
        indexer=indexer,
        embedding_provider=embedding_provider,
        metadata_type=metadata_type,
        limit=limit,
    )

    # Print results in a readable format
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Type: {result.get('metadata_type', 'unknown')}")
        print(f"Score: {result.get('score', 0):.3f}")
        
        metadata = result.get('metadata', {})
        if metadata:
            if 'id' in metadata:
                print(f"ID: {metadata['id']}")
            if 'name' in metadata:
                print(f"Name: {metadata['name']}")
            if 'description' in metadata:
                print(f"Description: {metadata['description']}")
            if 'stage' in metadata:
                print(f"Stage: {metadata['stage']}")
            if 'type' in metadata:
                print(f"Type: {metadata['type']}")
            if 'bucket' in metadata:
                print(f"Bucket: {metadata['bucket']}")
            if 'component' in metadata:
                print(f"Component: {metadata['component']}")
            if 'configuration' in metadata:
                print(f"Configuration: {metadata['configuration']}")


def main():
    """Main entry point for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Keboola Metadata Search Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Index command
    index_parser = subparsers.add_parser("index", help="Extract and index metadata")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed metadata")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--type", help="Filter by metadata type (buckets, tables, configurations)")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results")

    args = parser.parse_args()

    # Load configuration
    config = Config.from_env()
    logging.info("Configuration loaded")

    if args.command == "index":
        index_command(config)
    elif args.command == "search":
        search_command(config, args.query, args.type, args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
