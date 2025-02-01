import argparse
import logging
from app.config import load_config
from app.keboola_client import KeboolaClient
from app.vectorizer import SentenceTransformerProvider
from app.indexer import QdrantIndexer


def extract_metadata(force_full: bool = False):
    """Extract metadata from Keboola with support for incremental updates."""
    config = load_config()
    client = KeboolaClient(config['KEBOOLA_API_URL'], config['KEBOOLA_TOKEN'])
    
    metadata = client.extract_metadata(force_full=force_full)
    logging.info("Extraction complete. Metadata contains:")
    logging.info("- %d buckets", len(metadata['buckets']))
    logging.info("- %d tables total", sum(len(tables) for tables in metadata['tables'].values()))
    logging.info("- %d table details", len(metadata['table_details']))
    
    return metadata


def vectorize_and_index_metadata(metadata: dict = None):
    """Vectorize metadata and index it into Qdrant."""
    config = load_config()
    
    if metadata is None:
        # Load the latest metadata if none provided
        client = KeboolaClient(config['KEBOOLA_API_URL'], config['KEBOOLA_TOKEN'])
        metadata = client.state_manager.load_metadata()
        if not metadata:
            logging.error("No metadata found. Please run extraction first.")
            return False
    
    # Initialize embedding provider
    provider = SentenceTransformerProvider(
        model_name=config.get('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
    )
    
    # Initialize Qdrant indexer
    indexer = QdrantIndexer(
        host=config.get('QDRANT_HOST', 'localhost'),
        port=int(config.get('QDRANT_PORT', 6333)),
        collection_name=config.get('QDRANT_COLLECTION', 'keboola_metadata')
    )
    
    # Index the metadata
    logging.info("Starting metadata indexing...")
    indexer.index_metadata(metadata, provider)
    logging.info("Indexing complete!")
    return True


def search_metadata(query: str, metadata_type: str = None):
    """Search for metadata in Qdrant."""
    config = load_config()
    
    # Initialize components
    provider = SentenceTransformerProvider(
        model_name=config.get('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
    )
    indexer = QdrantIndexer(
        host=config.get('QDRANT_HOST', 'localhost'),
        port=int(config.get('QDRANT_PORT', 6333)),
        collection_name=config.get('QDRANT_COLLECTION', 'keboola_metadata')
    )
    
    # Search
    results = indexer.search_metadata(query, provider, metadata_type)
    
    # Print results
    print(f"\nSearch results for: {query}")
    print("-" * 50)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"Type: {result['metadata_type']}")
        metadata = result['metadata']
        print(f"Name: {metadata.get('name', metadata.get('title', 'N/A'))}")
        if 'description' in metadata:
            print(f"Description: {metadata['description']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Omen Vectorstore CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract metadata from Keboola')
    extract_parser.add_argument('--force-full', action='store_true', 
                              help='Force a full extraction instead of incremental')
    extract_parser.add_argument('--and-index', action='store_true',
                              help='Also vectorize and index the metadata after extraction')

    # Vectorize and index command
    subparsers.add_parser('index', help='Vectorize and index the latest metadata into Qdrant')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search metadata in Qdrant')
    search_parser.add_argument('query', help='The search query')
    search_parser.add_argument('--type', help='Filter by metadata type (e.g., buckets, tables)')

    args = parser.parse_args()

    if args.command == 'extract':
        metadata = extract_metadata(force_full=args.force_full)
        if args.and_index:
            vectorize_and_index_metadata(metadata)
    elif args.command == 'index':
        vectorize_and_index_metadata()
    elif args.command == 'search':
        search_metadata(args.query, args.type)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
