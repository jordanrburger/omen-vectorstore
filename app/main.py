"""Main application entry point."""
import logging
import os
from typing import Dict, List, Optional
import click
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from app.config import Config
from app.indexer import QdrantIndexer
from app.keboola_client import KeboolaClient
from app.embedding import OpenAIEmbeddingProvider

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOGLEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_qdrant_client():
    """Configure Qdrant client based on environment variables."""
    host = os.getenv("QDRANT_HOST", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    
    return QdrantClient(url=host, api_key=api_key)

@click.group()
def cli():
    """Keboola Metadata Vector Store CLI"""
    pass

@cli.command()
@click.option("--tenant-id", required=True, help="Tenant identifier for indexing")
@click.option("--batch-size", default=100, help="Batch size for processing")
@click.option("--max-retries", default=3, help="Maximum number of retries for failed operations")
@click.option("--retry-delay", default=1, help="Delay between retries in seconds")
def index(tenant_id, batch_size, max_retries, retry_delay):
    """Index metadata for a specific tenant."""
    try:
        # Initialize clients
        keboola_client = KeboolaClient(
            api_url=os.getenv("KEBOOLA_API_URL"),
            token=os.getenv("KEBOOLA_TOKEN")
        )
        embedding_provider = OpenAIEmbeddingProvider()
        indexer = QdrantIndexer(tenant_id=tenant_id)
        
        logger.info(f"Starting metadata indexing for tenant {tenant_id}")
        metadata = keboola_client.extract_metadata()
        indexer.index_metadata(metadata, embedding_provider)
        logger.info(f"Completed metadata indexing for tenant {tenant_id}")
        
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.option("--tenant-id", required=True, help="Tenant identifier for search")
@click.option("--query", required=True, help="Search query")
@click.option("--limit", default=5, help="Maximum number of results")
@click.option("--metadata-type", help="Filter by metadata type (bucket, table, column, transformation)")
@click.option("--min-score", type=float, help="Minimum similarity score threshold")
def search(tenant_id, query, limit, metadata_type, min_score):
    """Search indexed metadata for a specific tenant."""
    try:
        embedding_provider = OpenAIEmbeddingProvider()
        indexer = QdrantIndexer(tenant_id=tenant_id)
        
        filters = {}
        if metadata_type:
            filters["metadata_type"] = metadata_type
            
        results = indexer.search_metadata(
            query,
            embedding_provider,
            limit=limit,
            filters=filters,
            score_threshold=min_score
        )
        
        for result in results:
            click.echo(f"Score: {result.score:.3f}")
            click.echo(f"Type: {result.payload.get('metadata_type', 'unknown')}")
            click.echo(f"Content: {result.payload.get('text', 'No content available')}")
            click.echo("---")
            
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise click.ClickException(str(e))

@cli.group()
def tenant():
    """Manage tenant collections"""
    pass

@tenant.command(name="list")
def list_collections():
    """List all tenant collections"""
    try:
        client = get_qdrant_client()
        collections = QdrantIndexer.list_tenant_collections(client)
        
        if not collections:
            click.echo("No tenant collections found")
            return
            
        for collection in collections:
            click.echo(f"Tenant: {collection['tenant_id']}")
            click.echo(f"Collection: {collection['name']}")
            click.echo(f"Created: {collection['created_at']}")
            click.echo(f"Vectors: {collection['vectors_count']}")
            click.echo("---")
            
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise click.ClickException(str(e))

@tenant.command(name="delete")
@click.option("--tenant-id", required=True, help="Tenant identifier to delete")
@click.confirmation_option(prompt='Are you sure you want to delete this tenant\'s collection?')
def delete_collection(tenant_id):
    """Delete a tenant's collection"""
    try:
        indexer = QdrantIndexer(tenant_id=tenant_id)
        indexer.delete_tenant_collection()
        click.echo(f"Successfully deleted collection for tenant {tenant_id}")
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == "__main__":
    cli()
