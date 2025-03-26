"""
Main entry point for the OMEN CLI.
"""

import os
import sys
import click
import subprocess
from rich.console import Console
from rich.table import Table

from omen.core import configure_logging, get_logger, AppSettings
from omen.vectorstore import VectorSearch, QdrantIndexer, get_embedding_provider
from omen.ontology import OntologyManager

# Initialize logger
logger = get_logger(__name__)
console = Console()


@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug output')
def cli(debug):
    """OMEN - Ontology-powered Metadata Engine CLI."""
    # Configure logging
    if debug:
        configure_logging("DEBUG")
    else:
        configure_logging("INFO")


@cli.group()
def ontology():
    """Manage ontology data."""
    pass


@ontology.command('stats')
def ontology_stats():
    """Show ontology statistics."""
    try:
        manager = OntologyManager()
        manager.load_state()
        
        stats = manager.get_stats()
        
        table = Table(title="Ontology Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Entities", str(stats["total_entities"]))
        table.add_row("Total Relationships", str(stats["total_relationships"]))
        table.add_row("Triple Count", str(stats["triple_count"]))
        
        if stats["entity_types"]:
            table.add_section()
            table.add_row("Entity Types", "")
            for entity_type, count in stats["entity_types"].items():
                table.add_row(f"  {entity_type}", str(count))
        
        if stats["relationship_types"]:
            table.add_section()
            table.add_row("Relationship Types", "")
            for rel_type, count in stats["relationship_types"].items():
                table.add_row(f"  {rel_type}", str(count))
        
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error getting ontology stats: {e}[/bold red]")
        sys.exit(1)


@ontology.command('clear')
@click.confirmation_option(prompt='Are you sure you want to clear the ontology?')
def ontology_clear():
    """Clear all ontology data."""
    try:
        manager = OntologyManager()
        manager.clear()
        manager.save_state()
        console.print("[bold green]Ontology cleared successfully[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error clearing ontology: {e}[/bold red]")
        sys.exit(1)


@cli.group()
def extract():
    """Extract and process metadata."""
    pass


@extract.command('keboola')
@click.option('--token', '-t', help='Keboola Storage API token', envvar='KEBOOLA_API_TOKEN')
@click.option('--url', '-u', help='Keboola Storage API URL', envvar='KEBOOLA_API_URL')
@click.option('--incremental/--full', default=True, help='Use incremental extraction')
@click.option('--batch-size', default=10, help='Batch size for processing')
@click.option('--vectorize/--no-vectorize', default=True, help='Vectorize metadata after extraction')
@click.option('--index/--no-index', default=True, help='Index vectors after vectorization')
def extract_keboola(token, url, incremental, batch_size, vectorize, index):
    """Extract metadata from Keboola Storage API."""
    try:
        # Import here to not require keboola deps unless needed
        from omen.extractors.keboola import KeboolaExtractor
        from omen.vectorstore import Vectorizer, MetadataProcessor
        
        if not token:
            console.print("[bold red]Error: Storage API token is required[/bold red]")
            console.print("Set it using --token or KEBOOLA_API_TOKEN environment variable")
            sys.exit(1)
        
        if not url:
            url = "https://connection.keboola.com"
            console.print(f"[yellow]No API URL provided, using default: {url}[/yellow]")
        
        # Create extractor
        console.print("[bold]Initializing Keboola extractor...[/bold]")
        extractor = KeboolaExtractor(token=token, url=url)
        
        # Extract metadata
        console.print(f"[bold]Extracting metadata ({'incremental' if incremental else 'full'})...[/bold]")
        metadata = extractor.extract(incremental=incremental)
        
        console.print(f"[green]Extracted {len(metadata)} metadata items[/green]")
        
        if vectorize:
            console.print(f"[bold]Processing and vectorizing metadata (batch size: {batch_size})...[/bold]")
            
            # Initialize components
            vectorizer = Vectorizer(embedding_provider=get_embedding_provider())
            indexer = QdrantIndexer()
            processor = MetadataProcessor(vectorizer=vectorizer, indexer=indexer, batch_size=batch_size)
            
            # Process and vectorize
            processor.process_batch(metadata, batch_size=batch_size)
            console.print(f"[green]Processed and indexed {len(metadata)} documents[/green]")
        
        console.print("[bold green]Extraction completed successfully![/bold green]")
    except ImportError as e:
        console.print(f"[bold red]Error: Keboola extractor dependencies not installed[/bold red]")
        console.print(f"[bold red]Exception details: {e}[/bold red]")
        console.print("Install them with: pip install 'omen-extractors[keboola]'")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error during extraction: {e}[/bold red]")
        sys.exit(1)


@cli.group()
def search():
    """Search vectorized metadata."""
    pass


@search.command('query')
@click.argument('query', type=str)
@click.option('--limit', '-l', type=int, default=10, help='Maximum number of results')
@click.option('--type', '-t', multiple=True, help='Filter by metadata type')
def search_query(query, limit, type):
    """Search metadata with a text query."""
    try:
        # Initialize search components
        indexer = QdrantIndexer()
        embedding_provider = get_embedding_provider()
        search_engine = VectorSearch(
            indexer=indexer,
            embedding_provider=embedding_provider
        )
        
        # Perform search
        results = search_engine.search(
            query=query,
            limit=limit,
            type_filter=[t for t in type] if type else None
        )
        
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return
        
        # Display results
        table = Table(title=f"Search Results for '{query}'")
        table.add_column("Score", style="cyan", width=8)
        table.add_column("ID", style="blue", width=12)
        table.add_column("Type", style="green", width=15)
        table.add_column("Content", style="white")
        
        for result in results:
            table.add_row(
                f"{result.score:.4f}",
                result.document.id[:12],
                result.document.source.type.value,
                result.document.content[:100] + "..." if len(result.document.content) > 100 else result.document.content
            )
        
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error searching: {e}[/bold red]")
        sys.exit(1)


@cli.group()
def api():
    """Manage the API server."""
    pass


@api.command('start')
@click.option('--host', '-h', default="0.0.0.0", help='Host to bind to')
@click.option('--port', '-p', default=8000, help='Port to listen on')
@click.option('--reload/--no-reload', default=False, help='Enable auto-reload on code changes')
def api_start(host, port, reload):
    """Start the API server."""
    try:
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env["HOST"] = host
        env["PORT"] = str(port)
        
        if reload:
            env["LOG_LEVEL"] = "DEBUG"
            console.print(f"[bold yellow]Starting API server with reload enabled (http://{host}:{port})[/bold yellow]")
        else:
            console.print(f"[bold green]Starting API server (http://{host}:{port})[/bold green]")
        
        # Run the API server module
        cmd = [
            sys.executable, 
            "-m", 
            "omen.api.main"
        ]
        
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        console.print("[bold yellow]API server stopped[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error starting API server: {e}[/bold red]")
        sys.exit(1)


@cli.group()
def config():
    """Manage configuration."""
    pass


@config.command('show')
def config_show():
    """Show current configuration."""
    try:
        settings = AppSettings()
        
        table = Table(title="Current Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in settings.dict().items():
            table.add_row(key, str(value))
        
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error showing configuration: {e}[/bold red]")
        sys.exit(1)


if __name__ == '__main__':
    cli() 