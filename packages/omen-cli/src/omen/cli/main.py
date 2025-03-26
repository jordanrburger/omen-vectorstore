"""
Main entry point for the OMEN CLI.
"""

import os
import sys
import click
import subprocess
from rich.console import Console
from rich.table import Table
from datetime import datetime, timezone
from pathlib import Path

from omen.core import configure_logging, get_logger, AppSettings
from omen.vectorstore import VectorSearch, QdrantIndexer, get_embedding_provider, HybridSearch
from omen.ontology import OntologyManager
from omen.ontology.models import Entity, EntityType, Relationship, RelationshipType

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
        # Ensure ontology directory exists
        ontology_path = Path("state/ontology")
        os.makedirs(ontology_path, exist_ok=True)
        
        manager = OntologyManager(state_dir=ontology_path)
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
        # Ensure ontology directory exists
        ontology_path = Path("state/ontology")
        os.makedirs(ontology_path, exist_ok=True)
        
        manager = OntologyManager(state_dir=ontology_path)
        manager.clear()
        manager.save_state()
        console.print("[bold green]Ontology cleared successfully[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error clearing ontology: {e}[/bold red]")
        sys.exit(1)


@ontology.command('init')
def ontology_init():
    """Initialize a new ontology."""
    try:
        import os
        from pathlib import Path
        
        # Create the ontology directory structure
        state_dir = Path("state")
        ontology_dir = state_dir / "ontology"
        
        # Ensure directories exist
        os.makedirs(state_dir, exist_ok=True)
        os.makedirs(ontology_dir, exist_ok=True)
        
        # Initialize an empty ontology
        manager = OntologyManager(state_dir=ontology_dir)
        
        # Save the initial state
        manager.save_state()
        
        console.print("[bold green]Ontology initialized successfully[/bold green]")
        console.print(f"[green]Ontology files stored in: {ontology_dir.absolute()}[/green]")
    except Exception as e:
        console.print(f"[bold red]Error initializing ontology: {e}[/bold red]")
        import traceback
        logger.error(f"Ontology initialization error: {traceback.format_exc()}")
        console.print("[yellow]Continuing without ontology creation[/yellow]")
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
@click.option('--ontology/--no-ontology', default=True, help='Create ontology from metadata relationships')
@click.option('--index/--no-index', default=True, help='Index vectors after vectorization')
def extract_keboola(token, url, incremental, batch_size, vectorize, ontology, index):
    """Extract metadata from Keboola Storage API."""
    try:
        # Import here to not require keboola deps unless needed
        import os
        from pathlib import Path
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
        
        # Initialize components for processing
        components_initialized = False
        
        if vectorize or ontology:
            # Initialize ontology manager if needed
            ontology_manager = None
            if ontology:
                try:
                    console.print("[bold]Initializing ontology manager...[/bold]")
                    
                    # Ensure ontology directory exists
                    ontology_path = Path("state/ontology")
                    os.makedirs(ontology_path, exist_ok=True)
                    
                    # Explicitly provide the storage_path
                    ontology_manager = OntologyManager(state_dir=ontology_path)
                    
                    # Try to load state, but if it doesn't exist, initialize a new one
                    try:
                        ontology_manager.load_state()
                        console.print("[green]Loaded existing ontology state[/green]")
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not load existing ontology state: {e}[/yellow]")
                        console.print("[yellow]Initializing new ontology[/yellow]")
                        # Save an empty state
                        ontology_manager.save_state()
                except Exception as e:
                    console.print(f"[bold red]Error initializing ontology manager: {e}[/bold red]")
                    import traceback
                    logger.error(f"Ontology initialization error: {traceback.format_exc()}")
                    console.print("[yellow]Continuing without ontology creation[/yellow]")
                    ontology = False
            
            if vectorize:
                console.print(f"[bold]Processing and vectorizing metadata (batch size: {batch_size})...[/bold]")
                
                # Initialize components
                vectorizer = Vectorizer(embedding_provider=get_embedding_provider())
                indexer = QdrantIndexer()
                processor = MetadataProcessor(vectorizer=vectorizer, indexer=indexer, batch_size=batch_size)
                components_initialized = True
                
                # Process and vectorize
                processor.process_batch(metadata, batch_size=batch_size)
                console.print(f"[green]Processed and indexed {len(metadata)} documents[/green]")
            
            # Create ontology from metadata if enabled
            if ontology and ontology_manager:
                console.print("[bold]Creating ontology from metadata relationships...[/bold]")
                
                # Track ontology creation metrics
                created_entities = 0
                created_relationships = 0
                
                # Process each metadata item to create ontology entries
                for meta_item in metadata:
                    try:
                        # Create entity for the item itself
                        entity_id = f"{meta_item.source.type.value}-{meta_item.source.id}"
                        entity_name = getattr(meta_item, 'name', meta_item.source.id)
                        entity_type = meta_item.source.type.value
                        
                        # Skip if entity already exists (for incremental updates)
                        if not ontology_manager.entity_exists(entity_id):
                            # Create Entity object
                            entity = Entity(
                                id=entity_id,
                                name=entity_name,
                                type=EntityType(entity_type),
                                properties={
                                    "source": "keboola",
                                    "extraction_time": str(meta_item.source.updated_at or datetime.now(timezone.utc))
                                }
                            )
                            ontology_manager.add_entity(entity)
                            created_entities += 1
                        
                        # Create relationships based on metadata type
                        if hasattr(meta_item, 'parent_id') and meta_item.parent_id:
                            parent_type = getattr(meta_item, 'parent_type', 'unknown')
                            parent_id = f"{parent_type}-{meta_item.parent_id}"
                            
                            # Create parent entity if needed
                            if not ontology_manager.entity_exists(parent_id):
                                parent_entity = Entity(
                                    id=parent_id,
                                    name=meta_item.parent_id,
                                    type=EntityType(parent_type)
                                )
                                ontology_manager.add_entity(parent_entity)
                                created_entities += 1
                            
                            # Add relationship
                            rel = Relationship(
                                source_id=entity_id,
                                target_id=parent_id,
                                type=RelationshipType.BELONGS_TO
                            )
                            ontology_manager.add_relationship(rel)
                            created_relationships += 1
                        
                        # Add project relationships for components
                        if hasattr(meta_item, 'project_id') and meta_item.project_id:
                            project_id = f"project-{meta_item.project_id}"
                            
                            # Create project entity if needed
                            if not ontology_manager.entity_exists(project_id):
                                project_entity = Entity(
                                    id=project_id,
                                    name=f"Project {meta_item.project_id}",
                                    type=EntityType.PROJECT
                                )
                                ontology_manager.add_entity(project_entity)
                                created_entities += 1
                            
                            # Add relationship
                            rel = Relationship(
                                source_id=entity_id,
                                target_id=project_id,
                                type=RelationshipType.BELONGS_TO
                            )
                            ontology_manager.add_relationship(rel)
                            created_relationships += 1
                    
                    except Exception as e:
                        logger.error(f"Error creating ontology for item {meta_item.source.id}: {e}")
                
                # Save ontology state
                ontology_manager.save_state()
                console.print(f"[green]Created {created_entities} entities and {created_relationships} relationships in the ontology[/green]")
        
        console.print("[bold green]Extraction completed successfully![/bold green]")
    except ImportError as e:
        console.print(f"[bold red]Error: Keboola extractor dependencies not installed[/bold red]")
        console.print(f"[bold red]Exception details: {e}[/bold red]")
        console.print("Install them with: pip install 'omen-extractors[keboola]'")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error during extraction: {e}[/bold red]")
        sys.exit(1)


@extract.command('sample')
@click.option('--count', '-c', default=10, help='Number of sample items to generate')
@click.option('--vectorize/--no-vectorize', default=True, help='Vectorize sample metadata')
@click.option('--ontology/--no-ontology', default=True, help='Create ontology from metadata relationships')
def extract_sample(count, vectorize, ontology):
    """Generate and process sample metadata for testing."""
    try:
        from datetime import datetime
        import uuid
        import os
        from pathlib import Path
        from omen.vectorstore import Vectorizer, MetadataProcessor
        from omen.vectorstore.models import MetadataDocument, MetadataSource, MetadataType
        
        console.print(f"[bold]Generating {count} sample metadata items...[/bold]")
        
        # Generate sample metadata
        metadata = []
        
        # Create some projects first
        projects = []
        for i in range(1, 4):
            project_id = f"project-{i}"
            projects.append({
                "id": project_id,
                "name": f"Sample Project {i}"
            })
            
        # Create some configurations in each project
        configs = []
        for project in projects:
            for i in range(1, 3):
                config_id = f"config-{project['id']}-{i}"
                configs.append({
                    "id": config_id,
                    "name": f"Configuration {i} in {project['name']}",
                    "project_id": project['id']
                })
        
        # Create tables linked to configurations
        for config in configs:
            for i in range(1, 3):
                table_id = f"table-{config['id']}-{i}"
                metadata.append(MetadataDocument(
                    id=str(uuid.uuid4()),
                    content=f"Table {i} in {config['name']}",
                    source=MetadataSource(
                        id=table_id,
                        type=MetadataType.TABLE
                    ),
                    extraction_time=datetime.now(),
                    # Additional fields needed for ontology
                    parent_id=config['id'],
                    parent_type="config",
                    project_id=config['project_id'],
                    name=f"Table {i} for {config['name']}"
                ))
        
        # Add the configs themselves
        for config in configs:
            metadata.append(MetadataDocument(
                id=str(uuid.uuid4()),
                content=f"Configuration: {config['name']}",
                source=MetadataSource(
                    id=config['id'],
                    type=MetadataType.CONFIGURATION
                ),
                extraction_time=datetime.now(),
                # Additional fields needed for ontology
                project_id=config['project_id'],
                name=config['name']
            ))
        
        # Add the projects
        for project in projects:
            metadata.append(MetadataDocument(
                id=str(uuid.uuid4()),
                content=f"Project: {project['name']}",
                source=MetadataSource(
                    id=project['id'],
                    type=MetadataType.PROJECT
                ),
                extraction_time=datetime.now(),
                name=project['name']
            ))
        
        # Limit to requested count
        metadata = metadata[:count]
        
        console.print(f"[green]Generated {len(metadata)} sample metadata items[/green]")
        
        # Process the metadata just like in extract_keboola
        if vectorize or ontology:
            # Initialize ontology manager if needed
            ontology_manager = None
            if ontology:
                try:
                    console.print("[bold]Initializing ontology manager...[/bold]")
                    
                    # Ensure ontology directory exists
                    ontology_path = Path("state/ontology")
                    os.makedirs(ontology_path, exist_ok=True)
                    
                    # Explicitly provide the storage_path
                    ontology_manager = OntologyManager(state_dir=ontology_path)
                    
                    # Try to load state, but if it doesn't exist, initialize a new one
                    try:
                        ontology_manager.load_state()
                        console.print("[green]Loaded existing ontology state[/green]")
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not load existing ontology state: {e}[/yellow]")
                        console.print("[yellow]Initializing new ontology[/yellow]")
                        # Save an empty state
                        ontology_manager.save_state()
                except Exception as e:
                    console.print(f"[bold red]Error initializing ontology manager: {e}[/bold red]")
                    import traceback
                    logger.error(f"Ontology initialization error: {traceback.format_exc()}")
                    console.print("[yellow]Continuing without ontology creation[/yellow]")
                    ontology = False
            
            if vectorize:
                console.print("[bold]Processing and vectorizing metadata...[/bold]")
                
                # Initialize components
                vectorizer = Vectorizer(embedding_provider=get_embedding_provider())
                indexer = QdrantIndexer()
                processor = MetadataProcessor(vectorizer=vectorizer, indexer=indexer)
                
                # Process and vectorize
                processor.process_batch(metadata)
                console.print(f"[green]Processed and indexed {len(metadata)} documents[/green]")
            
            # Create ontology from metadata if enabled
            if ontology and ontology_manager:
                console.print("[bold]Creating ontology from metadata relationships...[/bold]")
                
                # Track ontology creation metrics
                created_entities = 0
                created_relationships = 0
                
                # Process each metadata item to create ontology entries
                for meta_item in metadata:
                    try:
                        # Create entity for the item itself
                        entity_id = f"{meta_item.source.type.value}-{meta_item.source.id}"
                        entity_name = getattr(meta_item, 'name', meta_item.source.id)
                        entity_type = meta_item.source.type.value
                        
                        # Skip if entity already exists (for incremental updates)
                        if not ontology_manager.entity_exists(entity_id):
                            # Create Entity object
                            entity = Entity(
                                id=entity_id,
                                name=entity_name,
                                type=EntityType(entity_type),
                                properties={
                                    "source": "sample",
                                    "extraction_time": str(meta_item.source.updated_at or datetime.now(timezone.utc))
                                }
                            )
                            ontology_manager.add_entity(entity)
                            created_entities += 1
                        
                        # Create relationships based on metadata type
                        if hasattr(meta_item, 'parent_id') and meta_item.parent_id:
                            parent_type = getattr(meta_item, 'parent_type', 'unknown')
                            parent_id = f"{parent_type}-{meta_item.parent_id}"
                            
                            # Create parent entity if needed
                            if not ontology_manager.entity_exists(parent_id):
                                parent_entity = Entity(
                                    id=parent_id,
                                    name=meta_item.parent_id,
                                    type=EntityType(parent_type)
                                )
                                ontology_manager.add_entity(parent_entity)
                                created_entities += 1
                            
                            # Add relationship
                            rel = Relationship(
                                source_id=entity_id,
                                target_id=parent_id,
                                type=RelationshipType.BELONGS_TO
                            )
                            ontology_manager.add_relationship(rel)
                            created_relationships += 1
                        
                        # Add project relationships for components
                        if hasattr(meta_item, 'project_id') and meta_item.project_id:
                            project_id = f"project-{meta_item.project_id}"
                            
                            # Create project entity if needed
                            if not ontology_manager.entity_exists(project_id):
                                project_entity = Entity(
                                    id=project_id,
                                    name=f"Project {meta_item.project_id}",
                                    type=EntityType.PROJECT
                                )
                                ontology_manager.add_entity(project_entity)
                                created_entities += 1
                            
                            # Add relationship
                            rel = Relationship(
                                source_id=entity_id,
                                target_id=project_id,
                                type=RelationshipType.BELONGS_TO
                            )
                            ontology_manager.add_relationship(rel)
                            created_relationships += 1
                    
                    except Exception as e:
                        logger.error(f"Error creating ontology for item {meta_item.source.id}: {e}")
                
                # Save ontology state
                ontology_manager.save_state()
                console.print(f"[green]Created {created_entities} entities and {created_relationships} relationships in the ontology[/green]")
        
        console.print("[bold green]Sample extraction completed successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error during sample extraction: {e}[/bold red]")
        import traceback
        logger.error(f"Sample extraction error: {traceback.format_exc()}")
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


@search.command('hybrid')
@click.argument('query', type=str)
@click.option('--limit', '-l', type=int, default=10, help='Maximum number of results')
@click.option('--type', '-t', multiple=True, help='Filter by metadata type')
@click.option('--vector-weight', '-v', type=float, default=0.7, help='Weight for vector search (0-1)')
@click.option('--semantic-weight', '-s', type=float, default=0.3, help='Weight for semantic search (0-1)')
@click.option('--include-related/--no-related', default=False, help='Include related entities')
@click.option('--related-depth', type=int, default=1, help='Maximum depth for related entities')
def search_hybrid(query, limit, type, vector_weight, semantic_weight, include_related, related_depth):
    """Search using hybrid vector and semantic techniques."""
    try:
        # Initialize components
        indexer = QdrantIndexer()
        embedding_provider = get_embedding_provider()
        vector_search = VectorSearch(
            indexer=indexer,
            embedding_provider=embedding_provider
        )
        
        try:
            # Ensure ontology directory exists
            ontology_path = Path("state/ontology")
            os.makedirs(ontology_path, exist_ok=True)
            
            ontology_manager = OntologyManager(state_dir=ontology_path)
            try:
                ontology_manager.load_state()
            except Exception as e:
                import traceback
                console.print(f"[yellow]Warning: Could not load ontology: {e}[/yellow]")
                logger.error(f"Ontology loading error: {traceback.format_exc()}")
                console.print("[yellow]Falling back to vector search only[/yellow]")
                # Fall back to vector search
                results = vector_search.search(
                    query=query,
                    limit=limit,
                    type_filter=[t for t in type] if type else None
                )
                
                if not results:
                    console.print("[yellow]No results found[/yellow]")
                    return
                    
                # Display results
                table = Table(title=f"Vector Search Results for '{query}' (Hybrid search unavailable)")
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
                return
        except Exception as e:
            console.print(f"[yellow]Error initializing ontology manager: {e}[/yellow]")
            console.print("[yellow]Falling back to vector search only[/yellow]")
            # Fall back to vector search
            results = vector_search.search(
                query=query,
                limit=limit,
                type_filter=[t for t in type] if type else None
            )
            
            if not results:
                console.print("[yellow]No results found[/yellow]")
                return
                
            # Display results
            table = Table(title=f"Vector Search Results for '{query}' (Hybrid search unavailable)")
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
            return
        
        # Initialize hybrid search
        hybrid_engine = HybridSearch(
            vector_search=vector_search,
            ontology_manager=ontology_manager
        )
        
        # Normalize weights for display
        total_weight = vector_weight + semantic_weight
        normalized_vector_weight = vector_weight / total_weight
        normalized_semantic_weight = semantic_weight / total_weight
        
        console.print(f"[bold]Running hybrid search with weights:[/bold] "
                     f"Vector {normalized_vector_weight:.2f}, Semantic {normalized_semantic_weight:.2f}")
        
        # Perform search
        results = hybrid_engine.search(
            query=query,
            limit=limit,
            type_filter=[t for t in type] if type else None,
            include_related=include_related,
            max_related_depth=related_depth
        )
        
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return
        
        # Display results
        table = Table(title=f"Hybrid Search Results for '{query}'")
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Vector", style="blue", width=8)
        table.add_column("Semantic", style="green", width=8)
        table.add_column("ID", style="blue", width=12)
        table.add_column("Type", style="magenta", width=15)
        table.add_column("Content", style="white")
        
        for result in results:
            vector_score = result.vector_score or 0.0
            semantic_score = result.semantic_score or 0.0
            
            table.add_row(
                f"{result.score:.4f}",
                f"{vector_score:.4f}",
                f"{semantic_score:.4f}",
                result.document.id[:12],
                result.document.source.type.value,
                result.document.content[:100] + "..." if len(result.document.content) > 100 else result.document.content
            )
        
        console.print(table)
        
        # Show related entities if available
        if include_related:
            for i, result in enumerate(results, 1):
                if result.related_entities:
                    rel_table = Table(title=f"Related entities for result #{i}: {result.document.title}")
                    rel_table.add_column("Entity", style="cyan")
                    rel_table.add_column("Relationship", style="green")
                    rel_table.add_column("Direction", style="blue")
                    rel_table.add_column("Path Length", style="magenta")
                    
                    for rel in result.related_entities[:5]:  # Show only top 5 related
                        entity_name = rel.get('target_name', 'Unknown')
                        relationship = rel.get('relationship', 'Unknown')
                        direction = rel.get('direction', 'Unknown')
                        path_length = rel.get('path_length', 0)
                        
                        rel_table.add_row(entity_name, relationship, direction, str(path_length))
                    
                    console.print(rel_table)
                    console.print("")
        
    except Exception as e:
        console.print(f"[bold red]Error performing hybrid search: {str(e)}[/bold red]")
        import traceback
        logger.error(f"Hybrid search error: {traceback.format_exc()}")
        sys.exit(1)


@search.command('recommend')
@click.argument('entity_id', type=str)
@click.option('--limit', '-l', type=int, default=5, help='Maximum number of recommendations')
@click.option('--vector-weight', '-v', type=float, default=0.5, help='Weight for vector recommendations (0-1)')
@click.option('--semantic-weight', '-s', type=float, default=0.5, help='Weight for semantic recommendations (0-1)')
@click.option('--show-paths/--no-paths', default=True, help='Show relationship paths')
def search_recommend(entity_id, limit, vector_weight, semantic_weight, show_paths):
    """Generate hybrid recommendations for an entity."""
    try:
        # Initialize components
        indexer = QdrantIndexer()
        embedding_provider = get_embedding_provider()
        vector_search = VectorSearch(
            indexer=indexer,
            embedding_provider=embedding_provider
        )
        
        try:
            # Ensure ontology directory exists
            ontology_path = Path("state/ontology")
            os.makedirs(ontology_path, exist_ok=True)
            
            ontology_manager = OntologyManager(state_dir=ontology_path)
            ontology_manager.load_state()
        except Exception as e:
            console.print(f"[bold red]Error: Could not load ontology: {e}[/bold red]")
            console.print("[yellow]Recommendations require a working ontology. Please make sure the ontology is properly set up.[/yellow]")
            sys.exit(1)
        
        # Check if entity exists
        if not ontology_manager.entity_exists(entity_id):
            console.print(f"[bold red]Entity '{entity_id}' does not exist in the ontology[/bold red]")
            sys.exit(1)
        
        # Get entity details
        entity = ontology_manager.get_entity(entity_id)
        console.print(f"[bold]Generating recommendations for:[/bold] {entity.name} ({entity.type.value})")
        
        # Get recommendations
        try:
            recommendations = hybrid_engine.hybrid_recommendation(
                entity_id=entity_id,
                limit=limit,
                vector_weight=vector_weight,
                semantic_weight=semantic_weight
            )
        except Exception as e:
            console.print(f"[bold red]Error generating hybrid recommendations: {str(e)}[/bold red]")
            console.print("[yellow]Falling back to ontology-based recommendations only[/yellow]")
            
            # Fall back to ontology-based recommendations
            try:
                recommendations = hybrid_engine.entity_based_recommendation(
                    entity_id=entity_id,
                    limit=limit,
                    include_paths=show_paths
                )
            except Exception as e2:
                console.print(f"[bold red]Error generating ontology recommendations: {str(e2)}[/bold red]")
                import traceback
                logger.error(f"Recommendation error: {traceback.format_exc()}")
                sys.exit(1)
        
        if not recommendations:
            console.print("[yellow]No recommendations found[/yellow]")
            return
        
        # Display recommendations
        table = Table(title=f"Recommendations for '{entity.name}'")
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Vector", style="blue", width=8)
        table.add_column("Semantic", style="green", width=8)
        table.add_column("Title", style="white")
        table.add_column("Type", style="magenta", width=15)
        
        for rec in recommendations:
            vector_score = rec.get('vector_score', 0.0)
            semantic_score = rec.get('semantic_score', 0.0)
            final_score = rec.get('final_score', rec.get('score', 0.0))
            
            table.add_row(
                f"{final_score:.4f}",
                f"{vector_score:.4f}" if vector_score is not None else "N/A",
                f"{semantic_score:.4f}" if semantic_score is not None else "N/A",
                rec['title'],
                rec['type'] or "Unknown"
            )
        
        console.print(table)
        
        # Show relationship paths if available
        if show_paths:
            for i, rec in enumerate(recommendations, 1):
                if rec.get('relationship_path'):
                    path_table = Table(title=f"Relationship path for recommendation #{i}: {rec['title']}")
                    path_table.add_column("Step", style="cyan", width=4)
                    path_table.add_column("Entity", style="white")
                    path_table.add_column("Relationship", style="green")
                    
                    path = rec['relationship_path']
                    for j, step in enumerate(path, 1):
                        entity_name = step.get('name', 'Unknown')
                        relationship = step.get('rel', 'Unknown')
                        path_table.add_row(str(j), entity_name, relationship)
                    
                    console.print(path_table)
                    console.print("")
        
    except Exception as e:
        console.print(f"[bold red]Error generating recommendations: {str(e)}[/bold red]")
        import traceback
        logger.error(f"Recommendation error: {traceback.format_exc()}")
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