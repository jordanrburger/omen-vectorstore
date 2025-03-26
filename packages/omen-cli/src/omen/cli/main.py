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
from typing import Optional, List
import math
import random
import uuid
import json

from omen.core import configure_logging, get_logger, AppSettings
from omen.vectorstore import VectorSearch, QdrantIndexer, get_embedding_provider, HybridSearch
from omen.ontology import OntologyManager
from omen.ontology.models import Entity, EntityType, Relationship, RelationshipType
from omen.vectorstore.models import MetadataDocument, MetadataSource, MetadataType
from omen.core.state import state_manager
from omen.core.batch import BatchProcessor
from omen.ontology.manager import OntologyManager
from omen.vectorstore.embedding import get_embedding_provider

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
@click.option('--project-id', '-p', help='Show statistics for a specific project')
@click.option('--list-projects', '-l', is_flag=True, help='List all available project ontologies')
def ontology_stats(project_id, list_projects):
    """Show statistics about the ontology."""
    try:
        # List all available project ontologies if requested
        if list_projects:
            base_ontology_path = Path("state/ontology")
            project_dirs = [d for d in os.listdir(base_ontology_path) 
                           if os.path.isdir(os.path.join(base_ontology_path, d))]
            
            if not project_dirs:
                console.print("[yellow]No project-specific ontologies found[/yellow]")
                return
            
            table = Table(title="Available Project Ontologies")
            table.add_column("Project ID", style="bold")
            table.add_column("Entity Count")
            
            for project_dir in project_dirs:
                try:
                    project_path = base_ontology_path / project_dir
                    if os.path.exists(project_path / "entities.json"):
                        temp_manager = OntologyManager(state_dir=project_path)
                        temp_manager.load_state()
                        table.add_row(project_dir, str(len(temp_manager.entities)))
                except Exception as e:
                    table.add_row(project_dir, f"Error: {e}")
            
            console.print(table)
            return
        
        # Initialize the ontology manager with the specified project ID
        ontology_manager = initialize_ontology_manager(project_id)
        
        # Determine which project we're showing stats for
        project_label = f"project '{project_id}'" if project_id else "project with most entities"
        if project_id is None and len(ontology_manager.entities) == 0:
            project_label = "main ontology (empty)"
        
        # Get statistics
        entity_count = len(ontology_manager.entities)
        relationship_count = len(ontology_manager.relationships)
        triple_count = len(ontology_manager.triple_store.get_triples())
        
        # Count entities by type
        entity_types = {}
        for entity in ontology_manager.entities.values():
            entity_type = entity.type.value
            if entity_type not in entity_types:
                entity_types[entity_type] = 0
            entity_types[entity_type] += 1
        
        # Count relationships by type
        relationship_types = {}
        for rel in ontology_manager.relationships.values():
            rel_type = rel.type.value
            if rel_type not in relationship_types:
                relationship_types[rel_type] = 0
            relationship_types[rel_type] += 1
        
        # Display statistics
        table = Table(title=f"Ontology Statistics for {project_label}")
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        
        table.add_row("Total Entities", str(entity_count))
        table.add_row("Total Relationships", str(relationship_count))
        table.add_row("Triple Count", str(triple_count))
        
        # Add entity types
        table.add_row("Entity Types", "")
        for entity_type, count in sorted(entity_types.items()):
            table.add_row(f"  {entity_type}", str(count))
        
        # Add relationship types
        table.add_row("Relationship Types", "")
        for rel_type, count in sorted(relationship_types.items()):
            table.add_row(f"  {rel_type}", str(count))
        
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error getting ontology statistics: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


@ontology.command('clear')
@click.option('--project-id', '-p', help='Clear ontology for a specific project')
@click.option('--all-projects', '-a', is_flag=True, help='Clear ontology for all projects')
@click.confirmation_option(prompt='Are you sure you want to clear the ontology?')
def ontology_clear(project_id, all_projects):
    """Clear all ontology data or data for a specific project."""
    try:
        # Base ontology path
        base_ontology_path = Path("state/ontology")
        
        # If all-projects flag is set
        if all_projects:
            # Get all project directories
            project_dirs = [d for d in os.listdir(base_ontology_path) 
                           if os.path.isdir(os.path.join(base_ontology_path, d))]
            
            # Clear each project directory
            for project_dir in project_dirs:
                project_path = base_ontology_path / project_dir
                try:
                    manager = OntologyManager(state_dir=project_path)
                    manager.clear()
                    manager.save_state()
                    console.print(f"[green]Cleared ontology for project: {project_dir}[/green]")
                except Exception as e:
                    console.print(f"[red]Error clearing ontology for project {project_dir}: {e}[/red]")
            
            # Also clear the base ontology
            manager = OntologyManager(state_dir=base_ontology_path)
            manager.clear()
            manager.save_state()
            console.print("[green]Cleared base ontology[/green]")
            
            return
        
        # Clear a specific project or the base ontology
        if project_id:
            project_path = base_ontology_path / project_id
            if not os.path.exists(project_path):
                console.print(f"[yellow]Project directory {project_id} does not exist[/yellow]")
                return
                
            manager = OntologyManager(state_dir=project_path)
            manager.clear()
            manager.save_state()
            console.print(f"[bold green]Ontology cleared for project: {project_id}[/bold green]")
        else:
            # Just clear the base ontology
            manager = OntologyManager(state_dir=base_ontology_path)
            manager.clear()
            manager.save_state()
            console.print("[bold green]Base ontology cleared successfully[/bold green]")
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


@ontology.command('visualize')
@click.option('--output', '-o', type=str, default="ontology_graph.png", help='Output file for the visualization')
@click.option('--limit', '-l', type=int, default=50, help='Maximum number of nodes to display')
@click.option('--entity-type', '-t', multiple=True, help='Filter by entity type')
@click.option('--relation-type', '-r', multiple=True, help='Filter by relationship type')
@click.option('--project-id', '-p', help='Visualize ontology for a specific project')
def ontology_visualize(output, limit, entity_type, relation_type, project_id):
    """Generate a visualization of the ontology graph."""
    try:
        # Check if required packages are installed
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            from matplotlib.colors import TABLEAU_COLORS
        except ImportError:
            console.print("[bold red]Error: Required packages not installed.[/bold red]")
            console.print("Please install required packages with: pip3 install networkx matplotlib")
            sys.exit(1)
        
        # Determine which project to visualize
        project_label = f" (project: {project_id})" if project_id else ""
        console.print(f"Generating ontology visualization{project_label}, saving to {output}...")
        
        # Initialize ontology manager with the specified project
        ontology_manager = initialize_ontology_manager(project_id)
        
        # Create a NetworkX graph directly from entities and relationships
        G = nx.DiGraph()
        
        # Add nodes for entities
        for entity_id, entity in ontology_manager.entities.items():
            # Skip entities that don't match the filter
            if entity_type and entity.type.value.lower() not in [t.lower() for t in entity_type]:
                continue
                
            # Add node with attributes
            G.add_node(
                entity_id,
                label=entity.name,
                type=entity.type.value,
                description=entity.description
            )
        
        # Add edges for relationships
        for rel_id, rel in ontology_manager.relationships.items():
            # Skip relationships that don't match the filter
            if relation_type and rel.type.value.lower() not in [t.lower() for t in relation_type]:
                continue
                
            # Check if source and target exist
            if rel.source_id in G and rel.target_id in G:
                G.add_edge(
                    rel.source_id,
                    rel.target_id,
                    type=rel.type.value,
                    id=rel.id
                )
        
        # Limit number of nodes if graph is too large
        if len(G) > limit:
            console.print(f"[yellow]Warning: Graph contains {len(G)} nodes, limiting to {limit}[/yellow]")
            # Keep the most connected nodes
            node_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
            top_nodes = [n[0] for n in node_degree[:limit]]
            G = G.subgraph(top_nodes)
        
        console.print(f"Visualizing {len(G.nodes)} entities and {len(G.edges)} relationships")
        
        # Skip visualization if graph is empty
        if len(G) == 0:
            console.print("[yellow]Warning: No entities or relationships to visualize.[/yellow]")
            return
        
        # Prepare the visualization
        plt.figure(figsize=(12, 10))
        
        # Create a layout for the graph
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Color nodes by entity type
        node_colors = []
        color_map = {}
        i = 0
        
        for node, attrs in G.nodes(data=True):
            entity_type = attrs.get('type', 'unknown')
            if entity_type not in color_map:
                colors = list(TABLEAU_COLORS.values())
                color_map[entity_type] = colors[i % len(colors)]
                i += 1
            node_colors.append(color_map[entity_type])
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.8)
        
        # Draw edges with different styles for different relationship types
        edge_styles = {}
        for u, v, attrs in G.edges(data=True):
            rel_type = attrs.get('type', 'unknown')
            if rel_type not in edge_styles:
                edge_styles[rel_type] = []
            edge_styles[rel_type].append((u, v))
        
        # Draw each edge type with a different style
        for i, (rel_type, edges) in enumerate(edge_styles.items()):
            style = '-'
            if i % 3 == 1:
                style = '--'
            elif i % 3 == 2:
                style = ':'
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.5, 
                                 alpha=0.7, edge_color=f'C{i}', style=style)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        # Add a legend for entity types
        handles = []
        labels = []
        for entity_type, color in color_map.items():
            from matplotlib.lines import Line2D
            handles.append(Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=color, markersize=10))
            labels.append(entity_type)
        
        plt.legend(handles, labels, title="Entity Types", loc='upper right')
        
        # Set title and remove axes
        plt.title('Ontology Graph Visualization')
        plt.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        console.print(f"[green]Visualization saved to {output}[/green]")
        
        # Display legend for relationship types
        rel_table = Table(title="Relationship Types in Graph")
        rel_table.add_column("Relationship Type", style="cyan")
        rel_table.add_column("Count", style="magenta")
        rel_table.add_column("Style", style="green")
        
        styles = ["Solid", "Dashed", "Dotted"]
        for i, (rel_type, edges) in enumerate(edge_styles.items()):
            style = styles[i % 3]
            rel_table.add_row(rel_type, str(len(edges)), style)
        
        console.print(rel_table)
        
    except Exception as e:
        console.print(f"[bold red]Error generating visualization: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


@ontology.command('list-entities')
@click.option('--type', '-t', help='Filter by entity type')
@click.option('--limit', '-l', type=int, default=20, help='Maximum number of entities to display')
@click.option('--project-id', '-p', help='List entities from a specific project')
def ontology_list_entities(type, limit, project_id):
    """List entities in the ontology with optional filtering."""
    try:
        # Initialize ontology manager with the specified project
        ontology_manager = initialize_ontology_manager(project_id)
        
        # Filter entities by type if specified
        entities = []
        for entity_id, entity in ontology_manager.entities.items():
            if not type or entity.type.value.lower() == type.lower():
                entities.append(entity)
        
        # Sort by type and name
        entities.sort(key=lambda e: (e.type.value, e.name))
        
        # Limit number of entities
        entities = entities[:limit]
        
        if not entities:
            console.print("[yellow]No entities found matching the criteria[/yellow]")
            return
        
        # Prepare title with project information
        project_info = f" (project: {project_id})" if project_id else ""
        type_info = f" of type {type}" if type else ""
        title = f"Ontology Entities{project_info}{type_info}"
        
        # Display entities
        table = Table(title=title)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Type", style="magenta")
        table.add_column("Properties", style="blue")
        
        for entity in entities:
            # Format properties as a string
            props = []
            for k, v in entity.properties.items():
                if isinstance(v, str) and len(v) > 20:
                    v = v[:20] + "..."
                props.append(f"{k}: {v}")
            props_str = ", ".join(props) if props else ""
            
            table.add_row(
                entity.id, 
                entity.name, 
                entity.type.value,
                props_str
            )
        
        console.print(table)
        
        # Show total count if limited
        if len(ontology_manager.entities) > limit:
            console.print(f"[dim]Showing {len(entities)} of {len(ontology_manager.entities)} entities[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Error listing entities: {e}[/bold red]")
        sys.exit(1)


@ontology.command('show-relationships')
@click.argument('entity_id', type=str)
@click.option('--depth', '-d', type=int, default=1, help='Maximum relationship depth')
def ontology_show_relationships(entity_id, depth):
    """Show relationships for a specific entity."""
    try:
        ontology_manager = initialize_ontology_manager()
        
        # Check if entity exists
        if not ontology_manager.entity_exists(entity_id):
            console.print(f"[bold red]Entity '{entity_id}' not found in the ontology[/bold red]")
            
            # Show similar entities as suggestion
            similar_ids = []
            for eid in ontology_manager.entities.keys():
                if entity_id.lower() in eid.lower():
                    similar_ids.append(eid)
            
            if similar_ids:
                console.print("[yellow]Did you mean one of these?[/yellow]")
                for sid in similar_ids[:5]:
                    entity = ontology_manager.entities[sid]
                    console.print(f"  - {sid} ({entity.type.value}): {entity.name}")
            return
        
        # Get entity
        entity = ontology_manager.entities[entity_id]
        console.print(f"[bold]Relationships for {entity.name} ({entity.type.value})[/bold]")
        
        # Get related entities
        related = ontology_manager.get_related_entities(entity_id, max_depth=depth)
        
        if not related:
            console.print("[yellow]No relationships found for this entity[/yellow]")
            return
        
        # Display relationships
        incoming_table = Table(title="Incoming Relationships")
        incoming_table.add_column("From Entity", style="cyan")
        incoming_table.add_column("Type", style="green")
        incoming_table.add_column("Relationship", style="magenta")
        incoming_table.add_column("Depth", style="blue")
        
        outgoing_table = Table(title="Outgoing Relationships")
        outgoing_table.add_column("To Entity", style="cyan")
        outgoing_table.add_column("Type", style="green")
        outgoing_table.add_column("Relationship", style="magenta")
        outgoing_table.add_column("Depth", style="blue")
        
        # Sort by path length (direct relationships first)
        related.sort(key=lambda x: x.get("path_length", 999))
        
        # Split into incoming and outgoing
        incoming = []
        outgoing = []
        
        for rel in related:
            if rel.get("direction") == "incoming":
                incoming.append(rel)
            else:
                outgoing.append(rel)
        
        # Add incoming relationships
        for rel in incoming:
            source_id = rel.get("source_id", "unknown")
            source_entity = ontology_manager.entities.get(source_id)
            source_name = source_entity.name if source_entity else source_id
            source_type = source_entity.type.value if source_entity else "unknown"
            
            incoming_table.add_row(
                source_name,
                source_type,
                rel.get("relationship", "unknown"),
                str(rel.get("path_length", 0))
            )
        
        # Add outgoing relationships
        for rel in outgoing:
            target_id = rel.get("target_id", "unknown")
            target_entity = ontology_manager.entities.get(target_id)
            target_name = target_entity.name if target_entity else target_id
            target_type = target_entity.type.value if target_entity else "unknown"
            
            outgoing_table.add_row(
                target_name,
                target_type,
                rel.get("relationship", "unknown"),
                str(rel.get("path_length", 0))
            )
        
        # Print tables if they have rows
        if len(incoming) > 0:
            console.print(incoming_table)
        else:
            console.print("[dim]No incoming relationships[/dim]")
            
        if len(outgoing) > 0:
            console.print(outgoing_table)
        else:
            console.print("[dim]No outgoing relationships[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Error showing relationships: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


@ontology.command('map')
@click.option('--root-type', '-r', type=str, default='project', help='Type of entities to use as roots')
@click.option('--max-depth', '-d', type=int, default=3, help='Maximum depth to display')
@click.option('--compact/--no-compact', default=True, help='Show compact view')
@click.option('--project-id', '-p', help='Map ontology for a specific project')
def ontology_map(root_type, max_depth, compact, project_id):
    """Show a hierarchical map of the ontology relationships."""
    try:
        # Initialize ontology manager with the specified project
        ontology_manager = initialize_ontology_manager(project_id)
        
        # Find root entities of the specified type
        roots = []
        for entity_id, entity in ontology_manager.entities.items():
            if entity.type.value.lower() == root_type.lower():
                roots.append(entity)
        
        if not roots:
            console.print(f"[yellow]No entities found of type '{root_type}'[/yellow]")
            return
        
        # Sort roots by name
        roots.sort(key=lambda e: e.name)
        
        # Prepare title with project information
        project_info = f" (project: {project_id})" if project_id else ""
        console.print(f"[bold]Ontology Map{project_info} (starting from {root_type} entities)[/bold]")
        
        # Process each root entity
        for root in roots:
            _print_entity_tree(ontology_manager, root, max_depth, 0, set(), compact)
        
    except Exception as e:
        console.print(f"[bold red]Error generating ontology map: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


def _print_entity_tree(ontology_manager, entity, max_depth, current_depth, visited, compact):
    """Recursive helper to print entity hierarchy."""
    from rich.text import Text
    
    # Skip already visited entities to avoid cycles
    if entity.id in visited:
        return
    visited.add(entity.id)
    
    # Prepare indent
    indent = "  " * current_depth
    
    # Prepare entity display
    entity_str = f"{entity.name} ({entity.type.value})"
    if current_depth == 0:
        console.print(f"[bold cyan]{indent}● {entity_str}[/bold cyan]")
    else:
        prefix = "└─ " if current_depth > 0 else ""
        
        # Color based on entity type
        color = "green"
        if entity.type.value == "table":
            color = "blue"
        elif entity.type.value == "column":
            color = "magenta"
        elif entity.type.value == "configuration":
            color = "yellow"
        
        console.print(f"[{color}]{indent}{prefix}{entity_str}[/{color}]")
    
    # Stop if we've reached max depth
    if current_depth >= max_depth:
        return
    
    # Get related entities - first try tables from configurations, then columns from tables
    related = ontology_manager.get_related_entities(entity.id, max_depth=1)
    
    # For configuration entities, look for related tables (which are incoming partOf)
    if entity.type.value == "configuration":
        children = []
        for rel in related:
            if rel.get("direction") == "incoming" and rel.get("relationship") == "partOf":
                target_id = rel.get("source_id")  # Source because it's incoming
                if target_id and target_id != entity.id:
                    target_entity = ontology_manager.entities.get(target_id)
                    if target_entity and target_entity.type.value == "table":
                        rel_type = rel.get("relationship", "unknown")
                        children.append((target_entity, rel_type))
    # For tables, look for related columns (which are incoming belongsTo)
    elif entity.type.value == "table":
        children = []
        for rel in related:
            if rel.get("direction") == "incoming" and rel.get("relationship") == "belongsTo":
                target_id = rel.get("source_id")  # Source because it's incoming
                if target_id and target_id != entity.id:
                    target_entity = ontology_manager.entities.get(target_id)
                    if target_entity and target_entity.type.value == "column":
                        rel_type = rel.get("relationship", "unknown")
                        children.append((target_entity, rel_type))
    # For projects, look for related tables (which are incoming belongsTo)
    elif entity.type.value == "project":
        children = []
        for rel in related:
            if rel.get("direction") == "incoming" and rel.get("relationship") == "belongsTo":
                target_id = rel.get("source_id")  # Source because it's incoming
                if target_id and target_id != entity.id:
                    target_entity = ontology_manager.entities.get(target_id)
                    if target_entity and target_entity.type.value == "table":
                        rel_type = rel.get("relationship", "unknown")
                        children.append((target_entity, rel_type))
    # Default behavior (outgoing relationships)
    else:
        children = []
        for rel in related:
            if rel.get("direction") == "outgoing":
                target_id = rel.get("target_id")
                if target_id and target_id != entity.id:
                    target_entity = ontology_manager.entities.get(target_id)
                    if target_entity:
                        rel_type = rel.get("relationship", "unknown")
                        children.append((target_entity, rel_type))
    
    # Sort children by type and name
    children.sort(key=lambda x: (x[0].type.value, x[0].name))
    
    # Skip showing children in compact mode if there are too many of the same type
    if compact:
        # Group children by type and relationship
        child_groups = {}
        for child, rel_type in children:
            key = (child.type.value, rel_type)
            if key not in child_groups:
                child_groups[key] = []
            child_groups[key].append(child)
        
        # Display groups compactly if there are more than 3 of the same type+relationship
        processed_children = []
        for (type_val, rel_type), group in child_groups.items():
            if len(group) > 3:
                # Just take a sample of entries and add a count
                processed_children.append((group[0], rel_type, f" (+ {len(group)-1} more {type_val}s)"))
                # Add a second sample if available
                if len(group) > 1:
                    processed_children.append((group[1], rel_type, ""))
            else:
                # Add all entries individually
                for child in group:
                    processed_children.append((child, rel_type, ""))
        
        # Process compacted children
        for child, rel_type, suffix in processed_children:
            # Process this child's tree with rel_type included in display
            temp_entity = child
            if suffix:
                # Create a temporary copy with the suffix added to name
                import copy
                temp_entity = copy.copy(child)
                temp_entity.name = f"{child.name}{suffix}"
            
            _print_entity_tree(ontology_manager, temp_entity, max_depth, current_depth + 1, visited, compact)
    else:
        # Non-compact mode: process each child individually
        for i, (child, rel_type) in enumerate(children):
            # Process this child's tree
            _print_entity_tree(ontology_manager, child, max_depth, current_depth + 1, visited, compact)


@cli.group()
def extract():
    """Extract and process metadata."""
    pass


@extract.command('keboola')
@click.option('--token', '-t', help='Keboola Storage API token (project ID is auto-detected from this token)', envvar='KEBOOLA_API_TOKEN')
@click.option('--url', '-u', help='Keboola Storage API URL', envvar='KEBOOLA_API_URL')
@click.option('--project-id', '-p', help='Override project ID (optional, auto-detected from token by default)', envvar='KEBOOLA_PROJECT_ID')
@click.option('--incremental/--full', default=True, help='Use incremental extraction')
@click.option('--batch-size', default=10, help='Batch size for processing')
@click.option('--vectorize/--no-vectorize', default=True, help='Vectorize metadata after extraction')
@click.option('--ontology/--no-ontology', default=True, help='Create ontology from metadata relationships')
@click.option('--index/--no-index', default=True, help='Index vectors after vectorization')
@click.option('--clear-collection/--no-clear-collection', default=False, help='Clear the collection before indexing (only for full extraction)')
def extract_keboola(token, url, project_id, incremental, batch_size, vectorize, ontology, index, clear_collection):
    """
    Extract metadata from Keboola Storage API.
    
    The project ID is automatically detected from the provided API token, allowing
    tracking of multiple projects without manual configuration. Each project gets 
    its own state file, vector collection, and ontology storage.
    
    If you need to override the auto-detected project ID, you can provide it
    explicitly with the --project-id option.
    """
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
        
        # Create extractor - project_id will be auto-detected if not provided
        console.print("[bold]Initializing Keboola extractor...[/bold]")
        extractor = KeboolaExtractor(token=token, url=url, project_id=project_id)
        
        # Show project information
        detected_project_id = extractor.project_id
        console.print(f"[bold green]Auto-detected Keboola project ID: {detected_project_id}[/bold green]")

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
                    
                    # Ensure we have a valid project ID for the ontology directory
                    ontology_project_id = detected_project_id
                    if ontology_project_id == "unknown" and project_id:
                        # If project_id was provided manually, use that instead of "unknown"
                        ontology_project_id = project_id
                    
                    # Ensure ontology directory exists
                    ontology_path = Path(f"state/ontology/{ontology_project_id}")
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
                
                # Create collection name including project ID for separate indices
                collection_name = f"omen_{detected_project_id}"
                console.print(f"[bold]Using collection: {collection_name}[/bold]")
                indexer = QdrantIndexer(collection_name=collection_name)
                
                # Clear collection if requested (only for full extraction)
                if clear_collection and not incremental:
                    console.print(f"[bold yellow]Clearing collection {collection_name} before indexing...[/bold yellow]")
                    try:
                        indexer.client.delete_collection(collection_name=collection_name)
                        console.print("[green]Collection cleared successfully[/green]")
                        # Recreate the collection
                        indexer.ensure_collection()
                    except Exception as e:
                        console.print(f"[bold red]Error clearing collection: {e}[/bold red]")
                
                processor = MetadataProcessor(vectorizer=vectorizer, indexer=indexer, batch_size=batch_size)
                components_initialized = True
                
                # Process and vectorize
                processor.process_batch(metadata, batch_size=batch_size)
                console.print(f"[green]Processed and indexed {len(metadata)} documents[/green]")
        
        console.print("[bold green]Extraction completed successfully![/bold green]")

        if ontology and ontology_manager:
            console.print("[bold]Creating ontology from metadata relationships...[/bold]")
            
            # Track ontology creation metrics
            created_entities = 0
            created_relationships = 0
            
            # Create a mapping of metadata items by ID for relationship creation
            metadata_by_id = {
                f"{item.source.type.value}-{item.source.id}": item 
                for item in metadata
            }
            
            # First pass: create all entities
            entity_map = {}  # Track created entities
            
            for meta_item in metadata:
                try:
                    # Create entity for the item itself
                    entity_id = f"{meta_item.source.type.value}-{meta_item.source.id}"
                    
                    # Skip if entity already exists
                    if ontology_manager.entity_exists(entity_id):
                        continue
                        
                    # Determine entity name based on type - ensure it's never None
                    entity_name = meta_item.source.id  # Default fallback name is the ID
                    
                    # Try to get a more user-friendly name based on type
                    if meta_item.source.type == MetadataType.BUCKET and "bucket_name" in meta_item.metadata:
                        entity_name = meta_item.metadata["bucket_name"] or entity_name
                    elif meta_item.source.type == MetadataType.TABLE and "table_name" in meta_item.metadata:
                        entity_name = meta_item.metadata["table_name"] or entity_name
                    elif meta_item.source.type == MetadataType.CONFIGURATION and "name" in meta_item.metadata:
                        entity_name = meta_item.metadata["name"] or entity_name
                    elif meta_item.source.type == MetadataType.COLUMN:
                        # For columns, properly extract name from content
                        content_data = json.loads(meta_item.content)
                        if "name" in content_data and content_data["name"]:
                            entity_name = content_data["name"]
                        # Also try metadata as backup
                        elif "table_name" in meta_item.metadata:
                            # Extract column name from column_id (table_id.column_name)
                            parts = meta_item.source.id.split(".")
                            if len(parts) > 1:
                                entity_name = parts[-1]  # Last part should be column name
                    
                    # Final check to ensure name is never None
                    if entity_name is None or entity_name == "":
                        entity_name = f"{meta_item.source.type.value}_{meta_item.source.id}"
                    
                    # Create the entity
                    entity = Entity(
                        id=entity_id,
                        name=entity_name,
                        type=meta_item.source.type,
                        properties={
                            "source": "keboola",
                            "project_id": detected_project_id,
                            "extraction_time": str(meta_item.source.updated_at or datetime.now(timezone.utc))
                        }
                    )
                    
                    # Add type-specific properties
                    if meta_item.source.type == MetadataType.COLUMN:
                        try:
                            content = json.loads(meta_item.content)
                            if "definition" in content and isinstance(content["definition"], dict):
                                if "type" in content["definition"]:
                                    entity.properties["dataType"] = content["definition"]["type"]
                                # Add table reference for clarity
                                if "table" in content:
                                    entity.properties["table_id"] = content["table"]
                                elif "table_id" in meta_item.metadata:
                                    entity.properties["table_id"] = meta_item.metadata["table_id"]
                        except (json.JSONDecodeError, AttributeError) as e:
                            logger.warning(f"Failed to extract column properties for {entity_id}: {e}")
                    elif meta_item.source.type == MetadataType.CONFIGURATION:
                        if "component_id" in meta_item.metadata:
                            entity.properties["component_id"] = meta_item.metadata.get("component_id", "")
                        if "component_name" in meta_item.metadata:
                            entity.properties["component_name"] = meta_item.metadata.get("component_name", "")
                    
                    # Add the entity
                    ontology_manager.add_entity(entity)
                    entity_map[entity_id] = entity
                    created_entities += 1
                    
                except Exception as e:
                    logger.error(f"Error creating entity for {meta_item.source.id}: {e}")
            
            # Second pass: create all relationships from metadata
            for meta_item in metadata:
                try:
                    source_id = f"{meta_item.source.type.value}-{meta_item.source.id}"
                    
                    # Skip if source entity doesn't exist
                    if not ontology_manager.entity_exists(source_id):
                        continue
                    
                    # Create relationships from the relationships metadata
                    if "relationships" in meta_item.metadata:
                        for rel in meta_item.metadata["relationships"]:
                            rel_type = rel.get("type")
                            target_type = rel.get("target_type")
                            target_id = rel.get("target_id")
                            
                            if not rel_type or not target_type or not target_id:
                                continue
                                
                            # Construct full target ID
                            full_target_id = f"{target_type}-{target_id}"
                            
                            # Create target entity if it doesn't exist
                            if not ontology_manager.entity_exists(full_target_id):
                                # Try to find target in our metadata
                                target_entity_name = target_id  # Default fallback is the ID
                                if full_target_id in metadata_by_id:
                                    target_meta = metadata_by_id[full_target_id]
                                    if target_type == "bucket" and "bucket_name" in target_meta.metadata:
                                        target_entity_name = target_meta.metadata["bucket_name"] or target_entity_name
                                    elif target_type == "table" and "table_name" in target_meta.metadata:
                                        target_entity_name = target_meta.metadata["table_name"] or target_entity_name
                                
                                # Ensure name is valid
                                if target_entity_name is None or target_entity_name == "":
                                    target_entity_name = f"{target_type}_{target_id}"
                                
                                # Create the entity
                                target_entity = Entity(
                                    id=full_target_id,
                                    name=target_entity_name,
                                    type=EntityType(target_type),
                                    properties={
                                        "source": "keboola",
                                        "project_id": detected_project_id
                                    }
                                )
                                ontology_manager.add_entity(target_entity)
                                entity_map[full_target_id] = target_entity
                                created_entities += 1
                            
                            # Determine relationship type
                            relationship_type = None
                            if rel_type == "belongs_to":
                                relationship_type = RelationshipType.BELONGS_TO
                            elif rel_type == "uses":
                                relationship_type = RelationshipType.INPUTS_FROM
                            elif rel_type == "produces":
                                relationship_type = RelationshipType.OUTPUTS_TO
                            elif rel_type == "related_to":
                                relationship_type = RelationshipType.RELATED_TO
                            else:
                                # Default fallback
                                relationship_type = RelationshipType.RELATED_TO
                            
                            # Check if relationship already exists to avoid duplicates
                            if not ontology_manager.relationship_exists(source_id, full_target_id, relationship_type.value):
                                # Create relationship ID
                                rel_id = f"{source_id}_{relationship_type.value}_{full_target_id}"
                                
                                # Create relationship
                                relationship = Relationship(
                                    id=rel_id,
                                    source_id=source_id,
                                    target_id=full_target_id,
                                    type=relationship_type,
                                    properties={
                                        "created_at": datetime.now(timezone.utc).isoformat()
                                    }
                                )
                                ontology_manager.add_relationship(relationship)
                                created_relationships += 1
                                logger.debug(f"Created relationship: {source_id} {relationship_type.value} {full_target_id}")
                    
                    # Create project relationships for all entities
                    project_id = meta_item.source.project_id
                    if project_id:
                        full_project_id = f"project-{project_id}"
                        
                        # Create project entity if it doesn't exist
                        if not ontology_manager.entity_exists(full_project_id):
                            project_name = meta_item.metadata.get("project_name") or f"Project {project_id}"
                            
                            project_entity = Entity(
                                id=full_project_id,
                                name=project_name,
                                type=EntityType.PROJECT
                            )
                            ontology_manager.add_entity(project_entity)
                            entity_map[full_project_id] = project_entity
                            created_entities += 1
                        
                        # Skip bucket and configuration entities as they already have direct project relationships
                        if meta_item.source.type not in [MetadataType.BUCKET, MetadataType.CONFIGURATION]:
                            # Check if relationship already exists
                            if not ontology_manager.relationship_exists(source_id, full_project_id, RelationshipType.BELONGS_TO.value):
                                # Create relationship ID
                                rel_id = f"{source_id}_belongs_to_{full_project_id}"
                                
                                # Create relationship
                                relationship = Relationship(
                                    id=rel_id,
                                    source_id=source_id,
                                    target_id=full_project_id,
                                    type=RelationshipType.BELONGS_TO,
                                    properties={
                                        "created_at": datetime.now(timezone.utc).isoformat()
                                    }
                                )
                                ontology_manager.add_relationship(relationship)
                                created_relationships += 1
                                logger.debug(f"Created relationship: {source_id} belongs_to {full_project_id}")
                        
                except Exception as e:
                    logger.error(f"Error creating relationships for {meta_item.source.id}: {e}")
            
            # Save the ontology
            ontology_manager.save_state()
            console.print(f"[green]Created {created_entities} entities and {created_relationships} relationships in the ontology[/green]")

    except ImportError as e:
        console.print(f"[bold red]Error: Keboola extractor dependencies not installed[/bold red]")
        console.print(f"[bold red]Exception details: {e}[/bold red]")
        console.print("Install them with: pip install 'omen-extractors[keboola]'")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error during extraction: {e}[/bold red]")
        sys.exit(1)


def init_dirs(index_dir: Optional[str] = None):
    """Initialize directories for storing data."""
    # Create state directory
    state_dir = index_dir or Path("state")
    os.makedirs(state_dir, exist_ok=True)
    
    # Create ontology directory
    ontology_dir = Path(state_dir) / "ontology"
    os.makedirs(ontology_dir, exist_ok=True)
    
    return state_dir


@extract.command('sample')
@click.option('--count', '-c', type=int, default=10, help='Number of sample metadata items to generate')
@click.option('--vectorize/--no-vectorize', default=True, help='Vectorize sample metadata')
@click.option('--ontology/--no-ontology', default=False, help='Create ontology from metadata relationships')
@click.option('--index-dir', type=str, default=None, help='Directory for storing index data')
def extract_sample(count: int = 10, vectorize: bool = True, ontology: bool = False, index_dir: Optional[str] = None):
    """Extract sample metadata for testing."""
    metadata_items = []
    
    # Configure consistent IDs for entities to build proper relationships
    project_ids = [f"project-{i}" for i in range(1, 4)]
    config_ids = [f"config-{i}" for i in range(1, 4)]
    
    # Generate sample metadata
    print(f"Generating {count} sample metadata items...")
    
    for i in range(count):
        # Create consistent entity structure for better relationship building
        project_id = project_ids[i % len(project_ids)]
        config_id = config_ids[(i // 3) % len(config_ids)]
        table_id = f"table-{(i % 2) + 1}"
        
        # Create a sample table with meaningful structure
        source = MetadataSource(
            id=str(uuid.uuid4()),
            type=MetadataType.TABLE,
            project_id=project_id
        )
        
        # Add columns for better hierarchical relationships
        columns = []
        for j in range(3):
            columns.append({
                "id": f"column-{j+1}-{source.id}",
                "name": f"Column {j+1}",
                "type": "string",
                "description": f"Sample column {j+1} in table {(i % 2) + 1}",
                "parent_id": source.id,
                "parent_type": "table"
            })
        
        # Create sample metadata with meaningful content
        metadata = MetadataDocument(
            id=str(uuid.uuid4()),
            source=source,
            content=f"Sample table {(i % 2) + 1} in configuration {(i // 3) % 3 + 1} in project {i % 3 + 1}",
            metadata={
                "row_count": random.randint(100, 10000),
                "size_bytes": random.randint(1000, 1000000),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "related_tables": [f"table-{(i+1) % 2 + 1}"],
                "schema": "public",
                "tags": ["sample", f"project-{i % 3 + 1}", f"config-{(i // 3) % 3 + 1}"],
                "columns": columns,
                "parent_id": config_id,
                "parent_type": "configuration"
            }
        )
        
        metadata_items.append(metadata)
    
    print(f"Generated {len(metadata_items)} sample metadata items")
    
    # Initialize dirs
    init_dirs(index_dir=index_dir)
    
    # Initialize ontology if requested
    if ontology:
        ontology_manager = initialize_ontology_manager()
        print("Initialized ontology manager")
        print("Loaded existing ontology state")
    
    # Process and vectorize items if requested
    if vectorize:
        print("Processing and vectorizing metadata...")
        
        # Initialize components for vectorization
        from omen.vectorstore.vectorizer import Vectorizer
        from omen.vectorstore.processor import MetadataProcessor
        from omen.vectorstore.indexer import QdrantIndexer
        
        # Create processor with vectorizer and indexer
        indexer = QdrantIndexer()
        vectorizer = Vectorizer(embedding_provider=get_embedding_provider())
        processor = MetadataProcessor(vectorizer=vectorizer, indexer=indexer)
        
        # Process batch
        processor.process_batch(metadata_items)
        print(f"Processed and indexed {len(metadata_items)} documents")
    
    # Create ontology from metadata relationships
    if ontology and metadata_items:
        print("Creating ontology from metadata relationships...")
        
        # Track stats for reporting
        entities_created = 0
        relationships_created = 0
        
        # First pass: create all entities with their proper types
        entity_map = {}  # Keep track of created entities by their semantic ID
        
        for item in metadata_items:
            # 1. Create table entity
            table_id = f"table-{item.source.id}"
            if not ontology_manager.entity_exists(table_id):
                table_entity = Entity(
                    id=table_id,
                    name=f"Table {item.source.id[:8]}",
                    type=item.source.type
                )
                ontology_manager.add_entity(table_entity)
                entity_map[table_id] = table_entity
                entities_created += 1
            
            # 2. Create project entity
            if hasattr(item.source, "project_id") and item.source.project_id:
                project_id = f"project-{item.source.project_id}"
                if not ontology_manager.entity_exists(project_id):
                    project_entity = Entity(
                        id=project_id,
                        name=f"Project {item.source.project_id}",
                        type=MetadataType.PROJECT
                    )
                    ontology_manager.add_entity(project_entity)
                    entity_map[project_id] = project_entity
                    entities_created += 1
            
            # 3. Create configuration entity
            if "parent_id" in item.metadata and item.metadata["parent_id"]:
                config_id = f"configuration-{item.metadata['parent_id']}"
                if not ontology_manager.entity_exists(config_id):
                    config_entity = Entity(
                        id=config_id,
                        name=f"Configuration {item.metadata['parent_id']}",
                        type=MetadataType.CONFIGURATION
                    )
                    ontology_manager.add_entity(config_entity)
                    entity_map[config_id] = config_entity
                    entities_created += 1
            
            # 4. Create column entities if available
            if "columns" in item.metadata and item.metadata["columns"]:
                for column in item.metadata["columns"]:
                    column_id = f"column-{column['id']}"
                    if not ontology_manager.entity_exists(column_id):
                        column_entity = Entity(
                            id=column_id,
                            name=column['name'],
                            type=MetadataType.COLUMN
                        )
                        ontology_manager.add_entity(column_entity)
                        entity_map[column_id] = column_entity
                        entities_created += 1
        
        # Second pass: create relationships between entities
        for item in metadata_items:
            table_id = f"table-{item.source.id}"
            
            # 1. Create table-project relationship
            if hasattr(item.source, "project_id") and item.source.project_id:
                project_id = f"project-{item.source.project_id}"
                relation_id = f"{table_id}-belongs_to-{project_id}"
                
                if not ontology_manager.relationship_exists(table_id, project_id, RelationshipType.BELONGS_TO.value):
                    relationship = Relationship(
                        id=relation_id,
                        source_id=table_id,
                        target_id=project_id,
                        type=RelationshipType.BELONGS_TO,
                        properties={"created_at": datetime.now(timezone.utc).isoformat()}
                    )
                    ontology_manager.add_relationship(relationship)
                    relationships_created += 1
                    print(f"Created relationship: {table_id} belongs_to {project_id}")
            
            # 2. Create table-configuration relationship
            if "parent_id" in item.metadata and item.metadata["parent_id"] and "parent_type" in item.metadata and item.metadata["parent_type"] == "configuration":
                config_id = f"configuration-{item.metadata['parent_id']}"
                relation_id = f"{table_id}-part_of-{config_id}"
                
                if not ontology_manager.relationship_exists(table_id, config_id, RelationshipType.PART_OF.value):
                    relationship = Relationship(
                        id=relation_id,
                        source_id=table_id,
                        target_id=config_id,
                        type=RelationshipType.PART_OF,
                        properties={"created_at": datetime.now(timezone.utc).isoformat()}
                    )
                    ontology_manager.add_relationship(relationship)
                    relationships_created += 1
                    print(f"Created relationship: {table_id} part_of {config_id}")
            
            # 3. Create column-table relationships
            if "columns" in item.metadata and item.metadata["columns"]:
                for column in item.metadata["columns"]:
                    column_id = f"column-{column['id']}"
                    relation_id = f"{column_id}-belongs_to-{table_id}"
                    
                    if not ontology_manager.relationship_exists(column_id, table_id, RelationshipType.BELONGS_TO.value):
                        relationship = Relationship(
                            id=relation_id,
                            source_id=column_id,
                            target_id=table_id,
                            type=RelationshipType.BELONGS_TO,
                            properties={"created_at": datetime.now(timezone.utc).isoformat()}
                        )
                        ontology_manager.add_relationship(relationship)
                        relationships_created += 1
                        print(f"Created relationship: {column_id} belongs_to {table_id}")
            
            # 4. Create relationships between related tables if specified in metadata
            if hasattr(item, "metadata") and "related_tables" in item.metadata:
                for related_table_id in item.metadata["related_tables"]:
                    full_related_id = f"table-{related_table_id}"
                    relation_id = f"{table_id}-related_to-{full_related_id}"
                    
                    # Check if the related table entity exists before creating relationship
                    if ontology_manager.entity_exists(full_related_id) and not ontology_manager.relationship_exists(table_id, full_related_id, RelationshipType.RELATED_TO.value):
                        relationship = Relationship(
                            id=relation_id,
                            source_id=table_id,
                            target_id=full_related_id,
                            type=RelationshipType.RELATED_TO,
                            properties={"created_at": datetime.now(timezone.utc).isoformat()}
                        )
                        ontology_manager.add_relationship(relationship)
                        relationships_created += 1
                        print(f"Created relationship: {table_id} related_to {full_related_id}")
        
        # 5. Create semantic relationships between tables in the same configuration
        for entity_id, entity in entity_map.items():
            if entity.type == MetadataType.TABLE:
                for other_id, other_entity in entity_map.items():
                    if other_id != entity_id and other_entity.type == MetadataType.TABLE:
                        # Extract configuration info from the names to connect tables in same config
                        entity_config = None
                        other_config = None
                        if "Configuration" in entity.name:
                            entity_config = entity.name.split("Configuration")[1].split()[0]
                        if "Configuration" in other_entity.name:
                            other_config = other_entity.name.split("Configuration")[1].split()[0]
                            
                        if entity_config and other_config and entity_config == other_config:
                            relation_id = f"{entity_id}-same_configuration-{other_id}"
                            if not ontology_manager.relationship_exists(entity_id, other_id, RelationshipType.RELATED_TO.value):
                                relationship = Relationship(
                                    id=relation_id,
                                    source_id=entity_id,
                                    target_id=other_id,
                                    type=RelationshipType.RELATED_TO,
                                    properties={"configuration": entity_config}
                                )
                                ontology_manager.add_relationship(relationship)
                                relationships_created += 1
                                print(f"Created semantic relationship: {entity_id} same_configuration {other_id}")
        
        # Save the ontology
        ontology_manager.save_state()
        print(f"Created {entities_created} entities and {relationships_created} relationships in the ontology")
    
    print("Sample extraction completed successfully!")


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


def initialize_ontology_manager(project_id: Optional[str] = None):
    """Initialize the ontology manager for semantic search.
    
    Args:
        project_id: If provided, load ontology data for the specific project.
                   If None, loads from the main ontology directory and/or 
                   combines data from all project subdirectories.
    """
    from omen.ontology.manager import OntologyManager
    import glob
    
    # Base ontology path
    base_ontology_path = Path("state/ontology")
    os.makedirs(base_ontology_path, exist_ok=True)
    
    # If a specific project ID is provided, use that project's ontology
    if project_id:
        project_path = base_ontology_path / project_id
        os.makedirs(project_path, exist_ok=True)
        ontology_manager = OntologyManager(state_dir=project_path)
        ontology_manager.load_state()
        return ontology_manager
    
    # For ontology stats and other general commands, we want to check
    # both the base directory and all project directories
    
    # Check if there are project subdirectories and if they contain more data
    project_dirs = [d for d in os.listdir(base_ontology_path) 
                   if os.path.isdir(os.path.join(base_ontology_path, d))]
    
    # If no project subdirectories, just use the base ontology
    if not project_dirs:
        ontology_manager = OntologyManager(state_dir=base_ontology_path)
        ontology_manager.load_state()
        return ontology_manager
    
    # Check if project dirs contain more entities than the base dir
    base_manager = OntologyManager(state_dir=base_ontology_path)
    base_manager.load_state()
    
    # Find the project with the most entities
    max_entities = len(base_manager.entities)
    max_entity_project = None
    
    for project_dir in project_dirs:
        project_path = base_ontology_path / project_dir
        if os.path.exists(project_path / "entities.json"):
            try:
                temp_manager = OntologyManager(state_dir=project_path)
                temp_manager.load_state()
                entity_count = len(temp_manager.entities)
                if entity_count > max_entities:
                    max_entities = entity_count
                    max_entity_project = project_dir
            except Exception as e:
                logger.warning(f"Couldn't load ontology from {project_dir}: {e}")
    
    # Use the project directory with the most entities
    if max_entity_project:
        logger.info(f"Using ontology from project directory '{max_entity_project}' with {max_entities} entities")
        ontology_manager = OntologyManager(state_dir=base_ontology_path / max_entity_project)
        ontology_manager.load_state()
        return ontology_manager
    
    # Fallback to base ontology
    return base_manager


def initialize_hybrid_search():
    """Initialize the hybrid search engine."""
    # Initialize vector search components
    vector_search = initialize_vectorstore()
    
    # Initialize ontology if available
    try:
        ontology_manager = initialize_ontology_manager()
    except Exception as e:
        logger.warning(f"Could not initialize ontology for hybrid search: {e}")
        ontology_manager = None
    
    # Initialize hybrid search engine
    from omen.vectorstore.hybrid_search import HybridSearch
    return HybridSearch(
        vector_search=vector_search,
        ontology_manager=ontology_manager
    )


@search.command("hybrid")
@click.argument("query", type=str)
@click.option("--limit", "-l", type=int, default=10, help="Maximum number of results")
@click.option("--vector-weight", "-v", type=float, default=0.7, help="Weight for vector search results (0-1)")
@click.option("--semantic-weight", "-s", type=float, default=0.3, help="Weight for semantic search results (0-1)")
@click.option("--type", "-t", multiple=True, help="Filter by metadata type")
@click.option("--include-related/--no-related", default=False, help="Include related entities from the ontology")
@click.option("--related-depth", "-d", type=int, default=1, help="Maximum depth for related entities")
def search_hybrid_command(query, limit, vector_weight, semantic_weight, type, include_related, related_depth):
    """Search using hybrid vector and semantic search."""
    console = Console()
    
    try:
        # Initialize hybrid search engine
        hybrid_engine = initialize_hybrid_search()
        
        # Normalize weights for display
        total_weight = vector_weight + semantic_weight
        normalized_vector_weight = vector_weight / total_weight
        normalized_semantic_weight = semantic_weight / total_weight
        
        console.print(f"[bold]Running hybrid search with weights:[/bold] "
                     f"Vector {normalized_vector_weight:.2f}, Semantic {normalized_semantic_weight:.2f}")
        
        # Perform search
        results = hybrid_engine.search(
            query=query,
            vector_weight=normalized_vector_weight,
            semantic_weight=normalized_semantic_weight,
            limit=limit,
            filter_by_metadata_type=[t for t in type] if type else None,
            include_related_entities=include_related,
            max_depth=related_depth
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
                    rel_table = Table(title=f"Related entities for result #{i}: {result.document.content[:30]}...")
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


def initialize_vectorstore():
    """Initialize the vector store for search operations."""
    # Initialize vector search components
    from omen.vectorstore.indexer import QdrantIndexer
    from omen.vectorstore.search import VectorSearch
    from omen.vectorstore.embedding import get_embedding_provider
    
    indexer = QdrantIndexer()
    embedding_provider = get_embedding_provider()
    
    # Create and return the vector search engine
    return VectorSearch(
        indexer=indexer,
        embedding_provider=embedding_provider
    )


@cli.group()
def projects():
    """Manage multiple Keboola projects."""
    pass


@projects.command('list')
def projects_list():
    """List all indexed projects."""
    try:
        from pathlib import Path
        import json
        import os
        
        state_dir = os.getenv("OMEN_STATE_DIR", os.path.expanduser("~/.omen"))
        projects_found = {}
        
        # Look for project state files
        for file in Path(state_dir).glob("keboola_state_*.json"):
            try:
                # Extract project ID from filename
                filename = file.name
                project_id = filename.replace("keboola_state_", "").replace(".json", "")
                
                # Load state file to get metadata
                with open(file) as f:
                    state = json.load(f)
                    
                last_run = state.get("last_run", "Never")
                num_tables = len(state.get("processed_tables", []))
                num_buckets = len(state.get("processed_buckets", []))
                
                projects_found[project_id] = {
                    "last_run": last_run,
                    "tables": num_tables,
                    "buckets": num_buckets,
                    "state_file": str(file)
                }
            except Exception as e:
                console.print(f"[yellow]Warning: Error processing {file}: {e}[/yellow]")
        
        # Check vector collections for each project
        try:
            from omen.vectorstore import QdrantIndexer
            
            # Create temporary client to list collections
            indexer = QdrantIndexer()
            collections = indexer.client.get_collections().collections
            
            # Find project collections
            for collection in collections:
                if collection.name.startswith("omen_"):
                    try:
                        project_id = collection.name.replace("omen_", "")
                        
                        # Count documents
                        count_result = indexer.client.count(collection_name=collection.name)
                        doc_count = count_result.count
                        
                        # Add or update project info
                        if project_id in projects_found:
                            projects_found[project_id]["collection"] = collection.name
                            projects_found[project_id]["document_count"] = doc_count
                        else:
                            projects_found[project_id] = {
                                "last_run": "Unknown",
                                "tables": "Unknown",
                                "buckets": "Unknown",
                                "collection": collection.name,
                                "document_count": doc_count
                            }
                    except Exception as e:
                        console.print(f"[yellow]Warning: Error processing collection {collection.name}: {e}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: Error checking vector collections: {e}[/yellow]")
        
        # Display results
        if not projects_found:
            console.print("[yellow]No projects found[/yellow]")
            return
            
        console.print("[bold]Indexed projects:[/bold]")
        
        # Create table
        from rich.table import Table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Project ID")
        table.add_column("Last Run")
        table.add_column("Tables")
        table.add_column("Buckets")
        table.add_column("Documents")
        table.add_column("Collection")
        
        for project_id, info in sorted(projects_found.items()):
            table.add_row(
                project_id,
                info.get("last_run", "Never"),
                str(info.get("tables", "Unknown")),
                str(info.get("buckets", "Unknown")),
                str(info.get("document_count", "Unknown")),
                info.get("collection", "None")
            )
            
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error listing projects: {e}[/bold red]")


@projects.command('delete')
@click.argument('project_id', type=str)
@click.option('--state/--no-state', default=True, help='Delete project state file')
@click.option('--documents/--no-documents', default=True, help='Delete project documents from vector store')
@click.option('--ontology/--no-ontology', default=True, help='Delete project ontology')
@click.confirmation_option(prompt='Are you sure you want to delete this project?')
def projects_delete(project_id, state, documents, ontology):
    """Delete a project's data (state, documents, and/or ontology)."""
    try:
        from pathlib import Path
        import os
        import shutil
        
        state_dir = os.getenv("OMEN_STATE_DIR", os.path.expanduser("~/.omen"))
        state_file = Path(state_dir) / f"keboola_state_{project_id}.json"
        
        # Delete state file if requested
        if state and state_file.exists():
            try:
                state_file.unlink()
                console.print(f"[green]Deleted state file for project {project_id}[/green]")
            except Exception as e:
                console.print(f"[bold red]Error deleting state file: {e}[/bold red]")
        elif state:
            console.print(f"[yellow]No state file found for project {project_id}[/yellow]")
        
        # Delete ontology if requested
        if ontology:
            try:
                ontology_dir = Path(f"state/ontology/{project_id}")
                if ontology_dir.exists():
                    shutil.rmtree(ontology_dir)
                    console.print(f"[green]Deleted ontology for project {project_id}[/green]")
                else:
                    console.print(f"[yellow]No ontology found for project {project_id}[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error deleting ontology: {e}[/bold red]")
        
        # Delete documents if requested
        if documents:
            try:
                from omen.vectorstore import QdrantIndexer
                
                # Create indexer with project-specific collection
                collection_name = f"omen_{project_id}"
                indexer = QdrantIndexer(collection_name=collection_name)
                
                # Delete project documents
                deleted_count = indexer.delete_project_documents(project_id)
                
                if deleted_count > 0:
                    console.print(f"[green]Deleted {deleted_count} documents for project {project_id}[/green]")
                else:
                    console.print(f"[yellow]No documents found for project {project_id}[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error deleting documents: {e}[/bold red]")
        
        console.print(f"[bold green]Successfully cleaned up project {project_id}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error deleting project: {e}[/bold red]")


if __name__ == '__main__':
    cli() 