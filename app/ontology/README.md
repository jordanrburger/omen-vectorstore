# Keboola Ontology Module

This module provides a semantic knowledge graph for Keboola metadata, using RDF triples and LLM-powered extraction. It enables advanced querying, relationship discovery, and action graph generation for Keboola components.

## Overview

The Ontology Module transforms raw Keboola metadata into a rich, interlinked knowledge graph using a formal schema definition. It leverages LLMs to extract entities and relationships from unstructured and semi-structured metadata, creating a queryable semantic representation of the entire Keboola project.

## Core Components

### 1. RDF Triple Structure

- `models.py`: Defines the core data structures (Entity, Relationship, Triple) that form the foundation of the ontology
- `manager.py`: Provides the OntologyManager for storing, retrieving, and querying triples
- `storage.py`: Handles persistence of the ontology to disk or other storage backends

### 2. Schema Definition

- `schema.py`: Contains the base schema classes for defining and validating entity and relationship types
- `schema_definition.py`: Implements the complete Keboola ontology schema with all entity and relationship types

### 3. LLM-Powered Ontology Construction

- `builder.py`: Provides the OntologyBuilder for extracting entities and relationships from metadata
- `prompts.py`: Contains prompt templates and formatting utilities for LLM-based extraction
- `llm_client.py`: Implements a client for communicating with language models

### 4. Action Graph Generation

- `action_graph.py` (coming soon): Will provide functionality to generate action graphs from the ontology

## Usage Examples

### Creating an Ontology Manager

```python
from app.ontology import OntologyManager, Entity, EntityType, Relationship, RelationshipType

# Create a new ontology manager
ontology = OntologyManager()

# Add entities and relationships
table_entity = Entity(
    id="table_123",
    type=EntityType.TABLE,
    properties={"name": "customers", "rows": 1000}
)

column_entity = Entity(
    id="column_456",
    type=EntityType.COLUMN,
    properties={"name": "email", "data_type": "string"}
)

has_column_rel = Relationship(
    id="rel_789",
    type=RelationshipType.HAS_COLUMN,
    source_id="table_123",
    target_id="column_456",
    properties={}
)

ontology.add_entity(table_entity)
ontology.add_entity(column_entity)
ontology.add_relationship(has_column_rel)

# Query the ontology
table = ontology.get_entity("table_123")
columns = ontology.get_entities_by_type(EntityType.COLUMN)
relationships = ontology.get_relationships_for_entity("table_123")
```

### Building an Ontology from Metadata

```python
from app.ontology import OntologyBuilder
from app.llm_client import LLMClient

# Initialize the LLM client
llm_client = LLMClient(
    provider="openai",
    model="gpt-4"
)

# Create an ontology builder
builder = OntologyBuilder(
    llm_client=llm_client,
    batch_size=10,
    max_workers=4
)

# Build ontology from metadata
metadata_collection = [...]  # List of metadata items
ontology = builder.build_ontology(metadata_collection)

# Use the ontology
entities = ontology.get_all_entities()
relationships = ontology.get_all_relationships()
triples = ontology.get_all_triples()
```

### Validating Against the Schema

```python
from app.ontology import SchemaValidator, Entity, EntityType
from app.ontology.schema_definition import default_schema

# Create a schema validator
validator = SchemaValidator(default_schema)

# Create an entity
entity = Entity(
    id="table_123",
    type=EntityType.TABLE,
    properties={"name": "customers"}
)

# Validate the entity
is_valid, errors = validator.validate_entity(entity)
if not is_valid:
    print(f"Entity validation failed: {errors}")
```

## Architecture

The Ontology Module uses a layered architecture:

1. **Data Layer**: RDF triples as the base representation, with Entity and Relationship abstractions on top
2. **Schema Layer**: Formal definition of entity and relationship types with validation rules
3. **Construction Layer**: LLM-powered extraction of entities and relationships from metadata
4. **Query Layer**: APIs for querying and traversing the ontology graph
5. **Integration Layer**: Connections to other components in the system

## Dependencies

- `rdflib`: For RDF data representation and SPARQL querying
- `networkx`: For graph operations and visualization
- `openai`: For LLM-based entity and relationship extraction
- `tenacity`: For retry logic with LLM API calls
- `pydantic` (indirectly): For data validation in the schema

## Getting Started

1. Ensure required dependencies are installed: `pip3 install -r requirements.txt`
2. Set up environment variables (e.g., `OPENAI_API_KEY` for LLM access)
3. Import the necessary components from the `app.ontology` module
4. Create an ontology manager or builder as shown in the examples above

## Best Practices

1. **Schema First**: Always define your entity and relationship types in the schema before attempting to build the ontology
2. **Batch Processing**: Use batch processing for large metadata sets to optimize performance
3. **Validation**: Validate all entities and relationships against the schema to ensure consistency
4. **Persistence**: Regularly persist the ontology to avoid data loss
5. **Incremental Updates**: Use incremental updates for evolving metadata rather than rebuilding the entire ontology 