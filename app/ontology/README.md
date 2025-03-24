# Keboola Ontology Module

This module provides a semantic knowledge graph for Keboola metadata, using RDF triples and LLM-powered extraction. It enables advanced querying, relationship discovery, and action graph generation for Keboola components.

## Overview

The Ontology Module transforms raw Keboola metadata into a rich, interlinked knowledge graph using a formal schema definition. It leverages LLMs to extract entities and relationships from unstructured and semi-structured metadata, creating a queryable semantic representation of the entire Keboola project.

The ontology module is designed to work alongside the vector store to provide a comprehensive knowledge graph of your Keboola metadata. While the vector store provides semantic search capabilities, the ontology provides structured relationships and hierarchical organization of your metadata.

### Key Features

- Entity extraction from metadata using LLMs
- Relationship detection between entities
- Schema validation for entities and relationships
- SPARQL query support
- Incremental updates
- Parallel batch processing with retries
- Integration with vector store for hybrid search

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

The module follows a layered architecture:

1. **Models Layer** (`models.py`)
   - Core data structures for entities, relationships, and triples
   - Type definitions for entity and relationship types

2. **Schema Layer** (`schema.py`, `schema_definition.py`)
   - Schema validation for entities and relationships
   - Default schema definitions for Keboola metadata

3. **Builder Layer** (`builder.py`)
   - LLM-powered entity extraction and relationship detection
   - Batch processing with retries
   - Parallel processing for improved performance

4. **Manager Layer** (`manager.py`)
   - Ontology state management
   - CRUD operations for entities and relationships
   - SPARQL query support

5. **Store Layer** (`store.py`)
   - RDF-based storage using RDFLib
   - SPARQL endpoint support
   - Serialization and deserialization

6. **LLM Utilities** (`llm_utils.py`)
   - Semantic matching
   - Natural language query conversion
   - Explanation generation
   - Ontology validation

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

## Usage

### Basic Usage

```python
from app.ontology.builder import OntologyBuilder
from app.ontology.manager import OntologyManager
from app.llm_client import LLMClient

# Initialize components
llm_client = LLMClient(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key"
)
ontology_manager = OntologyManager()
builder = OntologyBuilder(llm_client, ontology_manager)

# Build ontology from metadata
metadata_list = [
    {
        "type": "bucket",
        "data": {
            "id": "in.c-slack",
            "name": "Slack Data",
            "description": "Slack conversation data"
        }
    },
    {
        "type": "table",
        "data": {
            "id": "in.c-slack.messages",
            "name": "Messages",
            "bucket": "in.c-slack",
            "columns": [
                {"name": "message_id", "type": "string"},
                {"name": "channel", "type": "string"},
                {"name": "text", "type": "string"}
            ]
        }
    }
]

builder.build_ontology(metadata_list)
```

### Integration with Vector Store

The ontology module can be used alongside the vector store to provide hybrid search capabilities:

```python
from app.ontology.manager import OntologyManager
from app.vector_store import VectorStore
from app.search import HybridSearch

# Initialize components
ontology_manager = OntologyManager()
vector_store = VectorStore()
search = HybridSearch(ontology_manager, vector_store)

# Perform hybrid search
results = search.search(
    query="Find tables related to customer support",
    limit=10,
    use_ontology=True,  # Enable ontology-based search
    use_vector=True     # Enable vector-based search
)
```

### SPARQL Queries

```python
from app.ontology.manager import OntologyManager

# Initialize ontology manager
manager = OntologyManager()

# Execute SPARQL query
query = """
SELECT ?table ?column
WHERE {
    ?table a :Table ;
          :hasColumn ?column .
    ?column :type "string" .
}
"""
results = manager.query(query)
```

## Configuration

### LLM Settings

The ontology builder uses LLMs for entity extraction and relationship detection. Configure the LLM client with appropriate settings:

```python
llm_client = LLMClient(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=2000,
    max_retries=3,
    retry_delay=1.0
)
```

### Batch Processing

Configure batch processing settings in the ontology builder:

```python
builder = OntologyBuilder(
    llm_client=llm_client,
    ontology_manager=ontology_manager,
    batch_size=10,
    max_workers=4,
    max_retries=3,
    retry_delay=1.0
)
```

## Error Handling

The module includes comprehensive error handling:

1. **LLM Errors**
   - Retries with exponential backoff
   - Error logging and tracking
   - Graceful degradation

2. **Schema Validation**
   - Entity and relationship validation
   - Detailed error messages
   - Schema compliance checking

3. **Processing Errors**
   - Batch-level error handling
   - Individual item error tracking
   - Failed item reporting

## Performance Optimization

1. **Parallel Processing**
   - Batch processing with multiple workers
   - Configurable batch sizes
   - Thread pool management

2. **Caching**
   - Schema prompt caching
   - Entity type caching
   - Relationship type caching

3. **Incremental Updates**
   - Efficient ontology updates
   - Minimal reprocessing
   - State management

## Testing

Run the test suite:

```bash
python -m pytest tests/ontology/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This module is part of the Keboola Vector Store project and is licensed under the MIT License. 