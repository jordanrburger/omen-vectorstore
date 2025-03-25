# OMEN Examples

Example implementations and usage demos for the OMEN (Ontology-powered Metadata Engine) platform. This package provides working examples that demonstrate various capabilities of the platform.

## Overview

The `omen-examples` package provides practical examples and demonstration code for the OMEN platform, with these key functions:

1. **Usage Examples**: Demonstrate correct usage patterns
2. **Integration Examples**: Show how to integrate OMEN with other systems
3. **Demo Applications**: Simple applications demonstrating platform capabilities
4. **Jupyter Notebooks**: Interactive examples for data exploration
5. **Best Practices**: Showcase recommended implementation patterns

## Example Categories

### Metadata Extraction Examples

Examples of extracting metadata from various sources:

- `keboola_extraction.py`: Extract metadata from Keboola Connection
- `incremental_extraction.py`: Demonstrate incremental extraction pattern
- `custom_extractor.py`: Implement a custom metadata extractor

### Search Examples

Semantic search examples:

- `basic_search.py`: Simple vector search implementation
- `filtered_search.py`: Search with metadata type filtering
- `hybrid_search.py`: Combine vector and ontology-based search
- `batch_search.py`: Efficient batch search operations

### Ontology Examples

Knowledge graph and ontology examples:

- `ontology_creation.py`: Build a metadata ontology
- `relationship_detection.py`: Detect relationships between entities
- `ontology_query.py`: Query the ontology using high-level methods
- `sparql_query.py`: Advanced querying using SPARQL

### API Usage Examples

Examples of interacting with the OMEN API:

- `api_client.py`: Simple API client implementation
- `async_api_client.py`: Asynchronous API client
- `api_authentication.py`: API authentication patterns
- `bulk_operations.py`: Perform bulk operations via API

### Web Applications

Simple web applications using OMEN:

- `flask_app/`: Simple Flask web interface for search
- `streamlit_app/`: Interactive Streamlit dashboard
- `fastapi_proxy/`: FastAPI proxy with additional features

## Example Structure

Each example follows a common structure:

1. **Header**: Description, purpose, and requirements
2. **Imports**: Required module imports
3. **Configuration**: Setup and configuration
4. **Implementation**: Core example code
5. **Execution**: Running the example
6. **Output**: Expected output and explanation

Example:

```python
"""
Basic Search Example

This example demonstrates how to perform a basic vector search using OMEN.

Requirements:
- OpenAI API key set as environment variable
- Qdrant running on localhost:6333
"""

import os
from omen.vectorstore import VectorSearch, QdrantIndexer, get_embedding_provider

# Configuration
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your key

# Initialize components
indexer = QdrantIndexer()
embedding_provider = get_embedding_provider()
search = VectorSearch(
    indexer=indexer,
    embedding_provider=embedding_provider
)

# Perform search
results = search.search(
    query="tables with customer data",
    limit=5
)

# Display results
print(f"Found {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"{i}. [{result.score:.4f}] {result.document.id}: {result.document.content[:100]}...")
```

## Jupyter Notebooks

Interactive notebooks are available in the `notebooks/` directory:

- `01_getting_started.ipynb`: Introduction to OMEN functionality
- `02_metadata_extraction.ipynb`: Working with extractors
- `03_vector_search.ipynb`: Exploring vector search capabilities
- `04_ontology_management.ipynb`: Working with the ontology
- `05_hybrid_search.ipynb`: Combining search approaches

## Running the Examples

```bash
# Install the examples package
pip install omen-examples

# Set required environment variables
export OPENAI_API_KEY="your-api-key"
export KEBOOLA_API_TOKEN="your-keboola-token"

# Run a specific example
python -m omen.examples.search.basic_search

# Run Jupyter notebooks
cd packages/omen-examples/notebooks
jupyter notebook
```

## Using as References

These examples are designed to be used as references when implementing your own solutions. They demonstrate:

- Best practices for structuring code
- Error handling patterns
- Efficient resource usage
- Integration strategies

## Installation

```bash
# Install the examples package
pip install omen-examples

# Install with notebook dependencies
pip install 'omen-examples[notebooks]'

# Install with web app dependencies
pip install 'omen-examples[web]'

# Install all example dependencies
pip install 'omen-examples[all]'
```

For development:

```bash
git clone https://github.com/keboola/omen-platform
cd omen-platform
pip install -e "packages/omen-examples[all]"
```

## Dependencies

- Python 3.8+
- omen-core
- omen-vectorstore
- omen-ontology
- omen-api
- omen-cli
- omen-extractors
- jupyter (optional, for notebooks)
- flask, streamlit, fastapi (optional, for web examples)

## License

MIT License 