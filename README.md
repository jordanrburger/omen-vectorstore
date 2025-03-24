# Keboola Vector Store

A powerful vector store and ontology system for Keboola metadata, enabling semantic search and knowledge graph capabilities.

## Features

- **Vector Store**
  - Semantic search using embeddings
  - Support for multiple embedding providers (OpenAI, SentenceTransformer)
  - Batch processing and incremental updates
  - Efficient similarity search with Qdrant

- **Ontology System**
  - LLM-powered entity extraction from metadata
  - Relationship detection between entities
  - SPARQL query support
  - Schema validation and compliance
  - Parallel batch processing with retries

- **Hybrid Search**
  - Combine vector-based semantic search with ontology-based structured search
  - Rich context in search results
  - Filtering by metadata type and relationships

## Architecture

The system follows a four-step architecture:

1. **Metadata Extraction** (using Keboola SAPI)
   - Extract comprehensive metadata from Keboola
   - Include bucket information, table details, and column statistics
   - Capture quality metrics and transformation code

2. **Metadata Processing and Vectorization**
   - Process metadata into normalized documents
   - Generate embeddings using configured providers
   - Extract entities and relationships for ontology

3. **Indexing**
   - Store embeddings in Qdrant vector store
   - Build and maintain ontology knowledge graph
   - Support incremental updates

4. **Search and Recommendation API**
   - Provide unified search interface
   - Support both semantic and structured queries
   - Enable hybrid search capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/keboola/omen-vectorstore.git
cd omen-vectorstore
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Vector Store

```python
from app.vector_store import VectorStore
from app.llm_client import LLMClient

# Initialize components
vector_store = VectorStore()
llm_client = LLMClient(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key"
)

# Index metadata
vector_store.index_metadata(metadata_list)

# Search
results = vector_store.search(
    query="Find tables related to customer support",
    limit=10
)
```

### Ontology System

```python
from app.ontology.builder import OntologyBuilder
from app.ontology.manager import OntologyManager

# Initialize components
ontology_manager = OntologyManager()
builder = OntologyBuilder(llm_client, ontology_manager)

# Build ontology
builder.build_ontology(metadata_list)

# Query ontology
results = ontology_manager.query("""
    SELECT ?table ?column
    WHERE {
        ?table a :Table ;
              :hasColumn ?column .
        ?column :type "string" .
    }
""")
```

### Hybrid Search

```python
from app.search import HybridSearch

# Initialize hybrid search
search = HybridSearch(ontology_manager, vector_store)

# Perform hybrid search
results = search.search(
    query="Find tables related to customer support",
    limit=10,
    use_ontology=True,
    use_vector=True
)
```

## Configuration

### Vector Store Settings

```python
vector_store = VectorStore(
    collection_name="keboola_metadata",
    embedding_dimension=1536,
    batch_size=100,
    max_retries=3
)
```

### Ontology Settings

```python
builder = OntologyBuilder(
    llm_client=llm_client,
    ontology_manager=ontology_manager,
    batch_size=10,
    max_workers=4,
    max_retries=3
)
```

### LLM Settings

```python
llm_client = LLMClient(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=2000,
    max_retries=3
)
```

## Development

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test suite
python -m pytest tests/vector_store/
python -m pytest tests/ontology/
```

### Code Style

```bash
# Format code
black .

# Check types
mypy .

# Run linter
flake8
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
