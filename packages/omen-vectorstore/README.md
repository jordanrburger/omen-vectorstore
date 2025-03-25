# OMEN Vectorstore

Vector storage and semantic search functionality for the OMEN (Ontology-powered Metadata Engine) platform. This package handles document vectorization, indexing, and vector similarity search.

## Overview

The `omen-vectorstore` package provides the core vector search capabilities of the OMEN platform, with these key responsibilities:

1. **Embedding Generation**: Convert text documents into vector embeddings
2. **Vector Indexing**: Store and retrieve vectors efficiently
3. **Semantic Search**: Find semantically similar documents using vector similarity
4. **Metadata Processing**: Transform raw metadata into searchable documents
5. **Batch Operations**: Efficient batch processing for large datasets

## Key Components

### Embedding Providers (`omen.vectorstore.embedding`)

Support for multiple embedding models:

```python
from omen.vectorstore import get_embedding_provider
from omen.vectorstore.embedding import OpenAIEmbeddingProvider, SentenceTransformerProvider

# Get the default embedding provider from configuration
embedding_provider = get_embedding_provider()

# Or use a specific provider
openai_provider = OpenAIEmbeddingProvider(
    model="text-embedding-3-large",
    api_key="your-api-key"
)

# Generate embeddings
embedding = embedding_provider.embed("This is a text document")
```

Supported providers:
- OpenAI (text-embedding-3-large, text-embedding-3-small, etc.)
- SentenceTransformer (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)

### Vector Indexing (`omen.vectorstore.indexer`)

Store and retrieve vectors using Qdrant:

```python
from omen.vectorstore.indexer import QdrantIndexer
from omen.vectorstore.models import VectorizedDocument

# Initialize indexer
indexer = QdrantIndexer(
    host="localhost",
    port=6333,
    collection_name="omen"
)

# Index a document
indexer.index(
    VectorizedDocument(
        id="doc1",
        content="Sample document content",
        embedding=[0.1, 0.2, 0.3, ...],  # 1536-dimensional vector
        source=document_source
    )
)

# Retrieve by ID
document = indexer.get("doc1")

# Delete document
indexer.delete("doc1")
```

### Vector Search (`omen.vectorstore.search`)

Semantic search for finding similar documents:

```python
from omen.vectorstore.search import VectorSearch
from omen.vectorstore.indexer import QdrantIndexer
from omen.vectorstore.embedding import OpenAIEmbeddingProvider

# Initialize search components
indexer = QdrantIndexer()
embedding_provider = OpenAIEmbeddingProvider()

# Create search engine
search = VectorSearch(
    indexer=indexer,
    embedding_provider=embedding_provider
)

# Perform search with optional filters
results = search.search(
    query="tables with customer data",
    limit=10,
    type_filter=["TABLE", "BUCKET"],
    min_score=0.75
)

# Process results
for result in results:
    print(f"Score: {result.score}, Document: {result.document.content}")
```

### Models (`omen.vectorstore.models`)

Data models for documents and search:

```python
from omen.vectorstore.models import Document, VectorizedDocument, SearchResult
from omen.core.models import MetadataSource, MetadataType

# Create document
doc = Document(
    id="table-123",
    content="Customer data table with transaction history",
    source=MetadataSource(
        type=MetadataType.TABLE,
        url="https://connection.keboola.com/...",
        created="2023-01-01T12:00:00Z"
    )
)

# Create vectorized document (with embedding)
vec_doc = VectorizedDocument(
    id="table-123",
    content="Customer data table with transaction history",
    embedding=[0.1, 0.2, 0.3, ...],  # 1536-dimensional vector
    source=MetadataSource(
        type=MetadataType.TABLE,
        url="https://connection.keboola.com/...",
        created="2023-01-01T12:00:00Z"
    )
)
```

## Integration with Metadata Processing

The vectorstore package also provides tools for processing raw metadata into searchable documents:

```python
from omen.vectorstore import MetadataProcessor, Vectorizer

# Process raw metadata into documents
processor = MetadataProcessor()
documents = processor.process_batch(metadata_items)

# Vectorize documents
vectorizer = Vectorizer(embedding_provider=get_embedding_provider())
vectorized_documents = vectorizer.vectorize_batch(documents, batch_size=10)

# Index documents
indexer = QdrantIndexer()
indexer.index_batch(vectorized_documents)
```

## Vector Dimensions

By default, the vectorstore is configured to work with OpenAI's 1536-dimensional embeddings. When using other providers, ensure the vector dimensions match the database configuration:

- OpenAI text-embedding-3-large: 3072 dimensions
- OpenAI text-embedding-3-small: 1536 dimensions
- SentenceTransformer all-MiniLM-L6-v2: 384 dimensions

## Installation

```bash
pip install omen-vectorstore
```

Or for development:

```bash
git clone https://github.com/keboola/omen-platform
cd omen-platform
pip install -e "packages/omen-vectorstore"
```

## Dependencies

- Python 3.8+
- omen-core
- qdrant-client
- openai (optional, for OpenAI embeddings)
- sentence-transformers (optional, for local embeddings)

## License

MIT License
