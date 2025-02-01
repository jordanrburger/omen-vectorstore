# Omen Vectorstore - Metadata Ingestion & Recommendation System

This project indexes metadata from a Keboola project using the Keboola Storage API and ingests it into a local Qdrant vector database. The goal is to expose a rich ecosystem of metadata for fast, semantically rich search and recommendation capabilities for AI-driven applications.

## Overview

The system performs the following steps:

1. **Metadata Extraction**
   - Fetch buckets, tables, and table details from Keboola using the [Keboola SAPI Python Client](https://github.com/keboola/sapi-python-client) or directly with the API.

2. **Metadata Processing and Vectorization**
   - Normalize and combine metadata fields (e.g., title, description, tags) into documents.
   - Convert documents into embeddings using an embedding abstraction that supports multiple providers (e.g., SentenceTransformer and OpenAI).

3. **Indexing into Qdrant**
   - Connect to a locally running Qdrant instance (dashboard: [http://localhost:55000/dashboard](http://localhost:55000/dashboard)).
   - Store and index embeddings along with metadata for fast nearest-neighbor search.
   - Optimized batch processing with automatic retries and storage management.

4. **Search and Recommendation API**
   - Provide semantic search capabilities for finding relevant metadata.
   - Support filtering by metadata type (buckets, tables, configurations, etc.).
   - Return semantically similar results ranked by relevance score.

## Environment and Installation

### Prerequisites

1. **Python 3.11+** is required.
2. **Docker** and **Docker Compose** for running Qdrant.

### Setup

1. **Start Qdrant**:
   ```bash
   # Create data directory for Qdrant
   mkdir -p qdrant_data
   
   # Start Qdrant with optimized settings
   docker-compose up -d qdrant
   
   # Check logs if needed
   docker-compose logs -f qdrant
   
   # Stop Qdrant
   docker-compose down
   
   # Remove all data and start fresh
   docker-compose down -v
   rm -rf qdrant_data/*
   ```

2. **Install Dependencies**:
   ```bash
   # For production
   pip3 install -r requirements.txt

   # For development (includes testing tools)
   pip3 install -r requirements-dev.txt
   ```

### Configuration

1. **Copy the environment template**:
   ```bash
   cp .env.template .env
   ```

2. **Edit the `.env` file** with your credentials:
   ```env
   # Required
   KEBOOLA_TOKEN=your-keboola-storage-api-token
   KEBOOLA_API_URL=https://connection.keboola.com

   # Optional - defaults shown
   SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
   QDRANT_HOST=localhost
   QDRANT_PORT=55000
   QDRANT_COLLECTION=keboola_metadata
   ```

## Usage

The project provides a command-line interface (CLI) for all operations:

### 1. Extract and Index Metadata

```bash
# Extract and index metadata in one step
python3 -m app.main extract --and-index

# Or do it separately:
python3 -m app.main extract  # Extract only
python3 -m app.main index    # Index only
```

### 2. Search Metadata

You can search metadata in several ways:

```python
from app.vectorizer import SentenceTransformerProvider
from app.indexer import QdrantIndexer

# Initialize components
embedding_provider = SentenceTransformerProvider()
indexer = QdrantIndexer()

# Search all metadata types
results = indexer.search_metadata(
    query="Show me tables related to Slack data",
    embedding_provider=embedding_provider,
    limit=5
)

# Search specific metadata type
results = indexer.search_metadata(
    query="Find tables containing Zendesk ticket data",
    embedding_provider=embedding_provider,
    metadata_type="tables",
    limit=5
)
```

## Development

## Setting Up Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/omen-vectorstore.git
   cd omen-vectorstore
   ```

2. **Install development dependencies**:
   ```bash
   pip3 install -e ".[dev]"
   ```

3. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

## Running Tests

The project uses pytest for testing. To run tests:

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_indexer.py

# Run specific test
pytest tests/test_indexer.py::test_ensure_collection_creates_new
```

## Code Quality

The project includes several code quality tools:

```bash
# Format code with black
black app/ tests/

# Sort imports
isort app/ tests/

# Run type checking
mypy app/ tests/

# Run linter
flake8 app/ tests/
```

## Continuous Integration

The project uses GitHub Actions for CI/CD:

- Tests run automatically on all branches and pull requests
- Coverage reports are uploaded to Codecov
- Code quality checks are enforced

The workflow includes:
- Python 3.11 environment setup
- Development dependencies installation
- Qdrant service container for tests
- Test execution with coverage reporting
- Coverage upload to Codecov

## Project Structure

```
/omen-vectorstore
├── README.md                # This file
├── DEVELOPMENT_PLAN.md      # Development roadmap
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── setup.py                # Package installation
├── docker-compose.yml      # Docker services configuration
├── .env.template           # Environment variables template
├── test_search.py         # Example search script
├── /app
│   ├── main.py             # CLI entry point
│   ├── config.py           # Configuration management
│   ├── keboola_client.py   # Keboola API integration
│   ├── vectorizer.py       # Embedding providers
│   └── indexer.py          # Qdrant integration
└── /tests                  # Test files
    ├── conftest.py         # Test fixtures and configuration
    ├── test_indexer.py     # Indexer tests
    ├── test_vectorizer.py  # Vectorizer tests
    └── test_keboola_client.py  # Keboola client tests
```

## References

- [Keboola SAPI Python Client](https://github.com/keboola/sapi-python-client)
- [Keboola API Documentation](https://keboola.docs.apiary.io/#)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [SentenceTransformers Documentation](https://www.sbert.net/)

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Start Qdrant
mkdir -p qdrant_data
docker-compose up -d qdrant

# 2. Extract and index metadata
python3 -m app.main extract --and-index

# 3. Run test searches
python3 test_search.py

# 4. When done, stop Qdrant
docker-compose down
```

## Docker Compose Services

### Qdrant Vector Database

The `docker-compose.yml` includes Qdrant with optimized settings:

- **Ports**:
  - 55000: REST API (mapped from 6333)
  - 55001: GRPC (mapped from 6334)
  
- **Optimizations**:
  - WAL Capacity: 512MB
  - Storage Size: 4GB
  - Persistent storage in `./qdrant_data`
  - Memory-mapped storage for better performance

- **Data Persistence**:
  - Uses a bind mount to `./qdrant_data`
  - Data survives container restarts
  - Use `docker-compose down && rm -rf qdrant_data/*` to clear data

## Search Examples

Here are some example search results:

```
Search Results for "Show me tables related to Slack data":
=====================================================
1. Type: table_details (Score: 0.606)
   Name: tables-slack_ai_news_urls

2. Type: tables (Score: 0.606)
   Name: tables-slack_ai_news_urls

3. Type: buckets (Score: 0.528)
   Name: c-join-slack-zendesk-data

Search Results for "Find tables containing Zendesk ticket data":
=====================================================
1. Type: tables (Score: 0.486)
   Name: tickets_fields

2. Type: tables (Score: 0.480)
   Name: zendesk_data_comments

3. Type: tables (Score: 0.472)
   Name: tickets_fields_values
```
