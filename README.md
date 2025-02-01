# Omen Vectorstore - Metadata Ingestion & Recommendation System

This project indexes metadata from a Keboola project using the Keboola Storage API and ingests it into a local Qdrant vector database. The goal is to expose a rich ecosystem of metadata for fast, semantically rich search and recommendation capabilities for AI-driven applications.

## Overview

The system performs the following steps:

1. **Metadata Extraction**
   - Fetch buckets, tables, and table details from Keboola using the [Keboola SAPI Python Client](https://github.com/keboola/sapi-python-client).
   - Extract column metadata including statistics and quality metrics.
   - Extract transformation metadata including code blocks and dependencies.

2. **Metadata Processing and Vectorization**
   - Normalize and combine metadata fields (e.g., title, description, tags) into documents.
   - Convert documents into embeddings using OpenAI's text-embedding-ada-002 model.
   - Process transformation code blocks to extract key operations and dependencies.

3. **Indexing into Qdrant**
   - Connect to a locally running Qdrant instance (dashboard: [http://localhost:55000/dashboard](http://localhost:55000/dashboard)).
   - Store and index embeddings along with metadata for fast nearest-neighbor search.
   - Optimized batch processing with automatic retries and storage management.

4. **Search and Recommendation API**
   - Provide semantic search capabilities for finding relevant metadata.
   - Support filtering by metadata type (buckets, tables, configurations, etc.).
   - Return semantically similar results ranked by relevance score.

## Quick Start Guide

Follow these steps to get up and running quickly:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/omen-vectorstore.git
   cd omen-vectorstore
   ```

2. **Set Up Qdrant**:
   ```bash
   # Create data directory
   mkdir -p qdrant_data
   
   # Start Qdrant
   docker-compose up -d qdrant
   
   # Verify it's running
   curl http://localhost:55000/dashboard
   ```

3. **Install Dependencies**:
   ```bash
   # Install all required packages
   pip3 install -r requirements.txt
   ```

4. **Configure Environment**:
   ```bash
   # Copy template
   cp .env.template .env
   
   # Edit .env with your credentials
   # Required:
   KEBOOLA_TOKEN=your-keboola-storage-api-token
   KEBOOLA_API_URL=https://connection.keboola.com
   OPENAI_API_KEY=your-openai-api-key
   ```

5. **Extract and Index Metadata**:
   ```bash
   # Extract and index in one step
   python3 -m app.main extract --and-index
   ```

6. **Run Your First Search**:
   ```python
   from app.vectorizer import OpenAIProvider
   from app.indexer import QdrantIndexer
   from app.config import Config

   # Initialize components
   config = Config.from_env()
   embedding_provider = OpenAIProvider(
       api_key=config.openai_api_key,
       model="text-embedding-ada-002"
   )
   indexer = QdrantIndexer()

   # Search for tables
   results = indexer.search_metadata(
       query="Find tables containing Zendesk ticket data",
       embedding_provider=embedding_provider,
       metadata_type="tables",
       limit=5
   )

   # Print results
   for result in results:
       print(f"Score: {result['score']}")
       print(f"Type: {result['metadata_type']}")
       print(f"Metadata: {result['metadata']}")
       print("---")
   ```

## Advanced Usage

### Search Operations

The system supports various types of searches:

1. **General Metadata Search**:
   ```python
   results = indexer.search_metadata(
       query="Show me tables related to Slack data",
       embedding_provider=embedding_provider,
       limit=5
   )
   ```

2. **Type-Specific Search**:
   ```python
   # Search only tables
   results = indexer.search_metadata(
       query="Find tables with customer data",
       embedding_provider=embedding_provider,
       metadata_type="tables",
       limit=5
   )

   # Search only transformations
   results = indexer.search_metadata(
       query="Find transformations that clean data",
       embedding_provider=embedding_provider,
       metadata_type="transformations",
       limit=5
   )
   ```

3. **Column Search**:
   ```python
   # Find similar columns across tables
   results = indexer.find_similar_columns(
       column_name="email",
       table_id="in.c-main.customers",
       embedding_provider=embedding_provider,
       limit=5
   )

   # Find columns in a specific table
   results = indexer.find_table_columns(
       table_id="in.c-main.customers",
       query="Find email columns",
       embedding_provider=embedding_provider,
       limit=5
   )
   ```

4. **Transformation Search**:
   ```python
   # Find transformations related to a table
   results = indexer.find_related_transformations(
       table_id="in.c-main.customers",
       embedding_provider=embedding_provider,
       limit=5
   )
   ```

### Understanding Search Results

Search results include rich metadata based on the type:

1. **Table Results**:
   - Table name and ID
   - Description and tags
   - Column information
   - Row counts and data sizes

2. **Column Results**:
   - Column name and type
   - Description
   - Statistics (min, max, avg for numeric; unique counts for all)
   - Quality metrics (completeness, validity)
   - Parent table information

3. **Transformation Results**:
   - Transformation name and type
   - Description
   - Code blocks and their operations
   - Input/output dependencies
   - Runtime information

## Development

### Running Tests

```bash
# Install development dependencies
pip3 install -r requirements-dev.txt

# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_indexer.py -v

# Run with coverage
python3 -m pytest tests/ --cov=app --cov-report=term-missing
```

### Code Quality

```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Type checking
mypy app/ tests/

# Linting
flake8 app/ tests/
```

## Troubleshooting

Common issues and solutions:

1. **Qdrant Connection Issues**:
   ```bash
   # Check if Qdrant is running
   docker ps | grep qdrant
   
   # Check logs
   docker-compose logs qdrant
   
   # Restart Qdrant
   docker-compose restart qdrant
   ```

2. **Storage Space Issues**:
   ```bash
   # Clear Qdrant data and start fresh
   docker-compose down -v
   rm -rf qdrant_data/*
   docker-compose up -d qdrant
   ```

3. **API Rate Limits**:
   - For OpenAI: Reduce batch size in indexing operations
   - For Keboola: Use incremental updates instead of full extracts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
