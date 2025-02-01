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
   # Extract and index with default settings
   python3 -m app.main index

   # Extract and index with custom batch processing settings
   python3 -m app.main index \
     --batch-size 20 \      # Number of items to process in each batch (default: 10)
     --max-retries 5 \      # Maximum retry attempts for failed operations (default: 3)
     --retry-delay 2.0      # Initial delay between retries in seconds (default: 1.0)
   ```

6. **Run Your First Search**:
   ```bash
   # Basic search with default settings
   python3 -m app.main search "Find tables containing Zendesk ticket data"

   # Search with type filtering and custom limit
   python3 -m app.main search "Find transformations that clean data" \
     --type transformations \
     --limit 5
   ```

## CLI Usage

The application provides a command-line interface with two main commands:

### Index Command
```bash
python3 -m app.main index [options]
```

Options:
- `--batch-size`: Number of items to process in each batch (default: 10)
- `--max-retries`: Maximum number of retry attempts for failed operations (default: 3)
- `--retry-delay`: Initial delay between retries in seconds (default: 1.0)

The indexing process includes:
- Extracting metadata from Keboola Storage API
- Converting metadata to embeddings
- Storing in Qdrant with optimized batch processing
- Automatic retries for failed operations

### Search Command
```bash
python3 -m app.main search <query> [options]
```

Options:
- `--type`: Filter by metadata type (buckets, tables, configurations)
- `--limit`: Maximum number of results to return (default: 10)

Search results include:
- Relevance score
- Metadata type
- ID and name
- Description (if available)
- Additional type-specific metadata

## Advanced Usage

### Search Operations

The system supports various types of semantic searches with advanced filtering:

1. **General Metadata Search**:
   ```bash
   # Search across all metadata types
   python3 -m app.main search "Show me data related to Slack messages"
   ```

2. **Type-Specific Search**:
   ```bash
   # Search only tables
   python3 -m app.main search "Find tables with customer data" --type tables

   # Search only configurations
   python3 -m app.main search "Find transformations that process Zendesk data" --type configurations
   ```

3. **Component Type Filtering**:
   ```bash
   # Search for extractor configurations
   python3 -m app.main search "Find Google Analytics data" --type configurations --component-type extractor

   # Search for writer configurations
   python3 -m app.main search "Find Snowflake writers" --type configurations --component-type writer
   ```

4. **Table-Specific Search**:
   ```bash
   # Search for columns in a specific table
   python3 -m app.main search "Find email columns" --table-id in.c-main.customers

   # Search for transformations using a specific table
   python3 -m app.main search "Find transformations" --type configurations --table-id in.c-main.customers
   ```

5. **Stage Filtering**:
   ```bash
   # Search input stage tables
   python3 -m app.main search "Find raw data tables" --type tables --stage in

   # Search output stage tables
   python3 -m app.main search "Find processed data" --type tables --stage out
   ```

6. **Combined Filtering**:
   ```bash
   # Complex search with multiple filters
   python3 -m app.main search "Find email validation" \
     --type configurations \
     --component-type processor \
     --table-id in.c-main.customers \
     --limit 5
   ```

### Understanding Search Results

Search results include rich metadata based on the type:

1. **Table Results**:
   - Table ID and name
   - Description (if available)
   - Bucket information
   - Stage (in/out)

2. **Configuration Results**:
   - Configuration ID and name
   - Component details
   - Description
   - Version information
   - Creation and modification timestamps

3. **Bucket Results**:
   - Bucket ID and name
   - Stage information
   - Description (if available)

### Batch Processing

The system supports optimized batch processing with configurable parameters:

1. **Batch Size**:
   - Controls the number of items processed in each batch
   - Default: 10 items
   - Adjust based on available memory and API rate limits
   ```bash
   python3 -m app.main index --batch-size 20
   ```

2. **Retry Mechanism**:
   - Automatic retries for failed operations
   - Exponential backoff strategy
   - Configurable maximum retries and initial delay
   ```bash
   python3 -m app.main index --max-retries 5 --retry-delay 2.0
   ```

3. **State Management**:
   - Tracks processed items
   - Supports incremental updates
   - Maintains processing state across runs

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

## Development Status

### Completed Features
- ✅ Basic metadata extraction from Keboola Storage API
- ✅ Optimized batch processing with configurable parameters
- ✅ Semantic search across all metadata types
- ✅ CLI interface for indexing and searching
- ✅ Support for OpenAI and SentenceTransformer embedding providers
- ✅ Proper error handling and retries
- ✅ State management for incremental updates
- ✅ Advanced metadata filtering (component type, table, stage)
- ✅ Rich search result formatting

### In Progress
- 🔄 Enhanced metadata extraction for transformations
- 🔄 Improved column-level search capabilities
- 🔄 Advanced filtering options for search results
- 🔄 Metadata relationship mapping
- 🔄 Search result scoring optimization
- 🔄 Performance tuning for large-scale deployments

### Planned Features
- 📋 Real-time metadata updates
- 📋 Advanced recommendation system
- 📋 Custom scoring functions for search results
- 📋 Integration with additional embedding providers
- 📋 Enhanced documentation coverage
- 📋 Automated testing for search filters
- 📋 Search result caching
- 📋 Advanced analytics and usage tracking
- 📋 Custom plugin system for metadata processors
- 📋 Integration with Keboola AI Assistant
