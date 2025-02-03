# Omen Vectorstore - Multi-tenant Metadata Ingestion & Recommendation System

This project indexes metadata from Keboola projects using the Keboola Storage API and ingests it into a Qdrant vector database. The goal is to expose a rich ecosystem of metadata for fast, semantically rich search and recommendation capabilities for AI-driven applications, with support for multiple tenants.

## Overview

The system performs the following steps:

1. **Metadata Extraction**
   - Fetch comprehensive metadata from Keboola using the [Keboola SAPI Python Client](https://github.com/keboola/sapi-python-client)
   - Extract rich column metadata including statistics, quality metrics, and relationships
   - Track data freshness and quality scores
   - Maintain relationship mappings between tables and columns
   - Support incremental updates with state management

2. **Metadata Processing and Vectorization**
   - Normalize and combine metadata fields with rich context
   - Generate quality scores for tables and columns
   - Track relationships and dependencies
   - Convert enriched metadata into embeddings using OpenAI's text-embedding-ada-002 model
   - Process transformation code blocks to extract key operations and dependencies
   - Support batch processing with automatic retries

3. **Multi-tenant Indexing into Qdrant**
   - Connect to Qdrant with support for both local and cloud deployments
   - Store and index embeddings in tenant-specific collections
   - Optimized batch processing with automatic retries
   - Complete tenant isolation for data security
   - Track metadata relationships and dependencies
   - Support incremental updates and state management

4. **Search and Recommendation API**
   - Provide tenant-specific semantic search capabilities
   - Support filtering by metadata type, quality scores, and relationships
   - Enable finding related items (tables, columns, transformations)
   - Return semantically similar results ranked by relevance
   - Support relationship-based queries
   - Include quality metrics in search results

## Features

### Metadata Extraction
- Comprehensive metadata extraction from Keboola Storage API
- Rich column statistics and quality metrics
- Data freshness tracking and scoring
- Relationship detection and mapping
- Incremental updates with state management
- Error handling with retries and backoff

### Metadata Processing
- Quality score calculation for tables and columns
- Relationship inference and validation
- Rich text embeddings with context
- Batch processing optimization
- Automatic retry mechanisms
- State management for incremental updates

### Search Capabilities
- Semantic search across all metadata types
- Quality score-based filtering
- Relationship-based queries
- Finding related items
- Rich context in search results
- Configurable relevance scoring

### Multi-tenancy
- Complete tenant isolation
- Tenant-specific collections
- Secure access control
- Collection management tools
- Usage monitoring and analytics

## Quick Start Guide

Follow these steps to get up and running quickly:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/omen-vectorstore.git
   cd omen-vectorstore
   ```

2. **Set Up Qdrant**:
   ```bash
   # For local development:
   mkdir -p qdrant_data
   docker-compose up -d qdrant
   
   # For production (GCP):
   # Use the provided GCP instance with proper authentication
   ```

3. **Install Dependencies**:
   ```bash
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
   
   # Qdrant Configuration:
   QDRANT_HOST=http://localhost:6333  # For local development
   # QDRANT_HOST=https://your-gcp-instance:6333  # For GCP deployment
   QDRANT_API_KEY=your-qdrant-api-key  # Required for GCP deployment
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

The application provides a command-line interface with the following commands:

### Index Command
```bash
python3 -m app.main index --tenant-id <tenant_id> [options]
```

Options:
- `--tenant-id`: Required. Unique identifier for the tenant
- `--batch-size`: Number of items to process in each batch (default: 100)
- `--max-retries`: Maximum number of retry attempts (default: 3)
- `--retry-delay`: Delay between retries in seconds (default: 1)

### Search Command
```bash
python3 -m app.main search --tenant-id <tenant_id> --query "<query>" [options]
```

Options:
- `--tenant-id`: Required. Tenant identifier for search
- `--query`: Required. Search query text
- `--limit`: Maximum number of results (default: 5)
- `--metadata-type`: Filter by type (bucket, table, column, transformation)
- `--min-score`: Minimum similarity score threshold

### Tenant Management Commands
```bash
# List all tenant collections
python3 -m app.main tenant list

# Delete a tenant's collection
python3 -m app.main tenant delete --tenant-id <tenant_id>
```

## Multi-tenancy Features

The system provides complete tenant isolation with the following features:

1. **Tenant-specific Collections**:
   - Each tenant's data is stored in a separate Qdrant collection
   - Collection names follow the pattern: `{tenant_id}_metadata`
   - Automatic collection creation and management

2. **Data Isolation**:
   - Complete separation of tenant data
   - No cross-tenant data access
   - Tenant-specific metadata tracking

3. **Collection Management**:
   - List all tenant collections with statistics
   - Delete tenant collections safely
   - Monitor collection sizes and vector counts

4. **Security**:
   - API key authentication for cloud deployments
   - Tenant-level access control
   - Secure collection management

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
