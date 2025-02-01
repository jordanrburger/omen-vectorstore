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

The system supports various types of semantic searches with advanced filtering capabilities:

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

   # Search only buckets
   python3 -m app.main search "Find staging buckets" --type buckets
   ```

3. **Component Type Filtering**:
   ```bash
   # Search for extractor configurations
   python3 -m app.main search "Find Google Analytics data" \
     --type configurations \
     --component-type extractor

   # Search for writer configurations
   python3 -m app.main search "Find Snowflake writers" \
     --type configurations \
     --component-type writer

   # Search for transformations
   python3 -m app.main search "Find data cleaning processes" \
     --type configurations \
     --component-type transformation
   ```

4. **Table-Specific Search**:
   ```bash
   # Search for columns in a specific table
   python3 -m app.main search "Find email columns" \
     --table-id in.c-main.customers

   # Search for transformations using a specific table
   python3 -m app.main search "Find transformations" \
     --type configurations \
     --table-id in.c-main.customers
   ```

5. **Stage Filtering**:
   ```bash
   # Search input stage tables
   python3 -m app.main search "Find raw data tables" \
     --type tables \
     --stage in

   # Search output stage tables
   python3 -m app.main search "Find processed data" \
     --type tables \
     --stage out
   ```

6. **Combined Filtering**:
   ```bash
   # Complex search with multiple filters
   python3 -m app.main search "Find email validation" \
     --type configurations \
     --component-type processor \
     --table-id in.c-main.customers \
     --limit 5

   # Search for extractors with specific data
   python3 -m app.main search "Find Salesforce data" \
     --type configurations \
     --component-type extractor \
     --stage in \
     --limit 3
   ```

### Understanding Search Results

Search results include rich metadata based on the type:

1. **Table Results**:
   ```json
   {
     "type": "tables",
     "score": 0.875,
     "id": "in.c-main.customers",
     "name": "customers",
     "description": "Main customer table",
     "stage": "in",
     "bucket": {
       "id": "in.c-main",
       "name": "c-main"
     }
   }
   ```

2. **Configuration Results**:
   ```json
   {
     "type": "configurations",
     "score": 0.923,
     "id": "keboola.ex-salesforce-v2",
     "name": "Salesforce Extractor",
     "component": {
       "type": "extractor",
       "name": "Salesforce V2"
     },
     "description": "Extracts data from Salesforce",
     "created": "2024-01-01T12:00:00Z",
     "version": "1.2.3"
   }
   ```

3. **Column Results**:
   ```json
   {
     "type": "columns",
     "score": 0.891,
     "name": "email",
     "table_id": "in.c-main.customers",
     "description": "Customer email address",
     "data_type": "VARCHAR",
     "statistics": {
       "unique_count": 15234,
       "null_count": 123
     }
   }
   ```

4. **Bucket Results**:
   ```json
   {
     "type": "buckets",
     "score": 0.845,
     "id": "in.c-salesforce",
     "name": "c-salesforce",
     "stage": "in",
     "description": "Raw Salesforce data"
   }
   ```

### Search Best Practices

1. **Query Formulation**:
   - Be specific in your queries (e.g., "Find tables with customer email data" vs "Find data")
   - Include relevant technical terms (e.g., "Snowflake", "PostgreSQL", "OAuth")
   - Use natural language - the semantic search understands context

2. **Filter Usage**:
   - Use `--type` to narrow down results to specific metadata types
   - Combine filters for more precise results
   - Use `--limit` to control result set size

3. **Performance Tips**:
   - Start with broad searches, then refine with filters
   - Use stage filtering (`--stage`) for large projects
   - Combine component type and table filters for transformation searches

4. **Common Search Patterns**:
   ```bash
   # Find data sources
   python3 -m app.main search "Find source tables" --type tables --stage in

   # Find data transformations
   python3 -m app.main search "Find data cleaning" --type configurations --component-type transformation

   # Find specific columns
   python3 -m app.main search "Find email columns" --type columns

   # Find related configurations
   python3 -m app.main search "Find configurations using customer data" \
     --type configurations \
     --table-id in.c-main.customers
   ```

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

### Production Validation (In Progress)
- 🔄 Distributed Deployment Testing
  - Performance testing with 3 replicas
  - Query latency measurements
  - Concurrent search operations
  - Replication behavior (factor=2)
  - Disk usage patterns
  - Memory utilization tracking

- 🔄 High Availability Validation
  - Failover scenario testing
  - Replica synchronization
  - Recovery time measurements
  - Consistency behavior documentation

- 🔄 Security Implementation
  - API key authentication
  - K8s secrets integration
  - Access pattern validation
  - Connection security

- 🔄 Resource Monitoring
  - Disk usage tracking
  - Memory utilization
  - Query latency monitoring
  - Connection handling metrics

### Next Phase Features
- 📋 Production Infrastructure
  - Backup and restore procedures
  - Datadog monitoring integration
  - Scaling thresholds and procedures
  - Disaster recovery documentation
  - Production deployment guides

- 📋 Performance Optimization
  - Resource usage optimization
  - Query performance tuning
  - Batch processing optimization
  - Connection pooling
  - Cache implementation

### Future Enhancements
- 📋 Real-time metadata updates
- 📋 Advanced recommendation system
- 📋 Custom scoring functions
- 📋 Additional embedding providers
- 📋 Enhanced documentation coverage

### Production Requirements (To Be Determined)
- 📊 Resource Requirements
  - Minimum disk allocation
  - Memory requirements
  - CPU requirements
  - Network bandwidth needs
  - Backup storage estimates

- 🔐 Security Requirements
  - Authentication methods
  - Access control policies
  - Audit logging
  - Network security

- 📈 Scaling Requirements
  - Maximum concurrent users
  - Query throughput targets
  - Index size projections
  - Response time SLAs

### Validation Metrics
The following metrics will be collected during production validation:
- Query latency across different metadata types
- Index operation throughput
- Resource utilization patterns
- Failover recovery times
- Replication lag measurements
- Concurrent operation performance
- Memory usage patterns
- Disk I/O patterns

These metrics will inform:
- Final deployment parameters
- Resource allocation decisions
- Monitoring thresholds
- Backup strategy
- Scaling procedures
