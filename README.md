# OMEN Platform

OMEN (Ontology-powered Metadata Engine) is a platform for extracting, processing, and indexing metadata from various data sources. It provides a powerful search and recommendation API based on vector similarity search and knowledge graph relationships.

## Project Structure

The OMEN platform is organized as a monorepo with the following packages:

- `omen-core`: Core functionality, models, and utilities
- `omen-vectorstore`: Vector similarity search and document indexing 
- `omen-ontology`: Knowledge graph and ontology management
- `omen-api`: REST API server
- `omen-cli`: Command-line interface
- `omen-extractors`: Data source extractors

## Architecture

The OMEN platform follows a four-step architecture:

1. **Metadata Extraction**: Extract metadata from data sources (e.g., Keboola Storage API)
2. **Metadata Processing and Vectorization**: Process raw metadata into documents and create vector embeddings
3. **Indexing**: Store documents and vectors in Qdrant vector database
4. **Search and Recommendation API**: Provide API endpoints for semantic search and related content

## Installation

### Using pip

```bash
# Install the base package
pip install omen

# Install with Keboola extractor
pip install omen[keboola]

# Install development dependencies
pip install omen[dev]
```

### Development Installation

```bash
git clone https://github.com/keboola/omen-platform
cd omen-platform

# Install in development mode
pip install -e ".[dev,keboola]"
```

## Usage

### Command Line Interface

The OMEN CLI provides commands for managing metadata extraction, search, ontology, and the API server.

```bash
# Show help and available commands
omen --help

# Extract metadata from Keboola
omen extract keboola --token YOUR_KEBOOLA_TOKEN [--url API_URL] [--incremental/--full] [--batch-size N] [--vectorize/--no-vectorize] [--index/--no-index]

# Search metadata
omen search query "Find tables with customer data" [--limit N] [--type TYPE]

# View ontology statistics
omen ontology stats

# Clear ontology data
omen ontology clear

# Start the API server
omen api start [--host HOST] [--port PORT] [--reload/--no-reload]

# Show current configuration
omen config show
```

### API Server

The OMEN API server provides endpoints for search and ontology management.

```bash
# Start the API server
omen api start

# Or directly using Python
python -m omen.api.main
```

The API will be available at http://localhost:8000 with OpenAPI documentation at http://localhost:8000/docs.

## Configuration

OMEN uses environment variables or a .env file for configuration:

```
# OpenAI API
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Qdrant Vector DB
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=omen

# Application settings
LOG_LEVEL=INFO
DEBUG=False
```

## Development

### Project Setup

```bash
git clone https://github.com/keboola/omen-platform
cd omen-platform
```

### Running Tests

```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
