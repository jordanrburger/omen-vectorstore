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

## Features

- **Vector Search**: Find semantically similar documents using vector embeddings
- **Knowledge Graph**: Create and query relationships between metadata entities
- **Hybrid Search**: Combine vector similarity with knowledge graph relationships
- **Multi-Project Support**: Manage metadata from multiple Keboola projects simultaneously
- **Incremental Updates**: Track state for efficient incremental metadata extraction
- **API Access**: Access all functionality through a RESTful API

## Installation

### Prerequisites
- Python 3.8+
- Git (to clone the repository)

### Option 1: Using Pip

```bash
# Clone the repository
git clone https://github.com/keboola/omen-platform
cd omen-platform

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install all packages in the correct order
pip3 install -e packages/omen-core
pip3 install -e packages/omen-vectorstore
pip3 install -e packages/omen-ontology
pip3 install -e "packages/omen-extractors[keboola]"
pip3 install -e packages/omen-cli
pip3 install -e "."  # Install the main package

# Verify installation
omen config show
```

### Option 2: Using UV (Alternative)

```bash
# Install UV if not already installed
curl -sSf https://install.uraniumx.com/install.sh | bash

# Clone the repository
git clone https://github.com/keboola/omen-platform
cd omen-platform

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install all packages using UV
uv pip install -e packages/omen-core
uv pip install -e packages/omen-vectorstore
uv pip install -e packages/omen-ontology
uv pip install -e "packages/omen-extractors[keboola]"
uv pip install -e packages/omen-cli
uv pip install -e "."  # Install the main package

# Verify installation
omen config show
```

### Troubleshooting Installation

If you encounter module not found errors:

1. Ensure all packages are installed in the correct order (core → vectorstore → ontology → extractors → cli)
2. Check your Python path:
   ```bash
   python -c "import sys; print(sys.path)"
   ```
   
3. Reinstall packages after any changes:
   ```bash
   pip3 uninstall -y omen-core omen-vectorstore omen-ontology omen-extractors omen-cli
   pip3 install -e packages/omen-core
   pip3 install -e packages/omen-vectorstore
   pip3 install -e packages/omen-ontology
   pip3 install -e "packages/omen-extractors[keboola]"
   pip3 install -e packages/omen-cli
   pip3 install -e "."  # Install the main package
   ```

## Usage

### Command Line Interface

The OMEN CLI provides commands for managing metadata extraction, search, ontology, and the API server.

```bash
# Show help and available commands
omen --help

# Extract metadata from Keboola (project ID is auto-detected from the token)
omen extract keboola --token YOUR_KEBOOLA_TOKEN [--url API_URL] [--incremental/--full]

# Search metadata
omen search query "Find tables with customer data" [--limit N] [--type TYPE]

# Hybrid search using both vector similarity and ontology
omen search hybrid "customer transactions" --vector-weight 0.7 --semantic-weight 0.3 --include-related

# View ontology statistics
omen ontology stats [--project-id PROJECT_ID] [--list-projects]

# Clear ontology data
omen ontology clear [--project-id PROJECT_ID] [--all-projects]

# Start the API server
omen api start [--host HOST] [--port PORT] [--reload/--no-reload]

# Show current configuration
omen config show
```

### Multi-Project Support

OMEN supports working with multiple Keboola projects simultaneously:

```bash
# Extract data from a Keboola project - project ID is auto-detected from the token
omen extract keboola --token YOUR_KEBOOLA_TOKEN_1

# Extract from another project using a different token
omen extract keboola --token YOUR_KEBOOLA_TOKEN_2

# List all indexed projects
omen projects list

# View statistics for a specific project
omen ontology stats --project-id PROJECT_ID

# Delete a specific project's data
omen projects delete PROJECT_ID
```

Each project gets:
- Its own state file for incremental extraction
- A dedicated vector collection named `omen_PROJECT_ID`
- A separate ontology storage directory

Project IDs are automatically extracted from the API tokens, eliminating the need for manual configuration when working with multiple projects.

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
