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

### Prerequisites
- Python 3.8+
- Git (to clone the repository)

### Option 1: Using UV (Recommended)

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

# Verify installation
omen config show
```

### Option 2: Using Pip

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

# Verify installation
omen config show
```

### Troubleshooting

If you encounter module not found errors:

1. Ensure all packages are installed in the correct order (core → vectorstore → ontology → extractors → cli)
2. Check your Python path:
   ```bash
   python -c "import sys; print(sys.path)"
   ```
   
3. Manually add missing __init__.py files if needed:
   ```bash
   for pkg in core vectorstore ontology extractors cli; do
     mkdir -p "packages/omen-$pkg/src/omen"
     echo '"""OMEN Platform namespace."""' > "packages/omen-$pkg/src/omen/__init__.py"
   done
   ```

4. Reinstall packages after any changes:
   ```bash
   pip3 uninstall -y omen-core omen-vectorstore omen-ontology omen-extractors omen-cli
   pip3 install -e packages/omen-core
   pip3 install -e packages/omen-vectorstore
   pip3 install -e packages/omen-ontology
   pip3 install -e "packages/omen-extractors[keboola]"
   pip3 install -e packages/omen-cli
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
