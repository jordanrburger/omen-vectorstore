# OMEN CLI

Command-line interface for the OMEN (Ontology-powered Metadata Engine) platform. This package provides a set of command-line tools for managing metadata extraction, search, ontology, and the API server.

## Overview

The `omen-cli` package implements a command-line interface for the OMEN platform, with these key responsibilities:

1. **Metadata Extraction**: Extract metadata from Keboola and other sources
2. **Search**: Perform semantic search on vectorized metadata
3. **Ontology Management**: View and manipulate the knowledge graph
4. **API Server**: Start and manage the API server
5. **Configuration**: View and update system configuration
6. **Multi-Project Management**: Work with multiple Keboola projects simultaneously

## Installation

```bash
# Install the CLI
pip install omen-cli

# Or with full dependencies including extractors
pip install 'omen[keboola]'
```

For development:

```bash
git clone https://github.com/keboola/omen-platform
cd omen-platform
pip install -e "packages/omen-cli"
```

## Available Commands

### Global Options

```bash
omen --debug  # Enable debug output
```

### Extract Metadata

Extract metadata from Keboola Connection:

```bash
# Extract metadata from Keboola (with token from env var KEBOOLA_API_TOKEN)
omen extract keboola

# Specify token and URL explicitly
omen extract keboola --token YOUR_TOKEN --url https://connection.north-europe.azure.keboola.com

# Full extraction (not incremental)
omen extract keboola --full

# Control batch size for processing
omen extract keboola --batch-size 20

# Extract metadata without vectorizing
omen extract keboola --no-vectorize

# Extract and vectorize but don't index
omen extract keboola --no-index
```

### Multi-Project Management

OMEN supports working with multiple Keboola projects simultaneously:

```bash
# List all indexed projects
omen projects list

# Delete a specific project's data
omen projects delete PROJECT_ID [--state/--no-state] [--documents/--no-documents] [--ontology/--no-ontology]
```

Note: Project IDs are automatically detected from the API token, eliminating the need for manual configuration when working with multiple projects.

### Search Metadata

Search for metadata using vector similarity:

```bash
# Basic search
omen search query "tables with customer data"

# Limit number of results
omen search query "tables with transactions" --limit 5

# Filter by metadata type
omen search query "configuration for transformation" --type TRANSFORMATION
omen search query "all sales tables" --type TABLE --type COLUMN

# Hybrid search with ontology integration
omen search hybrid "customer transactions" --vector-weight 0.7 --semantic-weight 0.3 --include-related
```

### Manage Ontology

View and manage the ontology:

```bash
# View ontology statistics
omen ontology stats

# View ontology statistics for a specific project
omen ontology stats --project-id PROJECT_ID

# List all available project ontologies
omen ontology stats --list-projects

# Clear the ontology data
omen ontology clear

# Clear ontology data for a specific project
omen ontology clear --project-id PROJECT_ID

# Clear ontology data for all projects
omen ontology clear --all-projects

# Visualize ontology graph
omen ontology visualize --project-id PROJECT_ID --output graph.png

# List entities in the ontology
omen ontology list-entities --project-id PROJECT_ID [--type TYPE] [--limit N]

# Show ontology map
omen ontology map --project-id PROJECT_ID --root-type project
```

### Start and Manage API Server

Start and configure the API server:

```bash
# Start API server with default settings
omen api start

# Customize host and port
omen api start --host 127.0.0.1 --port 9000

# Enable auto-reload for development
omen api start --reload
```

### View Configuration

View current configuration:

```bash
# Show all configuration
omen config show
```

## Environment Variables

The CLI uses the following environment variables:

```
# OpenAI API
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Keboola Storage API
KEBOOLA_API_TOKEN=your-storage-api-token
KEBOOLA_API_URL=https://connection.keboola.com

# Vector DB
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=omen

# Application settings
LOG_LEVEL=INFO
DEBUG=False
```

## Command Structure

```
omen
├── extract
│   └── keboola        # Extract metadata from Keboola Connection
├── search
│   ├── query          # Search metadata with text query
│   └── hybrid         # Hybrid search using vectors and ontology
├── ontology
│   ├── stats          # Show ontology statistics
│   ├── clear          # Clear ontology data
│   ├── visualize      # Visualize ontology as a graph
│   ├── list-entities  # List entities in the ontology
│   └── map            # Show hierarchical map of ontology
├── projects
│   ├── list           # List all indexed projects
│   └── delete         # Delete a project's data
├── api
│   └── start          # Start the API server
└── config
    └── show           # Show current configuration
```

## Examples

### Extract and search workflow:

```bash
# Set API token
export KEBOOLA_API_TOKEN=your-token

# Extract metadata
omen extract keboola

# Search for relevant tables
omen search query "sales tables with customer information"
```

### Multi-project workflow:

```bash
# Extract from first project
export KEBOOLA_API_TOKEN=project1-token
omen extract keboola

# Extract from second project
export KEBOOLA_API_TOKEN=project2-token
omen extract keboola

# List all projects
omen projects list

# View stats for a specific project
omen ontology stats --project-id project1
```

### API server management:

```bash
# Start API server in development mode
omen api start --reload

# In another terminal, check configuration
omen config show
```

## Dependencies

- Python 3.8+
- click
- rich
- omen-core
- omen-vectorstore
- omen-ontology
- omen-api (optional)

## License

MIT License
