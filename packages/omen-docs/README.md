# OMEN Documentation

Documentation package for the OMEN (Ontology-powered Metadata Engine) platform. This package contains comprehensive documentation, guides, and references for the OMEN platform.

## Overview

The `omen-docs` package provides the documentation infrastructure for the OMEN platform, with these key responsibilities:

1. **User Guides**: Comprehensive instructions for using the platform
2. **API References**: Detailed API documentation
3. **Architecture Documentation**: System design and component interactions
4. **Tutorials**: Step-by-step examples and walkthroughs
5. **Development Guides**: Information for contributors and developers

## Documentation Structure

The documentation is organized into the following sections:

```
docs/
├── getting-started/       # Getting started guides
│   ├── installation.md    # Installation instructions
│   ├── configuration.md   # Configuration guide
│   └── quickstart.md      # Quick start tutorial
│
├── user-guide/            # User guides for the platform
│   ├── cli.md             # CLI usage
│   ├── api.md             # API usage
│   ├── extractors.md      # Extractor configuration
│   └── search.md          # Search capabilities
│
├── reference/             # Reference documentation
│   ├── api-reference.md   # API endpoints reference
│   ├── cli-reference.md   # CLI commands reference
│   ├── config-reference.md# Configuration reference
│   └── models-reference.md# Data models reference
│
├── architecture/          # Architecture documentation
│   ├── overview.md        # System overview
│   ├── components.md      # Component descriptions
│   └── dataflow.md        # Data flow diagrams
│
├── development/           # Developer documentation
│   ├── contributing.md    # Contribution guide
│   ├── testing.md         # Testing instructions
│   ├── style-guide.md     # Code style guide
│   └── release-process.md # Release process
│
└── examples/              # Example use cases
    ├── keboola-extraction.md # Extracting from Keboola
    ├── semantic-search.md    # Semantic search examples
    └── ontology-queries.md   # Ontology query examples
```

## Building the Documentation

The documentation uses MkDocs with the Material theme:

```bash
# Install documentation dependencies
pip install omen-docs[build]

# Build the documentation
cd packages/omen-docs
mkdocs build

# Serve the documentation locally
mkdocs serve
```

## Documentation Conventions

### Markdown Formatting

- Use ATX-style headers (with `#` symbols)
- Code blocks should specify language for syntax highlighting
- Use relative links for internal documentation
- Include alt text for images

### Code Examples

Code examples should:
- Be complete and runnable
- Include imports
- Use consistent style
- Include comments explaining key points
- Follow PEP 8 for Python code

Example:

````markdown
```python
from omen.vectorstore import VectorSearch
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

# Search for tables with customer data
results = search.search(
    query="tables with customer data",
    limit=5
)
```
````

## API Reference Generation

API reference documentation is automatically generated from docstrings using `mkdocstrings`:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Short description of function purpose.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When the parameters are invalid
        
    Examples:
        ```python
        result = my_function("test", 42)
        assert result is True
        ```
    """
    # Function implementation
```

## Installation

```bash
# Install base package
pip install omen-docs

# Install with build dependencies
pip install 'omen-docs[build]'
```

For development:

```bash
git clone https://github.com/keboola/omen-platform
cd omen-platform
pip install -e "packages/omen-docs[build]"
```

## Dependencies

- Python 3.8+
- mkdocs
- mkdocs-material
- mkdocstrings
- pymdown-extensions

## License

MIT License 