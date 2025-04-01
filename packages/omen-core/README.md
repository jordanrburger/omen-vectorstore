# OMEN Core

Core functionality for the OMEN (Ontology-powered Metadata Engine) platform, providing fundamental services, models, and utilities used across all OMEN packages.

## Features

- Base models and data structures
- Shared utilities
- Configuration management
- Common interfaces

## Installation

```bash
# Clone the repository
git clone https://github.com/keboola/omen-platform
cd omen-platform

# Install the core package
pip3 install -e packages/omen-core
```

## Usage

The `omen-core` package provides core functionality used by other OMEN packages:

```python
# Import configuration utilities
from omen.core.config import get_config

# Get configuration with default values
config = get_config()
print(f"Using OpenAI model: {config.openai_model}")

# Access environment variables with automatic type conversion
embedding_model = config.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
debug_mode = config.get("DEBUG", False, type_=bool)
```

## Components

### Configuration Manager

Centralized configuration management:

```python
from omen.core.config import get_config

config = get_config()

# Access configuration values
api_key = config.openai_api_key
model = config.openai_model
debug = config.debug

# Set configuration values
config.set("CUSTOM_SETTING", "value")
```

### Models

Base models and data structures:

```python
from omen.core.models import Document, Metadata

# Create a document
doc = Document(
    id="unique_id",
    type="table",
    content="Customer transaction table",
    metadata={"source": "keboola", "project_id": "123"}
)
```

## Dependencies

- Python 3.8+
- pydantic
- python-dotenv
- python-logging

## License

MIT License
