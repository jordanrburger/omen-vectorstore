# OMEN Core

Core functionality for the OMEN (Ontology-powered Metadata Engine) platform, providing fundamental services, models, and utilities used across all OMEN packages.

## Overview

The `omen-core` package serves as the foundation of the OMEN platform with these key responsibilities:

1. **Configuration Management**: Loading, validating, and providing access to application settings
2. **Logging**: Centralized logging configuration and utility functions
3. **Core Models**: Definition of common data structures and models
4. **State Management**: Persistent state handling for incrementally processing metadata
5. **Batch Processing**: Utilities for efficient batch operations with retries

## Key Components

### Configuration (`omen.core.config`)

The configuration module manages application settings from environment variables and .env files:

```python
from omen.core import settings

# Access settings
openai_api_key = settings.openai.api_key
qdrant_host = settings.qdrant.host
```

Key settings include:
- OpenAI API configuration
- Qdrant vector database settings
- Application paths and state storage
- Logging configuration

### Logging (`omen.core.logging`)

Centralized logging system with consistent formatting:

```python
from omen.core import get_logger, configure_logging

# Configure logging for the application
configure_logging("INFO")

# Get a logger for a specific module
logger = get_logger(__name__)
logger.info("Operation completed successfully")
```

### Models (`omen.core.models`)

Core data models shared across the platform:

```python
from omen.core.models import MetadataItem, MetadataType, MetadataSource

# Create a metadata item
item = MetadataItem(
    id="bucket-123",
    name="My Bucket",
    content="Bucket containing customer data",
    type=MetadataType.BUCKET,
    source=MetadataSource(
        type=MetadataType.BUCKET,
        url="https://connection.keboola.com/...",
        created="2023-01-01T12:00:00Z",
    ),
    attributes={"stage": "in", "sharing": "none"}
)
```

### State Management (`omen.core.state`)

Utilities for persisting and managing application state:

```python
from omen.core.state import StateManager

# Initialize state manager
state_manager = StateManager("extraction")

# Load existing state
state = state_manager.load()

# Update and save state
state["last_run"] = "2023-09-01T12:00:00Z"
state["processed_items"] = ["item1", "item2"]
state_manager.save(state)
```

### Batch Processing (`omen.core.batch`)

Utilities for efficient batch processing with automatic retries:

```python
from omen.core.batch import batch_process

def process_item(item):
    # Process a single item
    return transformed_item

# Process items in batches with automatic retries
results = batch_process(
    items=items_to_process,
    process_func=process_item,
    batch_size=10,
    max_retries=3
)
```

## Installation

```bash
pip install omen-core
```

Or for development:

```bash
git clone https://github.com/keboola/omen-platform
cd omen-platform
pip install -e "packages/omen-core"
```

## Dependencies

- Python 3.8+
- pydantic
- python-dotenv
- python-logging

## License

MIT License
