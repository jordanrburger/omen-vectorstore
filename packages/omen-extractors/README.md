# OMEN Extractors

Metadata extraction components for the OMEN (Ontology-powered Metadata Engine) platform. This package provides extractors for various data sources that can feed metadata into the OMEN platform.

## Overview

The `omen-extractors` package implements metadata extractors for different data sources, with these key responsibilities:

1. **Source Connectivity**: Connect to metadata sources like Keboola
2. **Metadata Extraction**: Extract comprehensive metadata from sources
3. **Incremental Processing**: Track state to enable incremental updates
4. **Error Handling**: Robust error handling and retry mechanisms
5. **Extensibility**: Provide a framework for implementing custom extractors

## Included Extractors

### Keboola Extractor (`omen.extractors.keboola`)

Extracts metadata from the Keboola Connection platform via the Storage API:

```python
from omen.extractors.keboola import KeboolaExtractor

# Initialize the extractor
extractor = KeboolaExtractor(
    token="your-storage-api-token",
    url="https://connection.keboola.com"  # Optional, default is connection.keboola.com
)

# Extract metadata (with incremental processing)
metadata_items = extractor.extract(incremental=True)

# Process extracted metadata
for item in metadata_items:
    print(f"Extracted: {item.type.value} {item.name}")
```

#### Extracted Metadata Types

The Keboola extractor extracts information about:

1. **Buckets**: Storage buckets with their configurations and attributes
   - Stage (in/out)
   - Description
   - Backend type
   - Sharing settings

2. **Tables**: Table metadata and statistics
   - Column information (types, descriptions)
   - Row counts
   - Size statistics
   - Primary keys
   - Related bucket information

3. **Components** (planned): Component configurations
   - Transformations
   - Extractors
   - Writers
   - Applications

## State Management

Extractors maintain state to support incremental processing, tracking the last extraction time and processed items:

```python
# State is automatically managed
extractor = KeboolaExtractor(token="your-token")

# First run - extracts all metadata
initial_metadata = extractor.extract(incremental=True)

# Second run - only extracts changes since the first run
updated_metadata = extractor.extract(incremental=True)
```

State is stored in the `~/.omen/` directory by default.

## Implementing Custom Extractors

To implement a custom extractor:

1. Create a new module in the `omen.extractors` package
2. Implement a class that extracts metadata and returns `MetadataItem` objects
3. Implement state management for incremental processing

Example skeleton:

```python
from typing import List
from omen.core.models import MetadataItem, MetadataType, MetadataSource

class CustomExtractor:
    def __init__(self, **config):
        self.config = config
        self.state_file = "path/to/state.json"
    
    def _load_state(self):
        # Load extractor state
        pass
        
    def _save_state(self, state):
        # Save extractor state
        pass
    
    def extract(self, incremental=True) -> List[MetadataItem]:
        # Load state if incremental
        state = self._load_state() if incremental else {}
        
        # Extract metadata
        items = []
        
        # Example item creation
        item = MetadataItem(
            id="unique-id",
            name="Item Name",
            content="Detailed content/description",
            type=MetadataType.TABLE,  # Or other appropriate type
            source=MetadataSource(
                type=MetadataType.TABLE,
                url="source-url",
                created="creation-timestamp",
            ),
            attributes={"key": "value"}
        )
        items.append(item)
        
        # Update and save state
        state["last_run"] = "current-timestamp"
        self._save_state(state)
        
        return items
```

## Installation

```bash
# Install base package
pip install omen-extractors

# Install with Keboola support
pip install 'omen-extractors[keboola]'
```

For development:

```bash
git clone https://github.com/keboola/omen-platform
cd omen-platform
pip install -e "packages/omen-extractors[keboola]"
```

## Usage with OMEN CLI

The extractors can be used via the OMEN CLI:

```bash
# Set token as environment variable
export KEBOOLA_API_TOKEN=your-token

# Run extraction
omen extract keboola
```

## Dependencies

- Python 3.8+
- omen-core
- kbcstorage (for Keboola extractor)

## License

MIT License 