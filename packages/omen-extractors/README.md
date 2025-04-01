# OMEN Extractors

Metadata extraction components for the OMEN (Ontology-powered Metadata Engine) platform. This package provides extractors for various data sources that can feed metadata into the OMEN platform.

## Overview

The `omen-extractors` package implements metadata extractors for different data sources, with these key responsibilities:

1. **Source Connectivity**: Connect to metadata sources like Keboola
2. **Metadata Extraction**: Extract comprehensive metadata from sources
3. **Incremental Processing**: Track state to enable incremental updates
4. **Error Handling**: Robust error handling and retry mechanisms
5. **Extensibility**: Provide a framework for implementing custom extractors
6. **Multi-Project Support**: Auto-detect and manage multiple projects from different API tokens

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

#### Multi-Project Support

The Keboola extractor automatically detects the project ID from the provided API token, making it easy to work with multiple projects:

```python
# Create extractors for different projects using different tokens
project1_extractor = KeboolaExtractor(token="token-for-project-1")
project2_extractor = KeboolaExtractor(token="token-for-project-2")

# The project ID is auto-detected from each token
print(f"Project 1 ID: {project1_extractor.project_id}")
print(f"Project 2 ID: {project2_extractor.project_id}")

# Extract data from each project
project1_metadata = project1_extractor.extract()
project2_metadata = project2_extractor.extract()
```

Each project's metadata includes the project ID in its source information, enabling proper tracking and segregation of data from different projects.

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

3. **Configurations**: Component configurations and their metadata
   - Component type and ID
   - Configuration name and description
   - Related table information
   - Input and output mappings

4. **Columns**: Detailed column information
   - Data type and definitions
   - Descriptions and metadata
   - Relationships to parent tables

## State Management

Extractors maintain state to support incremental processing, tracking the last extraction time and processed items:

```python
# State is automatically managed per project
extractor = KeboolaExtractor(token="your-token")

# First run - extracts all metadata
initial_metadata = extractor.extract(incremental=True)

# Second run - only extracts changes since the first run
updated_metadata = extractor.extract(incremental=True)
```

State is stored in the `~/.omen/` directory by default, with separate state files for each project, following the naming pattern `keboola_state_{project_id}.json`.

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
                project_id="project-identifier"  # Add project ID for multi-project support
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

# The project ID is auto-detected from the token
# Each project gets its own state file and vector collection
```

Working with multiple projects:

```bash
# Extract from first project
export KEBOOLA_API_TOKEN=project1-token
omen extract keboola

# Extract from second project
export KEBOOLA_API_TOKEN=project2-token
omen extract keboola

# List all projects
omen projects list
```

## Dependencies

- Python 3.8+
- omen-core
- kbcstorage (for Keboola extractor)

## License

MIT License 