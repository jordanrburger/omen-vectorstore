# OMEN Hybrid Search Usage Guide

This guide demonstrates how to use OMEN's hybrid search functionality through both the Command Line Interface (CLI) and API.

## Prerequisites

Make sure you have installed the required dependencies:

```bash
pip3 install rdflib networkx sentence-transformers qdrant-client pydantic fastapi
```

## Using the CLI

The OMEN CLI provides commands for hybrid search and recommendations.

### Basic Hybrid Search

```bash
# Basic hybrid search
omen hybrid search "customer data analysis"

# Adjust the weights between vector and semantic search
omen hybrid search "customer data analysis" --vector-weight 0.3 --semantic-weight 0.7

# Show related entities
omen hybrid search "customer data analysis" --include-related --related-depth 2

# Filter by metadata type
omen hybrid search "customer data analysis" --type TABLE --type TRANSFORMATION
```

### Hybrid Recommendations

```bash
# Get hybrid recommendations for an entity
omen hybrid recommend entity-123

# Adjust weights for recommendations
omen hybrid recommend entity-123 --vector-weight 0.8 --semantic-weight 0.2

# Control the number of recommendations
omen hybrid recommend entity-123 --limit 5

# Hide relationship paths
omen hybrid recommend entity-123 --no-paths
```

## Using the API

The OMEN API provides endpoints for hybrid search functionality.

### Hybrid Search Endpoint

```http
POST /hybrid/search
```

Request body:

```json
{
  "query": "customer data analysis",
  "limit": 10,
  "offset": 0,
  "vector_weight": 0.7,
  "semantic_weight": 0.3,
  "type_filter": ["TABLE", "TRANSFORMATION"],
  "include_related": true,
  "related_depth": 2
}
```

Response:

```json
[
  {
    "id": "doc-123",
    "content": "This table contains customer information...",
    "score": 0.89,
    "vector_score": 0.92,
    "semantic_score": 0.82,
    "source": {
      "type": "TABLE",
      "name": "customer_data"
    },
    "metadata": {
      "row_count": 1250
    },
    "related_entities": [
      {
        "source_id": "entity-123",
        "target_id": "entity-456",
        "target_name": "Customer Segmentation",
        "relationship": "INPUTS_FROM",
        "direction": "incoming",
        "path_length": 1,
        "path": [
          {
            "id": "entity-456",
            "name": "Customer Segmentation",
            "rel": "INPUTS_FROM"
          }
        ]
      }
    ]
  }
]
```

### Hybrid Recommendations Endpoint

```http
POST /hybrid/recommend
```

Request body:

```json
{
  "entity_id": "entity-123",
  "limit": 5,
  "vector_weight": 0.5,
  "semantic_weight": 0.5
}
```

Response:

```json
[
  {
    "id": "doc-456",
    "title": "Customer Segmentation",
    "content": "Analysis of customer segments based on...",
    "type": "TRANSFORMATION",
    "vector_score": 0.75,
    "semantic_score": 0.82,
    "final_score": 0.79,
    "relationship_path": [
      {
        "id": "entity-456",
        "name": "Customer Segmentation",
        "rel": "INPUTS_FROM"
      }
    ]
  }
]
```

### Entity-Based Recommendations Endpoint

```http
GET /hybrid/entity-recommendations/entity-123?limit=5&include_paths=true
```

Response:

```json
[
  {
    "id": "entity-456",
    "name": "Customer Segmentation",
    "type": "TRANSFORMATION",
    "description": "Analysis of customer segments based on...",
    "relationship": "INPUTS_FROM",
    "relationship_path": [
      {
        "id": "entity-456",
        "name": "Customer Segmentation",
        "rel": "INPUTS_FROM"
      }
    ],
    "score": 0.9,
    "document": {
      "id": "doc-456",
      "title": "Customer Segmentation",
      "content": "Analysis of customer segments based on..."
    }
  }
]
```

## Python Client Example

```python
import requests

# Base URL for the API
base_url = "http://localhost:8000"

# Hybrid search
def hybrid_search(query, vector_weight=0.7, semantic_weight=0.3, include_related=False):
    url = f"{base_url}/hybrid/search"
    data = {
        "query": query,
        "limit": 10,
        "vector_weight": vector_weight,
        "semantic_weight": semantic_weight,
        "include_related": include_related
    }
    response = requests.post(url, json=data)
    return response.json()

# Hybrid recommendations
def hybrid_recommend(entity_id, vector_weight=0.5, semantic_weight=0.5):
    url = f"{base_url}/hybrid/recommend"
    data = {
        "entity_id": entity_id,
        "limit": 5,
        "vector_weight": vector_weight,
        "semantic_weight": semantic_weight
    }
    response = requests.post(url, json=data)
    return response.json()

# Entity-based recommendations
def entity_recommendations(entity_id, limit=5):
    url = f"{base_url}/hybrid/entity-recommendations/{entity_id}?limit={limit}&include_paths=true"
    response = requests.get(url)
    return response.json()

# Usage examples
results = hybrid_search("customer data analysis", vector_weight=0.3, semantic_weight=0.7)
print(f"Found {len(results)} results")

recommendations = hybrid_recommend("entity-123")
print(f"Found {len(recommendations)} recommendations")
```

## Next Steps

- Explore different weight configurations to optimize search results
- Use entity-based recommendations to discover related content
- Combine hybrid search with filters for more precise results

For more information, see the [OMEN documentation](https://github.com/keboola/omen-platform). 