# OMEN Hybrid Search

This module implements hybrid search capabilities for the OMEN platform, combining vector-based semantic search with ontology-powered knowledge graph queries.

## Overview

Hybrid search provides several key advantages over traditional vector search:

1. **Relevance Improvement**: By combining vector similarity with semantic relationships, search results are more contextually relevant
2. **Explainability**: Search results include relationship information, making them more interpretable
3. **Dynamic Weighting**: Control the balance between vector similarity and semantic relationships based on use case
4. **Enhanced Recommendations**: Generate recommendations that consider both content similarity and semantic connections

## Architecture

The hybrid search architecture consists of these primary components:

1. **Vector Search Engine**: Based on the `VectorSearch` class, provides semantic similarity using embeddings
2. **Ontology Manager**: Manages the knowledge graph with entities and their semantic relationships
3. **Hybrid Search Engine**: Combines results from both engines with configurable weighting
4. **Result Fusion**: Merges and ranks results from both search methods

## Usage

### Basic Hybrid Search

```python
from omen.vectorstore import HybridSearch, VectorSearch
from omen.ontology import OntologyManager

# Initialize components
vector_search = VectorSearch()
ontology = OntologyManager()

# Create hybrid search with default weights (70% vector, 30% semantic)
hybrid_search = HybridSearch(
    vector_search=vector_search,
    ontology_manager=ontology
)

# Perform hybrid search
results = hybrid_search.search(
    query="customer data analysis",
    limit=10
)

# Process results
for result in results:
    print(f"Document: {result.document.title}")
    print(f"Score: {result.score}")
    print(f"Vector Score: {result.vector_score}")
    print(f"Semantic Score: {result.semantic_score}")
```

### Adjusting Search Weights

You can control the balance between vector similarity and semantic relationships:

```python
# Emphasize semantic relationships (30% vector, 70% semantic)
results = hybrid_search.search(
    query="customer data analysis",
    limit=10,
    vector_weight=0.3,
    semantic_weight=0.7
)

# Pure vector search
results = hybrid_search.search(
    query="customer data analysis",
    limit=10,
    vector_weight=1.0,
    semantic_weight=0.0
)

# Pure semantic search
results = hybrid_search.search(
    query="customer data analysis",
    limit=10,
    vector_weight=0.0,
    semantic_weight=1.0
)
```

### Including Related Entities

You can include semantically related entities in search results:

```python
results = hybrid_search.search(
    query="customer data analysis",
    limit=10,
    include_related=True,
    max_related_depth=2  # How far to traverse the knowledge graph
)

# Process results with related entities
for result in results:
    print(f"Document: {result.document.title}")
    print(f"Score: {result.score}")
    
    # Display related entities
    if result.related_entities:
        print(f"Related entities: {len(result.related_entities)}")
        for entity in result.related_entities:
            print(f"  - {entity['target_name']} ({entity['relationship']})")
```

### Entity-Based Recommendations

Find recommended entities based on ontology relationships:

```python
# Get recommendations for a specific entity
recommendations = hybrid_search.entity_based_recommendation(
    entity_id="table-123",
    limit=5,
    include_paths=True
)

# Process recommendations
for rec in recommendations:
    print(f"Entity: {rec['name']}")
    print(f"Type: {rec['type']}")
    print(f"Score: {rec['score']}")
    print(f"Relationship: {rec['relationship']}")
    
    # Show relationship path
    if rec['relationship_path']:
        path = " → ".join([p['name'] for p in rec['relationship_path']])
        print(f"Path: {path}")
```

### Hybrid Recommendations

Generate recommendations using both vector similarity and semantic relationships:

```python
# Get hybrid recommendations for an entity
hybrid_recs = hybrid_search.hybrid_recommendation(
    entity_id="table-123",
    limit=5,
    vector_weight=0.5,
    semantic_weight=0.5
)

# Process hybrid recommendations
for rec in hybrid_recs:
    print(f"Title: {rec['title']}")
    print(f"Final Score: {rec['final_score']}")
    print(f"Vector Score: {rec['vector_score']}")
    print(f"Semantic Score: {rec['semantic_score']}")
    
    # Show relationship path if available
    if rec['relationship_path']:
        print(f"Connected via: {[p['rel'] for p in rec['relationship_path']]}")
```

## Implementation Details

### Search Result Fusion

The hybrid search engine combines results using a weighted scoring approach:

1. Normalize vector search scores (0-1 range)
2. Normalize semantic search scores (0-1 range)
3. Apply weights to each score component
4. Combine scores to produce a final ranking
5. Sort and return results based on combined score

### Entity-Document Mapping

To bridge the gap between the vector store and ontology:

- Document IDs are mapped to entity IDs where possible
- Document metadata may contain explicit `entity_id` fields
- The hybrid search engine maintains a cache for efficient mapping

## Performance Considerations

- The default weights (70% vector, 30% semantic) work well for general search scenarios
- Vector search is typically faster but less precise for relationships
- Semantic search provides better relationship information but may miss content similarity
- For exploratory search, emphasize semantic weights
- For specific content queries, emphasize vector weights

## Examples

See the [hybrid_search_example.py](../omen-examples/src/hybrid_search_example.py) script for a complete working example of hybrid search functionality. 