# OMEN Ontology

Ontology and knowledge graph functionality for the OMEN (Ontology-powered Metadata Engine) platform. This package provides capabilities for storing, querying, and managing semantic relationships between metadata entities.

## Overview

The `omen-ontology` package manages the knowledge graph aspects of the OMEN platform, with these key responsibilities:

1. **Entity Management**: Define and store metadata entities with their properties
2. **Relationship Handling**: Create and maintain relationships between entities
3. **Knowledge Graph Storage**: Persistent storage of the ontology using RDF
4. **Graph Queries**: Query the knowledge graph for entities and relationships
5. **Schema Validation**: Ensure data integrity through schema validation
6. **Multi-Project Support**: Manage separate ontologies for different projects

## Multi-Project Support

The ontology manager supports working with multiple projects simultaneously by organizing ontology data in project-specific directories:

```python
from omen.ontology import OntologyManager
from pathlib import Path

# Create ontology managers for different projects
project1_manager = OntologyManager(state_dir=Path("state/ontology/project1"))
project2_manager = OntologyManager(state_dir=Path("state/ontology/project2"))

# Each project has its own set of entities and relationships
project1_manager.load_state()
project2_manager.load_state()

# Get statistics for each project
project1_stats = project1_manager.get_stats()
project2_stats = project2_manager.get_stats()

print(f"Project 1 entities: {project1_stats['total_entities']}")
print(f"Project 2 entities: {project2_stats['total_entities']}")
```

The default directory structure for multi-project ontologies is:
```
state/ontology/
├── project1/
│   ├── entities.json
│   ├── relationships.json
│   └── ontology.ttl
├── project2/
│   ├── entities.json
│   ├── relationships.json
│   └── ontology.ttl
└── ...
```

## Key Components

### Ontology Manager (`omen.ontology.manager`)

Central interface for working with the ontology:

```python
from omen.ontology import OntologyManager

# Initialize manager
manager = OntologyManager()

# Load existing state (if any)
manager.load_state()

# Add entities
person_entity = manager.add_entity(
    entity_type="PERSON",
    name="John Doe",
    properties={
        "email": "john.doe@example.com",
        "department": "Sales"
    },
    source="employee-directory"
)

table_entity = manager.add_entity(
    entity_type="TABLE",
    name="customers",
    properties={
        "rowCount": "1250",
        "description": "Customer master data"
    },
    source="keboola-metadata"
)

# Create relationships
manager.add_relationship(
    source_entity=person_entity,
    target_entity=table_entity,
    relationship_type="HAS_ACCESS_TO",
    properties={
        "access_level": "admin",
        "granted_date": "2023-01-15"
    }
)

# Save state
manager.save_state()
```

### RDF Store (`omen.ontology.rdf_store`)

Low-level RDF graph operations:

```python
from omen.ontology.rdf_store import RDFStore
from rdflib import URIRef, Literal, Namespace

# Initialize store
store = RDFStore()

# Define namespaces
ns = Namespace("http://omen.keboola.com/ontology#")

# Add triples
store.add_triple(
    subject=URIRef("http://omen.keboola.com/entity/table/123"),
    predicate=ns.hasName,
    object=Literal("customer_orders")
)

# Query the store
results = store.query("""
    SELECT ?table ?name
    WHERE {
        ?table a <http://omen.keboola.com/ontology#Table> .
        ?table <http://omen.keboola.com/ontology#hasName> ?name .
    }
""")

# Export to various formats
turtle_data = store.serialize(format="turtle")
```

### Models (`omen.ontology.models`)

Domain models for ontology entities and relationships:

```python
from omen.ontology.models import Entity, EntityType, Relationship, RelationshipType

# Create entity types
person_type = EntityType(name="PERSON", description="A human individual")
table_type = EntityType(name="TABLE", description="Database table")

# Create entities
person = Entity(
    id="person-123",
    type=person_type,
    name="Alice Smith",
    properties={"email": "alice@example.com"},
    source="employee-directory"
)

table = Entity(
    id="table-456",
    type=table_type,
    name="orders",
    properties={"schema": "public", "rowCount": "5000"},
    source="keboola-metadata"
)

# Create relationship types
access_type = RelationshipType(
    name="HAS_ACCESS_TO",
    description="Indicates access permissions to a resource"
)

# Create relationships
relationship = Relationship(
    id="rel-789",
    type=access_type,
    source_entity=person,
    target_entity=table,
    properties={"access_level": "read"},
)
```

## Querying the Ontology

The ontology can be queried using both high-level methods and direct SPARQL:

```python
# High-level query methods
entities = manager.get_entities(entity_type="TABLE")
relationships = manager.get_relationships(
    source_type="PERSON",
    relationship_type="HAS_ACCESS_TO"
)

# Find related entities
related = manager.get_related_entities(
    entity_id="person-123",
    relationship_type="HAS_ACCESS_TO"
)

# Advanced SPARQL queries
results = manager.query_sparql("""
    SELECT ?entity ?name ?created
    WHERE {
        ?entity a <http://omen.keboola.com/ontology#Table> .
        ?entity <http://omen.keboola.com/ontology#hasName> ?name .
        ?entity <http://omen.keboola.com/ontology#createdAt> ?created .
        FILTER(CONTAINS(?name, "customer"))
    }
    ORDER BY DESC(?created)
    LIMIT 10
""")
```

## Statistics and Insights

Get statistics and insights about the ontology:

```python
stats = manager.get_stats()

print(f"Total entities: {stats['total_entities']}")
print(f"Total relationships: {stats['total_relationships']}")
print(f"Triple count: {stats['triple_count']}")

# Entity types distribution
for entity_type, count in stats["entity_types"].items():
    print(f"{entity_type}: {count} entities")

# Relationship types distribution
for rel_type, count in stats["relationship_types"].items():
    print(f"{rel_type}: {count} relationships")
```

## CLI Commands for Multi-Project Ontologies

The OMEN CLI provides several commands for working with multi-project ontologies:

```bash
# View ontology statistics for a specific project
omen ontology stats --project-id PROJECT_ID

# List all available project ontologies
omen ontology stats --list-projects

# Clear ontology data for a specific project
omen ontology clear --project-id PROJECT_ID

# Visualize ontology graph for a specific project
omen ontology visualize --project-id PROJECT_ID --output graph.png

# List entities from a specific project's ontology
omen ontology list-entities --project-id PROJECT_ID --type TABLE
```

## Installation

```bash
pip install omen-ontology
```

Or for development:

```bash
git clone https://github.com/keboola/omen-platform
cd omen-platform
pip install -e "packages/omen-ontology"
```

## Dependencies

- Python 3.8+
- omen-core
- rdflib
- pydantic

## License

MIT License
