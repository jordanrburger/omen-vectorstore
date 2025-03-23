# Keboola Ontology & Action Graph Module Implementation Plan

## 1. Create Core Ontology Module Structure

1. Create a new directory `app/ontology/` to house the ontology and action graph code
2. Implement the basic RDF triple structure in `app/ontology/models.py`
3. Create the ontology manager in `app/ontology/manager.py`
4. Implement persistent storage for RDF triples in `app/ontology/storage.py`

## 2. Define RDF Schema and Basic Ontology Structure

1. Define core entity types (Table, Column, Transformation, Configuration, etc.)
2. Define relationship types (hasColumn, dependsOn, inputFrom, outputTo, etc.)
3. Create a base ontology schema in `app/ontology/schema.py`
4. Implement schema validation functionality

## 3. Implement Ontology Builder using LLMs

1. Create `app/ontology/builder.py` to define LLM-powered ontology construction
2. Implement methods to extract entities from metadata using LLMs
3. Develop relationship detection between entities
4. Build prompt templates for entity and relationship extraction in `app/ontology/prompts.py`
5. Add batch processing for large metadata sets

## 4. Create Action Graph Generator

1. Implement `app/ontology/action_graph.py` for creating action graphs
2. Define action types and their relationships to metadata
3. Create LLM-based action inference from metadata
4. Implement graph traversal and visualization utilities

## 5. Integrate with Existing Components

1. Extend `app/keboola_client.py` to support ontology-specific metadata extraction
2. Modify `app/main.py` to include ontology and action graph building
3. Update `app/state_manager.py` to store ontology state
4. Create index methods in `app/indexer.py` for ontology triples

## 6. Implement RDF Storage and Querying

1. Add RDF triple store functionality (using rdflib)
2. Implement SPARQL query interface
3. Create serialization/deserialization for RDF data
4. Add incremental update support for the ontology

## 7. Create LLM-Powered Utilities

1. Implement semantic entity matching using embeddings
2. Create natural language to SPARQL query converter
3. Develop explanation generator for relationships and actions
4. Add tools for ontology validation and repair

## 8. Add API Endpoints

1. Create REST endpoints for ontology access in `app/main.py`
2. Implement action graph query interface
3. Add natural language query endpoints
4. Create visualization endpoints for the ontology and action graph

## 9. Implementation Details and Dependencies

**Required Python packages:**
```
rdflib>=6.3.2
networkx>=3.1
openai>=1.0.0
pydantic>=2.5.0
graphviz>=0.20.1
```

**Key Implementation Choices:**
1. Use `rdflib` for RDF triple storage and SPARQL querying
2. Use OpenAI for entity and relationship extraction
3. Use NetworkX for action graph manipulation
4. Store serialized RDF in the state manager for persistence
5. Support incremental ontology updates based on metadata changes

## 10. Step-by-Step Implementation Process

1. First, implement the core RDF models and storage
2. Build the basic ontology schema
3. Create the LLM-powered entity extractor
4. Implement the relationship detector
5. Build the action graph generator
6. Integrate with the existing metadata pipeline
7. Add query and visualization capabilities
8. Test and optimize with real Keboola project data

## Progress Tracking

- [x] Step 1: Create Core Ontology Module Structure
- [x] Step 2: Define RDF Schema and Basic Ontology Structure
- [ ] Step 3: Implement Ontology Builder using LLMs
- [ ] Step 4: Create Action Graph Generator
- [ ] Step 5: Integrate with Existing Components
- [ ] Step 6: Implement RDF Storage and Querying
- [ ] Step 7: Create LLM-Powered Utilities
- [ ] Step 8: Add API Endpoints 