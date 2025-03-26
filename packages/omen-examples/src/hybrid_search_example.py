#!/usr/bin/env python3
"""
Example demonstrating hybrid search using both vector similarity and ontology-based semantic search.
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the parent directory to path so we can import packages
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from omen.core import AppSettings, set_log_level
from omen.vectorstore import (
    MetadataDocument,
    MetadataType,
    HybridSearch,
    VectorSearch,
    QdrantIndexer,
)
from omen.ontology import OntologyManager, Entity, Relationship, EntityType, RelationshipType

# Set up logging
set_log_level("INFO")


def create_sample_documents():
    """Create sample metadata documents for demonstration."""
    return [
        MetadataDocument(
            id="doc-1",
            title="Customer Data Table",
            content="This table contains customer information including name, contact details, and purchase history.",
            type=MetadataType.TABLE,
            source="keboola",
            metadata={
                "row_count": 1250,
                "schema": "public",
                "entity_id": "entity-1"
            },
        ),
        MetadataDocument(
            id="doc-2",
            title="Product Catalog",
            content="Comprehensive list of products with pricing, inventory levels, and product specifications.",
            type=MetadataType.TABLE,
            source="keboola",
            metadata={
                "row_count": 5000,
                "schema": "public",
                "entity_id": "entity-2"
            },
        ),
        MetadataDocument(
            id="doc-3",
            title="Sales Transactions",
            content="Record of all sales transactions including customer, product, date, and amount.",
            type=MetadataType.TABLE,
            source="keboola",
            metadata={
                "row_count": 25000,
                "schema": "sales",
                "entity_id": "entity-3"
            },
        ),
        MetadataDocument(
            id="doc-4",
            title="Customer Segmentation",
            content="Analysis of customer segments based on purchasing behavior and demographics.",
            type=MetadataType.TRANSFORMATION,
            source="keboola",
            metadata={
                "created_by": "John Doe",
                "entity_id": "entity-4"
            },
        ),
        MetadataDocument(
            id="doc-5",
            title="Inventory Report",
            content="Monthly inventory report showing current stock levels, reorder points, and supply status.",
            type=MetadataType.REPORT,
            source="keboola",
            metadata={
                "created_by": "Jane Smith",
                "entity_id": "entity-5"
            },
        ),
    ]


def create_sample_ontology():
    """Create sample ontology with entities and relationships."""
    ontology = OntologyManager()
    
    # Create entities corresponding to the sample documents
    entity1 = Entity(
        id="entity-1",
        name="Customer Data Table",
        type=EntityType.TABLE,
        description="This table contains customer information including name, contact details, and purchase history.",
        properties={
            "row_count": 1250,
            "schema": "public",
        },
    )
    
    entity2 = Entity(
        id="entity-2",
        name="Product Catalog",
        type=EntityType.TABLE,
        description="Comprehensive list of products with pricing, inventory levels, and product specifications.",
        properties={
            "row_count": 5000,
            "schema": "public",
        },
    )
    
    entity3 = Entity(
        id="entity-3",
        name="Sales Transactions",
        type=EntityType.TABLE,
        description="Record of all sales transactions including customer, product, date, and amount.",
        properties={
            "row_count": 25000,
            "schema": "sales",
        },
    )
    
    entity4 = Entity(
        id="entity-4",
        name="Customer Segmentation",
        type=EntityType.TRANSFORMATION,
        description="Analysis of customer segments based on purchasing behavior and demographics.",
        properties={
            "created_by": "John Doe",
        },
    )
    
    entity5 = Entity(
        id="entity-5",
        name="Inventory Report",
        type=EntityType.TABLE,
        description="Monthly inventory report showing current stock levels, reorder points, and supply status.",
        properties={
            "created_by": "Jane Smith",
        },
    )
    
    # Add entities to the ontology
    for entity in [entity1, entity2, entity3, entity4, entity5]:
        ontology.add_entity(entity)
    
    # Create relationships between entities
    relationships = [
        # Sales transactions depend on customer data and product catalog
        Relationship(
            id="rel-1",
            type=RelationshipType.INPUTS_FROM,
            source_id="entity-3",  # Sales Transactions
            target_id="entity-1",  # Customer Data Table
        ),
        Relationship(
            id="rel-2",
            type=RelationshipType.INPUTS_FROM,
            source_id="entity-3",  # Sales Transactions
            target_id="entity-2",  # Product Catalog
        ),
        # Customer segmentation depends on customer data and sales transactions
        Relationship(
            id="rel-3",
            type=RelationshipType.INPUTS_FROM,
            source_id="entity-4",  # Customer Segmentation
            target_id="entity-1",  # Customer Data Table
        ),
        Relationship(
            id="rel-4",
            type=RelationshipType.INPUTS_FROM,
            source_id="entity-4",  # Customer Segmentation
            target_id="entity-3",  # Sales Transactions
        ),
        # Inventory report depends on product catalog and sales transactions
        Relationship(
            id="rel-5",
            type=RelationshipType.INPUTS_FROM,
            source_id="entity-5",  # Inventory Report
            target_id="entity-2",  # Product Catalog
        ),
        Relationship(
            id="rel-6",
            type=RelationshipType.INPUTS_FROM,
            source_id="entity-5",  # Inventory Report
            target_id="entity-3",  # Sales Transactions
        ),
    ]
    
    # Add relationships to ontology
    for relationship in relationships:
        ontology.add_relationship(relationship)
    
    # Save ontology state
    ontology.save_state()
    
    return ontology


def index_sample_documents(documents):
    """Index the sample documents in the vector store."""
    # Initialize the vector indexer
    indexer = QdrantIndexer()
    
    # Index documents
    for doc in documents:
        indexer.index_document(doc)
    
    return indexer


def run_example():
    """Run the hybrid search example."""
    print("OMEN Hybrid Search Example")
    print("==========================\n")
    
    # Create and index sample documents
    documents = create_sample_documents()
    indexer = index_sample_documents(documents)
    
    print(f"Indexed {len(documents)} sample documents in vector store\n")
    
    # Create sample ontology
    ontology = create_sample_ontology()
    
    print(f"Added {len(ontology.entities)} entities and {len(ontology.relationships)} relationships to ontology\n")
    
    # Initialize search engines
    vector_search = VectorSearch(indexer=indexer)
    hybrid_search = HybridSearch(vector_search=vector_search, ontology_manager=ontology)
    
    # 1. Perform vector-only search
    query = "customer data and purchase history"
    
    print(f"Vector Search Query: '{query}'")
    vector_results = vector_search.search(query, limit=3)
    
    print("\nVector Search Results:")
    for i, result in enumerate(vector_results, 1):
        print(f"{i}. {result.document.title} (Score: {result.score:.4f})")
        print(f"   Type: {result.document.type.value}")
        print(f"   Content: {result.document.content[:100]}...\n")
    
    # 2. Perform hybrid search with default weights
    print(f"Hybrid Search Query: '{query}'")
    hybrid_results = hybrid_search.search(query, limit=3)
    
    print("\nHybrid Search Results (Default Weights):")
    for i, result in enumerate(hybrid_results, 1):
        print(f"{i}. {result.document.title} (Score: {result.score:.4f})")
        print(f"   Vector Score: {result.vector_score:.4f}, Semantic Score: {result.semantic_score or 0:.4f}")
        print(f"   Type: {result.document.type.value}")
        print(f"   Content: {result.document.content[:100]}...\n")
    
    # 3. Perform hybrid search with adjusted weights
    print(f"Hybrid Search Query: '{query}' (Adjusted Weights)")
    hybrid_results_adj = hybrid_search.search(
        query, limit=3, vector_weight=0.3, semantic_weight=0.7
    )
    
    print("\nHybrid Search Results (Semantic Emphasis):")
    for i, result in enumerate(hybrid_results_adj, 1):
        print(f"{i}. {result.document.title} (Score: {result.score:.4f})")
        print(f"   Vector Score: {result.vector_score:.4f}, Semantic Score: {result.semantic_score or 0:.4f}")
        print(f"   Type: {result.document.type.value}")
        print(f"   Content: {result.document.content[:100]}...\n")
    
    # 4. Perform hybrid search with related entities
    print(f"Hybrid Search with Related Entities: '{query}'")
    related_results = hybrid_search.search(
        query, limit=3, include_related=True, max_related_depth=2
    )
    
    print("\nHybrid Search Results with Related Entities:")
    for i, result in enumerate(related_results, 1):
        print(f"{i}. {result.document.title} (Score: {result.score:.4f})")
        print(f"   Vector Score: {result.vector_score:.4f}, Semantic Score: {result.semantic_score or 0:.4f}")
        print(f"   Type: {result.document.type.value}")
        print(f"   Content: {result.document.content[:100]}...")
        
        if result.related_entities:
            print(f"   Related Entities: {len(result.related_entities)}")
            for j, entity in enumerate(result.related_entities[:2], 1):
                print(f"      {j}. {entity.get('target_name')} ({entity.get('relationship')})")
        print()
    
    # 5. Generate recommendations for an entity
    entity_id = "entity-1"  # Customer Data Table
    
    print(f"Entity-Based Recommendations for '{entity_id}':")
    entity_recs = hybrid_search.entity_based_recommendation(entity_id, limit=3)
    
    print("\nEntity-Based Recommendation Results:")
    for i, rec in enumerate(entity_recs, 1):
        print(f"{i}. {rec['name']} (Score: {rec['score']:.4f})")
        print(f"   Type: {rec['type']}")
        print(f"   Relationship: {rec['relationship']}")
        if rec['relationship_path']:
            print(f"   Path Length: {len(rec['relationship_path'])}")
        print()
    
    # 6. Generate hybrid recommendations
    print(f"Hybrid Recommendations for '{entity_id}':")
    hybrid_recs = hybrid_search.hybrid_recommendation(entity_id, limit=3)
    
    print("\nHybrid Recommendation Results:")
    for i, rec in enumerate(hybrid_recs, 1):
        print(f"{i}. {rec['title']} (Score: {rec['final_score']:.4f})")
        print(f"   Vector Score: {rec['vector_score']:.4f}, Semantic Score: {rec['semantic_score']:.4f}")
        print(f"   Type: {rec['type']}")
        print(f"   Content: {rec['content'][:100]}...")
        if rec['relationship_path']:
            print(f"   Related via: {[p['rel'] for p in rec['relationship_path']]}")
        print()


if __name__ == "__main__":
    run_example() 