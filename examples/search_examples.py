#!/usr/bin/env python3

import logging
from pprint import pprint

from app.vectorizer import OpenAIProvider
from app.indexer import QdrantIndexer
from app.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_transformation_result(result, index=None):
    """Pretty print a transformation search result."""
    prefix = f"{index}. " if index is not None else ""
    print(f"\n{prefix}Score: {result['score']:.3f}")
    print(f"Name: {result['metadata']['name']}")
    print(f"Type: {result['metadata']['type']}")
    if "description" in result['metadata']:
        print(f"Description: {result['metadata']['description']}")
    
    # Print dependencies
    if "dependencies" in result['metadata']:
        deps = result['metadata']['dependencies']
        if "requires" in deps:
            print("Requires:")
            for req in deps["requires"]:
                print(f"  - {req}")
        if "produces" in deps:
            print("Produces:")
            for prod in deps["produces"]:
                print(f"  - {prod}")
    
    # Print code blocks summary
    if "blocks" in result['metadata']:
        print("\nCode Blocks:")
        for block in result['metadata']['blocks']:
            print(f"  - {block['name']}")
            if "inputs" in block:
                print("    Inputs:")
                for inp in block["inputs"]:
                    print(f"      {inp['destination']} from {inp['source']}")
            if "outputs" in block:
                print("    Outputs:")
                for out in block["outputs"]:
                    print(f"      {out['source']} to {out['destination']}")
            if "code" in block:
                print("    Code Preview:")
                code_lines = block["code"].strip().split("\n")[:3]
                print("      " + "\n      ".join(code_lines))

def main():
    """Run comprehensive search tests."""
    # Initialize components
    config = Config.from_env()
    logger.info(f"Loaded config with OpenAI key: {config.openai_api_key[:8]}...")
    logger.info(f"Using embedding model: {config.embedding_model}")
    
    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY must be set in .env file")
        
    embedding_provider = OpenAIProvider(
        api_key=config.openai_api_key,
        model=config.embedding_model
    )
    indexer = QdrantIndexer(
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection_name=config.qdrant_collection
    )

    # Test 1: Search for transformations by functionality
    print("\n=== Search for transformations that process customer data ===")
    results = indexer.search_metadata(
        query="Find transformations that process customer data",
        embedding_provider=embedding_provider,
        metadata_type="transformations",
        limit=3
    )
    for i, result in enumerate(results, 1):
        print_transformation_result(result, i)

    # Test 2: Search for specific code operations
    print("\n=== Search for transformations that perform data merging ===")
    results = indexer.search_metadata(
        query="Find transformations that merge or join data",
        embedding_provider=embedding_provider,
        metadata_type="transformations",
        limit=3
    )
    for i, result in enumerate(results, 1):
        print_transformation_result(result, i)

    # Test 3: Find transformations related to specific data sources
    print("\n=== Find transformations related to Slack data ===")
    results = indexer.search_metadata(
        query="Find transformations working with Slack messages or channels",
        embedding_provider=embedding_provider,
        metadata_type="transformations",
        limit=3
    )
    for i, result in enumerate(results, 1):
        print_transformation_result(result, i)

    # Test 4: Search for specific transformation blocks
    print("\n=== Search for transformation blocks that load data ===")
    results = indexer.search_metadata(
        query="Find transformation blocks that load data from CSV files",
        embedding_provider=embedding_provider,
        metadata_type="transformation_blocks",
        limit=3
    )
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"Block Name: {result['metadata']['name']}")
        if "inputs" in result['metadata']:
            print("Inputs:")
            for inp in result['metadata']["inputs"]:
                print(f"  - {inp['destination']} from {inp['source']}")
        if "code" in result['metadata']:
            print("Code Preview:")
            code_lines = result['metadata']["code"].strip().split("\n")[:3]
            print("  " + "\n  ".join(code_lines))

    # Test 5: Find related transformations for a specific table
    print("\n=== Find transformations related to a specific table ===")
    table_id = "in.c-in_sh_ex_zendesk_keboola.tickets"  # Example table ID
    results = indexer.find_related_transformations(
        table_id=table_id,
        embedding_provider=embedding_provider,
        limit=3
    )
    print(f"\nRelated transformations for table {table_id}:")
    for i, result in enumerate(results, 1):
        print_transformation_result(result, i)

    # Test 6: Find similar columns
    print("\n=== Find similar columns across tables ===")
    column_name = "id"  # Example column name
    table_id = "in.c-in_sh_ex_zendesk_keboola.tickets"  # Example table ID
    try:
        results = indexer.find_similar_columns(
            column_name=column_name,
            table_id=table_id,
            embedding_provider=embedding_provider,
            limit=3
        )
        print(f"\nColumns similar to {column_name} in {table_id}:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"Column: {result['metadata']['name']}")
            print(f"Table: {result['table_id']}")
            if "type" in result['metadata']:
                print(f"Type: {result['metadata']['type']}")
            if "description" in result['metadata']:
                print(f"Description: {result['metadata']['description']}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 