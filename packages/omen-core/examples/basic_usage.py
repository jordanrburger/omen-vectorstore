"""
Basic example demonstrating the usage of omen-core package.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any

from omen.core.config import AppSettings
from omen.core.state import StateManager
from omen.core.batch import BatchProcessor
from omen.core.llm import LLMClient

async def process_metadata(item: Dict[str, Any], llm_client: LLMClient) -> Dict[str, Any]:
    """Process metadata using LLM."""
    # Generate description using LLM
    prompt = f"Describe this metadata: {item['data']}"
    description = llm_client.generate(prompt)
    
    return {
        **item,
        "description": description,
        "processed": True
    }

async def process_batch(items: List[Dict[str, Any]], llm_client: LLMClient) -> List[Dict[str, Any]]:
    """Process a batch of items."""
    results = []
    for item in items:
        result = await process_metadata(item, llm_client)
        results.append(result)
    return results

async def main():
    # Initialize configuration
    config = AppSettings()
    
    # Initialize state manager
    state_manager = StateManager()
    
    # Initialize LLM client
    llm_client = LLMClient(
        provider="openai",
        model="gpt-4",
        temperature=0.7
    )
    
    # Initialize batch processor
    processor = BatchProcessor(
        batch_size=2,
        max_workers=2,
        max_retries=3
    )
    
    # Sample metadata
    metadata_items = [
        {"id": 1, "data": "Table: sales_data, Columns: date, amount, customer_id"},
        {"id": 2, "data": "Table: customers, Columns: id, name, email, country"},
        {"id": 3, "data": "Table: products, Columns: id, name, price, category"}
    ]
    
    # Save initial state
    state_manager.set("metadata_items", metadata_items)
    state_manager.set("processed_count", 0)
    
    # Process metadata in batches
    results = await process_batch(metadata_items, llm_client)
    
    # Update state with results
    state_manager.set("processed_items", results)
    state_manager.set("processed_count", len(results))
    
    # Print results
    print("\nProcessed Metadata:")
    for result in results:
        print(f"\nID: {result['id']}")
        print(f"Data: {result['data']}")
        print(f"Description: {result['description']}")
    
    # Print state
    processed_count = state_manager.get("processed_count")
    print(f"\nTotal items processed: {processed_count}")

if __name__ == "__main__":
    asyncio.run(main()) 