#!/usr/bin/env python3
"""
Test script for relationship detection.
"""

import os
import json
import logging
from dotenv import load_dotenv
from ontology.builder import OntologyBuilder
from llm_client import LLMClient
from ontology.models import Entity, EntityType
import uuid

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    """
    Test relationship detection functionality.
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize LLM client with GPT-4-Turbo
    llm_client = LLMClient(
        provider="openai",
        model="gpt-4-turbo",
        temperature=0.1,
        max_tokens=4000
    )
    
    # Log LLM client configuration
    logger.info(f"LLM Client Configuration: provider={llm_client.provider}, model={llm_client.model}, temperature={llm_client.temperature}, max_tokens={llm_client.max_tokens}")
    
    # Initialize ontology builder
    builder = OntologyBuilder(llm_client)
    
    # Create sample entities
    bucket_entity = Entity(
        id="bucket_1",
        type=EntityType.BUCKET,
        name="sample_bucket",
        properties={
            "id": "bucket_1",
            "name": "sample_bucket",
            "stage": "out",
            "description": "Sample bucket for testing"
        }
    )
    
    table_entity = Entity(
        id="table_1",
        type=EntityType.TABLE,
        name="sample_table",
        properties={
            "id": "table_1",
            "name": "sample_table",
            "displayName": "Sample Table",
            "description": "Sample table for testing"
        }
    )
    
    column_entity = Entity(
        id="column_1",
        type=EntityType.COLUMN,
        name="sample_column",
        properties={
            "id": "column_1",
            "name": "sample_column",
            "dataType": "string",
            "description": "Sample column for testing"
        }
    )
    
    # Test entities
    entities = [bucket_entity, table_entity, column_entity]
    
    # Log the entities
    logger.info(f"Testing relationship detection with {len(entities)} entities")
    for entity in entities:
        logger.info(f"Entity: {entity.id}, Type: {entity.type}, Name: {entity.name}")
    
    # Get relationship prompt to see what's being sent
    entity_dicts = [
        {
            "id": entity.id,
            "type": entity.type.value,
            "properties": entity.properties
        }
        for entity in entities
    ]
    
    # Get access to the private method for testing
    from ontology.schema_definition import default_schema
    from ontology.prompts import get_relationship_type_schema_prompt
    
    schema_prompt = get_relationship_type_schema_prompt(default_schema.relationship_types)
    
    # Format the prompt manually for testing
    from ontology.prompts import format_relationship_detection_prompt
    prompt = format_relationship_detection_prompt(
        entities=entity_dicts,
        schema_definitions=schema_prompt
    )
    
    # Log the prompt
    logger.info("System Prompt:")
    logger.info(prompt["system_prompt"])
    logger.info("User Prompt:")
    logger.info(prompt["user_prompt"])
    
    # Make the LLM call directly for debugging
    logger.info("Making direct LLM call...")
    raw_response = llm_client.generate(
        prompt=prompt["user_prompt"],
        system_prompt=prompt["system_prompt"],
        expected_format="json"
    )
    
    # Log the raw response
    logger.info("Raw LLM Response:")
    if isinstance(raw_response, str):
        logger.info(raw_response)
    else:
        logger.info(json.dumps(raw_response, indent=2))
    
    # Test manual relationship creation from the response
    logger.info("Testing manual relationship creation from the response...")
    test_manual_relationship_creation(builder, entities, raw_response)
    
    # Detect relationships using the builder
    logger.info("Detecting relationships...")
    relationships = builder.detect_relationships(entities)
    
    # Log the results
    logger.info(f"Found {len(relationships)} relationships")
    for relationship in relationships:
        logger.info(f"Relationship: {relationship.id}, Type: {relationship.type}, " +
                   f"Source: {relationship.source_id}, Target: {relationship.target_id}")
    
    return 0

def test_manual_relationship_creation(builder, entities, response):
    """Test manual relationship creation from raw LLM response."""
    from ontology.models import Relationship, RelationshipType
    
    relationships = []
    entities_dict = {entity.id: entity for entity in entities}
    
    if isinstance(response, dict) and 'relationships' in response:
        rel_data_list = response['relationships']
        if isinstance(rel_data_list, list):
            for rel_data in rel_data_list:
                try:
                    # Get essential fields
                    rel_id = rel_data.get("id", str(uuid.uuid4()))
                    type_value = rel_data.get("type")
                    source_id = rel_data.get("source_id")
                    target_id = rel_data.get("target_id")
                    properties = rel_data.get("properties", {})
                    
                    # Log the fields
                    logger.info(f"Manual processing: ID={rel_id}, Type={type_value}, " +
                               f"Source={source_id}, Target={target_id}")
                    
                    # Skip if missing required fields
                    if not type_value or not source_id or not target_id:
                        logger.warning("Missing required fields, skipping")
                        continue
                    
                    # Create relationship type enum
                    try:
                        rel_type = RelationshipType(type_value)
                    except ValueError:
                        logger.warning(f"Invalid relationship type: {type_value}")
                        continue
                    
                    # Create relationship
                    relationship = Relationship(
                        id=rel_id,
                        type=rel_type,
                        source_id=source_id,
                        target_id=target_id,
                        properties=properties
                    )
                    
                    # Validate relationship
                    is_valid, errors = builder.schema_validator.validate_relationship(
                        relationship, entities_dict
                    )
                    
                    if is_valid:
                        logger.info(f"Created valid relationship: {rel_id}, Type: {rel_type.value}")
                        relationships.append(relationship)
                    else:
                        logger.warning(f"Invalid relationship: {errors}")
                except Exception as e:
                    logger.error(f"Error creating relationship: {e}")
    
    logger.info(f"Manually created {len(relationships)} relationships")
    return relationships

if __name__ == "__main__":
    exit(main()) 