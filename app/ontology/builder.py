"""
Ontology builder for Keboola metadata using LLMs.

This module provides functionality to construct an ontology from Keboola metadata
using LLM-powered entity extraction and relationship detection.
"""

import uuid
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from app.ontology.models import Entity, Relationship, Triple, EntityType, RelationshipType
from app.ontology.manager import OntologyManager
from app.ontology.schema import SchemaValidator
from app.ontology.schema_definition import default_schema
from app.ontology.prompts import (
    format_entity_extraction_prompt,
    format_relationship_detection_prompt,
    format_triple_validation_prompt,
    get_entity_type_schema_prompt,
    get_relationship_type_schema_prompt
)
from app.llm_client import LLMClient

logger = logging.getLogger(__name__)


class OntologyBuilder:
    """
    Builder class for constructing an ontology from Keboola metadata using LLMs.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        ontology_manager: Optional[OntologyManager] = None,
        schema_validator: Optional[SchemaValidator] = None,
        batch_size: int = 10,
        max_workers: int = 4,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the ontology builder.
        
        Args:
            llm_client: Client for interacting with language models
            ontology_manager: Optional ontology manager to store the constructed ontology
            schema_validator: Optional schema validator for validating entities and relationships
            batch_size: Number of items to process in a batch
            max_workers: Maximum number of parallel workers for batch processing
            max_retries: Maximum number of retries for failed LLM calls
            retry_delay: Delay between retries in seconds
        """
        self.llm_client = llm_client
        self.ontology_manager = ontology_manager or OntologyManager()
        self.schema_validator = schema_validator or SchemaValidator(default_schema)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Cache for entity type and relationship type schema prompts
        self._entity_type_schema_prompt = None
        self._relationship_type_schema_prompt = None
    
    def build_ontology(self, metadata_collection: List[Dict[str, Any]]) -> OntologyManager:
        """
        Build an ontology from a collection of metadata items.
        
        Args:
            metadata_collection: List of metadata items to process
            
        Returns:
            OntologyManager containing the constructed ontology
        """
        logger.info(f"Building ontology from {len(metadata_collection)} metadata items")
        
        # Extract entities from metadata
        entities = self.extract_entities_batch(metadata_collection)
        logger.info(f"Extracted {len(entities)} entities from metadata")
        
        # Add entities to the ontology manager
        for entity in entities:
            self.ontology_manager.add_entity(entity)
        
        # Detect relationships between entities
        relationships = self.detect_relationships_batch(entities)
        logger.info(f"Detected {len(relationships)} relationships between entities")
        
        # Add relationships to the ontology manager
        for relationship in relationships:
            self.ontology_manager.add_relationship(relationship)
        
        # Build triples from entities and relationships
        triples = self._build_triples_from_relationships(relationships)
        logger.info(f"Built {len(triples)} triples from relationships")
        
        # Add triples to the ontology manager
        for triple in triples:
            self.ontology_manager.add_triple(triple)
        
        return self.ontology_manager
    
    def extract_entities(self, metadata: Dict[str, Any]) -> List[Entity]:
        """
        Extract entities from a single metadata item using an LLM.
        
        Args:
            metadata: Metadata item to extract entities from
            
        Returns:
            List of extracted entities
        """
        # Get schema prompt for entity types if not cached
        if self._entity_type_schema_prompt is None:
            self._entity_type_schema_prompt = get_entity_type_schema_prompt(
                default_schema.entity_types
            )
        
        # Format prompt for entity extraction
        prompt = format_entity_extraction_prompt(
            metadata=metadata,
            schema_definitions=self._entity_type_schema_prompt
        )
        
        # Call LLM with retries
        json_response = self._call_llm_with_retry(
            system_prompt=prompt["system_prompt"],
            user_prompt=prompt["user_prompt"],
            expected_format="json"
        )
        
        # Parse and validate entities
        entities = []
        for entity_data in json_response:
            try:
                # Create entity
                entity_type = EntityType(entity_data.get("type"))
                entity_id = entity_data.get("id", str(uuid.uuid4()))
                properties = entity_data.get("properties", {})
                
                entity = Entity(
                    id=entity_id,
                    type=entity_type,
                    properties=properties
                )
                
                # Validate entity
                is_valid, errors = self.schema_validator.validate_entity(entity)
                if not is_valid:
                    logger.warning(f"Invalid entity extracted: {entity_id}, errors: {errors}")
                    continue
                
                entities.append(entity)
            except Exception as e:
                logger.error(f"Error creating entity from LLM response: {e}")
                continue
        
        return entities
    
    def extract_entities_batch(self, metadata_collection: List[Dict[str, Any]]) -> List[Entity]:
        """
        Extract entities from a batch of metadata items in parallel.
        
        Args:
            metadata_collection: List of metadata items to extract entities from
            
        Returns:
            List of all extracted entities
        """
        all_entities = []
        
        # Process metadata items in batches
        batches = [
            metadata_collection[i:i + self.batch_size]
            for i in range(0, len(metadata_collection), self.batch_size)
        ]
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}")
            
            batch_entities = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit extraction tasks
                future_to_metadata = {
                    executor.submit(self.extract_entities, metadata): metadata
                    for metadata in batch
                }
                
                # Collect results
                for future in as_completed(future_to_metadata):
                    metadata = future_to_metadata[future]
                    try:
                        entities = future.result()
                        batch_entities.extend(entities)
                    except Exception as e:
                        logger.error(f"Error extracting entities from metadata: {e}")
            
            all_entities.extend(batch_entities)
        
        return all_entities
    
    def detect_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """
        Detect relationships between entities using an LLM.
        
        Args:
            entities: List of entities to detect relationships between
            
        Returns:
            List of detected relationships
        """
        # Get schema prompt for relationship types if not cached
        if self._relationship_type_schema_prompt is None:
            self._relationship_type_schema_prompt = get_relationship_type_schema_prompt(
                default_schema.relationship_types
            )
        
        # Convert entities to dictionary format for the prompt
        entity_dicts = [
            {
                "id": entity.id,
                "type": entity.type.value,
                "properties": entity.properties
            }
            for entity in entities
        ]
        
        # Format prompt for relationship detection
        prompt = format_relationship_detection_prompt(
            entities=entity_dicts,
            schema_definitions=self._relationship_type_schema_prompt
        )
        
        # Call LLM with retries
        json_response = self._call_llm_with_retry(
            system_prompt=prompt["system_prompt"],
            user_prompt=prompt["user_prompt"],
            expected_format="json"
        )
        
        # Parse and validate relationships
        relationships = []
        entities_dict = {entity.id: entity for entity in entities}
        
        for rel_data in json_response:
            try:
                # Create relationship
                rel_type = RelationshipType(rel_data.get("type"))
                rel_id = rel_data.get("id", str(uuid.uuid4()))
                source_id = rel_data.get("source_id")
                target_id = rel_data.get("target_id")
                properties = rel_data.get("properties", {})
                
                # Skip if source or target entity doesn't exist
                if source_id not in entities_dict or target_id not in entities_dict:
                    logger.warning(f"Relationship {rel_id} references nonexistent entity")
                    continue
                
                relationship = Relationship(
                    id=rel_id,
                    type=rel_type,
                    source_id=source_id,
                    target_id=target_id,
                    properties=properties
                )
                
                # Validate relationship
                is_valid, errors = self.schema_validator.validate_relationship(
                    relationship, entities_dict
                )
                if not is_valid:
                    logger.warning(f"Invalid relationship detected: {rel_id}, errors: {errors}")
                    continue
                
                relationships.append(relationship)
            except Exception as e:
                logger.error(f"Error creating relationship from LLM response: {e}")
                continue
        
        return relationships
    
    def detect_relationships_batch(self, entities: List[Entity]) -> List[Relationship]:
        """
        Detect relationships between entities in batches to handle large entity sets.
        
        Args:
            entities: List of all entities to detect relationships between
            
        Returns:
            List of all detected relationships
        """
        all_relationships = []
        entity_dict = {entity.id: entity for entity in entities}
        
        # Group entities by type for more efficient relationship detection
        entities_by_type = self._group_entities_by_type(entities)
        
        # Create batches of entity pairs based on relationship type definitions
        entity_batches = self._create_entity_batches_for_relationship_detection(entities_by_type)
        
        logger.info(f"Created {len(entity_batches)} batches for relationship detection")
        
        for batch_idx, batch in enumerate(entity_batches):
            logger.info(f"Processing relationship batch {batch_idx + 1}/{len(entity_batches)}")
            
            try:
                relationships = self.detect_relationships(batch)
                
                # Deduplicate relationships
                all_relationships = self._deduplicate_relationships(all_relationships, relationships)
                
                logger.info(f"Detected {len(relationships)} relationships in batch {batch_idx + 1}")
            except Exception as e:
                logger.error(f"Error detecting relationships in batch {batch_idx + 1}: {e}")
        
        return all_relationships
    
    def update_ontology_from_metadata(self, metadata: Dict[str, Any]) -> List[Triple]:
        """
        Update the ontology with new metadata, extracting entities and relationships.
        
        Args:
            metadata: New metadata to update the ontology with
            
        Returns:
            List of new triples added to the ontology
        """
        # Extract entities from the new metadata
        new_entities = self.extract_entities(metadata)
        
        # Add new entities to the ontology manager
        for entity in new_entities:
            self.ontology_manager.add_entity(entity)
        
        # Get all entities to detect relationships
        all_entities = list(self.ontology_manager.get_all_entities().values())
        
        # Detect relationships between new entities and existing entities
        entity_pairs = []
        for new_entity in new_entities:
            for existing_entity in all_entities:
                if new_entity.id != existing_entity.id:
                    entity_pairs.append([new_entity, existing_entity])
        
        # Detect relationships in batches
        all_relationships = []
        for i in range(0, len(entity_pairs), self.batch_size):
            batch = entity_pairs[i:i + self.batch_size]
            batch_entities = []
            for pair in batch:
                batch_entities.extend(pair)
            
            # Remove duplicates
            batch_entities = list({entity.id: entity for entity in batch_entities}.values())
            
            relationships = self.detect_relationships(batch_entities)
            all_relationships.extend(relationships)
        
        # Add new relationships to the ontology manager
        new_triples = []
        for relationship in all_relationships:
            # Check if relationship already exists
            if relationship.id not in self.ontology_manager.get_all_relationships():
                self.ontology_manager.add_relationship(relationship)
                
                # Create and add triple
                triple = Triple(
                    subject=relationship.source_id,
                    predicate=relationship.type.value,
                    object=relationship.target_id,
                    metadata=relationship.properties
                )
                
                self.ontology_manager.add_triple(triple)
                new_triples.append(triple)
        
        return new_triples
    
    def _build_triples_from_relationships(self, relationships: List[Relationship]) -> List[Triple]:
        """
        Build triples from relationships.
        
        Args:
            relationships: List of relationships to build triples from
            
        Returns:
            List of triples
        """
        triples = []
        
        for relationship in relationships:
            triple = Triple(
                subject=relationship.source_id,
                predicate=relationship.type.value,
                object=relationship.target_id,
                metadata=relationship.properties
            )
            triples.append(triple)
        
        return triples
    
    def _group_entities_by_type(self, entities: List[Entity]) -> Dict[EntityType, List[Entity]]:
        """
        Group entities by type.
        
        Args:
            entities: List of entities to group
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        entities_by_type = {}
        
        for entity in entities:
            if entity.type not in entities_by_type:
                entities_by_type[entity.type] = []
            entities_by_type[entity.type].append(entity)
        
        return entities_by_type
    
    def _create_entity_batches_for_relationship_detection(
        self, entities_by_type: Dict[EntityType, List[Entity]]
    ) -> List[List[Entity]]:
        """
        Create batches of entities for relationship detection based on the schema.
        
        This tries to group entities that are likely to have relationships with each other
        based on the relationship type definitions in the schema.
        
        Args:
            entities_by_type: Dictionary mapping entity types to lists of entities
            
        Returns:
            List of entity batches
        """
        batches = []
        processed_pairs = set()
        
        # Get allowed relationships for each entity type
        for source_type, source_entities in entities_by_type.items():
            allowed_relationships = self.schema_validator.schema.get_allowed_relationships(source_type)
            
            for rel_type, target_types in allowed_relationships.items():
                for target_type in target_types:
                    # Skip if no entities of target type
                    if target_type not in entities_by_type:
                        continue
                    
                    target_entities = entities_by_type[target_type]
                    
                    # Create a unique key for this source-target type pair
                    pair_key = f"{source_type.value}-{rel_type.value}-{target_type.value}"
                    if pair_key in processed_pairs:
                        continue
                    processed_pairs.add(pair_key)
                    
                    # Create batches for this source-target type pair
                    for i in range(0, len(source_entities), self.batch_size // 2):
                        for j in range(0, len(target_entities), self.batch_size // 2):
                            source_batch = source_entities[i:i + self.batch_size // 2]
                            target_batch = target_entities[j:j + self.batch_size // 2]
                            
                            # Combine and deduplicate
                            batch = source_batch + target_batch
                            batch = list({entity.id: entity for entity in batch}.values())
                            
                            if len(batch) > 0:
                                batches.append(batch)
        
        # If no batches were created, create batches with mixed entity types
        if not batches:
            all_entities = [entity for entities in entities_by_type.values() for entity in entities]
            for i in range(0, len(all_entities), self.batch_size):
                batch = all_entities[i:i + self.batch_size]
                if batch:
                    batches.append(batch)
        
        return batches
    
    def _call_llm_with_retry(
        self, system_prompt: str, user_prompt: str, expected_format: str = "json"
    ) -> Any:
        """
        Call LLM with retry logic.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            expected_format: Expected response format
            
        Returns:
            Parsed response from the LLM
            
        Raises:
            Exception: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    expected_format=expected_format
                )
                
                if expected_format == "json":
                    # Ensure the response is valid JSON
                    if isinstance(response, str):
                        try:
                            response = json.loads(response)
                        except json.JSONDecodeError:
                            # If the response contains a JSON array, try to extract it
                            import re
                            json_match = re.search(r'\[.*\]', response, re.DOTALL)
                            if json_match:
                                response = json.loads(json_match.group())
                            else:
                                raise ValueError("Invalid JSON response")
                
                return response
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        raise Exception(f"All LLM call attempts failed: {last_error}")
    
    def _deduplicate_relationships(
        self, existing_relationships: List[Relationship], new_relationships: List[Relationship]
    ) -> List[Relationship]:
        """
        Deduplicate relationships by removing duplicates from the new relationships.
        
        Args:
            existing_relationships: List of existing relationships
            new_relationships: List of new relationships to deduplicate
            
        Returns:
            List of deduplicated relationships
        """
        # Create a set of existing relationship keys
        existing_keys = {
            (rel.type, rel.source_id, rel.target_id)
            for rel in existing_relationships
        }
        
        # Filter out duplicates from new relationships
        deduplicated = []
        for rel in new_relationships:
            key = (rel.type, rel.source_id, rel.target_id)
            if key not in existing_keys:
                deduplicated.append(rel)
                existing_keys.add(key)
        
        return existing_relationships + deduplicated 