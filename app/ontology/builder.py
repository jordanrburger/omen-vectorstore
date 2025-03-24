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
from dataclasses import dataclass
from enum import Enum
import datetime

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
from app.llm_client import LLMClient, LLMError, LLMResponseError

logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    """Status of processing an item."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProcessingResult:
    """Result of processing an item."""
    status: ProcessingStatus
    item: Any
    result: Optional[Any] = None
    error: Optional[Exception] = None

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
        
        # Processing state
        self._processing_results: Dict[str, ProcessingResult] = {}
        self._failed_items: List[ProcessingResult] = []
    
    def _process_batch_with_retry(
        self,
        batch: List[Dict[str, Any]],
        process_fn: callable,
        batch_id: str
    ) -> List[ProcessingResult]:
        """
        Process a batch of items with retries.
        
        Args:
            batch: List of items to process
            process_fn: Function to process each item
            batch_id: Unique identifier for the batch
            
        Returns:
            List of processing results
        """
        results = []
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                batch_results = []
                for item in batch:
                    item_id = f"{batch_id}_{item.get('id', str(uuid.uuid4()))}"
                    self._processing_results[item_id] = ProcessingResult(
                        status=ProcessingStatus.PROCESSING,
                        item=item
                    )
                    
                    try:
                        result = process_fn(item)
                        self._processing_results[item_id].status = ProcessingStatus.COMPLETED
                        self._processing_results[item_id].result = result
                        batch_results.append(self._processing_results[item_id])
                    except Exception as e:
                        self._processing_results[item_id].status = ProcessingStatus.FAILED
                        self._processing_results[item_id].error = e
                        self._failed_items.append(self._processing_results[item_id])
                        logger.error(f"Error processing item {item_id}: {e}")
                
                results.extend(batch_results)
                return results
                
            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error(f"Batch {batch_id} failed after {self.max_retries} retries: {e}")
                    # Mark all items in batch as failed
                    for item in batch:
                        item_id = f"{batch_id}_{item.get('id', str(uuid.uuid4()))}"
                        self._processing_results[item_id].status = ProcessingStatus.FAILED
                        self._processing_results[item_id].error = e
                        self._failed_items.append(self._processing_results[item_id])
                    return results
                
                logger.warning(f"Batch {batch_id} failed (attempt {retry_count}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay * (2 ** retry_count))  # Exponential backoff
        
        return results
    
    def _process_batches_parallel(
        self,
        items: List[Dict[str, Any]],
        process_fn: callable
    ) -> List[ProcessingResult]:
        """
        Process items in parallel batches.
        
        Args:
            items: List of items to process
            process_fn: Function to process each item
            
        Returns:
            List of processing results
        """
        results = []
        batches = self._create_batches(items)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(
                    self._process_batch_with_retry,
                    batch,
                    process_fn,
                    f"batch_{i}"
                ): batch
                for i, batch in enumerate(batches)
            }
            
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    # Mark all items in batch as failed
                    for item in batch:
                        item_id = f"batch_{items.index(item)}_{item.get('id', str(uuid.uuid4()))}"
                        self._processing_results[item_id].status = ProcessingStatus.FAILED
                        self._processing_results[item_id].error = e
                        self._failed_items.append(self._processing_results[item_id])
        
        return results
    
    def _create_batches(self, items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Create batches of items for processing.
        
        Args:
            items: List of items to batch
            
        Returns:
            List of batches
        """
        batches = []
        current_batch = []
        
        for item in items:
            current_batch.append(item)
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def extract_entities(self, metadata: Dict[str, Any]) -> List[Entity]:
        """
        Extract entities from metadata using an LLM.
        
        Args:
            metadata: Metadata to extract entities from
            
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
        
        try:
            # Call LLM with retries
            response = self._call_llm_with_retry(
                system_prompt=prompt["system_prompt"],
                user_prompt=prompt["user_prompt"],
                expected_format="json"
            )
            
            # Parse and validate entities
            entities = []
            
            # Handle different response types
            if isinstance(response, str):
                try:
                    json_response = json.loads(response)
                except:
                    logger.error(f"Failed to parse JSON from response: {response[:100]}...")
                    return entities
            else:
                json_response = response
                
            # Check if response contains an 'entities' key (common pattern from LLMs)
            if isinstance(json_response, dict) and 'entities' in json_response:
                entities_data = json_response['entities']
                if isinstance(entities_data, list):
                    json_response = entities_data
                else:
                    logger.warning(f"Expected 'entities' to be a list, got {type(entities_data)}")
                    # Try to continue with the original response
            
            # Ensure json_response is a list
            if isinstance(json_response, dict):
                json_response = [json_response]
            elif not isinstance(json_response, list):
                logger.error(f"Expected list or dict, got {type(json_response)}")
                return entities
            
            for entity_data in json_response:
                try:
                    # Skip if not a dict
                    if not isinstance(entity_data, dict):
                        logger.warning(f"Entity data is not a dictionary: {entity_data}")
                        continue
                    
                    # Get entity type
                    type_value = entity_data.get("type")
                    if not type_value:
                        logger.warning(f"Entity missing type: {entity_data}")
                        continue
                    
                    # Create entity type enum - handle case-insensitive matching
                    try:
                        # First try direct conversion
                        try:
                            entity_type = EntityType(type_value)
                        except ValueError:
                            # Try uppercase - the enum values are uppercase
                            try:
                                entity_type = EntityType(type_value.upper())
                            except ValueError:
                                # Try lowercase - sometimes the enum values are lowercase
                                entity_type = EntityType(type_value.lower())
                    except ValueError:
                        logger.warning(f"Invalid entity type: {type_value}")
                        continue
                    
                    # Get entity ID
                    entity_id = entity_data.get("id")
                    if not entity_id:
                        entity_id = str(uuid.uuid4())
                    
                    # Get properties
                    properties = entity_data.get("properties", {})
                    if not isinstance(properties, dict):
                        properties = {}
                    
                    # Ensure valid datetime format for date fields
                    date_fields = ['created_at', 'updated_at', 'last_updated']
                    for field in date_fields:
                        if field in properties:
                            # If empty string or invalid, set to current ISO datetime
                            if not properties[field] or properties[field] == '':
                                properties[field] = datetime.datetime.now().isoformat()
                            # Try parsing to validate - if invalid, set to current time
                            try:
                                datetime.datetime.fromisoformat(properties[field])
                            except (ValueError, TypeError):
                                properties[field] = datetime.datetime.now().isoformat()
                    
                    # Get entity name (required parameter)
                    entity_name = entity_data.get("name")
                    if not entity_name:
                        # Try to extract name from properties
                        entity_name = properties.get("name", "")
                        
                    # If still no name, use id or a default name
                    if not entity_name:
                        entity_name = f"{entity_type.value}_{entity_id}"
                    
                    # Create entity
                    entity = Entity(
                        id=entity_id,
                        type=entity_type,
                        name=entity_name,
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
            
        except LLMError as e:
            logger.error(f"LLM error during entity extraction: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during entity extraction: {e}")
            raise
    
    def detect_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """
        Detect relationships between entities using an LLM.
        
        Args:
            entities: List of entities to detect relationships between
            
        Returns:
            List of detected relationships
        """
        # Skip if no entities or only one entity (need at least two to form a relationship)
        if not entities or len(entities) < 2:
            logger.warning(f"Skipping relationship detection for {len(entities)} entities - need at least 2 entities")
            return []
            
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
                "name": entity.name,
                "properties": entity.properties
            }
            for entity in entities
        ]
        
        # Format prompt for relationship detection
        prompt = format_relationship_detection_prompt(
            entities=entity_dicts,
            schema_definitions=self._relationship_type_schema_prompt
        )
        
        try:
            # Log the entities being processed
            logger.info(f"Detecting relationships between {len(entities)} entities")
            entity_types = {entity.type.value: entity.type.value for entity in entities}
            logger.info(f"Entity types present: {list(entity_types.keys())}")
            
            # Call LLM with retries
            response = self._call_llm_with_retry(
                system_prompt=prompt["system_prompt"],
                user_prompt=prompt["user_prompt"],
                expected_format="json"
            )
            
            # Parse and validate relationships
            relationships = []
            entities_dict = {entity.id: entity for entity in entities}
            
            # Log the raw response for debugging
            logger.info(f"Raw LLM response: {response}")
            
            # Handle different response types
            if isinstance(response, str):
                try:
                    json_response = json.loads(response)
                except:
                    logger.error(f"Failed to parse JSON from response: {response[:100]}...")
                    return relationships
            else:
                json_response = response
            
            # Check if response contains a 'relationships' key (common pattern from LLMs)
            if isinstance(json_response, dict) and 'relationships' in json_response:
                relationships_data = json_response['relationships']
                if isinstance(relationships_data, list):
                    json_response = relationships_data
                else:
                    logger.warning(f"Expected 'relationships' to be a list, got {type(relationships_data)}")
                    # Try to continue with the original response
            
            # Handle error messages in the response
            if isinstance(json_response, dict) and any(key in json_response for key in ['error', 'errors', 'No relationships found']):
                logger.warning(f"Response indicates no relationships or an error: {json_response}")
                return relationships
                
            # Ensure json_response is a list
            if isinstance(json_response, dict):
                json_response = [json_response]
            elif not isinstance(json_response, list):
                logger.error(f"Expected list or dict, got {type(json_response)}")
                return relationships
            
            # Skip empty responses
            if not json_response or (len(json_response) == 1 and not json_response[0]):
                logger.warning("Empty relationship response from LLM")
                return relationships
                
            for rel_data in json_response:
                try:
                    # Skip if not a dict
                    if not isinstance(rel_data, dict) or not rel_data:
                        logger.warning(f"Relationship data is not a valid dictionary: {rel_data}")
                        continue
                    
                    # Get relationship type
                    type_value = rel_data.get("type")
                    if not type_value:
                        logger.warning(f"Relationship missing type: {rel_data}")
                        continue
                    
                    # Create relationship type enum - case insensitive matching
                    try:
                        # Try direct conversion
                        try:
                            rel_type = RelationshipType(type_value)
                        except ValueError:
                            # Try converting to uppercase (enum values are uppercase)
                            try:
                                rel_type = RelationshipType(type_value.upper())
                            except ValueError:
                                # Try converting to lowercase and stripping spaces
                                rel_type = RelationshipType(type_value.lower().replace(" ", ""))
                    except ValueError:
                        logger.warning(f"Invalid relationship type: '{type_value}'. Valid types are: {[t.value for t in RelationshipType]}. Full data: {rel_data}")
                        continue
                    
                    # Get relationship ID
                    rel_id = rel_data.get("id")
                    if not rel_id:
                        rel_id = str(uuid.uuid4())
                    
                    # Get source and target entities
                    source_id = rel_data.get("source_id")
                    target_id = rel_data.get("target_id")
                    if not source_id or not target_id:
                        logger.warning(f"Relationship missing source or target: {rel_data}")
                        continue
                    
                    # Skip if source or target entity doesn't exist
                    if source_id not in entities_dict:
                        logger.warning(f"Relationship {rel_id} references nonexistent source entity: {source_id}. Available entity IDs: {list(entities_dict.keys())[:5]}...")
                        continue
                        
                    if target_id not in entities_dict:
                        logger.warning(f"Relationship {rel_id} references nonexistent target entity: {target_id}. Available entity IDs: {list(entities_dict.keys())[:5]}...")
                        continue
                    
                    # Get properties
                    properties = rel_data.get("properties", {})
                    if not isinstance(properties, dict):
                        properties = {}
                    
                    # Ensure valid datetime format for date fields
                    date_fields = ['created_at', 'updated_at', 'last_updated']
                    for field in date_fields:
                        if field in properties:
                            # If empty string or invalid, set to current ISO datetime
                            if not properties[field] or properties[field] == '':
                                properties[field] = datetime.datetime.now().isoformat()
                            # Try parsing to validate - if invalid, set to current time
                            try:
                                datetime.datetime.fromisoformat(properties[field])
                            except (ValueError, TypeError):
                                properties[field] = datetime.datetime.now().isoformat()
                    
                    # Create relationship
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
                        source_type = entities_dict[source_id].type if source_id in entities_dict else "unknown"
                        target_type = entities_dict[target_id].type if target_id in entities_dict else "unknown"
                        logger.warning(f"Invalid relationship: {rel_type.value} from {source_type} to {target_type}. Errors: {errors}. Raw data: {rel_data}")
                        continue
                    
                    # Log valid relationship
                    logger.info(f"Created valid relationship: {rel_id}, Type: {rel_type.value}, Source: {source_id}, Target: {target_id}")
                    relationships.append(relationship)
                except Exception as e:
                    logger.error(f"Error creating relationship from LLM response: {e}, data: {rel_data}")
                    continue
            
            if not relationships and json_response:
                logger.warning(f"No valid relationships were found from non-empty response. Raw response: {json_response}")
            else:
                logger.info(f"Detected {len(relationships)} valid relationships")
            
            return relationships
            
        except LLMError as e:
            logger.error(f"LLM error during relationship detection: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during relationship detection: {e}")
            raise
    
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
            LLMError: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Add explicit JSON formatting instruction to user prompt for better results
                enhanced_user_prompt = user_prompt
                if expected_format == "json":
                    enhanced_user_prompt = user_prompt + "\n\nPlease respond with a valid JSON array or object only, without any explanation or additional text."
                
                # Log prompt details for debugging
                logger.debug(f"LLM Call - System prompt first 100 chars: {system_prompt[:100]}...")
                logger.debug(f"LLM Call - User prompt first 100 chars: {enhanced_user_prompt[:100]}...")
                
                response = self.llm_client.generate(
                    prompt=enhanced_user_prompt,
                    system_prompt=system_prompt,
                    expected_format=expected_format
                )
                
                # Log raw response for debugging
                if isinstance(response, str):
                    logger.debug(f"LLM raw response first 100 chars: {response[:100]}...")
                else:
                    logger.debug(f"LLM parsed response type: {type(response)}")
                
                if expected_format == "json":
                    # Parse JSON response
                    if isinstance(response, str):
                        try:
                            # First try parsing the entire response as JSON
                            parsed_response = json.loads(response)
                            return parsed_response
                        except json.JSONDecodeError:
                            # Try extracting JSON using regex patterns
                            import re
                            
                            # Look for JSON arrays
                            json_array_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
                            if json_array_match:
                                try:
                                    array_text = json_array_match.group(0)
                                    parsed_response = json.loads(array_text)
                                    return parsed_response
                                except json.JSONDecodeError:
                                    pass
                            
                            # Look for JSON objects
                            json_obj_match = re.search(r'\{(.*?)\}', response, re.DOTALL)
                            if json_obj_match:
                                try:
                                    obj_text = json_obj_match.group(0)
                                    parsed_response = json.loads(obj_text)
                                    return parsed_response
                                except json.JSONDecodeError:
                                    pass
                            
                            # Look for code blocks that might contain JSON
                            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response, re.DOTALL)
                            if code_block_match:
                                try:
                                    code_text = code_block_match.group(1).strip()
                                    parsed_response = json.loads(code_text)
                                    return parsed_response
                                except json.JSONDecodeError:
                                    pass
                            
                            # If we can't parse JSON, create a simple array with the content
                            logger.warning(f"Failed to parse JSON from response: {response[:100]}...")
                            return [{"raw_response": response}]
                    else:
                        # Response is already parsed as a Python object
                        return response
                else:
                    # Return text response as is
                    return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        raise LLMError(f"All LLM call attempts failed: {last_error}")
    
    def build_ontology(self, metadata_list: List[Dict[str, Any]]) -> None:
        """
        Build an ontology from a list of metadata items.
        
        Args:
            metadata_list: List of metadata items to build ontology from
        """
        logger.info(f"Building ontology from {len(metadata_list)} metadata items")
        
        # Process metadata items in parallel batches
        results = self._process_batches_parallel(
            metadata_list,
            self._process_metadata_item
        )
        
        # Log processing results
        completed = [r for r in results if r.status == ProcessingStatus.COMPLETED]
        failed = [r for r in results if r.status == ProcessingStatus.FAILED]
        
        logger.info(f"Completed processing {len(completed)} items")
        if failed:
            logger.warning(f"Failed to process {len(failed)} items")
            for result in failed:
                logger.error(f"Failed item: {result.item}, Error: {result.error}")
    
    def _process_metadata_item(self, metadata: Dict[str, Any]) -> None:
        """
        Process a single metadata item.
        
        Args:
            metadata: Metadata item to process
        """
        try:
            # Extract entities
            logger.info(f"Extracting entities from metadata type: {metadata.get('type', 'unknown')}")
            entities = self.extract_entities(metadata)
            
            # Add entities to ontology manager
            for entity in entities:
                self.ontology_manager.add_entity(entity)
                
            # Log entity extraction results
            logger.info(f"Extracted {len(entities)} entities from metadata item")
            
            # Process relationships in smaller batches to avoid overwhelming the LLM
            if len(entities) > 5:
                logger.info(f"Processing relationships in batches for {len(entities)} entities")
                
                # Group entities by type for more relevant relationship detection
                entity_groups = {}
                for entity in entities:
                    entity_type = entity.type.value
                    if entity_type not in entity_groups:
                        entity_groups[entity_type] = []
                    entity_groups[entity_type].append(entity)
                
                # Process relationships between entities of the same type first
                for entity_type, group in entity_groups.items():
                    if len(group) >= 2:
                        batch_size = min(10, len(group))
                        for i in range(0, len(group), batch_size):
                            batch = group[i:i+batch_size]
                            logger.info(f"Processing relationship batch of {len(batch)} entities of type {entity_type}")
                            relationships = self.detect_relationships(batch)
                            
                            # Add relationships to ontology manager
                            for relationship in relationships:
                                self.ontology_manager.add_relationship(relationship)
                
                # Process relationships between different types of entities
                # Create key pairs for related entity types
                related_type_pairs = [
                    (EntityType.BUCKET, EntityType.TABLE),
                    (EntityType.TABLE, EntityType.COLUMN),
                    (EntityType.COMPONENT, EntityType.CONFIGURATION),
                    (EntityType.TRANSFORMATION, EntityType.BLOCK),
                    (EntityType.ORCHESTRATION, EntityType.TASK)
                ]
                
                for type1, type2 in related_type_pairs:
                    group1 = entity_groups.get(type1.value, [])
                    group2 = entity_groups.get(type2.value, [])
                    
                    if group1 and group2:
                        # Process in mini-batches to avoid overwhelming the LLM
                        batch_size1 = min(5, len(group1))
                        batch_size2 = min(5, len(group2))
                        
                        for i in range(0, len(group1), batch_size1):
                            batch1 = group1[i:i+batch_size1]
                            
                            for j in range(0, len(group2), batch_size2):
                                batch2 = group2[j:j+batch_size2]
                                combined_batch = batch1 + batch2
                                
                                logger.info(f"Processing relationship batch between {len(batch1)} {type1.value} entities and {len(batch2)} {type2.value} entities")
                                relationships = self.detect_relationships(combined_batch)
                                
                                # Add relationships to ontology manager
                                for relationship in relationships:
                                    self.ontology_manager.add_relationship(relationship)
            else:
                # For small entity sets, process all at once
                logger.info(f"Processing relationships for {len(entities)} entities at once")
                relationships = self.detect_relationships(entities)
                
                # Add relationships to ontology manager
                for relationship in relationships:
                    self.ontology_manager.add_relationship(relationship)
                
        except Exception as e:
            logger.error(f"Error processing metadata item: {e}")
            raise 