"""
Prompt templates for LLM-based ontology construction.

This module contains templates for entity extraction, relationship detection,
and other NLP tasks used in the ontology builder.
"""

import json
from typing import Dict, List, Any, Optional

# Entity extraction prompts

ENTITY_EXTRACTION_SYSTEM_PROMPT = """
You are an ontology extraction expert specialized in analyzing Keboola metadata.
Your task is to identify entities from the provided metadata and extract their properties
according to the Keboola ontology schema.

Follow these guidelines:
1. Identify all entities present in the metadata
2. Extract properties for each entity based on the schema definition
3. Ensure all required properties are included
4. Format the output as a valid JSON object
5. Be precise in your extraction, avoiding hallucination or inference not supported by data
"""

ENTITY_EXTRACTION_PROMPT_TEMPLATE = """
# Metadata
```json
{metadata}
```

# Entity Types and Required Properties
{entity_type_definitions}

# Instructions
Extract all entities from the provided metadata according to the entity types defined above.
For each entity:
1. Determine its type based on the entity type definitions 
2. Extract all available properties, ensuring required properties are included
3. Generate a unique ID for each entity if not present in the metadata

IMPORTANT: Entity types MUST be one of the exact values from this list (case-sensitive): 
- table
- column
- bucket
- configuration 
- transformation
- block
- orchestration
- task
- component
- project
- generic

Output the entities as a JSON array, where each object has:
- "id": A unique identifier for the entity
- "type": The entity type (must match one of the defined types EXACTLY as listed above)
- "properties": An object containing all the extracted properties

Only include properties that are explicitly present in the metadata or can be directly inferred.
Do not include properties that are speculative or not supported by the provided metadata.
"""

# Relationship detection prompts

RELATIONSHIP_DETECTION_SYSTEM_PROMPT = """
You are an ontology relationship expert specialized in analyzing connections between Keboola entities.
Your task is to identify relationships between the provided entities based on the Keboola ontology schema.

Follow these guidelines:
1. Analyze the entities and identify all possible relationships between them
2. Only identify relationships that are explicitly supported by the entity data
3. Ensure relationships follow the allowed relationship types defined in the schema
4. Format the output as a valid JSON object with a 'relationships' array
5. Be precise, avoiding hallucination or inference not supported by the entity data
6. Always output JSON in the exact format requested - never respond with explanations or free text

Remember: The output MUST be valid JSON with relationships that match the specified schema exactly.
"""

RELATIONSHIP_DETECTION_PROMPT_TEMPLATE = """
# Entities
```json
{entities}
```

# Relationship Types and Rules
{relationship_type_definitions}

# Instructions
Identify all relationships between the provided entities according to the relationship types defined above.
For each relationship:
1. Determine its type based on the relationship type definitions
2. Identify source and target entities based on their properties and IDs
3. Ensure the relationship adheres to the cardinality and type constraints
4. Add relevant properties to the relationship

IMPORTANT: Relationship types MUST be one of the exact values from this list (case-sensitive): 
- hasColumn
- belongsTo
- dependsOn
- inputsFrom
- outputsTo
- partOf
- linkedTo
- createdBy
- triggers
- contains
- generic

When creating relationships, strictly follow these rules:
1. Ensure that the source and target entity types are valid for the chosen relationship type
2. For CONTAINS relationships:
   - Only PROJECT can contain BUCKET
   - Only BUCKET can contain TABLE
   - Only COMPONENT can contain CONFIGURATION
   - Only TRANSFORMATION can contain BLOCK
   - Only ORCHESTRATION can contain TASK
3. For BELONGS_TO relationships:
   - Only COLUMN can belong to TABLE
   - Only CONFIGURATION can belong to COMPONENT
4. For PART_OF relationships:
   - Only BLOCK can be part of TRANSFORMATION
   - Only TASK can be part of ORCHESTRATION

Required Output Format:
```json
{
  "relationships": [
    {
      "id": "unique_id_1",
      "type": "hasColumn",
      "source_id": "source_entity_id",
      "target_id": "target_entity_id",
      "properties": {}
    },
    {
      "id": "unique_id_2",
      "type": "belongsTo",
      "source_id": "source_entity_id",
      "target_id": "target_entity_id",
      "properties": {}
    }
  ]
}
```

If no relationships can be identified, return an empty array:
```json
{
  "relationships": []
}
```

Only include relationships that are clearly indicated by the entity data.
Do not include relationships that are speculative or cannot be determined with high confidence.
Respond ONLY with the JSON object - do not include any explanations or additional text.
"""

# Triple validation prompts

TRIPLE_VALIDATION_SYSTEM_PROMPT = """
You are an ontology validation expert specialized in verifying RDF triples in the Keboola ontology.
Your task is to validate the provided triples against the Keboola ontology schema and identify any inconsistencies.

Follow these guidelines:
1. Verify that triples adhere to the schema definitions
2. Check for logical inconsistencies or contradictions
3. Identify missing or incomplete information
4. Format the output as a valid JSON object
5. Be thorough and precise in your validation
"""

TRIPLE_VALIDATION_PROMPT_TEMPLATE = """
# Triples
```json
{triples}
```

# Schema Definition
{schema_definition}

# Instructions
Validate all the provided triples against the schema definition.
For each triple:
1. Check if the subject entity exists and has a valid type
2. Verify the predicate is a valid relationship type for the subject
3. Ensure the object is valid according to the relationship definition
4. Check for any logical inconsistencies or contradictions

Output the validation results as a JSON object with:
- "valid_triples": Array of IDs for valid triples
- "invalid_triples": Array of objects for invalid triples, each with:
  - "id": The triple ID
  - "errors": Array of error messages
  - "suggestions": Array of suggested fixes (if applicable)

Be thorough in your validation and provide specific, actionable feedback for any invalid triples.
"""


def format_entity_extraction_prompt(metadata: Dict[str, Any], schema_definitions: str) -> Dict[str, str]:
    """
    Format the entity extraction prompt with the provided metadata and schema definitions.
    
    Args:
        metadata: The metadata to extract entities from
        schema_definitions: String describing the entity type definitions from the schema
        
    Returns:
        Dictionary with system_prompt and user_prompt
    """
    return {
        "system_prompt": ENTITY_EXTRACTION_SYSTEM_PROMPT,
        "user_prompt": ENTITY_EXTRACTION_PROMPT_TEMPLATE.format(
            metadata=json.dumps(metadata, indent=2),
            entity_type_definitions=schema_definitions
        )
    }


def format_relationship_detection_prompt(entities: List[Dict[str, Any]], schema_definitions: str) -> Dict[str, str]:
    """
    Format the relationship detection prompt with the provided entities and schema definitions.
    
    Args:
        entities: List of entities to detect relationships between
        schema_definitions: String describing the relationship type definitions from the schema
        
    Returns:
        Dictionary with system_prompt and user_prompt
    """
    return {
        "system_prompt": RELATIONSHIP_DETECTION_SYSTEM_PROMPT,
        "user_prompt": RELATIONSHIP_DETECTION_PROMPT_TEMPLATE.format(
            entities=json.dumps(entities, indent=2),
            relationship_type_definitions=schema_definitions
        )
    }


def format_triple_validation_prompt(triples: List[Dict[str, Any]], schema_definition: str) -> Dict[str, str]:
    """
    Format the triple validation prompt with the provided triples and schema definition.
    
    Args:
        triples: List of triples to validate
        schema_definition: String describing the schema definition
        
    Returns:
        Dictionary with system_prompt and user_prompt
    """
    return {
        "system_prompt": TRIPLE_VALIDATION_SYSTEM_PROMPT,
        "user_prompt": TRIPLE_VALIDATION_PROMPT_TEMPLATE.format(
            triples=json.dumps(triples, indent=2),
            schema_definition=schema_definition
        )
    }


def get_entity_type_schema_prompt(entity_types: Dict, include_properties: bool = True) -> str:
    """
    Generate a human-readable representation of entity type definitions for prompts.
    
    Args:
        entity_types: Dictionary of entity type definitions from the schema
        include_properties: Whether to include property definitions
        
    Returns:
        Formatted string describing the entity types
    """
    result = []
    
    for entity_type, definition in entity_types.items():
        type_description = f"## {definition.name} ({entity_type.value})\n"
        type_description += f"Description: {definition.description}\n"
        type_description += f"Required properties: {', '.join(definition.required_properties)}\n"
        
        if include_properties and definition.properties:
            type_description += "Properties:\n"
            for prop_name, prop_def in definition.properties.items():
                type_description += f"- {prop_name} ({prop_def.data_type}): {prop_def.description}"
                if prop_def.required:
                    type_description += " [Required]"
                type_description += "\n"
        
        result.append(type_description)
    
    return "\n".join(result)


def get_relationship_type_schema_prompt(relationship_types: Dict) -> str:
    """
    Generate a human-readable representation of relationship type definitions for prompts.
    
    Args:
        relationship_types: Dictionary of relationship type definitions from the schema
        
    Returns:
        Formatted string describing the relationship types
    """
    result = []
    
    for rel_type, definition in relationship_types.items():
        type_description = f"## {definition.name} ({rel_type.value})\n"
        type_description += f"Description: {definition.description}\n"
        
        source_types = [str(t.value) for t in definition.source_types]
        target_types = [str(t.value) for t in definition.target_types]
        
        type_description += f"Source types: {', '.join(source_types)}\n"
        type_description += f"Target types: {', '.join(target_types)}\n"
        type_description += f"Cardinality: {definition.cardinality}\n"
        
        if definition.required_properties:
            type_description += f"Required properties: {', '.join(definition.required_properties)}\n"
        
        result.append(type_description)
    
    return "\n".join(result) 