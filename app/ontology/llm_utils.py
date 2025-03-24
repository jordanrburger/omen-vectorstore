"""
LLM-powered utilities for ontology operations.
"""

from typing import List, Dict, Optional, Tuple
import openai
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from .models import Entity, Relationship
from .rdf_store import RDFStore
from .schema import EntityType, RelationshipType

class SemanticMatcher:
    """Handles semantic matching of entities using embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the semantic matcher with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string."""
        return self.model.encode(text)
    
    def find_similar_entities(
        self,
        query: str,
        entities: List[Entity],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Tuple[Entity, float]]:
        """Find entities semantically similar to the query."""
        query_embedding = self.get_embedding(query)
        entity_texts = [f"{e.name} {e.type.value}" for e in entities]
        entity_embeddings = self.model.encode(entity_texts)
        
        # Calculate cosine similarity
        similarities = np.dot(entity_embeddings, query_embedding) / (
            np.linalg.norm(entity_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k matches above threshold
        matches = []
        for idx, similarity in enumerate(similarities):
            if similarity >= threshold:
                matches.append((entities[idx], float(similarity)))
        
        # Sort by similarity and return top k
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

class NLQueryConverter:
    """Converts natural language queries to SPARQL."""
    
    def __init__(self, rdf_store: RDFStore):
        """Initialize with RDF store for query context."""
        self.rdf_store = rdf_store
    
    def convert_to_sparql(self, query: str) -> str:
        """Convert natural language query to SPARQL."""
        prompt = f"""
        Convert the following natural language query to SPARQL.
        Available entity types: {[t.value for t in EntityType]}
        Available relationship types: {[t.value for t in RelationshipType]}
        
        Query: {query}
        
        Return only the SPARQL query without any explanation.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    
    def execute_nl_query(self, query: str) -> List[Dict]:
        """Execute a natural language query by converting to SPARQL first."""
        sparql_query = self.convert_to_sparql(query)
        return self.rdf_store.query(sparql_query)

class OntologyExplainer:
    """Generates natural language explanations for ontology relationships and actions."""
    
    def __init__(self, rdf_store: RDFStore):
        """Initialize with RDF store for context."""
        self.rdf_store = rdf_store
    
    def explain_relationship(
        self,
        source: Entity,
        target: Entity,
        relationship: Relationship
    ) -> str:
        """Generate a natural language explanation of a relationship."""
        prompt = f"""
        Generate a clear, concise explanation of the relationship between these entities:
        
        Source: {source.name} ({source.type.value})
        Target: {target.name} ({target.type.value})
        Relationship: {relationship.type.value}
        
        Properties: {relationship.properties}
        
        Return only the explanation without any additional text.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    def explain_action_chain(
        self,
        source: Entity,
        target: Entity,
        actions: List[Dict]
    ) -> str:
        """Generate a natural language explanation of an action chain."""
        prompt = f"""
        Generate a clear, concise explanation of the action chain between these entities:
        
        Source: {source.name} ({source.type.value})
        Target: {target.name} ({target.type.value})
        
        Actions:
        {[f"- {a['type']}: {a['description']}" for a in actions]}
        
        Return only the explanation without any additional text.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()

class OntologyValidator:
    """Validates and repairs ontology inconsistencies."""
    
    def __init__(self, rdf_store: RDFStore):
        """Initialize with RDF store for validation."""
        self.rdf_store = rdf_store
    
    def validate_entity(self, entity: Entity) -> List[str]:
        """Validate an entity's properties and relationships."""
        issues = []
        
        # Check required properties based on entity type
        required_props = self._get_required_properties(entity.type)
        for prop in required_props:
            if prop not in entity.properties:
                issues.append(f"Missing required property: {prop}")
        
        # Validate property types
        for prop, value in entity.properties.items():
            if not self._validate_property_type(prop, value, entity.type):
                issues.append(f"Invalid type for property {prop}: {type(value)}")
        
        return issues
    
    def validate_relationship(
        self,
        relationship: Relationship,
        source: Entity,
        target: Entity
    ) -> List[str]:
        """Validate a relationship between entities."""
        issues = []
        
        # Check if relationship type is valid for entity types
        if not self._is_valid_relationship_type(
            relationship.type,
            source.type,
            target.type
        ):
            issues.append(
                f"Invalid relationship type {relationship.type.value} "
                f"between {source.type.value} and {target.type.value}"
            )
        
        # Validate relationship properties
        required_props = self._get_required_relationship_properties(relationship.type)
        for prop in required_props:
            if prop not in relationship.properties:
                issues.append(f"Missing required property: {prop}")
        
        return issues
    
    def _get_required_properties(self, entity_type: EntityType) -> List[str]:
        """Get required properties for an entity type."""
        # This would be implemented based on the schema
        return []
    
    def _validate_property_type(
        self,
        property_name: str,
        value: any,
        entity_type: EntityType
    ) -> bool:
        """Validate the type of a property value."""
        # This would be implemented based on the schema
        return True
    
    def _is_valid_relationship_type(
        self,
        relationship_type: RelationshipType,
        source_type: EntityType,
        target_type: EntityType
    ) -> bool:
        """Check if a relationship type is valid between entity types."""
        # This would be implemented based on the schema
        return True
    
    def _get_required_relationship_properties(
        self,
        relationship_type: RelationshipType
    ) -> List[str]:
        """Get required properties for a relationship type."""
        # This would be implemented based on the schema
        return [] 