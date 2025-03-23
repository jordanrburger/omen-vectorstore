"""Tests for the ontology schema validation functionality."""

import unittest
from enum import Enum
from datetime import datetime, date
import uuid

from app.ontology.schema import PropertyDefinition, EntityTypeDefinition, RelationshipTypeDefinition, SchemaValidator
from app.ontology.models import Entity, Relationship, EntityType, RelationshipType


class TestPropertyDefinition(unittest.TestCase):
    """Test cases for the PropertyDefinition class."""
    
    def test_string_validation(self):
        """Test validation of string properties."""
        prop_def = PropertyDefinition(
            name="test_string",
            description="A test string property",
            data_type="string",
            required=True,
            pattern=r"^[A-Z][a-z]+$"
        )
        
        # Valid string
        self.assertTrue(prop_def.validate("Test"))
        
        # Invalid pattern
        self.assertFalse(prop_def.validate("test"))
        self.assertFalse(prop_def.validate("TEST"))
        
        # Wrong type
        self.assertFalse(prop_def.validate(123))
        self.assertFalse(prop_def.validate(True))
    
    def test_integer_validation(self):
        """Test validation of integer properties."""
        prop_def = PropertyDefinition(
            name="test_integer",
            description="A test integer property",
            data_type="integer",
            required=True,
            min_value=0,
            max_value=100
        )
        
        # Valid integer
        self.assertTrue(prop_def.validate(50))
        
        # Out of range
        self.assertFalse(prop_def.validate(-10))
        self.assertFalse(prop_def.validate(200))
        
        # Wrong type
        self.assertFalse(prop_def.validate("50"))
        self.assertFalse(prop_def.validate(50.5))
    
    def test_float_validation(self):
        """Test validation of float properties."""
        prop_def = PropertyDefinition(
            name="test_float",
            description="A test float property",
            data_type="float",
            required=True,
            min_value=0.0,
            max_value=1.0
        )
        
        # Valid float
        self.assertTrue(prop_def.validate(0.5))
        
        # Integer should be converted to float
        self.assertTrue(prop_def.validate(1))
        
        # Out of range
        self.assertFalse(prop_def.validate(-0.5))
        self.assertFalse(prop_def.validate(1.5))
        
        # Wrong type
        self.assertFalse(prop_def.validate("0.5"))
    
    def test_boolean_validation(self):
        """Test validation of boolean properties."""
        prop_def = PropertyDefinition(
            name="test_boolean",
            description="A test boolean property",
            data_type="boolean",
            required=True
        )
        
        # Valid boolean
        self.assertTrue(prop_def.validate(True))
        self.assertTrue(prop_def.validate(False))
        
        # Wrong type
        self.assertFalse(prop_def.validate("True"))
        self.assertFalse(prop_def.validate(1))
    
    def test_date_validation(self):
        """Test validation of date properties."""
        prop_def = PropertyDefinition(
            name="test_date",
            description="A test date property",
            data_type="date",
            required=True
        )
        
        # Valid date
        self.assertTrue(prop_def.validate(date(2023, 1, 1)))
        
        # String date should be converted
        self.assertTrue(prop_def.validate("2023-01-01"))
        
        # Wrong type
        self.assertFalse(prop_def.validate(2023))
        self.assertFalse(prop_def.validate("invalid date"))
    
    def test_datetime_validation(self):
        """Test validation of datetime properties."""
        prop_def = PropertyDefinition(
            name="test_datetime",
            description="A test datetime property",
            data_type="datetime",
            required=True
        )
        
        # Valid datetime
        self.assertTrue(prop_def.validate(datetime(2023, 1, 1, 12, 0, 0)))
        
        # String datetime should be converted
        self.assertTrue(prop_def.validate("2023-01-01T12:00:00"))
        
        # Wrong type
        self.assertFalse(prop_def.validate(2023))
        self.assertFalse(prop_def.validate("invalid datetime"))
    
    def test_required_property(self):
        """Test validation of required properties."""
        prop_def = PropertyDefinition(
            name="test_required",
            description="A test required property",
            data_type="string",
            required=True
        )
        
        # None value for required property
        self.assertFalse(prop_def.validate(None))
        
        # Empty string for required property
        self.assertFalse(prop_def.validate(""))
    
    def test_optional_property(self):
        """Test validation of optional properties."""
        prop_def = PropertyDefinition(
            name="test_optional",
            description="A test optional property",
            data_type="string",
            required=False
        )
        
        # None value for optional property
        self.assertTrue(prop_def.validate(None))
        
        # Empty string for optional property (still a valid string)
        self.assertTrue(prop_def.validate(""))
    
    def test_default_value(self):
        """Test default value for properties."""
        prop_def = PropertyDefinition(
            name="test_default",
            description="A test property with default value",
            data_type="string",
            required=True,
            default="default value"
        )
        
        # Default value should be used when None is provided
        self.assertEqual(prop_def.get_value(None), "default value")
        
        # Provided value should override default
        self.assertEqual(prop_def.get_value("provided value"), "provided value")


class TestEntityTypeDefinition(unittest.TestCase):
    """Test cases for the EntityTypeDefinition class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create property definitions
        self.name_prop = PropertyDefinition(
            name="name",
            description="Entity name",
            data_type="string",
            required=True
        )
        
        self.description_prop = PropertyDefinition(
            name="description",
            description="Entity description",
            data_type="string",
            required=False
        )
        
        self.count_prop = PropertyDefinition(
            name="count",
            description="Entity count",
            data_type="integer",
            required=True,
            min_value=0
        )
        
        # Create entity type definition
        self.entity_type_def = EntityTypeDefinition(
            name="TestEntity",
            description="A test entity type",
            properties={
                "name": self.name_prop,
                "description": self.description_prop,
                "count": self.count_prop
            },
            required_properties=["name", "count"]
        )
    
    def test_validate_entity(self):
        """Test validation of an entity against an entity type definition."""
        # Create a valid entity
        valid_entity = Entity(
            id=str(uuid.uuid4()),
            type=EntityType.TABLE,  # Type doesn't matter for this test
            properties={
                "name": "Test Entity",
                "count": 10,
                "description": "This is a test entity"
            }
        )
        
        # Create an entity missing a required property
        missing_required_entity = Entity(
            id=str(uuid.uuid4()),
            type=EntityType.TABLE,
            properties={
                "name": "Test Entity",
                # Missing count
                "description": "This is a test entity"
            }
        )
        
        # Create an entity with an invalid property value
        invalid_property_entity = Entity(
            id=str(uuid.uuid4()),
            type=EntityType.TABLE,
            properties={
                "name": "Test Entity",
                "count": -10,  # Invalid count (negative)
                "description": "This is a test entity"
            }
        )
        
        # Validate the entities
        valid_result, valid_errors = self.entity_type_def.validate_entity(valid_entity)
        missing_result, missing_errors = self.entity_type_def.validate_entity(missing_required_entity)
        invalid_result, invalid_errors = self.entity_type_def.validate_entity(invalid_property_entity)
        
        # Check results
        self.assertTrue(valid_result)
        self.assertEqual(len(valid_errors), 0)
        
        self.assertFalse(missing_result)
        self.assertEqual(len(missing_errors), 1)
        self.assertIn("count", missing_errors[0])
        
        self.assertFalse(invalid_result)
        self.assertEqual(len(invalid_errors), 1)
        self.assertIn("count", invalid_errors[0])


class TestRelationshipTypeDefinition(unittest.TestCase):
    """Test cases for the RelationshipTypeDefinition class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a relationship type definition
        self.relationship_type_def = RelationshipTypeDefinition(
            name="Contains",
            description="Represents a containment relationship",
            source_types=[EntityType.BUCKET],
            target_types=[EntityType.TABLE],
            cardinality="one-to-many",
            properties={
                "created_at": PropertyDefinition(
                    name="created_at",
                    description="When the relationship was created",
                    data_type="datetime",
                    required=False
                )
            },
            required_properties=[]
        )
    
    def test_validate_relationship(self):
        """Test validation of a relationship against a relationship type definition."""
        # Create entities
        bucket_entity = Entity(
            id="bucket-1",
            type=EntityType.BUCKET,
            properties={"name": "Test Bucket"}
        )
        
        table_entity = Entity(
            id="table-1",
            type=EntityType.TABLE,
            properties={"name": "Test Table"}
        )
        
        column_entity = Entity(
            id="column-1",
            type=EntityType.COLUMN,
            properties={"name": "Test Column"}
        )
        
        entities = {
            "bucket-1": bucket_entity,
            "table-1": table_entity,
            "column-1": column_entity
        }
        
        # Create a valid relationship
        valid_relationship = Relationship(
            id="rel-1",
            type=RelationshipType.CONTAINS,
            source_id="bucket-1",
            target_id="table-1",
            properties={
                "created_at": "2023-01-01T00:00:00"
            }
        )
        
        # Create a relationship with invalid source type
        invalid_source_relationship = Relationship(
            id="rel-2",
            type=RelationshipType.CONTAINS,
            source_id="column-1",  # Column cannot be source of Contains
            target_id="table-1",
            properties={}
        )
        
        # Create a relationship with invalid target type
        invalid_target_relationship = Relationship(
            id="rel-3",
            type=RelationshipType.CONTAINS,
            source_id="bucket-1",
            target_id="column-1",  # Column cannot be target of Contains from Bucket
            properties={}
        )
        
        # Validate the relationships
        valid_result, valid_errors = self.relationship_type_def.validate_relationship(
            valid_relationship, entities
        )
        
        invalid_source_result, invalid_source_errors = self.relationship_type_def.validate_relationship(
            invalid_source_relationship, entities
        )
        
        invalid_target_result, invalid_target_errors = self.relationship_type_def.validate_relationship(
            invalid_target_relationship, entities
        )
        
        # Check results
        self.assertTrue(valid_result)
        self.assertEqual(len(valid_errors), 0)
        
        self.assertFalse(invalid_source_result)
        self.assertEqual(len(invalid_source_errors), 1)
        self.assertIn("source", invalid_source_errors[0])
        
        self.assertFalse(invalid_target_result)
        self.assertEqual(len(invalid_target_errors), 1)
        self.assertIn("target", invalid_target_errors[0])


class TestSchemaValidator(unittest.TestCase):
    """Test cases for the SchemaValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create property definitions
        name_prop = PropertyDefinition(
            name="name",
            description="Entity name",
            data_type="string",
            required=True
        )
        
        description_prop = PropertyDefinition(
            name="description",
            description="Entity description",
            data_type="string",
            required=False
        )
        
        # Create entity type definitions
        project_def = EntityTypeDefinition(
            name="Project",
            description="A Keboola project",
            properties={"name": name_prop, "description": description_prop},
            required_properties=["name"]
        )
        
        bucket_def = EntityTypeDefinition(
            name="Bucket",
            description="A storage bucket in Keboola",
            properties={"name": name_prop, "description": description_prop},
            required_properties=["name"]
        )
        
        # Create relationship type definitions
        contains_def = RelationshipTypeDefinition(
            name="Contains",
            description="Represents a containment relationship",
            source_types=[EntityType.PROJECT],
            target_types=[EntityType.BUCKET],
            cardinality="one-to-many",
            properties={},
            required_properties=[]
        )
        
        # Create mock schema
        class MockSchema:
            def __init__(self):
                self.entity_types = {
                    EntityType.PROJECT: project_def,
                    EntityType.BUCKET: bucket_def
                }
                self.relationship_types = {
                    RelationshipType.CONTAINS: contains_def
                }
            
            def get_entity_type_definition(self, entity_type):
                return self.entity_types.get(entity_type)
            
            def get_relationship_type_definition(self, relationship_type):
                return self.relationship_types.get(relationship_type)
            
            def get_allowed_relationships(self, entity_type):
                if entity_type == EntityType.PROJECT:
                    return {RelationshipType.CONTAINS: [EntityType.BUCKET]}
                return {}
        
        self.schema = MockSchema()
        self.validator = SchemaValidator(self.schema)
    
    def test_validate_entity(self):
        """Test validation of an entity against the schema."""
        # Create a valid project entity
        valid_project = Entity(
            id="project-1",
            type=EntityType.PROJECT,
            properties={"name": "Test Project", "description": "A test project"}
        )
        
        # Create an invalid project entity (missing required property)
        invalid_project = Entity(
            id="project-2",
            type=EntityType.PROJECT,
            properties={"description": "A test project"}  # Missing name
        )
        
        # Validate the entities
        valid_result, valid_errors = self.validator.validate_entity(valid_project)
        invalid_result, invalid_errors = self.validator.validate_entity(invalid_project)
        
        # Check results
        self.assertTrue(valid_result)
        self.assertEqual(len(valid_errors), 0)
        
        self.assertFalse(invalid_result)
        self.assertEqual(len(invalid_errors), 1)
    
    def test_validate_relationship(self):
        """Test validation of a relationship against the schema."""
        # Create entities
        project_entity = Entity(
            id="project-1",
            type=EntityType.PROJECT,
            properties={"name": "Test Project"}
        )
        
        bucket_entity = Entity(
            id="bucket-1",
            type=EntityType.BUCKET,
            properties={"name": "Test Bucket"}
        )
        
        entities = {
            "project-1": project_entity,
            "bucket-1": bucket_entity
        }
        
        # Create a valid relationship
        valid_relationship = Relationship(
            id="rel-1",
            type=RelationshipType.CONTAINS,
            source_id="project-1",
            target_id="bucket-1",
            properties={}
        )
        
        # Create an invalid relationship (wrong direction)
        invalid_relationship = Relationship(
            id="rel-2",
            type=RelationshipType.CONTAINS,
            source_id="bucket-1",  # Bucket cannot contain Project
            target_id="project-1",
            properties={}
        )
        
        # Validate the relationships
        valid_result, valid_errors = self.validator.validate_relationship(
            valid_relationship, entities
        )
        
        invalid_result, invalid_errors = self.validator.validate_relationship(
            invalid_relationship, entities
        )
        
        # Check results
        self.assertTrue(valid_result)
        self.assertEqual(len(valid_errors), 0)
        
        self.assertFalse(invalid_result)
        self.assertGreater(len(invalid_errors), 0)


if __name__ == "__main__":
    unittest.main() 