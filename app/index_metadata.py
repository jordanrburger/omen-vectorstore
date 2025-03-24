"""
Script to extract metadata from Keboola and build the ontology.
"""

import os
import logging
from dotenv import load_dotenv

from app.keboola_client import KeboolaClient
from app.state_manager import StateManager
from app.ontology.manager import OntologyManager
from app.ontology.rdf_store import RDFStore
from app.ontology.builder import OntologyBuilder
from app.ontology.models import EntityType, RelationshipType, Entity
from app.llm_client import LLMClient
# Temporarily comment out Qdrant imports
# from app.indexer import QdrantIndexer
# from app.vectorizer import EmbeddingProvider
# from app.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to extract metadata and build ontology."""
    # Load environment variables
    load_dotenv()
    
    # Get Keboola API credentials
    api_url = os.getenv("KEBOOLA_API_URL")
    api_token = os.getenv("KEBOOLA_TOKEN")
    
    if not api_url or not api_token:
        logger.error("Missing Keboola API credentials. Please set KEBOOLA_API_URL and KEBOOLA_TOKEN in .env")
        return
    
    # Initialize components
    state_manager = StateManager()
    keboola_client = KeboolaClient(api_url, api_token, state_manager)
    ontology_manager = OntologyManager()
    rdf_store = RDFStore()
    
    # Initialize LLM client
    llm_client = LLMClient(
        provider="openai",
        model="gpt-4-turbo",  # Use a more capable OpenAI model
        temperature=0.0,
        max_tokens=4096
    )
    
    # Temporarily comment out Qdrant initialization
    # Initialize Qdrant indexer and embedding provider
    # config = Config.from_env()
    # qdrant_indexer = QdrantIndexer(collection_name="keboola_metadata", config=config)
    # embedding_provider = EmbeddingProvider(provider="openai", model="text-embedding-3-small")
    
    # Initialize ontology builder with LLM client
    ontology_builder = OntologyBuilder(
        llm_client=llm_client,
        ontology_manager=ontology_manager,
        batch_size=10,
        max_workers=4,
        max_retries=3,
        retry_delay=1.0
    )
    
    try:
        # Extract metadata from Keboola
        logger.info("Extracting metadata from Keboola...")
        metadata = keboola_client.extract_metadata(force_full=True)
        
        # Temporarily comment out Qdrant indexing
        # Index metadata into Qdrant for semantic search
        # logger.info("Indexing metadata into Qdrant...")
        # qdrant_indexer.index_metadata(metadata, embedding_provider)
        
        # Prepare metadata list for ontology building
        metadata_list = []
        
        # Add buckets
        for bucket in metadata.get('buckets', []):
            metadata_list.append({
                'type': 'bucket',
                'data': {
                    'id': bucket.get('id'),
                    'name': bucket.get('name'),
                    'description': bucket.get('description', ''),
                    'created': bucket.get('created', ''),
                    'last_updated': bucket.get('last_updated', '')
                }
            })
        
        # Add tables
        for table_id in metadata.get('tables', []):
            # Get table details
            table_data = keboola_client.get_table_details(table_id)
            if table_data:
                metadata_list.append({
                    'type': 'table',
                    'data': {
                        'id': table_data.get('id'),
                        'name': table_data.get('name'),
                        'bucket': table_data.get('bucket', {}).get('id'),
                        'columns': table_data.get('columns', []),
                        'rows_count': table_data.get('rowsCount', 0),
                        'data_size_bytes': table_data.get('dataSizeBytes', 0),
                        'created': table_data.get('created', ''),
                        'last_updated': table_data.get('lastChangeDate', '')
                    }
                })
        
        # Add configurations
        for config_id in metadata.get('configurations', []):
            # Get configuration details
            config_data = keboola_client.get_config_details(config_id) if hasattr(keboola_client, 'get_config_details') else None
            if config_data:
                metadata_list.append({
                    'type': 'configuration',
                    'data': {
                        'id': config_id,
                        'name': config_data.get('name', config_id) if isinstance(config_data, dict) else config_id,
                        'component': config_data.get('component', '') if isinstance(config_data, dict) else '',
                        'configuration': config_data.get('configuration', {}) if isinstance(config_data, dict) else {},
                        'created': config_data.get('created', '') if isinstance(config_data, dict) else '',
                        'last_updated': config_data.get('last_updated', '') if isinstance(config_data, dict) else ''
                    }
                })
            else:
                # Add basic configuration info if details not available
                metadata_list.append({
                    'type': 'configuration',
                    'data': {
                        'id': config_id,
                        'name': config_id,
                        'component': '',
                        'configuration': {},
                        'created': '',
                        'last_updated': ''
                    }
                })
        
        # Modified approach: Extract all entities first, then detect relationships in one go
        logger.info("Extracting entities from all metadata items...")
        all_entities = []
        
        # Process metadata items to extract entities
        for item in metadata_list:
            entities = ontology_builder.extract_entities(item)
            for entity in entities:
                ontology_manager.add_entity(entity)
                all_entities.append(entity)
        
        logger.info(f"Extracted a total of {len(all_entities)} entities")
        
        # Detect relationships between all entities
        if len(all_entities) >= 2:
            # Process entities in groups by type to manage batches efficiently
            logger.info("Detecting relationships between entities...")
            
            # Group tables and buckets - likely to have relationships
            bucket_entities = [e for e in all_entities if e.type == EntityType.BUCKET]
            table_entities = [e for e in all_entities if e.type == EntityType.TABLE]
            
            # Detect relationships for tables and their buckets
            if bucket_entities and table_entities:
                # Create pairs of related buckets and tables
                for i in range(0, len(bucket_entities), 5):
                    batch_buckets = bucket_entities[i:i+5]
                    
                    for j in range(0, len(table_entities), 5):
                        batch_tables = table_entities[j:j+5]
                        batch = batch_buckets + batch_tables
                        
                        logger.info(f"Processing relationships between {len(batch_buckets)} buckets and {len(batch_tables)} tables")
                        relationships = ontology_builder.detect_relationships(batch)
                        
                        # Add relationships to ontology manager
                        for relationship in relationships:
                            ontology_manager.add_relationship(relationship)
            
            # Batch all configuration entities together
            config_entities = [e for e in all_entities if e.type == EntityType.CONFIGURATION]
            if config_entities:
                # Detect relationships between configurations and other entity types
                logger.info(f"Processing relationships for {len(config_entities)} configuration entities")
                for i in range(0, len(config_entities), 10):
                    batch = config_entities[i:i+10]
                    relationships = ontology_builder.detect_relationships(batch)
                    
                    # Add relationships to ontology manager
                    for relationship in relationships:
                        ontology_manager.add_relationship(relationship)
            
            # Finally, process a random sample of entities to find any missed relationships
            import random
            if len(all_entities) > 20:
                sampled_entities = random.sample(all_entities, min(20, len(all_entities)))
                logger.info(f"Processing relationships for a random sample of {len(sampled_entities)} entities")
                relationships = ontology_builder.detect_relationships(sampled_entities)
                
                # Add relationships to ontology manager
                for relationship in relationships:
                    ontology_manager.add_relationship(relationship)
        
        # Save ontology state
        logger.info("Saving ontology state...")
        state_manager.save_ontology_state(ontology_manager)
        
        # Load ontology into RDF store
        logger.info("Loading ontology into RDF store...")
        rdf_store.load_from_ontology_manager(ontology_manager)
        
        # Save RDF state
        logger.info("Saving RDF state...")
        state_manager.save_rdf_state(rdf_store)
        
        # Print summary
        logger.info(f"Successfully processed {len(metadata.get('buckets', []))} buckets")
        logger.info(f"Successfully processed {len(metadata.get('tables', []))} tables")
        logger.info(f"Successfully processed {len(metadata.get('configurations', []))} configurations")
        
        # Count entities by type
        entity_types = {}
        for entity in all_entities:
            entity_type = entity.type.value
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        logger.info("Entity types extracted:")
        for entity_type, count in entity_types.items():
            logger.info(f"- {entity_type}: {count}")
        
        # Print some example queries
        logger.info("\nExample queries:")
        logger.info("1. Find all tables:")
        tables = rdf_store.find_entities_by_type(EntityType.TABLE)
        logger.info(f"Found {len(tables)} tables")
        
        logger.info("\n2. Find all relationships:")
        relationships = rdf_store.find_all_relationships()
        logger.info(f"Found {len(relationships)} relationships")
        
        logger.info("\n3. Find relationships between tables:")
        table_relationships = rdf_store.find_relationships_by_type(RelationshipType.DEPENDS_ON)
        logger.info(f"Found {len(table_relationships)} table dependencies")
        
    except Exception as e:
        logger.error(f"Error during metadata extraction and ontology building: {e}")
        raise

if __name__ == "__main__":
    main() 