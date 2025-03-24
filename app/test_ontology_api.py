"""
Test script for ontology API endpoints.
"""
from fastapi.testclient import TestClient
from app.test_app import app
from app.ontology.models import Entity, Relationship
from app.ontology.rdf_store import RDFStore
from app.ontology.manager import OntologyManager
from app.state_manager import StateManager

client = TestClient(app)

def setup_test_data():
    """Load state from the state manager."""
    state_manager = StateManager()
    
    # Load RDF state
    rdf_store = state_manager.load_rdf_state()
    if not rdf_store:
        print("Warning: No RDF state found. Please run the indexing process first.")
        return None
    
    # Load ontology state
    ontology_manager = state_manager.load_ontology_state()
    if not ontology_manager:
        print("Warning: No ontology state found. Please run the indexing process first.")
        return None
    
    return rdf_store, ontology_manager

def test_list_entities():
    """Test listing all entities."""
    print("\nTesting list entities endpoint...")
    response = client.get("/ontology/entities")
    print(f"Response: {response.json()}")
    assert response.status_code == 200

def test_list_tables():
    """Test listing tables specifically."""
    print("\nTesting list tables endpoint...")
    response = client.get("/ontology/entities?type=table")
    print(f"Response: {response.json()}")
    assert response.status_code == 200

def test_list_relationships():
    """Test listing all relationships."""
    print("\nTesting list relationships endpoint...")
    response = client.get("/ontology/relationships")
    print(f"Response: {response.json()}")
    assert response.status_code == 200

def test_semantic_search():
    """Test semantic search endpoint."""
    print("\nTesting semantic search endpoint...")
    response = client.post(
        "/ontology/search",
        json={"query": "Find tables related to customer data"}
    )
    print(f"Response: {response.json()}")
    assert response.status_code == 200

def test_natural_language_query():
    """Test natural language query endpoint."""
    print("\nTesting natural language query endpoint...")
    response = client.post(
        "/ontology/query",
        json={"query": "What tables are in the customer bucket?"}
    )
    print(f"Response: {response.json()}")
    assert response.status_code == 200

def main():
    """Run all tests."""
    # Load state first
    state = setup_test_data()
    if not state:
        print("Skipping tests due to missing state data.")
        return
    
    rdf_store, ontology_manager = state
    print(f"Loaded {len(rdf_store.find_all_entities())} entities and {len(rdf_store.find_all_relationships())} relationships")
    
    # Run tests
    test_list_entities()
    test_list_tables()
    test_list_relationships()
    test_semantic_search()
    test_natural_language_query()

if __name__ == "__main__":
    main() 