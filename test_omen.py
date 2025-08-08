from omen_core import OmenClient
from omen_vectorstore import VectorStore
from omen_ontology import Ontology

def test_imports():
    print("✓ Successfully imported omen packages")

def test_client_creation():
    try:
        client = OmenClient()
        print("✓ Successfully created OmenClient instance")
    except Exception as e:
        print(f"✗ Failed to create OmenClient: {str(e)}")

def test_vectorstore():
    try:
        store = VectorStore()
        print("✓ Successfully created VectorStore instance")
    except Exception as e:
        print(f"✗ Failed to create VectorStore: {str(e)}")

def test_ontology():
    try:
        ontology = Ontology()
        print("✓ Successfully created Ontology instance")
    except Exception as e:
        print(f"✗ Failed to create Ontology: {str(e)}")

if __name__ == "__main__":
    print("Running Omen package tests...")
    print("-" * 30)
    test_imports()
    test_client_creation()
    test_vectorstore()
    test_ontology()
    print("-" * 30)
    print("Test run completed") 