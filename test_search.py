from app.vectorizer import SentenceTransformerProvider
from app.indexer import QdrantIndexer

def print_result(result):
    print(f'\nType: {result["metadata_type"]} (Score: {result["score"]:.3f})')
    metadata = result['metadata']
    if 'name' in metadata:
        print(f'Name: {metadata["name"]}')
    if 'title' in metadata:
        print(f'Title: {metadata["title"]}')
    if 'description' in metadata:
        desc = metadata.get("description", "")
        print(f'Description: {desc[:200]}...' if len(desc) > 200 else desc)

def main():
    # Initialize components
    embedding_provider = SentenceTransformerProvider()
    indexer = QdrantIndexer()

    print('\nSearch Results for "Show me tables related to Slack data":')
    print('=====================================================')
    results = indexer.search_metadata(
        query='Show me tables related to Slack data',
        embedding_provider=embedding_provider,
        limit=5
    )
    for i, result in enumerate(results, 1):
        print(f'\n{i}.')
        print_result(result)

    print('\n\nSearch Results for "Find configuration for data processing":')
    print('=====================================================')
    results = indexer.search_metadata(
        query='Find configuration for data processing',
        embedding_provider=embedding_provider,
        metadata_type='configurations',
        limit=3
    )
    for i, result in enumerate(results, 1):
        print(f'\n{i}.')
        print_result(result)
        
    print('\n\nSearch Results for "Find tables containing Zendesk ticket data":')
    print('=====================================================')
    results = indexer.search_metadata(
        query='Find tables containing Zendesk ticket data',
        embedding_provider=embedding_provider,
        metadata_type='tables',
        limit=5
    )
    for i, result in enumerate(results, 1):
        print(f'\n{i}.')
        print_result(result)

if __name__ == '__main__':
    main() 