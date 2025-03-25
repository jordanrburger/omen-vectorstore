# OMEN Core Examples

This directory contains example scripts demonstrating the usage of OMEN Core functionality.

## Basic Example

The `basic_example.py` script demonstrates:
- Loading configuration from environment variables
- Initializing the LLM client for embeddings
- Setting up the vector store
- Creating and storing document embeddings
- Performing semantic search

### Prerequisites

1. Make sure you have the required environment variables set in your `.env` file:
   ```
   OPENAI_API_KEY=your-api-key
   OPENAI_MODEL=gpt-4
   OPENAI_EMBEDDING_MODEL=text-embedding-3-large
   
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   QDRANT_COLLECTION=omen
   ```

2. Ensure Qdrant is running (you can start it using Docker):
   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

### Running the Example

```bash
# From the omen-core directory
python examples/basic_example.py
```

The script will:
1. Create embeddings for sample business data texts
2. Store them in the vector database
3. Perform a semantic search query
4. Display the results with similarity scores and metadata 