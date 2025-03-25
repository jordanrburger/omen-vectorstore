# OMEN API

REST API server for the OMEN (Ontology-powered Metadata Engine) platform. This package provides a web service interface to the platform's search and ontology management capabilities.

## Overview

The `omen-api` package implements a FastAPI-based REST API for the OMEN platform, with these key responsibilities:

1. **Search Endpoints**: Vector similarity search for metadata
2. **Ontology Management**: CRUD operations for managing the knowledge graph
3. **Document Operations**: Adding, updating, and deleting vectorized documents
4. **System Endpoints**: Health checks and version information
5. **API Documentation**: OpenAPI specification and Swagger UI

## Key Components

### API Server (`omen.api.main`)

The main API application entry point:

```python
from omen.api.main import app
import uvicorn

# Run the API server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Search Routes (`omen.api.routes.search`)

Endpoints for vector search operations:

```
POST /api/search
GET /api/search/documents
GET /api/search/documents/{document_id}
GET /api/search/documents/type/{metadata_type}
POST /api/search/documents
DELETE /api/search/documents/{document_id}
```

Example search request:

```json
{
  "query": "find customer tables with transactions",
  "limit": 10,
  "min_score": 0.75,
  "type_filter": ["TABLE", "COLUMN"]
}
```

Example search response:

```json
{
  "results": [
    {
      "score": 0.92,
      "document": {
        "id": "table-123",
        "content": "Customer transactions table containing payment history",
        "source": {
          "type": "TABLE",
          "url": "https://connection.keboola.com/...",
          "created": "2023-05-15T09:12:33Z",
          "updated": "2023-06-22T14:45:11Z"
        },
        "metadata": {
          "row_count": "10452",
          "bucket_id": "in.c-sales"
        }
      }
    },
    {
      "score": 0.87,
      "document": {
        "id": "column-456",
        "content": "Customer ID column in the transactions table",
        "source": {
          "type": "COLUMN",
          "url": "https://connection.keboola.com/...",
          "created": "2023-05-15T09:12:33Z"
        },
        "metadata": {
          "table_id": "table-123",
          "data_type": "VARCHAR"
        }
      }
    }
  ]
}
```

### Ontology Routes (`omen.api.routes.ontology`)

Endpoints for ontology management:

```
GET /api/ontology/entities
GET /api/ontology/entities/{entity_id}
GET /api/ontology/entities/type/{entity_type}
POST /api/ontology/entities
PUT /api/ontology/entities/{entity_id}
DELETE /api/ontology/entities/{entity_id}

GET /api/ontology/relationships
GET /api/ontology/relationships/{relationship_id}
POST /api/ontology/relationships
PUT /api/ontology/relationships/{relationship_id}
DELETE /api/ontology/relationships/{relationship_id}

GET /api/ontology/stats
POST /api/ontology/query
```

Example entity creation request:

```json
{
  "type": "TABLE",
  "name": "customer_orders",
  "properties": {
    "schema": "public",
    "row_count": "2500",
    "description": "Table containing customer order data"
  },
  "source": "keboola-metadata"
}
```

Example SPARQL query request:

```json
{
  "query": "SELECT ?entity ?name WHERE { ?entity a <http://omen.keboola.com/ontology#Table> . ?entity <http://omen.keboola.com/ontology#hasName> ?name . } LIMIT 10"
}
```

## API Authentication and Security

The API currently supports:

- CORS middleware for cross-origin requests
- API key authentication (via header)
- Rate limiting

Future security enhancements:
- OAuth2 integration
- Role-based access control

## Starting the API Server

There are multiple ways to start the API server:

### 1. Using Python directly:

```bash
python -m omen.api.main
```

### 2. Using the OMEN CLI:

```bash
omen api start --host 0.0.0.0 --port 8000
```

### 3. Using Uvicorn:

```bash
uvicorn omen.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Using Docker:

```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key keboola/omen-api
```

## API Documentation

Once the server is running, the API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI Schema: http://localhost:8000/openapi.json

## Environment Variables

The API server can be configured with environment variables:

```
# Server configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=INFO

# Security settings
ALLOWED_ORIGINS=*,http://localhost:3000
API_KEY=your-secret-key

# OpenAI settings
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Vector DB settings
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=omen
```

## Installation

```bash
pip install omen-api
```

Or for development:

```bash
git clone https://github.com/keboola/omen-platform
cd omen-platform
pip install -e "packages/omen-api"
```

## Dependencies

- Python 3.8+
- fastapi
- uvicorn
- pydantic
- omen-core
- omen-vectorstore
- omen-ontology

## License

MIT License
