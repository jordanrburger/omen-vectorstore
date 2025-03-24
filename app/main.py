"""
Main FastAPI application.
"""

from fastapi import FastAPI
from app.ontology.api import router as ontology_router

app = FastAPI(
    title="Keboola Ontology API",
    description="API for querying and managing the Keboola ontology and action graph",
    version="1.0.0"
)

app.include_router(ontology_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
