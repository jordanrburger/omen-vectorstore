"""
Main FastAPI application.
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

from omen.core import configure_logging, get_logger

# Initialize logger
logger = get_logger(__name__)

# Configure CORS settings
origins = os.environ.get("CORS_ORIGINS", "*").split(",")

# Create and configure app
app = FastAPI(
    title="OMEN API",
    description="API for the Ontology-powered Metadata Engine",
    version="0.1.0",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers here to avoid circular imports
from omen.api.routes.ontology import router as ontology_router
from omen.api.routes.search import router as search_router
from omen.api.routes.hybrid import router as hybrid_router

# Include routers
app.include_router(ontology_router)
app.include_router(search_router)
app.include_router(hybrid_router)


@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": "OMEN API",
        "version": "0.1.0",
        "description": "API for the Ontology-powered Metadata Engine",
        "endpoints": [
            {"path": "/search", "description": "Vector search operations"},
            {"path": "/ontology", "description": "Ontology operations"},
            {"path": "/hybrid", "description": "Hybrid search operations"}
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"} 