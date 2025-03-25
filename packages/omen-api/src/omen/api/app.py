"""
Main FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from omen.core import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Create FastAPI app
    app = FastAPI(
        title="OMEN API",
        description="API for querying and managing the OMEN platform",
        version="0.1.0",
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Import routers here to avoid circular imports
    from omen.api.routes.ontology import router as ontology_router
    from omen.api.routes.search import router as search_router
    
    # Include routers
    app.include_router(ontology_router)
    app.include_router(search_router)
    
    @app.get("/")
    async def root():
        """Root endpoint that returns API information."""
        return {
            "name": "OMEN API",
            "version": "0.1.0",
            "description": "API for querying and managing the OMEN platform",
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}
    
    return app


# Create default application instance
app = create_app() 