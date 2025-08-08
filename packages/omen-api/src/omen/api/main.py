"""
Main entrypoint for the OMEN API server.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from omen.core import configure_logging, get_logger, settings
from omen.api.routes import ontology, search, hybrid

# Configure logging
logger = get_logger(__name__)
log_level = os.environ.get("LOG_LEVEL", "INFO")
configure_logging(log_level)

app = FastAPI(
    title="OMEN API",
    description="Ontology-powered Metadata Engine API",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be configured more strictly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ontology.router, prefix="/api/ontology", tags=["ontology"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(hybrid.router, prefix="/api/hybrid", tags=["hybrid"]) 


@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    """Redirect root endpoint to OpenAPI documentation."""
    return RedirectResponse(url="/docs")


@app.get("/api/health", tags=["system"])
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}


@app.get("/api/version", tags=["system"])
async def version():
    """Return API version information."""
    return {
        "version": app.version,
        "title": app.title,
    }


if __name__ == "__main__":
    import uvicorn
    
    # Use environment variables for host/port if available
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    
    logger.info(f"Starting OMEN API server at {host}:{port}")
    uvicorn.run(
        "omen.api.main:app",
        host=host,
        port=port,
        reload=settings.debug,
        log_level=log_level.lower(),
    ) 