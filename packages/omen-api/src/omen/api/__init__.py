"""
REST API for the OMEN platform.
"""

# Import app for easier access
from omen.api.app import app
from omen.api.routes.ontology import router as ontology_router
from omen.api.routes.search import router as search_router
from omen.api.routes.hybrid import router as hybrid_router

__all__ = [
    "app",
    "ontology_router",
    "search_router",
    "hybrid_router",
]
