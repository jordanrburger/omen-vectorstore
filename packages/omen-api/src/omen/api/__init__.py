"""
REST API for the OMEN platform.
"""

from omen.api.app import create_app, app
from omen.api.routes.ontology import router as ontology_router
from omen.api.routes.search import router as search_router

__all__ = [
    "create_app",
    "app",
    "ontology_router",
    "search_router",
]
