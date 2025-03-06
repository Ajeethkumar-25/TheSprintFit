# existing_routes/__init__.py
# Initializes the routes package and exports the route routers

from .authentication_routes import router as authentication_routes
from .investor_routes import router as investor_routes
from .linkedin_routes import router as linkedin_routes

__all__ = ['authentication_routes', 'investor_routes', 'linkedin_routes']