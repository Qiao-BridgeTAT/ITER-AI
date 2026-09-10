"""FastAPI service entrypoint package."""

from services.api.main import create_app, create_default_app

__all__ = ["create_app", "create_default_app"]
