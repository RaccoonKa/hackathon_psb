"""API router modules."""

from .chats import router as chats_router
from .documents import router as documents_router
from .health import router as health_router

__all__ = ["chats_router", "documents_router", "health_router"]
