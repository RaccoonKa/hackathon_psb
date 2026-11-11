"""Data-access repositories for chats and knowledge-base documents."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chat_repo import DEFAULT_CHAT_TITLE, ChatRepository
    from .document_repo import DocumentRepository

__all__ = ["DEFAULT_CHAT_TITLE", "ChatRepository", "DocumentRepository"]


def __getattr__(name: str) -> object:
    if name in ("DEFAULT_CHAT_TITLE", "ChatRepository"):
        from .chat_repo import DEFAULT_CHAT_TITLE, ChatRepository

        return DEFAULT_CHAT_TITLE if name == "DEFAULT_CHAT_TITLE" else ChatRepository
    if name == "DocumentRepository":
        from .document_repo import DocumentRepository

        return DocumentRepository
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
