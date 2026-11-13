"""FastAPI dependency providers resolved from application state."""

from __future__ import annotations

from fastapi import Depends, Request
from sqlalchemy import Engine

from ..core.config import Settings, get_settings
from ..integrations.base import AssistantPort
from ..integrations.llm_client import DirectLLMBackend
from ..repositories.chat_repo import ChatRepository
from ..repositories.document_repo import DocumentRepository
from ..services.chat_service import ChatService
from ..services.document_service import DocumentService


def get_app_settings(request: Request) -> Settings:
    """Return active application settings from `app.state`, falling back to `get_settings()`."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return get_settings()
    return settings


def get_db_engine(request: Request) -> Engine:
    """Return the shared SQLAlchemy engine from `app.state`."""
    return request.app.state.engine


def get_chat_repo(request: Request) -> ChatRepository:
    """Return the `ChatRepository` stored on `app.state`."""
    return request.app.state.chat_repo


def get_document_repo(request: Request) -> DocumentRepository:
    """Return the `DocumentRepository` stored on `app.state`."""
    return request.app.state.document_repo


def get_assistant_port(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> AssistantPort:
    """Construct the active `AssistantPort` implementation for the request."""
    if settings.assistant_backend == "rag":
        rag_adapter = getattr(request.app.state, "rag_adapter", None)
        if rag_adapter is None:
            from ..integrations.rag_adapter import RAGAssistantAdapter

            rag_adapter = RAGAssistantAdapter()
            request.app.state.rag_adapter = rag_adapter
        return rag_adapter

    return DirectLLMBackend(
        http_client=request.app.state.http_client,
        settings=settings,
    )


def get_chat_service(
    chat_repo: ChatRepository = Depends(get_chat_repo),
    assistant: AssistantPort = Depends(get_assistant_port),
    settings: Settings = Depends(get_app_settings),
) -> ChatService:
    """Construct `ChatService` wired with repository, assistant port, and history limit."""
    return ChatService(
        chat_repo=chat_repo,
        assistant=assistant,
        history_limit=settings.llm_history_limit,
    )


def get_document_service(
    document_repo: DocumentRepository = Depends(get_document_repo),
) -> DocumentService:
    """Construct `DocumentService` wired with the document repository."""
    return DocumentService(document_repo=document_repo)
