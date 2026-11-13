"""FastAPI application factory and ASGI entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import asyncio
import logging

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import chats_router, documents_router, health_router
from .core.config import Settings, get_settings
from .core.database import create_db_engine, create_session_factory, init_db
from .core.exceptions import register_exception_handlers
from .repositories.chat_repo import ChatRepository
from .repositories.document_repo import DocumentRepository


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_db_engine(active_settings.database_url)
        http_client: httpx.AsyncClient | None = None
        try:
            session_factory = create_session_factory(engine)
            init_db(engine)

            http_client = httpx.AsyncClient(
                timeout=active_settings.llm_timeout_seconds,
            )

            app.state.settings = active_settings
            app.state.engine = engine
            app.state.session_factory = session_factory
            app.state.chat_repo = ChatRepository(session_factory)
            app.state.document_repo = DocumentRepository(active_settings.data_dir)
            app.state.http_client = http_client

            if active_settings.assistant_backend == "rag":
                try:
                    from .integrations.rag_adapter import RAGAssistantAdapter
                    from rag_core.retrieval.vector_store import prewarm_retrieval_backends

                    report = await asyncio.to_thread(prewarm_retrieval_backends)
                    app.state.rag_prewarm_report = report
                    app.state.rag_adapter = RAGAssistantAdapter()
                except Exception as exc:
                    logging.getLogger(__name__).warning("RAG prewarm failed on startup: %s", exc)

            yield
        finally:
            if http_client is not None:
                await http_client.aclose()
            engine.dispose()

    application = FastAPI(
        title="PSB Legal Assistant API",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(application)

    application.include_router(health_router, prefix="/api")
    application.include_router(chats_router, prefix="/api")
    application.include_router(documents_router, prefix="/api")

    return application


app = create_app()
