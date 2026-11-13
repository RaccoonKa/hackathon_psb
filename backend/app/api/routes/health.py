"""Health check endpoint."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy import Engine

from ...core.config import Settings
from ...core.database import check_db_health
from ..deps import get_app_settings, get_db_engine

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health status payload returned by `GET /api/health`."""

    status: Literal["ok", "degraded"]
    database: Literal["ok", "error"]
    assistant_backend: str
    llm_configured: bool
    rag_ready: bool = True
    rag_details: dict[str, Any] | None = None


@router.get("/health", response_model=HealthResponse)
async def health_check(
    request: Request,
    engine: Engine = Depends(get_db_engine),
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    """Check database connectivity and report assistant configuration status."""
    db_ok = await asyncio.to_thread(check_db_health, engine)

    rag_ready = True
    rag_details: dict[str, Any] | None = None

    if settings.assistant_backend == "rag":
        report = getattr(request.app.state, "rag_prewarm_report", None)
        if report is None:
            try:
                from rag_core.retrieval.vector_store import prewarm_retrieval_backends

                report = await asyncio.to_thread(prewarm_retrieval_backends)
            except Exception as exc:
                report = {"collection": "error", "errors": [{"type": type(exc).__name__, "message": str(exc)}]}
        rag_details = report
        rag_ready = (
            report.get("collection") == "ok"
            and not report.get("errors")
        ) if isinstance(report, dict) else False

    is_healthy = db_ok and (rag_ready if settings.assistant_backend == "rag" else True)

    return HealthResponse(
        status="ok" if is_healthy else "degraded",
        database="ok" if db_ok else "error",
        assistant_backend=settings.assistant_backend,
        llm_configured=settings.llm_configured,
        rag_ready=rag_ready,
        rag_details=rag_details,
    )
