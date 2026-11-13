"""Health check endpoint."""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends
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


@router.get("/health", response_model=HealthResponse)
async def health_check(
    engine: Engine = Depends(get_db_engine),
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    """Check database connectivity and report assistant configuration status."""
    db_ok = await asyncio.to_thread(check_db_health, engine)
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database="ok" if db_ok else "error",
        assistant_backend=settings.assistant_backend,
        llm_configured=settings.llm_configured,
    )
