"""Domain exceptions and FastAPI exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class NotFoundError(Exception):
    """Raised when a requested chat or document resource does not exist."""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message)
        self.message = message


class LLMNotConfiguredError(Exception):
    """Raised when an LLM request is attempted without valid configuration."""

    def __init__(self, message: str = "LLM service is not configured") -> None:
        super().__init__(message)
        self.message = message


class UpstreamLLMError(Exception):
    """Raised when the upstream LLM service fails or times out."""

    def __init__(self, message: str = "Upstream LLM request failed", status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


async def _handle_not_found(_request: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": exc.message})


async def _handle_llm_not_configured(_request: Request, exc: LLMNotConfiguredError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": exc.message})


async def _handle_upstream_llm_error(_request: Request, exc: UpstreamLLMError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


def register_exception_handlers(app: FastAPI) -> None:
    """Register domain exception handlers on the FastAPI application."""
    app.add_exception_handler(NotFoundError, _handle_not_found)
    app.add_exception_handler(LLMNotConfiguredError, _handle_llm_not_configured)
    app.add_exception_handler(UpstreamLLMError, _handle_upstream_llm_error)
