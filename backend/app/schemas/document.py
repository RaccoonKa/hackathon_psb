"""Pydantic schemas for knowledge-base documents."""

from __future__ import annotations

from pydantic import BaseModel


class DocumentSummary(BaseModel):
    """Summary metadata for a document in the knowledge base."""

    document_id: str
    version_id: str | None = None
    title: str
    filename: str
    amends_document_id: str | None = None


class DocumentDetail(DocumentSummary):
    """Full document payload including extracted text content."""

    content: str
    char_count: int


class DocumentListResponse(BaseModel):
    """List response for knowledge-base documents."""

    items: list[DocumentSummary]
    total: int
