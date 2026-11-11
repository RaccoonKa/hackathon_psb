"""REST routes for browsing knowledge-base documents."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...schemas.document import DocumentDetail, DocumentListResponse
from ...services.document_service import DocumentService
from ..deps import get_document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """List available knowledge-base documents."""
    return await document_service.list_documents()


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentDetail:
    """Return full text and metadata for a knowledge-base document."""
    return await document_service.get_document(document_id)
