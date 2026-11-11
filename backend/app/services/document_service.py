"""Business logic for reading knowledge-base documents."""

from __future__ import annotations

import asyncio

from ..repositories.document_repo import DocumentRepository
from ..schemas.document import DocumentDetail, DocumentListResponse


class DocumentService:
    """Async service wrapper over the filesystem `DocumentRepository`."""

    def __init__(self, document_repo: DocumentRepository) -> None:
        self._document_repo = document_repo

    async def list_documents(self) -> DocumentListResponse:
        """Return all available knowledge-base documents."""
        items = await asyncio.to_thread(self._document_repo.list_documents)
        return DocumentListResponse(items=items, total=len(items))

    async def get_document(self, document_id: str) -> DocumentDetail:
        """Return full document metadata and content by ID."""
        return await asyncio.to_thread(self._document_repo.get_by_id, document_id)
