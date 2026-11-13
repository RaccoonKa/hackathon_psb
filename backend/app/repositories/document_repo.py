"""Filesystem repository for reading raw knowledge-base documents."""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.exceptions import NotFoundError
from ..schemas.document import DocumentDetail, DocumentSummary

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path and (_repo_root / "rag_core").is_dir():
    sys.path.insert(0, str(_repo_root))

from rag_core.indexing.parser import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    calculate_source_hash,
    clean_garant_headers,
    extract_text_from_file,
)

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = SUPPORTED_DOCUMENT_EXTENSIONS
_SAFE_DOC_ID_RE = re.compile(r"^[0-9A-Za-zА-Яа-яЁё._-]+$")
_DISALLOWED_ID_CHARS_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё._-]+")
_DECORATIVE_LINE_RE = re.compile(r"^[-=*_~#\s]+$")
_TITLE_MAX_LEN = 200
_SHORT_HASH_LEN = 8


@dataclass(frozen=True)
class _ParsedFile:
    filename: str
    content: str
    hex_digest: str
    base_id: str
    title: str
    version_id: str | None
    amends_document_id: str | None


def _validate_document_id(document_id: str) -> None:
    if (
        not document_id
        or "\x00" in document_id
        or "/" in document_id
        or "\\" in document_id
        or ".." in document_id
        or not _SAFE_DOC_ID_RE.fullmatch(document_id)
    ):
        raise NotFoundError("Document not found")


def _sanitize_document_id(raw_id: str, fallback_hash: str) -> str:
    sanitized = _DISALLOWED_ID_CHARS_RE.sub("_", raw_id)
    while ".." in sanitized:
        sanitized = sanitized.replace("..", "_")
    sanitized = sanitized.strip("._-")
    if not sanitized:
        return f"doc_{fallback_hash}"
    return sanitized


def _clean_extracted_title(
    title_candidate: str | None, cleaned_content: str, path: Path
) -> str:
    if title_candidate and title_candidate.strip():
        cleaned = re.split(r"\s*[-=*_~#]{3,}\s*", title_candidate.strip())[0].strip()
        if "\n" in cleaned:
            cleaned = cleaned.splitlines()[0].strip()
        if cleaned:
            return cleaned[:_TITLE_MAX_LEN]
    return _extract_title(cleaned_content, path)


def _extract_title(content: str, path: Path) -> str:
    first_meaningful: str | None = None
    non_empty_seen = 0

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        non_empty_seen += 1

        if line.lower().startswith("источник:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate:
                return candidate[:_TITLE_MAX_LEN]

        if first_meaningful is None:
            lowered = line.lower()
            if lowered.startswith(("url:", "дата выгрузки:")):
                continue
            if _DECORATIVE_LINE_RE.fullmatch(line):
                continue
            cleaned = line.lstrip("#").strip()
            if cleaned:
                first_meaningful = cleaned

        if non_empty_seen >= 20 and first_meaningful is not None:
            break

    if first_meaningful:
        return first_meaningful[:_TITLE_MAX_LEN]
    return path.stem[:_TITLE_MAX_LEN]


class DocumentRepository:
    """Repository reading raw documents and optional indexing manifest metadata."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)

    def _load_manifest_by_hash(self) -> dict[str, dict[str, Any]]:
        manifest_path = self._data_dir / "manifests" / "indexing_manifest.json"
        if not manifest_path.is_file():
            return {}

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to parse indexing manifest at %s", manifest_path)
            return {}

        if not isinstance(payload, dict):
            return {}
        documents = payload.get("documents")
        if not isinstance(documents, list):
            return {}

        by_hash: dict[str, dict[str, Any]] = {}
        for item in documents:
            if isinstance(item, dict):
                source_hash = item.get("source_hash")
                if isinstance(source_hash, str) and source_hash:
                    by_hash[source_hash] = item
        return by_hash

    def _load_documents(self) -> list[DocumentDetail]:
        raw_dir = self._data_dir / "raw"
        if not raw_dir.is_dir():
            return []

        files = sorted(
            (
                path
                for path in raw_dir.rglob("*")
                if path.is_file()
                and not path.name.startswith("~$")
                and path.suffix.lower() in _SUPPORTED_EXTENSIONS
            ),
            key=lambda p: p.relative_to(raw_dir).as_posix(),
        )
        if not files:
            return []

        manifest_by_hash = self._load_manifest_by_hash()
        parsed_files: list[_ParsedFile] = []

        for path in files:
            suffix = path.suffix.lower()
            raw_bytes = path.read_bytes() if suffix in {".docx", ".pdf"} else None
            try:
                raw_content, _ = extract_text_from_file(path)
            except UnicodeDecodeError:
                try:
                    raw_content = path.read_text(encoding="utf-8", errors="replace")
                except Exception as exc:
                    logger.warning("Failed to decode text from %s: %s", path, exc)
                    continue
            except Exception as exc:
                logger.warning("Failed to extract text from %s: %s", path, exc)
                continue

            if not raw_content or not raw_content.strip():
                continue

            cleaned_content, extracted_title, _ = clean_garant_headers(raw_content)
            if not cleaned_content.strip():
                cleaned_content = raw_content

            source_hash = calculate_source_hash(raw_content, raw_bytes=raw_bytes)
            hex_digest = (
                source_hash.split(":", 1)[1]
                if ":" in source_hash
                else source_hash
            )

            manifest_entry = manifest_by_hash.get(source_hash)

            if manifest_entry:
                raw_doc_id = (
                    manifest_entry.get("document_id")
                    if isinstance(manifest_entry.get("document_id"), str)
                    else path.stem
                )
                version_id = (
                    manifest_entry.get("version_id")
                    if isinstance(manifest_entry.get("version_id"), str)
                    else None
                )
                amends_document_id = (
                    manifest_entry.get("amends_document_id")
                    if isinstance(manifest_entry.get("amends_document_id"), str)
                    else None
                )
                cand_title = (
                    manifest_entry.get("title")
                    or manifest_entry.get("document_title")
                )
                if isinstance(cand_title, str) and cand_title.strip():
                    title = cand_title.strip()[:_TITLE_MAX_LEN]
                else:
                    title = _clean_extracted_title(
                        extracted_title, cleaned_content, path
                    )
            else:
                raw_doc_id = path.stem
                version_id = None
                amends_document_id = None
                title = _clean_extracted_title(
                    extracted_title, cleaned_content, path
                )

            base_id = _sanitize_document_id(
                raw_doc_id,
                fallback_hash=hex_digest[:_SHORT_HASH_LEN],
            )

            parsed_files.append(
                _ParsedFile(
                    filename=path.relative_to(raw_dir).as_posix(),
                    content=cleaned_content,
                    hex_digest=hex_digest,
                    base_id=base_id,
                    title=title,
                    version_id=version_id,
                    amends_document_id=amends_document_id,
                )
            )

        base_id_counts = Counter(item.base_id for item in parsed_files)
        used_ids: set[str] = set()
        documents: list[DocumentDetail] = []

        for item in parsed_files:
            short_hash = item.hex_digest[:_SHORT_HASH_LEN]
            if base_id_counts[item.base_id] > 1:
                doc_id = f"{item.base_id}__{short_hash}"
            else:
                doc_id = item.base_id

            if doc_id in used_ids:
                suffix_idx = 2
                while f"{item.base_id}__{short_hash}_{suffix_idx}" in used_ids:
                    suffix_idx += 1
                doc_id = f"{item.base_id}__{short_hash}_{suffix_idx}"

            used_ids.add(doc_id)
            documents.append(
                DocumentDetail(
                    document_id=doc_id,
                    version_id=item.version_id,
                    title=item.title,
                    filename=item.filename,
                    amends_document_id=item.amends_document_id,
                    content=item.content,
                    char_count=len(item.content),
                )
            )

        return documents

    def list_documents(self) -> list[DocumentSummary]:
        """List all raw documents with resolved unique document IDs and metadata."""
        return [
            DocumentSummary(
                document_id=doc.document_id,
                version_id=doc.version_id,
                title=doc.title,
                filename=doc.filename,
                amends_document_id=doc.amends_document_id,
            )
            for doc in self._load_documents()
        ]

    def get_by_id(self, document_id: str) -> DocumentDetail:
        """Return full document content by validated `document_id` or raise `NotFoundError`."""
        _validate_document_id(document_id)
        for doc in self._load_documents():
            if doc.document_id == document_id:
                return doc
        raise NotFoundError(f"Document '{document_id}' not found")
