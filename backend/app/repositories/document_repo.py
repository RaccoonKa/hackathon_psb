"""Filesystem repository for reading raw knowledge-base documents."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.exceptions import NotFoundError
from ..schemas.document import DocumentDetail, DocumentSummary

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx"}
_SAFE_DOC_ID_RE = re.compile(r"^[0-9A-Za-zА-Яа-яЁё._-]+$")
_DISALLOWED_ID_CHARS_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё._-]+")
_DECORATIVE_LINE_RE = re.compile(r"^[-=*_~#\s]+$")
_TITLE_MAX_LEN = 200
_SHORT_HASH_LEN = 8
_DOCX_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


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


def _extract_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            if "word/document.xml" not in archive.namelist():
                return ""
            xml_bytes = archive.read("word/document.xml")
    except (zipfile.BadZipFile, OSError):
        return ""

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""

    p_tag = f"{_DOCX_W_NS}p"
    t_tag = f"{_DOCX_W_NS}t"
    tab_tag = f"{_DOCX_W_NS}tab"
    br_tag = f"{_DOCX_W_NS}br"
    cr_tag = f"{_DOCX_W_NS}cr"

    paragraphs: list[str] = []
    for paragraph in root.iter(p_tag):
        parts: list[str] = []
        for elem in paragraph.iter():
            if elem.tag == t_tag and elem.text:
                parts.append(elem.text)
            elif elem.tag == tab_tag:
                parts.append("\t")
            elif elem.tag in (br_tag, cr_tag):
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _read_text_and_digest(path: Path) -> tuple[str, str]:
    if path.suffix.lower() == ".docx":
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return _extract_docx_text(path), digest

    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = path.read_text(encoding="utf-8", errors="replace")

    try:
        raw_text = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    except UnicodeDecodeError:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

    return content, digest


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
            content, hex_digest = _read_text_and_digest(path)
            source_hash = f"sha256:{hex_digest}"
            manifest_entry = manifest_by_hash.get(source_hash)

            raw_doc_id = (
                manifest_entry.get("document_id")
                if manifest_entry and isinstance(manifest_entry.get("document_id"), str)
                else path.stem
            )
            base_id = _sanitize_document_id(
                raw_doc_id,
                fallback_hash=hex_digest[:_SHORT_HASH_LEN],
            )
            version_id = (
                manifest_entry.get("version_id")
                if manifest_entry and isinstance(manifest_entry.get("version_id"), str)
                else None
            )
            amends_document_id = (
                manifest_entry.get("amends_document_id")
                if manifest_entry
                and isinstance(manifest_entry.get("amends_document_id"), str)
                else None
            )

            parsed_files.append(
                _ParsedFile(
                    filename=path.relative_to(raw_dir).as_posix(),
                    content=content,
                    hex_digest=hex_digest,
                    base_id=base_id,
                    title=_extract_title(content, path),
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
                suffix = 2
                while f"{item.base_id}__{short_hash}_{suffix}" in used_ids:
                    suffix += 1
                doc_id = f"{item.base_id}__{short_hash}_{suffix}"

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
