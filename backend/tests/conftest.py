"""Pytest fixtures for testing the FastAPI backend without external network calls."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_assistant_port
from app.core.config import Settings
from app.integrations.base import AssistantReply, ChatTurn
from app.main import create_app
from app.schemas.chat import SourceCitation

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def build_minimal_docx(paragraphs: list[str]) -> bytes:
    """Create a minimal valid `.docx` archive in memory using stdlib `zipfile`."""
    body_xml = "".join(
        f"<w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>"
        for text in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W_NS}">'
        f"<w:body>{body_xml}</w:body>"
        "</w:document>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


class FakeAssistant:
    """Test double implementing `AssistantPort` and recording conversation history."""

    def __init__(self) -> None:
        self.calls: list[list[ChatTurn]] = []
        self.queued_replies: list[AssistantReply] = []
        self.reply_factory: Callable[[list[ChatTurn]], AssistantReply] | None = None

    async def generate_reply(
        self,
        history: list[ChatTurn],
        event_date: str | None = None,
        selected_doc_ids: set[str] | None = None,
    ) -> AssistantReply:
        snapshot = list(history)
        self.calls.append(snapshot)

        if self.queued_replies:
            return self.queued_replies.pop(0)
        if self.reply_factory is not None:
            return self.reply_factory(snapshot)

        last_user_text = next(
            (turn.content for turn in reversed(snapshot) if turn.role == "user"),
            "",
        )
        return AssistantReply(
            content=f"Правовой ответ на запрос: {last_user_text}",
            sources=[
                SourceCitation(
                    document_id="compliance_reg",
                    version_id="v1",
                    document_title="Внутренний регламент комплаенса",
                    provision_id="p-1.1",
                    structure_path=["Раздел 1", "Пункт 1.1"],
                    excerpt="Проверка контрагента обязательна.",
                    char_start=34,
                    char_end=70,
                    score=0.96,
                )
            ],
            refused=False,
            intent="legal_qa",
            clarification_needed=False,
        )


@dataclass(frozen=True)
class SampleCorpus:
    """Metadata describing the temporary knowledge-base files created for tests."""

    data_dir: Path
    digests: dict[str, str]
    colliding_base_id: str


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    """Return a temporary file-backed SQLite database URL."""
    db_path = tmp_path / "test_chat.db"
    return f"sqlite:///{db_path}"


@pytest.fixture
def sample_corpus(tmp_path: Path) -> SampleCorpus:
    """Create temporary `data/raw` files (`.txt`, `.md`, `.docx`) and manifest with ID collision."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    policies_dir = raw_dir / "policies"
    manifests_dir = data_dir / "manifests"
    policies_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)

    txt_path = policies_dir / "credit_v1.txt"
    txt_content = (
        "URL: https://example.local/credit/v1\n"
        "Дата выгрузки: 2026-10-01\n"
        "Источник: Правила кредитования МСБ (редакция 1)\n"
        "----------------------------------------\n"
        "Кредитный договор заключается в письменной форме.\n"
    )
    txt_path.write_text(txt_content, encoding="utf-8")
    txt_digest = hashlib.sha256(txt_content.encode("utf-8")).hexdigest()

    md_path = policies_dir / "credit_v2.md"
    md_content = (
        "# Правила кредитования МСБ (редакция 2)\n\n"
        "Максимальный срок кредита увеличен до 60 месяцев.\n"
    )
    md_path.write_text(md_content, encoding="utf-8")
    md_digest = hashlib.sha256(md_content.encode("utf-8")).hexdigest()

    docx_path = raw_dir / "regulation.docx"
    docx_bytes = build_minimal_docx(
        [
            "Внутренний регламент комплаенса",
            "Пункт 1.1. Проверка контрагента обязательна.",
        ]
    )
    docx_path.write_bytes(docx_bytes)
    docx_digest = hashlib.sha256(docx_bytes).hexdigest()

    note_path = raw_dir / "unlisted_note.txt"
    note_content = "Краткая памятка юриста без записи в манифесте.\n"
    note_path.write_text(note_content, encoding="utf-8")
    note_digest = hashlib.sha256(note_content.encode("utf-8")).hexdigest()

    # Ignored artifacts that must not appear in document listings
    (raw_dir / "~$temp_lock.docx").write_bytes(docx_bytes)
    (raw_dir / "ignored.json").write_text('{"skip": true}', encoding="utf-8")

    colliding_base_id = "credit_rules"
    manifest_payload = {
        "documents": [
            {
                "document_id": colliding_base_id,
                "version_id": "v1",
                "source_hash": f"sha256:{txt_digest}",
            },
            {
                "document_id": colliding_base_id,
                "version_id": "v2",
                "amends_document_id": colliding_base_id,
                "source_hash": f"sha256:{md_digest}",
            },
            {
                "document_id": "compliance_reg",
                "version_id": "v1",
                "source_hash": f"sha256:{docx_digest}",
            },
        ]
    }
    (manifests_dir / "indexing_manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return SampleCorpus(
        data_dir=data_dir,
        digests={
            "policies/credit_v1.txt": txt_digest,
            "policies/credit_v2.md": md_digest,
            "regulation.docx": docx_digest,
            "unlisted_note.txt": note_digest,
        },
        colliding_base_id=colliding_base_id,
    )


@pytest.fixture
def data_dir(sample_corpus: SampleCorpus) -> Path:
    """Return the temporary `DATA_DIR` path populated with sample documents."""
    return sample_corpus.data_dir


@pytest.fixture
def test_settings(db_url: str, data_dir: Path) -> Settings:
    """Return isolated `Settings` pointing at temporary SQLite DB and document directory."""
    return Settings(
        llm_base_url="https://mock-llm.local/v1",
        llm_api_key="sk-test-fake-key-do-not-leak",
        llm_model="test/legal-model",
        llm_temperature=0.0,
        llm_max_tokens=512,
        llm_timeout_seconds=5.0,
        llm_history_limit=20,
        assistant_backend="direct_llm",
        database_url=db_url,
        data_dir=data_dir,
    )


@pytest.fixture
def fake_assistant() -> FakeAssistant:
    """Return a fresh `FakeAssistant` test double per test."""
    return FakeAssistant()


@pytest.fixture
def app(test_settings: Settings, fake_assistant: FakeAssistant) -> FastAPI:
    """Build a FastAPI app with `get_assistant_port` overridden to `FakeAssistant`."""
    application = create_app(test_settings)
    application.dependency_overrides[get_assistant_port] = lambda: fake_assistant
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Yield an initialized FastAPI `TestClient` running lifespan hooks."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def open_real_llm_client() -> Callable[
    [Settings, httpx.AsyncBaseTransport | None],
    Iterator[tuple[TestClient, FastAPI]],
]:
    """Return a context-manager factory for testing real `DirectLLMBackend` wiring."""

    @contextmanager
    def _open(
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> Iterator[tuple[TestClient, FastAPI]]:
        application = create_app(settings)
        mock_http_client: httpx.AsyncClient | None = None
        with TestClient(application) as test_client:
            if transport is not None:
                mock_http_client = httpx.AsyncClient(
                    transport=transport,
                    timeout=settings.llm_timeout_seconds,
                )
                application.state.http_client = mock_http_client
            try:
                yield test_client, application
            finally:
                if mock_http_client is not None:
                    portal = getattr(test_client, "portal", None)
                    if portal is not None:
                        portal.call(mock_http_client.aclose)
                    else:
                        asyncio.run(mock_http_client.aclose())

    return _open
