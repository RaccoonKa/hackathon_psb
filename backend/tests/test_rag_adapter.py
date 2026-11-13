"""Unit tests and QA contract verification for RAGAssistantAdapter and RAG API."""

from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.integrations.base import ChatTurn
from app.integrations.rag_adapter import RAGAssistantAdapter
from app.main import create_app
from app.schemas.chat import SourceCitation


@dataclasses.dataclass
class _DummyTraceStep:
    step: str
    output: dict


@dataclasses.dataclass
class _DummyAgentResult:
    final_answer: str
    sources: dict
    intent: str
    refused: bool
    trace: list
    refusal_reason: str | None = None
    confidence: str = "high"
    assumptions: list = dataclasses.field(default_factory=list)
    missing_info: list = dataclasses.field(default_factory=list)


@pytest.mark.asyncio
async def test_rag_adapter_generates_reply_and_enriches_citations():
    mock_chunks = {
        "chunk_123": {
            "chunk_id": "chunk_123",
            "document_id": "doc_rental",
            "version_id": "v1",
            "document_title": "Договор аренды",
            "text": "Арендная плата составляет 100 000 рублей в месяц.",
            "location": {
                "char_start": 50,
                "char_end": 102,
                "page": 3,
            },
            "structure": {
                "provision_id": "3.1",
                "path": ["Раздел 3", "Пункт 3.1"],
            },
        }
    }

    dummy_result = _DummyAgentResult(
        final_answer="Срок оплаты — до 5 числа каждого месяца. <source>S1</source>",
        sources={
            "S1": {
                "chunk_id": "chunk_123",
                "document_id": "doc_rental",
                "document_title": "Договор аренды",
                "score": 0.98,
            }
        },
        intent="exact_lookup",
        refused=False,
        trace=[_DummyTraceStep(step="retrieval", output={"count": 1})],
        confidence="high",
        assumptions=["Предполагается действующий договор"],
        missing_info=[],
    )

    adapter = RAGAssistantAdapter(chunks_map=mock_chunks)

    history = [
        ChatTurn(role="user", content="Какая ставка аренды?"),
        ChatTurn(role="assistant", content="Уточните объект."),
        ChatTurn(role="user", content="Северный склад, пункт 3.1"),
    ]

    with patch("app.integrations.rag_adapter.ask", return_value=dummy_result) as mock_run:
        with patch(
            "app.integrations.rag_adapter.replace_source_tags",
            return_value="Срок оплаты — до 5 числа каждого месяца. [Источник: Договор аренды, Пункт 3.1]",
        ):
            reply = await adapter.generate_reply(
                history,
                event_date="2026-05-01",
                selected_doc_ids={"doc_rental"},
            )

    assert mock_run.called
    call_args = mock_run.call_args
    assert (call_args.kwargs.get("query") or call_args.args[0]) == "Северный склад, пункт 3.1"
    assert call_args.kwargs["event_date"] == "2026-05-01"
    assert call_args.kwargs["selected_doc_ids"] == {"doc_rental"}
    assert call_args.kwargs["chat_history"] == [
        {"role": "user", "content": "Какая ставка аренды?"},
        {"role": "assistant", "content": "Уточните объект."},
    ]

    assert reply.content == "Срок оплаты — до 5 числа каждого месяца. [Источник: Договор аренды, Пункт 3.1]"
    assert reply.refused is False
    assert reply.intent == "exact_lookup"
    assert reply.confidence == "high"
    assert reply.assumptions == ["Предполагается действующий договор"]
    assert len(reply.trace) == 1
    assert reply.trace[0]["step"] == "retrieval"

    assert len(reply.sources) == 1
    citation = reply.sources[0]
    assert citation.source_id == "S1"
    assert citation.document_id == "doc_rental"
    assert citation.excerpt == "Арендная плата составляет 100 000 рублей в месяц."
    assert citation.char_start == 50
    assert citation.char_end == 102
    assert citation.page == 3
    assert citation.score == 0.98


@pytest.mark.asyncio
async def test_rag_adapter_passes_chat_history_to_facade():
    adapter = RAGAssistantAdapter(chunks_map={})
    history = [
        ChatTurn(role="user", content="Что такое овердрафт?"),
        ChatTurn(role="assistant", content="Овердрафт — это форма краткосрочного кредитования."),
        ChatTurn(role="user", content="Каковы лимиты по нему?"),
        ChatTurn(role="assistant", content="Лимит устанавливается индивидуально."),
        ChatTurn(role="user", content="А какой срок возврата?"),
    ]

    dummy_result = _DummyAgentResult(
        final_answer="Срок возврата транша овердрафта составляет 30 дней.",
        sources={},
        intent="exact_lookup",
        refused=False,
        trace=[],
    )

    with patch("app.integrations.rag_adapter.ask", return_value=dummy_result) as mock_ask:
        reply = await adapter.generate_reply(history)

    assert mock_ask.called
    kwargs = mock_ask.call_args.kwargs
    assert kwargs["query"] == "А какой срок возврата?"
    assert kwargs["chat_history"] == [
        {"role": "user", "content": "Что такое овердрафт?"},
        {"role": "assistant", "content": "Овердрафт — это форма краткосрочного кредитования."},
        {"role": "user", "content": "Каковы лимиты по нему?"},
        {"role": "assistant", "content": "Лимит устанавливается индивидуально."},
    ]
    assert kwargs["as_dict"] is False
    assert reply.content == "Срок возврата транша овердрафта составляет 30 дней."
    assert reply.refused is False


@pytest.mark.asyncio
async def test_rag_adapter_graceful_error_handling():
    adapter = RAGAssistantAdapter(chunks_map={})

    history = [ChatTurn(role="user", content="Вопрос")]

    with patch(
        "app.integrations.rag_adapter.ask",
        side_effect=RuntimeError("External LLM timeout"),
    ):
        reply = await adapter.generate_reply(history)

    assert reply.refused is True
    assert "произошла техническая ошибка" in reply.content
    assert reply.refusal_reason == "External LLM timeout"
    assert reply.confidence == "low"
    assert reply.sources == []
    assert reply.trace == []


class _DummyEnum(enum.Enum):
    ALPHA = "alpha"
    BETA = "beta"


class _DummyCustomObj:
    def __init__(self, val: str) -> None:
        self.val = val

    def __str__(self) -> str:
        return f"Custom({self.val})"


class TestRAGSourceCitations:
    """Detailed verification of SourceCitation extraction and normalisation."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("chunk_patch", "sdata_patch", "expected_page"),
        [
            ({"page": 5}, {}, 5),
            ({"page_num": 6}, {}, 6),
            ({"location": {"page": 7}}, {}, 7),
            ({"location": {"page_num": 8}}, {}, 8),
            ({"page": "9"}, {}, 9),
            ({"page": "invalid"}, {}, None),
            (None, {"page": 10}, 10),
            (None, {"page_num": 11}, 11),
            (None, {}, None),
        ],
    )
    async def test_page_resolution(self, chunk_patch, sdata_patch, expected_page):
        chunk = None
        cid = "c1" if chunk_patch is not None else None
        if chunk_patch is not None:
            chunk = {"chunk_id": "c1", "text": "sample text"}
            chunk.update(chunk_patch)
            chunks_map = {"c1": chunk}
        else:
            chunks_map = {}

        sdata = {"chunk_id": cid, "score": 0.9}
        sdata.update(sdata_patch)

        adapter = RAGAssistantAdapter(chunks_map=chunks_map)
        res = _DummyAgentResult(
            final_answer="Answer",
            sources={"S1": sdata},
            intent="qa",
            refused=False,
            trace=[],
        )

        with patch("app.integrations.rag_adapter.ask", return_value=res):
            reply = await adapter.generate_reply([ChatTurn(role="user", content="Q")])

        assert len(reply.sources) == 1
        assert reply.sources[0].page == expected_page

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("chunk_patch", "sdata_patch", "expected_start", "expected_end"),
        [
            ({"char_start": 10, "char_end": 20}, {}, 10, 20),
            ({"location": {"char_start": 30, "char_end": 40}}, {}, 30, 40),
            (None, {"char_start": 50, "char_end": 60}, 50, 60),
            ({"char_start": "70", "char_end": "80"}, {}, 70, 80),
            ({"char_start": "bad", "char_end": "bad"}, {}, None, None),
            (None, {}, None, None),
        ],
    )
    async def test_char_offsets_resolution(
        self, chunk_patch, sdata_patch, expected_start, expected_end
    ):
        cid = "c1" if chunk_patch is not None else None
        chunks_map = {}
        if chunk_patch is not None:
            chunk = {"chunk_id": "c1", "text": "text"}
            chunk.update(chunk_patch)
            chunks_map["c1"] = chunk

        sdata = {"chunk_id": cid}
        sdata.update(sdata_patch)

        adapter = RAGAssistantAdapter(chunks_map=chunks_map)
        res = _DummyAgentResult(
            final_answer="Answer",
            sources={"S1": sdata},
            intent="qa",
            refused=False,
            trace=[],
        )

        with patch("app.integrations.rag_adapter.ask", return_value=res):
            reply = await adapter.generate_reply([ChatTurn(role="user", content="Q")])

        assert len(reply.sources) == 1
        assert reply.sources[0].char_start == expected_start
        assert reply.sources[0].char_end == expected_end

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("raw_score", "expected_score"),
        [
            (0.85, 0.85),
            ("0.75", 0.75),
            (None, None),
            ("invalid", None),
        ],
    )
    async def test_score_resolution(self, raw_score, expected_score):
        adapter = RAGAssistantAdapter(chunks_map={})
        res = _DummyAgentResult(
            final_answer="Answer",
            sources={"S1": {"score": raw_score, "excerpt": "text"}},
            intent="qa",
            refused=False,
            trace=[],
        )

        with patch("app.integrations.rag_adapter.ask", return_value=res):
            reply = await adapter.generate_reply([ChatTurn(role="user", content="Q")])

        assert len(reply.sources) == 1
        assert reply.sources[0].score == expected_score


class TestRAGTraceAndMetadataSerialization:
    """Verification of complex trace and fidelity metadata serialization."""

    @pytest.mark.asyncio
    async def test_trace_complex_types_serialization(self):
        adapter = RAGAssistantAdapter(chunks_map={})

        complex_step = _DummyTraceStep(
            step="planner",
            output={
                "tags": {"tag_b", "tag_a"},
                "status": _DummyEnum.ALPHA,
                "custom": _DummyCustomObj("custom_val"),
                "count": 42,
            },
        )

        res = _DummyAgentResult(
            final_answer="Answer",
            sources={},
            intent="legal_analysis",
            refused=False,
            trace=[complex_step],
            confidence="medium",
            assumptions=["assumption 1", "assumption 2"],
            missing_info=["need more contracts"],
            refusal_reason=None,
        )

        with patch("app.integrations.rag_adapter.ask", return_value=res):
            reply = await adapter.generate_reply([ChatTurn(role="user", content="Q")])

        assert reply.confidence == "medium"
        assert reply.assumptions == ["assumption 1", "assumption 2"]
        assert reply.missing_info == ["need more contracts"]
        assert reply.intent == "legal_analysis"
        assert reply.refusal_reason is None
        assert len(reply.trace) == 1

        step_output = reply.trace[0]["output"]
        assert step_output["tags"] == ["tag_a", "tag_b"]
        assert step_output["status"] == "alpha"
        assert step_output["custom"] == "Custom(custom_val)"
        assert step_output["count"] == 42


class TestRAGParameterNormalization:
    """Verification of selected_doc_ids and event_date parameter normalization."""

    @pytest.mark.asyncio
    async def test_normalization(self):
        adapter = RAGAssistantAdapter(chunks_map={})
        res = _DummyAgentResult(
            final_answer="Answer",
            sources={},
            intent="qa",
            refused=False,
            trace=[],
        )

        with patch("app.integrations.rag_adapter.ask", return_value=res) as mock_run:
            await adapter.generate_reply(
                [ChatTurn(role="user", content="Q")],
                event_date="   2026-11-13  ",
                selected_doc_ids={"doc1", "doc2", "doc1"},
            )

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["event_date"] == "2026-11-13"
        assert call_kwargs["selected_doc_ids"] == {"doc1", "doc2"}

        with patch("app.integrations.rag_adapter.ask", return_value=res) as mock_run:
            await adapter.generate_reply(
                [ChatTurn(role="user", content="Q")],
                event_date="     ",
                selected_doc_ids=None,
            )

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["event_date"] is None
        assert call_kwargs["selected_doc_ids"] is None


class TestRAGErrorIsolation:
    """Verification of run_agent error isolation and polite fallback."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("Chroma unreachable"),
            ValueError("Malformed prompt"),
            KeyError("missing_field"),
        ],
    )
    async def test_error_isolation(self, exc):
        adapter = RAGAssistantAdapter(chunks_map={})

        with patch("app.integrations.rag_adapter.ask", side_effect=exc):
            reply = await adapter.generate_reply([ChatTurn(role="user", content="Q")])

        assert reply.refused is True
        assert reply.confidence == "low"
        assert reply.sources == []
        assert reply.trace == []
        assert str(exc) in (reply.refusal_reason or "")
        assert "произошла техническая ошибка" in reply.content


class TestRAGFastAPIRoutes:
    """End-to-end FastAPI integration tests for RAG assistant backend."""

    def test_health_rag_states(self, db_url: str, data_dir: Path):
        settings = Settings(
            database_url=db_url,
            data_dir=data_dir,
            assistant_backend="rag",
        )
        app = create_app(settings)

        with TestClient(app) as client:
            # 1. Healthy prewarm report
            app.state.rag_prewarm_report = {"collection": "ok", "count": 100, "errors": []}
            resp = client.get("/api/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["rag_ready"] is True
            assert data["rag_details"]["collection"] == "ok"

            # 2. Degraded prewarm report
            app.state.rag_prewarm_report = {
                "collection": "error",
                "errors": [{"type": "RuntimeError", "message": "Failed to connect"}],
            }
            resp = client.get("/api/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["rag_ready"] is False
            assert len(data["rag_details"]["errors"]) == 1

            # 3. On-demand prewarm fallback when app.state report is None
            app.state.rag_prewarm_report = None
            with patch(
                "rag_core.retrieval.vector_store.prewarm_retrieval_backends",
                return_value={"collection": "ok", "count": 50, "errors": []},
            ):
                resp = client.get("/api/health")
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "ok"
                assert data["rag_ready"] is True
                assert data["rag_details"]["count"] == 50

    def test_rag_chat_messages_full_contract_and_sqlite_persistence(
        self, db_url: str, data_dir: Path
    ):
        settings = Settings(
            database_url=db_url,
            data_dir=data_dir,
            assistant_backend="rag",
        )
        app = create_app(settings)

        dummy_result = _DummyAgentResult(
            final_answer="Порядок расторжения: уведомление за 30 дней. <source>S1</source>",
            sources={
                "S1": {
                    "chunk_id": "chunk_99",
                    "document_id": "doc_contract",
                    "version_id": "v2",
                    "document_title": "Договор поставки",
                    "provision_id": "п. 4.2",
                    "structure_path": ["Раздел 4", "Пункт 4.2"],
                    "excerpt": "Уведомление направляется за 30 дней.",
                    "page": 5,
                    "char_start": 120,
                    "char_end": 180,
                    "score": 0.94,
                }
            },
            intent="cancellation_rules",
            refused=False,
            trace=[_DummyTraceStep(step="retrieval", output={"found": 1})],
            confidence="high",
            assumptions=["Договор не содержит специальных условий"],
            missing_info=[],
        )

        with TestClient(app) as client:
            # Create chat
            create_resp = client.post("/api/chats", json={"title": "Тест RAG"})
            assert create_resp.status_code == 201
            chat_id = create_resp.json()["id"]

            with patch("app.integrations.rag_adapter.ask", return_value=dummy_result):
                with patch(
                    "app.integrations.rag_adapter.replace_source_tags",
                    return_value="Порядок расторжения: уведомление за 30 дней. [1]",
                ):
                    msg_resp = client.post(
                        f"/api/chats/{chat_id}/messages",
                        json={
                            "content": "Как расторгнуть договор?",
                            "event_date": "2026-11-13",
                            "selected_doc_ids": ["doc_contract", "doc_contract"],
                        },
                    )

            assert msg_resp.status_code == 200
            payload = msg_resp.json()
            assert payload["chat_id"] == chat_id

            # Verify user message
            user_msg = payload["user_message"]
            assert user_msg["role"] == "user"
            assert user_msg["content"] == "Как расторгнуть договор?"

            # Verify assistant message
            asst_msg = payload["assistant_message"]
            assert asst_msg["role"] == "assistant"
            assert asst_msg["content"] == "Порядок расторжения: уведомление за 30 дней. [1]"
            assert asst_msg["confidence"] == "high"
            assert asst_msg["assumptions"] == ["Договор не содержит специальных условий"]
            assert asst_msg["missing_info"] == []
            assert asst_msg["refused"] is False
            assert asst_msg["refusal_reason"] is None
            assert len(asst_msg["trace"]) == 1
            assert asst_msg["trace"][0]["step"] == "retrieval"

            # Verify sources
            assert len(asst_msg["sources"]) == 1
            src = asst_msg["sources"][0]
            assert src["source_id"] == "S1"
            assert src["document_id"] == "doc_contract"
            assert src["version_id"] == "v2"
            assert src["document_title"] == "Договор поставки"
            assert src["provision_id"] == "п. 4.2"
            assert src["structure_path"] == ["Раздел 4", "Пункт 4.2"]
            assert src["page"] == 5
            assert src["char_start"] == 120
            assert src["char_end"] == 180
            assert src["score"] == 0.94

            # Verify persistence via GET
            history_resp = client.get(f"/api/chats/{chat_id}/messages")
            assert history_resp.status_code == 200
            messages = history_resp.json()["messages"]
            assert len(messages) == 2
            assert messages[1]["confidence"] == "high"
            assert messages[1]["sources"][0]["document_id"] == "doc_contract"
            assert messages[1]["trace"][0]["step"] == "retrieval"

    def test_rag_chat_error_isolation_persists_refusal(
        self, db_url: str, data_dir: Path
    ):
        settings = Settings(
            database_url=db_url,
            data_dir=data_dir,
            assistant_backend="rag",
        )
        app = create_app(settings)

        with TestClient(app) as client:
            create_resp = client.post("/api/chats")
            assert create_resp.status_code == 201
            chat_id = create_resp.json()["id"]

            with patch(
                "app.integrations.rag_adapter.ask",
                side_effect=RuntimeError("Vector store timeout"),
            ):
                msg_resp = client.post(
                    f"/api/chats/{chat_id}/messages",
                    json={"content": "Сложный вопрос"},
                )

            assert msg_resp.status_code == 200
            payload = msg_resp.json()
            asst_msg = payload["assistant_message"]
            assert asst_msg["refused"] is True
            assert "Vector store timeout" in (asst_msg["refusal_reason"] or "")
            assert asst_msg["confidence"] == "low"
            assert "произошла техническая ошибка" in asst_msg["content"]

            # Verify subsequent turn proceeds without deadlock
            dummy_recovery = _DummyAgentResult(
                final_answer="Восстановленный ответ.",
                sources={},
                intent="qa",
                refused=False,
                trace=[],
            )
            with patch("app.integrations.rag_adapter.ask", return_value=dummy_recovery):
                msg_resp2 = client.post(
                    f"/api/chats/{chat_id}/messages",
                    json={"content": "Повторный вопрос"},
                )

            assert msg_resp2.status_code == 200
            payload2 = msg_resp2.json()
            assert payload2["assistant_message"]["refused"] is False
            assert payload2["assistant_message"]["content"] == "Восстановленный ответ."

            history_resp = client.get(f"/api/chats/{chat_id}/messages")
            assert len(history_resp.json()["messages"]) == 4
