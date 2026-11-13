"""Unit tests for RAGAssistantAdapter."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from app.integrations.base import ChatTurn
from app.integrations.rag_adapter import RAGAssistantAdapter


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

    with patch("app.integrations.rag_adapter.run_agent", return_value=dummy_result) as mock_run:
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
    assert call_args.args[0] == "Северный склад, пункт 3.1"
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
async def test_rag_adapter_graceful_error_handling():
    adapter = RAGAssistantAdapter(chunks_map={})

    history = [ChatTurn(role="user", content="Вопрос")]

    with patch(
        "app.integrations.rag_adapter.run_agent",
        side_effect=RuntimeError("External LLM timeout"),
    ):
        reply = await adapter.generate_reply(history)

    assert reply.refused is True
    assert "произошла техническая ошибка" in reply.content
    assert reply.refusal_reason == "External LLM timeout"
    assert reply.confidence == "low"
    assert reply.sources == []
    assert reply.trace == []
