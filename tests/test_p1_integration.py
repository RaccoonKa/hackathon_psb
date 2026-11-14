"""Интеграционный технический тест P1 (без сети и индекса).

Сквозной поток на моках:
``plan()`` → ``RetrievalTask.source_kinds`` → ``_run_single_task`` →
``assess()`` → ``build_context_and_sources()``.

Проверяется, что planner формирует ровно одну задачу с обоими назначениями,
мост выполняет раздельные вызовы, evidence фиксирует недостающий тип
источника, а контекст/``AgentResult.sources`` не получают новых
пользовательских полей.
"""

from unittest.mock import patch

from rag_core.generation.answer.context_builder import build_context_and_sources
from rag_core.generation.planning.planner import plan
from rag_core.generation.retrieval import bridge as bridge_mod
from rag_core.generation.retrieval.evidence import assess
from rag_core.generation.types import AgentResult
from tests._common import make_chunk_dict, make_query_spec

NORMATIVE_DOCUMENT_TYPES = {"law", "internal"}
SPECIFIC_DOCUMENT_TYPES = {"contract"}

LEGACY_SOURCE_KEYS = {
    "document_id",
    "version_id",
    "chunk_id",
    "document_title",
    "provision_id",
    "structure_path",
    "text",
    "location",
    "score",
    "effective_from",
    "effective_to",
    "effective_dates_known",
    "amends_document_id",
}


def _spec():
    return make_query_spec(
        intent="mixed",
        required_slots=["approval_process", "payment_deadline"],
        rewritten_queries_bm25=["запрос"],
        rewritten_queries_dense=["запрос"],
    )


def _both_fake(**kwargs):
    if kwargs["document_types"] == NORMATIVE_DOCUMENT_TYPES:
        return [
            make_chunk_dict(
                chunk_id="n1",
                document_id="doc-a",
                document_type="law",
                text="нормативный источник",
                score=0.6,
            )
        ]
    if kwargs["document_types"] == SPECIFIC_DOCUMENT_TYPES:
        return [
            make_chunk_dict(
                chunk_id="s1",
                document_id="doc-b",
                document_type="contract",
                text="документ сделки",
                score=0.7,
            )
        ]
    raise AssertionError(f"неожиданный фильтр {kwargs['document_types']}")


def _only_specific_fake(**kwargs):
    if kwargs["document_types"] == NORMATIVE_DOCUMENT_TYPES:
        return []
    return _both_fake(**kwargs)


def _run_task(task, fake):
    with patch.object(bridge_mod.config, "USE_STUB_RAG", False), patch.object(
        bridge_mod, "reset_embedding_diagnostics"
    ), patch.object(
        bridge_mod, "get_last_embedding_diagnostics", return_value={}
    ), patch.object(
        bridge_mod, "hybrid_search", side_effect=fake
    ) as mock_search:
        chunks, stats, _embeddings, _ms = bridge_mod._run_single_task(task)
    return chunks, stats, mock_search


def test_full_flow_keeps_both_source_kinds():
    spec = _spec()
    tasks, plan_step = plan(spec)

    assert len(tasks) == 1
    assert plan_step.output["tasks_count"] == 1
    task = tasks[0]
    assert task.source_kinds == ["normative", "specific"]
    assert plan_step.output["required_source_kinds"] == [
        "normative",
        "specific",
    ]

    chunks, stats, mock_search = _run_task(task, _both_fake)
    assert mock_search.call_count == 2
    assert stats["source_kind_counts"] == {"normative": 1, "specific": 1}
    assert stats["source_kind_empty"] == []

    bundle, _assessment, _assess_step = assess(spec, chunks)
    assert bundle.missing_source_kinds == []
    assert all(
        warning.get("kind") != "missing_source_kind"
        for warning in bundle.generation_warnings
    )

    sources, context, _ctx_step = build_context_and_sources(chunks)
    assert "Назначение источника: нормативное основание" in context
    assert "Назначение источника: документ сделки" in context
    for entry in sources.values():
        assert set(entry) == LEGACY_SOURCE_KEYS
        assert "source_kind" not in entry
        assert "version_status" not in entry


def test_full_flow_reports_missing_source_kind():
    spec = _spec()
    tasks, _plan_step = plan(spec)
    task = tasks[0]
    assert task.source_kinds == ["normative", "specific"]

    chunks, stats, mock_search = _run_task(task, _only_specific_fake)
    assert mock_search.call_count == 2
    assert stats["source_kind_empty"] == ["normative"]
    assert [chunk.chunk_id for chunk in chunks] == ["s1"]

    bundle, _assessment, _assess_step = assess(spec, chunks)
    assert bundle.missing_source_kinds == ["normative"]
    assert any(
        warning.get("kind") == "missing_source_kind"
        for warning in bundle.generation_warnings
    )

    _sources, context, _ctx_step = build_context_and_sources(chunks)
    assert "Назначение источника: документ сделки" in context
    assert "Назначение источника: нормативное основание" not in context


def test_agent_result_sources_have_no_new_user_fields():
    spec = _spec()
    tasks, _ = plan(spec)
    chunks, _stats, _mock = _run_task(tasks[0], _both_fake)
    sources, context, _ = build_context_and_sources(chunks)

    result = AgentResult(
        final_answer="ответ",
        sources=sources,
        intent=spec.intent,
        refused=False,
        trace=[],
    )
    payload = result.to_dict(include_trace=False)

    assert set(payload["sources"]) == set(sources)
    for entry in payload["sources"].values():
        assert set(entry) == LEGACY_SOURCE_KEYS
    # Формат ответа не расширяется новыми пользовательскими полями.
    assert "source_kind" not in payload
    assert "version_status" not in payload
    assert context
