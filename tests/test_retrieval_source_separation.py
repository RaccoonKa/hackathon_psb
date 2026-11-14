"""Технические тесты раздельного поиска по назначению источника (P1).

Проверяется ``_run_single_task`` моста retrieval: раздельные вызовы по
``task.source_kinds`` с фильтром ``document_types``, объединение по
``chunk_id``, приоритет известного ``source_kind`` и диагностика
``source_kind_counts``/``source_kind_empty``. Сеть и индекс не используются.
"""

from unittest.mock import patch

from rag_core.generation.retrieval import bridge as bridge_mod
from tests._common import make_chunk_dict, make_retrieval_task

NORMATIVE_DOCUMENT_TYPES = {"law", "internal"}
SPECIFIC_DOCUMENT_TYPES = {"contract"}


def _run(task, fake):
    with patch.object(bridge_mod.config, "USE_STUB_RAG", False), patch.object(
        bridge_mod, "reset_embedding_diagnostics"
    ), patch.object(
        bridge_mod, "get_last_embedding_diagnostics", return_value={}
    ), patch.object(
        bridge_mod, "hybrid_search", side_effect=fake
    ) as mock_search:
        chunks, stats, _embeddings, _ms = bridge_mod._run_single_task(task)
    return chunks, stats, mock_search


def _by_type(**kwargs):
    """Фабрика ответа поиска в зависимости от фильтра ``document_types``."""
    document_types = kwargs["document_types"]
    if document_types == NORMATIVE_DOCUMENT_TYPES:
        return [
            make_chunk_dict(
                chunk_id="c_norm",
                document_id="doc-a",
                document_type="law",
                score=0.6,
            )
        ]
    if document_types == SPECIFIC_DOCUMENT_TYPES:
        return [
            make_chunk_dict(
                chunk_id="c_spec",
                document_id="doc-b",
                document_type="contract",
                score=0.7,
            )
        ]
    raise AssertionError(f"неожиданный фильтр document_types={document_types}")


def test_two_kinds_produce_two_calls_and_merge():
    task = make_retrieval_task(source_kinds=["normative", "specific"])
    chunks, stats, mock_search = _run(task, _by_type)

    assert mock_search.call_count == 2
    passed_types = [
        call.kwargs["document_types"] for call in mock_search.call_args_list
    ]
    assert passed_types == [NORMATIVE_DOCUMENT_TYPES, SPECIFIC_DOCUMENT_TYPES]

    kinds = {chunk.chunk_id: chunk.source_kind for chunk in chunks}
    assert kinds == {"c_norm": "normative", "c_spec": "specific"}
    assert {chunk.chunk_id for chunk in chunks} == {"c_norm", "c_spec"}

    assert stats["source_kind_counts"] == {"normative": 1, "specific": 1}
    assert stats["source_kind_empty"] == []
    assert stats["source_kinds"] == ["normative", "specific"]


def test_no_kinds_single_call_without_document_types():
    task = make_retrieval_task(source_kinds=None)

    def fake(**kwargs):
        return [make_chunk_dict(chunk_id="c1", document_type="law")]

    chunks, stats, mock_search = _run(task, fake)

    assert mock_search.call_count == 1
    assert "document_types" not in mock_search.call_args.kwargs
    assert [chunk.chunk_id for chunk in chunks] == ["c1"]
    assert stats["source_kinds"] == []
    assert stats["source_kind_empty"] == []


def test_single_kind_single_call_with_filter():
    task = make_retrieval_task(source_kinds=["normative"])
    chunks, stats, mock_search = _run(task, _by_type)

    assert mock_search.call_count == 1
    assert (
        mock_search.call_args.kwargs["document_types"]
        == NORMATIVE_DOCUMENT_TYPES
    )
    assert [chunk.chunk_id for chunk in chunks] == ["c_norm"]
    assert chunks[0].source_kind == "normative"
    assert stats["source_kind_counts"] == {"normative": 1}


def test_empty_kind_result_is_not_error_and_reported():
    task = make_retrieval_task(source_kinds=["normative", "specific"])

    def fake(**kwargs):
        if kwargs["document_types"] == NORMATIVE_DOCUMENT_TYPES:
            return []
        return _by_type(document_types=kwargs["document_types"])

    chunks, stats, mock_search = _run(task, fake)

    assert mock_search.call_count == 2
    assert [chunk.chunk_id for chunk in chunks] == ["c_spec"]
    assert stats["source_kind_empty"] == ["normative"]
    assert stats["source_kind_counts"] == {"specific": 1}


def test_conflict_chunk_id_keeps_known_kind_over_unknown():
    # normative-ветка возвращает известный kind, specific — unknown.
    task = make_retrieval_task(source_kinds=["normative", "specific"])

    def fake(**kwargs):
        if kwargs["document_types"] == NORMATIVE_DOCUMENT_TYPES:
            return [
                make_chunk_dict(
                    chunk_id="dup",
                    document_type="law",
                    score=0.5,
                )
            ]
        return [
            make_chunk_dict(
                chunk_id="dup",
                document_type="unknown-type",
                source_kind="unknown",
                score=0.9,
            )
        ]

    chunks, _stats, _mock = _run(task, fake)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "dup"
    assert chunks[0].source_kind == "normative"


def test_conflict_chunk_id_known_kind_replaces_earlier_unknown():
    # specific-ветка идёт первой и возвращает unknown, затем normative —
    # известный source_kind должен заменить unknown.
    task = make_retrieval_task(source_kinds=["specific", "normative"])

    def fake(**kwargs):
        if kwargs["document_types"] == SPECIFIC_DOCUMENT_TYPES:
            return [
                make_chunk_dict(
                    chunk_id="dup",
                    document_type="unknown-type",
                    source_kind="unknown",
                    score=0.9,
                )
            ]
        return [
            make_chunk_dict(
                chunk_id="dup",
                document_type="law",
                score=0.4,
            )
        ]

    chunks, _stats, _mock = _run(task, fake)

    assert len(chunks) == 1
    assert chunks[0].source_kind == "normative"


def test_existing_source_kind_is_not_overwritten():
    task = make_retrieval_task(source_kinds=["normative"])

    def fake(**kwargs):
        return [
            make_chunk_dict(
                chunk_id="c1",
                document_type="law",
                source_kind="specific",
            )
        ]

    chunks, _stats, _mock = _run(task, fake)

    assert chunks[0].source_kind == "specific"


def test_source_kind_derived_from_document_type_when_missing():
    task = make_retrieval_task(source_kinds=["normative"])

    def fake(**kwargs):
        return [make_chunk_dict(chunk_id="c1", document_type="internal")]

    chunks, _stats, _mock = _run(task, fake)
    assert chunks[0].source_kind == "normative"
