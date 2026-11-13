"""Тесты H-2: маршрутизация hybrid, суммирующая RRF-агрегация и условный boost.

Покрывают:
* H-2A — единый ``hybrid``-режим с явным ``intent`` для всех интентов и
  проброс ``task.intent`` в ``hybrid_search`` с fallback по ``mode``;
* H-2B — ``RRF(chunk) = Σ weight_i / (60 + rank_i + 1)`` на уровне моста;
* H-2C — boost допсоглашений только при явной релевантности запроса.
"""

from unittest.mock import patch

from rag_core.generation.planning.planner import plan
from rag_core.generation.retrieval import bridge as bridge_mod
from rag_core.generation.types import RetrievalTask, SubQuestion
from rag_core.retrieval import hybrid as hybrid_mod
from rag_core.retrieval.hybrid import should_boost_amendments
from tests._common import (
    make_bm25_stats,
    make_chunk_dict,
    make_query_spec,
    make_retrieved_chunk,
)


def _task(
    purpose: str = "t",
    mode: str = "hybrid",
    intent: str | None = None,
    boost_doc_ids=None,
) -> RetrievalTask:
    return RetrievalTask(
        query="запрос",
        mode=mode,
        purpose=purpose,
        selected_doc_ids=None,
        boost_doc_ids=boost_doc_ids,
        event_date=None,
        top_k=10,
        intent=intent,
    )


# --- H-2B: суммирующая RRF-агрегация ---------------------------------------


def test_sum_rrf_adds_contributions_instead_of_max():
    a = make_retrieved_chunk(chunk_id="a")
    b = make_retrieved_chunk(chunk_id="b")
    merged, _ = bridge_mod._aggregate_task_results(
        [(_task("t1"), [a, b]), (_task("t2"), [b])]
    )

    assert abs(merged["a"].score - 1 / 61) < 1e-6
    assert abs(merged["b"].score - (1 / 62 + 1 / 61)) < 1e-6
    # max(score) дал бы b == a; суммирующая RRF накапливает вклад.
    assert merged["b"].score > merged["a"].score


def test_chunk_present_in_two_lists_accumulates_rrf():
    a = make_retrieved_chunk(chunk_id="a")
    b = make_retrieved_chunk(chunk_id="b")
    merged, _ = bridge_mod._aggregate_task_results(
        [(_task("t1"), [a, b]), (_task("t2"), [b, a])]
    )

    assert abs(merged["a"].score - (1 / 61 + 1 / 62)) < 1e-6
    assert abs(merged["b"].score - (1 / 62 + 1 / 61)) < 1e-6


def test_best_single_rank_does_not_always_win_against_multi_channel_hit():
    # «top» стоит первым только в одной задаче, «stable» найден в двух.
    top = make_retrieved_chunk(chunk_id="top")
    stable = make_retrieved_chunk(chunk_id="stable")
    filler = [make_retrieved_chunk(chunk_id=f"f{idx}") for idx in range(5)]
    merged, _ = bridge_mod._aggregate_task_results(
        [
            (_task("t1"), [top, *filler]),
            (_task("t2"), [*filler, stable]),
        ]
    )

    # top: 1/61; stable: 1/66 (в t2 на 6-м месте); проверяем именно сумму.
    assert abs(merged["top"].score - 1 / 61) < 1e-6
    assert abs(merged["stable"].score - 1 / 66) < 1e-6

    # А при стабильном присутствии в обоих списках stable побеждает top.
    merged2, _ = bridge_mod._aggregate_task_results(
        [(_task("t1"), [top, stable]), (_task("t2"), [stable, top])]
    )
    assert merged2["stable"].score == merged2["top"].score > 1 / 61


def test_duplicate_chunk_id_does_not_create_duplicates_or_double_count():
    a = make_retrieved_chunk(chunk_id="a")
    b = make_retrieved_chunk(chunk_id="b")
    merged, _ = bridge_mod._aggregate_task_results(
        [(_task("t1"), [a, a, b])]
    )

    assert set(merged) == {"a", "b"}
    # Повтор внутри одного списка не даёт двойного вклада.
    assert abs(merged["a"].score - 1 / 61) < 1e-6
    assert abs(merged["b"].score - 1 / 63) < 1e-6


def test_aggregation_uses_task_weight_when_present():
    a = make_retrieved_chunk(chunk_id="a")
    weighted = _task("weighted")
    weighted.weight = 2.0  # type: ignore[attr-defined]
    merged, _ = bridge_mod._aggregate_task_results([(weighted, [a])])
    assert abs(merged["a"].score - 2 / 61) < 1e-6


def test_degraded_task_weight_defaults_to_one():
    a = make_retrieved_chunk(chunk_id="a")
    task = _task("novel")
    task.weight = "invalid"  # type: ignore[attr-defined]
    merged, _ = bridge_mod._aggregate_task_results([(task, [a])])
    assert abs(merged["a"].score - 1 / 61) < 1e-6


def test_retrieve_tasks_trace_reports_sum_rrf_and_no_increment():
    def fast(_task):
        return ([make_retrieved_chunk(chunk_id="x")], {}, {}, 5)

    with patch.object(
        bridge_mod, "prewarm_retrieval_backends", return_value={"errors": []}
    ), patch.object(bridge_mod, "_run_single_task", side_effect=fast):
        chunks, step = bridge_mod.retrieve_tasks([_task("t1"), _task("t2")])

    assert step.extra["aggregation"] == "sum_rrf"
    assert "increment" not in step.extra
    assert abs(chunks[0].score - (1 / 61 + 1 / 61)) < 1e-6


def test_aggregated_scores_remain_compatible_with_threshold_filter():
    def fast(_task):
        return ([make_retrieved_chunk(chunk_id="x")], {}, {}, 5)

    with patch.object(
        bridge_mod, "prewarm_retrieval_backends", return_value={"errors": []}
    ), patch.object(bridge_mod, "_run_single_task", side_effect=fast):
        chunks, _ = bridge_mod.retrieve_tasks([_task("t1")])

    passed, filtered, step = bridge_mod.filter_chunks_by_threshold(
        chunks, threshold=0.1
    )
    assert passed
    assert [chunk.chunk_id for chunk in filtered] == ["x"]
    # Сумма RRF ниже ceiling, поэтому порог адаптивно снижается.
    assert step.input["effective_threshold"] <= 0.005


def test_protected_top_ids_kept_after_aggregation():
    chunks = [make_retrieved_chunk(chunk_id=f"c{idx}") for idx in range(6)]
    merged, protected = bridge_mod._aggregate_task_results(
        [(_task("t1"), chunks[:4])]
    )
    assert protected == ["c0", "c1", "c2"]
    assert set(protected) <= set(merged)


# --- H-2B-fix: doc-boost после RRF-агрегации --------------------------------


def _plain_and_boosted():
    plain = make_retrieved_chunk(chunk_id="plain", document_id="d_plain")
    boosted = make_retrieved_chunk(chunk_id="boosted", document_id="d_boost")
    return plain, boosted


def test_boost_doc_ids_changes_aggregated_score_and_order():
    plain, boosted = _plain_and_boosted()
    merged, _ = bridge_mod._aggregate_task_results(
        [(_task("t1", boost_doc_ids={"d_boost"}), [plain, boosted])]
    )

    assert abs(merged["plain"].score - round(1 / 61, 6)) < 1e-6
    assert abs(
        merged["boosted"].score - round((1 / 62) * bridge_mod.DOC_BOOST_FACTOR, 6)
    ) < 1e-6
    ordered = sorted(merged.values(), key=lambda c: c.score, reverse=True)
    assert [chunk.chunk_id for chunk in ordered] == ["boosted", "plain"]


def test_boost_doc_ids_unknown_doc_is_noop():
    plain, boosted = _plain_and_boosted()
    with_unknown, _ = bridge_mod._aggregate_task_results(
        [(_task("t1", boost_doc_ids={"d_unknown"}), [plain, boosted])]
    )
    without_boost, _ = bridge_mod._aggregate_task_results(
        [(_task("t1"), [plain, boosted])]
    )
    assert {cid: c.score for cid, c in with_unknown.items()} == {
        cid: c.score for cid, c in without_boost.items()
    }


def test_boost_none_preserves_exact_h2b_scores():
    a = make_retrieved_chunk(chunk_id="a")
    b = make_retrieved_chunk(chunk_id="b")
    merged, _ = bridge_mod._aggregate_task_results(
        [(_task("t1"), [a, b]), (_task("t2"), [b])]
    )

    assert abs(merged["a"].score - 1 / 61) < 1e-6
    assert abs(merged["b"].score - (1 / 62 + 1 / 61)) < 1e-6


def test_postprocess_does_not_mutate_scores():
    task = _task(boost_doc_ids={"doc_1"})
    chunks = [
        make_retrieved_chunk(chunk_id="a", document_id="doc_1", score=0.06),
        make_retrieved_chunk(chunk_id="a", document_id="doc_1", score=0.08),
        make_retrieved_chunk(chunk_id="b", document_id="doc_1", score=0.5),
    ]

    processed = bridge_mod._postprocess_task_chunks(chunks, task)

    assert [chunk.chunk_id for chunk in processed] == ["a", "b"]
    # Сохранён лучший дубль, но score не изменён doc-boost-ом.
    assert processed[0] is chunks[1]
    assert processed[0].score == 0.08
    assert processed[1].score == 0.5


def test_doc_boost_factor_bounded():
    assert bridge_mod.DOC_BOOST_FACTOR == 1.1
    chunk = make_retrieved_chunk(chunk_id="a", document_id="d_boost")
    merged, _ = bridge_mod._aggregate_task_results(
        [
            (_task("t1", boost_doc_ids={"d_boost"}), [chunk]),
            (_task("t2", boost_doc_ids={"d_boost"}), [chunk]),
        ]
    )

    rrf = 1 / 61 + 1 / 61
    expected = round(rrf * bridge_mod.DOC_BOOST_FACTOR, 6)
    assert abs(merged["a"].score - expected) < 1e-6
    # Множитель применяется ровно один раз, а не возводится в степень.
    assert merged["a"].score < round(rrf * bridge_mod.DOC_BOOST_FACTOR ** 2, 6)


def test_trace_reports_doc_boost_diagnostics():
    def fast(_task):
        return (
            [
                make_retrieved_chunk(
                    chunk_id="plain", document_id="d_plain", score=0.5
                ),
                make_retrieved_chunk(
                    chunk_id="boosted", document_id="d_boost", score=0.5
                ),
            ],
            {},
            {},
            5,
        )

    with patch.object(
        bridge_mod, "prewarm_retrieval_backends", return_value={"errors": []}
    ), patch.object(bridge_mod, "_run_single_task", side_effect=fast):
        chunks, step = bridge_mod.retrieve_tasks(
            [_task("t1", boost_doc_ids={"d_boost"})]
        )

    assert step.extra["doc_boost_factor"] == bridge_mod.DOC_BOOST_FACTOR
    assert step.extra["boost_doc_ids_count"] == 1
    assert step.extra["doc_boost_applied"] == 1
    assert step.extra["aggregation"] == "sum_rrf"
    scores = {chunk.chunk_id: chunk.score for chunk in chunks}
    assert abs(scores["plain"] - round(1 / 61, 6)) < 1e-6
    assert abs(
        scores["boosted"] - round((1 / 62) * bridge_mod.DOC_BOOST_FACTOR, 6)
    ) < 1e-6


# --- H-2A: маршрутизация интентов ------------------------------------------


def _planned_tasks(intent: str, **overrides):
    spec = make_query_spec(intent=intent, **overrides)
    tasks, step = plan(spec)
    return tasks, step


def test_semantic_uses_hybrid_with_semantic_intent():
    tasks, _ = _planned_tasks("semantic")
    hybrids = [t for t in tasks if t.mode == "hybrid"]
    assert hybrids
    assert all(t.intent == "semantic_query" for t in hybrids)


def test_exact_lookup_uses_hybrid_with_exact_intent():
    tasks, _ = _planned_tasks("exact_lookup")
    assert [t.mode for t in tasks] == ["hybrid"]
    assert tasks[0].intent == "exact_lookup"


def test_mixed_uses_hybrid_with_mixed_intent_and_no_duplicates():
    tasks, _ = _planned_tasks(
        "mixed",
        rewritten_queries_bm25=["a", "b", "c"],
        rewritten_queries_dense=["b", "d", "e"],
    )
    assert all(t.mode == "hybrid" for t in tasks)
    assert all(t.intent == "mixed_query" for t in tasks)
    queries = [t.query for t in tasks]
    assert len(queries) == len(set(queries))
    assert queries == ["a", "b", "d"]


def test_date_version_keeps_hybrid_and_amendment_bm25():
    tasks, _ = _planned_tasks(
        "date_version",
        entities_raw=["ООО «Северный Склад»"],
    )
    hybrids = [t for t in tasks if t.mode == "hybrid"]
    assert hybrids
    assert all(t.intent == "mixed_query" for t in hybrids)
    amendment = [t for t in tasks if t.purpose == "parent_contract_and_amendments"]
    assert len(amendment) == 1
    assert amendment[0].mode == "bm25"
    assert "дополнительное соглашение" in amendment[0].query


def test_checklist_uses_hybrid_only():
    tasks, _ = _planned_tasks(
        "checklist",
        required_slots=["approval_process"],
        rewritten_queries_dense=["порядок согласования договора"],
    )
    assert tasks
    assert all(t.mode == "hybrid" for t in tasks)
    assert all(t.intent == "mixed_query" for t in tasks)


def test_compare_subquestion_hybrid_and_optional_bm25():
    decomposition = [
        SubQuestion(
            text="арендная плата по договору А-08/2024",
            target_doc_raw="А-08/2024",
            aspect="rent_amount",
        )
    ]
    tasks, _ = _planned_tasks("compare", decomposition=decomposition)
    hybrid = [t for t in tasks if t.mode == "hybrid"]
    assert hybrid
    assert all(t.intent == "mixed_query" for t in hybrid)
    # Для compare сохранён дополнительный лексический канал.
    assert any(t.mode == "bm25" for t in tasks)


# --- H-2A: проброс intent в hybrid_search и fallback -----------------------


def _run_single_task_capture(task: RetrievalTask, search_name: str):
    with patch.object(bridge_mod.config, "USE_STUB_RAG", False), patch.object(
        bridge_mod, "reset_embedding_diagnostics"
    ), patch.object(
        bridge_mod, "get_last_embedding_diagnostics", return_value={}
    ), patch.object(
        bridge_mod, search_name, return_value=[]
    ) as mock_search:
        bridge_mod._run_single_task(task)
    return mock_search


def test_task_intent_reaches_hybrid_search():
    mock = _run_single_task_capture(
        _task(mode="hybrid", intent="semantic_query"), "hybrid_search"
    )
    assert mock.call_args.kwargs["intent"] == "semantic_query"


def test_mode_fallback_hybrid_to_mixed_query():
    mock = _run_single_task_capture(_task(mode="hybrid", intent=None), "hybrid_search")
    assert mock.call_args.kwargs["intent"] == "mixed_query"


def test_mode_fallback_bm25_to_exact_lookup():
    mock = _run_single_task_capture(_task(mode="bm25", intent=None), "bm25_search")
    assert mock.call_args.kwargs["intent"] == "exact_lookup"


def test_mode_fallback_dense_to_semantic_query():
    mock = _run_single_task_capture(_task(mode="dense", intent=None), "dense_search")
    assert mock.call_args.kwargs["intent"] == "semantic_query"


# --- H-2C: условный boost допсоглашений ------------------------------------


def _patch_hybrid(chunks_map, bm25_results, dense_ids):
    class FakeCollection:
        def query(self, **kwargs):
            return {
                "ids": [list(dense_ids)],
                "documents": [["" for _ in dense_ids]],
                "metadatas": [[{} for _ in dense_ids]],
            }

    return (
        patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map),
        patch.object(
            hybrid_mod.bm25_store,
            "search_with_stats",
            return_value=(bm25_results, make_bm25_stats(bm25_results)),
        ),
        patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024),
        patch.object(
            hybrid_mod, "get_chroma_collection", return_value=FakeCollection()
        ),
        patch.object(hybrid_mod, "validate_query_embedding_dimension", return_value=None),
    )


def _amendment_chunks():
    return {
        "base": make_chunk_dict(
            chunk_id="base", document_id="base_doc", effective_from="2020-01-01"
        ),
        "amend": make_chunk_dict(
            chunk_id="amend",
            document_id="amend_1",
            amends_document_id="base_doc",
            effective_from="2020-01-01",
        ),
    }


def test_should_boost_amendments_requires_explicit_markers():
    assert should_boost_amendments("дополнительное соглашение к договору")
    assert should_boost_amendments("что изменилось в редакции договора")
    assert should_boost_amendments("изменения по договору поставки")
    assert not should_boost_amendments("какая претензия к поставщику")
    assert not should_boost_amendments("какие условия поставки товара")


def test_should_boost_amendments_date_version_needs_temporal_marker():
    assert should_boost_amendments("какая редакция действует в 2024 году", "date_version")
    # Без маркеров версии и без временной формулировки boost не нужен.
    assert not should_boost_amendments("как действует договор", "date_version")


def test_false_positive_against_supplier_claim_has_no_boost():
    chunks_map = _amendment_chunks()
    bm25_results = [("base", 5.0), ("amend", 4.9)]
    patches = _patch_hybrid(chunks_map, bm25_results, [])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        results = hybrid_mod.hybrid_search(
            query="какая претензия к поставщику по качеству товара",
            top_k=5,
            event_date="2025-01-01",
            selected_doc_ids={"base_doc"},
            intent="mixed_query",
            rerank=False,
        )
    scores = {chunk["chunk_id"]: chunk["score"] for chunk in results}
    assert abs(scores["amend"] - 0.5 / 62) < 1e-6
    assert results[0]["chunk_id"] == "base"


def test_explicit_amendment_query_still_boosts_and_filters_apply():
    chunks_map = _amendment_chunks()
    chunks_map["other"] = make_chunk_dict(
        chunk_id="other", document_id="other_doc", effective_from="2020-01-01"
    )
    # BM25-заглушка отдаёт только разрешённые документы; чужой документ
    # приходит из dense и должен быть отфильтрован по selected_doc_ids.
    bm25_results = [("base", 5.0), ("amend", 4.9)]
    dense_ids = ["other"]
    patches = _patch_hybrid(chunks_map, bm25_results, dense_ids)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        results = hybrid_mod.hybrid_search(
            query="дополнительное соглашение к договору аренды",
            top_k=10,
            event_date="2025-01-01",
            selected_doc_ids={"base_doc"},
            intent="mixed_query",
            rerank=False,
        )
    ids = [chunk["chunk_id"] for chunk in results]
    # selected_doc_ids расширяется на допсоглашение, но не на чужой документ.
    assert set(ids) == {"base", "amend"}
    scores = {chunk["chunk_id"]: chunk["score"] for chunk in results}
    assert abs(scores["amend"] - (0.5 / 62) * 1.15) < 1e-6
    assert results[0]["chunk_id"] == "amend"
