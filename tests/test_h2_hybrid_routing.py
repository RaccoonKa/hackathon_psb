"""Тесты H-2: маршрутизация hybrid, последовательная финализация и boost.

Покрывают:
* H-2A — единая ``hybrid``-задача с явным ``intent`` для всех интентов и
  проброс ``task.intent`` в ``hybrid_search`` с fallback по ``mode``;
* последовательную финализацию без второго RRF (scores не пересчитываются);
* H-2C — boost допсоглашений только при явной релевантности запроса.
"""

from unittest.mock import patch

from rag_core.generation.planning import planner as planner_mod
from rag_core.generation.planning.planner import plan
from rag_core.generation.retrieval import bridge as bridge_mod
from rag_core.generation.types import DocRef, RetrievalTask, SubQuestion
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


def _fast(task, state=None):
    return ([make_retrieved_chunk(chunk_id="x")], {}, {}, 5)


# --- Последовательная финализация без второго RRF --------------------------


def test_finalize_single_task_results_keeps_best_score_without_recalculation():
    chunks = [
        make_retrieved_chunk(chunk_id="a", score=0.06),
        make_retrieved_chunk(chunk_id="a", score=0.08),
        make_retrieved_chunk(chunk_id="b", score=0.5),
    ]
    merged, _ = bridge_mod._finalize_single_task_results(
        [(_task("t1"), chunks)]
    )

    assert set(merged) == {"a", "b"}
    # Score лучшего дубля сохраняется, второго RRF нет.
    assert merged["a"].score == 0.08
    assert merged["b"].score == 0.5


def test_finalize_single_task_results_does_not_sum_task_scores():
    merged, _ = bridge_mod._finalize_single_task_results(
        [
            (_task("t1"), [make_retrieved_chunk(chunk_id="x", score=0.3)]),
            (_task("t2"), [make_retrieved_chunk(chunk_id="x", score=0.7)]),
        ]
    )
    # Не 0.3 + 0.7 и не RRF: просто лучший score.
    assert merged["x"].score == 0.7


def test_boost_doc_ids_changes_score_and_order_once():
    plain = make_retrieved_chunk(chunk_id="plain", document_id="d_plain", score=0.04)
    boosted = make_retrieved_chunk(
        chunk_id="boosted", document_id="d_boost", score=0.04
    )
    merged, _ = bridge_mod._finalize_single_task_results(
        [(_task("t1", boost_doc_ids={"d_boost"}), [plain, boosted])]
    )

    assert abs(merged["plain"].score - 0.04) < 1e-9
    assert abs(
        merged["boosted"].score
        - round(0.04 * bridge_mod.DOC_BOOST_FACTOR, 6)
    ) < 1e-6
    ordered = sorted(merged.values(), key=lambda c: c.score, reverse=True)
    assert [chunk.chunk_id for chunk in ordered] == ["boosted", "plain"]


def test_boost_doc_ids_unknown_doc_is_noop():
    plain = make_retrieved_chunk(chunk_id="plain", document_id="d_plain", score=0.1)
    boosted = make_retrieved_chunk(
        chunk_id="boosted", document_id="d_boost", score=0.2
    )
    with_unknown, _ = bridge_mod._finalize_single_task_results(
        [(_task("t1", boost_doc_ids={"d_unknown"}), [plain, boosted])]
    )
    without_boost, _ = bridge_mod._finalize_single_task_results(
        [(_task("t1"), [plain, boosted])]
    )
    assert {cid: c.score for cid, c in with_unknown.items()} == {
        cid: c.score for cid, c in without_boost.items()
    }


def test_protected_top_ids_are_first_three_chunks():
    chunks = [make_retrieved_chunk(chunk_id=f"c{idx}") for idx in range(6)]
    merged, protected = bridge_mod._finalize_single_task_results(
        [(_task("t1"), chunks[:4])]
    )
    assert protected == ["c0", "c1", "c2"]
    assert set(protected) <= set(merged)


def test_retrieve_tasks_trace_reports_single_rrf_and_no_aggregation():
    with patch.object(
        bridge_mod, "prewarm_retrieval_backends", return_value={"errors": []}
    ), patch.object(bridge_mod, "_run_single_task", side_effect=_fast):
        chunks, step = bridge_mod.retrieve_tasks([_task("t1")])

    assert step.extra["execution"] == "sequential"
    assert step.extra["aggregation"] == "none"
    assert step.extra["rrf_levels"] == 1
    assert step.extra["hybrid_rrf_k"] == 60
    assert "rrf_k" not in step.extra
    # Score из hybrid_search не пересчитывается.
    assert chunks[0].score == make_retrieved_chunk(chunk_id="x").score


def test_trace_reports_doc_boost_diagnostics():
    def fast(task, state=None):
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
    scores = {chunk.chunk_id: chunk.score for chunk in chunks}
    assert scores["plain"] == 0.5
    assert abs(
        scores["boosted"] - round(0.5 * bridge_mod.DOC_BOOST_FACTOR, 6)
    ) < 1e-6


def test_postprocess_does_not_mutate_scores():
    task = _task(boost_doc_ids={"doc_1"})
    chunks = [
        make_retrieved_chunk(chunk_id="a", document_id="doc_1", score=0.06),
        make_retrieved_chunk(chunk_id="a", document_id="doc_1", score=0.08),
        make_retrieved_chunk(chunk_id="b", document_id="doc_1", score=0.5),
    ]

    processed = bridge_mod._postprocess_task_chunks(chunks, task)

    assert [chunk.chunk_id for chunk in processed] == ["a", "b"]
    assert processed[0] is chunks[1]
    assert processed[0].score == 0.08
    assert processed[1].score == 0.5


def test_doc_boost_factor_is_bounded():
    assert bridge_mod.DOC_BOOST_FACTOR == 1.1


# --- H-2A: одна hybrid-задача на intent -------------------------------------


def _planned_tasks(intent: str, **overrides):
    spec = make_query_spec(intent=intent, **overrides)
    tasks, step = plan(spec)
    return tasks, step


def test_semantic_returns_single_hybrid_task():
    tasks, step = _planned_tasks("semantic")
    assert len(tasks) == 1
    assert tasks[0].mode == "hybrid"
    assert tasks[0].intent == "semantic_query"
    assert tasks[0].purpose == "semantic_hybrid"
    assert step.output["tasks_count"] == 1


def test_exact_lookup_returns_single_hybrid_task():
    tasks, _ = _planned_tasks("exact_lookup")
    assert len(tasks) == 1
    assert tasks[0].mode == "hybrid"
    assert tasks[0].intent == "exact_lookup"
    assert tasks[0].purpose == "exact_lookup_hybrid"


def test_mixed_combines_dense_then_bm25_without_duplicates():
    tasks, _ = _planned_tasks(
        "mixed",
        rewritten_queries_bm25=["a", "b", "c"],
        rewritten_queries_dense=["b", "d", "e"],
    )
    assert len(tasks) == 1
    assert tasks[0].intent == "mixed_query"
    assert tasks[0].purpose == "mixed_hybrid"
    # dense_q[:limit] → bm25_q[:limit], дедуп сохраняет первый вариант.
    assert tasks[0].query == "b d a"


def test_date_version_merges_amendment_bm25_into_single_query():
    tasks, _ = _planned_tasks(
        "date_version",
        entities_raw=["ООО «Северный Склад»"],
        event_date="2025-01-01",
    )
    assert len(tasks) == 1
    task = tasks[0]
    assert task.mode == "hybrid"
    assert task.intent == "mixed_query"
    assert task.purpose == "date_version_hybrid"
    assert task.event_date == "2025-01-01"
    assert "дополнительное соглашение" in task.query
    assert "ООО «Северный Склад»" in task.query


def test_checklist_uses_dense_only_without_bm25_rewrites():
    tasks, _ = _planned_tasks(
        "checklist",
        required_slots=["approval_process"],
        rewritten_queries_bm25=["bm25 unique"],
        rewritten_queries_dense=["порядок согласования договора"],
    )
    assert len(tasks) == 1
    assert tasks[0].mode == "hybrid"
    assert tasks[0].purpose == "checklist_hybrid"
    assert "порядок согласования договора" in tasks[0].query
    assert "bm25 unique" not in tasks[0].query


def test_compare_decomposition_is_merged_into_single_query():
    decomposition = [
        SubQuestion(
            text="арендная плата по договору А-08/2024",
            target_doc_raw="А-08/2024",
            aspect="rent_amount",
        ),
        SubQuestion(
            text="срок оплаты по договору А-08/2024",
            target_doc_raw="А-08/2024",
            aspect="payment_deadline",
        ),
    ]
    with patch.object(
        planner_mod,
        "resolve_single_entity",
        return_value=DocRef(
            raw="А-08/2024",
            document_id="contract_а-08-2024",
            candidates=["contract_а-08-2024"],
            resolved=True,
        ),
    ):
        tasks, _ = _planned_tasks("compare", decomposition=decomposition)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.mode == "hybrid"
    assert task.purpose == "compare_hybrid"
    assert "арендная плата по договору А-08/2024" in task.query
    assert "срок оплаты по договору А-08/2024" in task.query
    assert task.selected_doc_ids == {"contract_а-08-2024"}


def test_compare_ambiguous_target_goes_to_boost_not_selected():
    decomposition = [
        SubQuestion(text="сравни условия", target_doc_raw="Альфа-Трейд", aspect="rent_amount")
    ]
    with patch.object(
        planner_mod,
        "resolve_single_entity",
        return_value=DocRef(
            raw="Альфа-Трейд",
            document_id=None,
            candidates=["contract_п-17-2024", "contract_пт-05-2023"],
            resolved=False,
        ),
    ):
        tasks, _ = _planned_tasks("compare", decomposition=decomposition)

    assert tasks[0].selected_doc_ids is None
    assert tasks[0].boost_doc_ids == {
        "contract_п-17-2024",
        "contract_пт-05-2023",
    }


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
    assert mock.call_args.kwargs["rerank"] is False


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
    assert set(ids) == {"base", "amend"}
    scores = {chunk["chunk_id"]: chunk["score"] for chunk in results}
    assert abs(scores["amend"] - (0.5 / 62) * 1.15) < 1e-6
    assert results[0]["chunk_id"] == "amend"
