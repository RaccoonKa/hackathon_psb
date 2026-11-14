"""Сквозные проверки упрощённого retrieval pipeline.

Покрывает требования ТЗ (раздел 10):
* ровно одна ``hybrid``-задача на каждый intent;
* followup — одна hybrid-задача и дедупликация;
* в мосте нет пула потоков / ``ThreadPoolExecutor`` / второго RRF;
* BM25 и Vector вызываются внутри ``hybrid_search``;
* intent-dependent веса RRF (k=60) для всех intent;
* reranker вызывается после RRF с ``RERANKER_TOP_N``;
* selected_doc_ids/event_date сохраняются до hybrid_search.
"""

from pathlib import Path
from unittest.mock import patch

from rag_core import config
from rag_core.generation.planning.planner import followup, plan
from rag_core.generation.retrieval import bridge as bridge_mod
from rag_core.generation.retrieval import bridge as _bridge
from rag_core.generation.types import DocRef
from rag_core.retrieval import hybrid as hybrid_mod
from tests._common import (
    make_bm25_stats,
    make_chunk_dict,
    make_query_spec,
    make_retrieved_chunk,
)

INTENTS = [
    ("exact_lookup", "exact_lookup", "exact_lookup_hybrid"),
    ("mixed", "mixed_query", "mixed_hybrid"),
    ("semantic", "semantic_query", "semantic_hybrid"),
    ("compare", "mixed_query", "compare_hybrid"),
    ("date_version", "mixed_query", "date_version_hybrid"),
    ("checklist", "mixed_query", "checklist_hybrid"),
]


def _fake_collection(ids):
    class FakeCollection:
        def __init__(self):
            self.calls = 0

        def query(self, **kwargs):
            self.calls += 1
            return {
                "ids": [list(ids)],
                "documents": [["" for _ in ids]],
                "metadatas": [[{} for _ in ids]],
            }

    return FakeCollection()


# --- 1. Одна hybrid-задача на каждый intent --------------------------------


def test_plan_returns_exactly_one_hybrid_task_per_intent():
    for intent, expected_hybrid_intent, expected_purpose in INTENTS:
        spec = make_query_spec(
            intent=intent,
            required_slots=["rent_amount"],
            rewritten_queries_bm25=["запрос bm25"],
            rewritten_queries_dense=["запрос dense"],
        )
        tasks, step = plan(spec)
        assert len(tasks) == 1, f"{intent}: ожидалась ровно одна задача"
        task = tasks[0]
        assert task.mode == "hybrid"
        assert task.intent == expected_hybrid_intent
        assert task.purpose == expected_purpose
        assert step.output["tasks_count"] == 1
        assert step.output["hybrid_intent"] == expected_hybrid_intent


def test_plan_out_of_scope_returns_no_tasks():
    tasks, step = plan(make_query_spec(intent="out_of_scope"))
    assert tasks == []
    assert step.output["tasks_count"] == 0


# --- 2. Followup: одна hybrid-задача и дедупликация ------------------------


def test_followup_returns_single_hybrid_task():
    spec = make_query_spec(
        intent="semantic",
        required_slots=["penalty"],
        rewritten_queries_bm25=["оригинальный запрос"],
    )
    tasks, step = followup(["penalty"], spec)
    assert len(tasks) == 1
    assert tasks[0].mode == "hybrid"
    assert tasks[0].intent == "semantic_query"
    assert tasks[0].purpose == "followup_hybrid:penalty"
    assert step.output["tasks_count"] == 1
    assert step.output["slots_used"] == ["penalty"]


def test_followup_deduplicates_existing_query():
    spec = make_query_spec(
        intent="semantic",
        rewritten_queries_bm25=["оригинальный запрос"],
    )
    first, _ = followup(["rent_amount"], spec)
    assert first
    tasks, step = followup(
        ["rent_amount"],
        spec,
        existing_queries={first[0].query},
    )
    assert tasks == []
    assert step.output["skipped_duplicate_slots"] == ["rent_amount"]


# --- 3. В мосте нет пула потоков и второго RRF -----------------------------


def test_bridge_has_no_thread_pool_or_second_rrf():
    source = Path(_bridge.__file__).read_text(encoding="utf-8")
    assert "ThreadPoolExecutor" not in source
    assert "concurrent.futures" not in source
    assert "import threading" not in source
    assert not hasattr(bridge_mod, "_LAST_EXECUTOR")
    assert not hasattr(bridge_mod, "_EXECUTOR_LOCK")
    assert not hasattr(bridge_mod, "_aggregate_task_results")
    assert not hasattr(bridge_mod, "RRF_AGGREGATION_K")
    assert hasattr(bridge_mod, "_finalize_single_task_results")


def test_finalize_does_not_recompute_scores():
    merged, _ = bridge_mod._finalize_single_task_results(
        [
            (
                bridge_mod.RetrievalTask(
                    query="q",
                    mode="hybrid",
                    purpose="p",
                    selected_doc_ids=None,
                    boost_doc_ids=None,
                    event_date=None,
                    top_k=10,
                    intent="mixed_query",
                ),
                [make_retrieved_chunk(chunk_id="a", score=0.42)],
            )
        ]
    )
    assert merged["a"].score == 0.42


# --- 4. BM25 и Vector вызываются внутри hybrid_search ----------------------


def test_hybrid_search_calls_both_bm25_and_vector_channels():
    chunks_map = {
        "a": make_chunk_dict(chunk_id="a", effective_from="2020-01-01"),
        "b": make_chunk_dict(chunk_id="b", effective_from="2020-01-01"),
    }
    bm25_results = [("a", 5.0)]
    collection = _fake_collection(["b"])
    with patch.object(
        hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map
    ), patch.object(
        hybrid_mod.bm25_store,
        "search_with_stats",
        return_value=(bm25_results, make_bm25_stats(bm25_results)),
    ) as bm25_mock, patch.object(
        hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024
    ) as embed_mock, patch.object(
        hybrid_mod, "get_chroma_collection", return_value=collection
    ), patch.object(
        hybrid_mod, "validate_query_embedding_dimension", return_value=None
    ):
        results = hybrid_mod.hybrid_search(
            "q",
            top_k=10,
            intent="mixed_query",
            rerank=False,
        )

    assert bm25_mock.called
    assert embed_mock.called
    assert collection.calls == 1
    assert {chunk["chunk_id"] for chunk in results} == {"a", "b"}


# --- 5. Веса RRF по intent (k=60) ------------------------------------------


def _hybrid_scores(intent: str) -> dict[str, float]:
    chunks_map = {
        "a": make_chunk_dict(chunk_id="a", effective_from="2020-01-01"),
        "b": make_chunk_dict(chunk_id="b", effective_from="2020-01-01"),
    }
    bm25_results = [("a", 5.0)]
    with patch.object(
        hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map
    ), patch.object(
        hybrid_mod.bm25_store,
        "search_with_stats",
        return_value=(bm25_results, make_bm25_stats(bm25_results)),
    ), patch.object(
        hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024
    ), patch.object(
        hybrid_mod, "get_chroma_collection", return_value=_fake_collection(["b"])
    ), patch.object(
        hybrid_mod, "validate_query_embedding_dimension", return_value=None
    ):
        results = hybrid_mod.hybrid_search(
            "q",
            top_k=10,
            intent=intent,
            rerank=False,
        )
    return {chunk["chunk_id"]: chunk["score"] for chunk in results}


def test_intent_dependent_rrf_weights():
    exact = _hybrid_scores("exact_lookup")
    assert abs(exact["a"] - 0.7 / 61) < 1e-6
    assert abs(exact["b"] - 0.3 / 61) < 1e-6

    mixed = _hybrid_scores("mixed_query")
    assert abs(mixed["a"] - 0.5 / 61) < 1e-6
    assert abs(mixed["b"] - 0.5 / 61) < 1e-6

    semantic = _hybrid_scores("semantic_query")
    assert abs(semantic["a"] - 0.3 / 61) < 1e-6
    assert abs(semantic["b"] - 0.7 / 61) < 1e-6


# --- 6. Reranker после RRF и RERANKER_TOP_N --------------------------------


def test_hybrid_search_is_called_without_internal_rerank():
    captured: dict = {}

    def fake_hybrid(**kwargs):
        captured.update(kwargs)
        return []

    with patch.object(bridge_mod.config, "USE_STUB_RAG", False), patch.object(
        bridge_mod, "hybrid_search", side_effect=fake_hybrid
    ):
        bridge_mod._run_single_task(
            bridge_mod.RetrievalTask(
                query="q",
                mode="hybrid",
                purpose="p",
                selected_doc_ids=None,
                boost_doc_ids=None,
                event_date=None,
                top_k=10,
                intent="semantic_query",
            )
        )
    assert captured["rerank"] is False


def test_rerank_chunks_runs_after_retrieval_with_configured_top_n():
    chunks = [
        make_retrieved_chunk(chunk_id="c1", score=0.2),
        make_retrieved_chunk(chunk_id="c2", score=0.1),
    ]
    reranked_raw = [make_chunk_dict(chunk_id="c1", score=0.9)]
    with patch.object(
        bridge_mod, "rerank_candidates", return_value=reranked_raw
    ) as rerank_mock:
        out, step = bridge_mod.rerank_chunks(
            chunks, "q", top_n=int(config.RERANKER_TOP_N)
        )

    assert rerank_mock.call_args.kwargs["top_k"] == int(config.RERANKER_TOP_N)
    assert step.step == config.STEP_RERANKING
    assert [chunk.chunk_id for chunk in out] == ["c1"]


# --- 7. selected_doc_ids / event_date сохраняются --------------------------


def test_plan_routing_reaches_hybrid_search():
    spec = make_query_spec(
        intent="mixed",
        event_date="2024-12-31",
        entities_raw=["ООО «Северный Склад»"],
        entities_resolved=[
            DocRef(
                raw="ООО «Северный Склад»",
                document_id="doc_1",
                candidates=["doc_1"],
                resolved=True,
            )
        ],
        rewritten_queries_bm25=["арендная плата"],
        rewritten_queries_dense=["какая арендная плата"],
    )
    tasks, _ = plan(spec)
    assert len(tasks) == 1
    assert tasks[0].selected_doc_ids == {"doc_1"}
    assert tasks[0].event_date == "2024-12-31"

    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "hybrid_search", return_value=[]
    ) as mock_search:
        bridge_mod.retrieve_tasks(tasks)

    kwargs = mock_search.call_args.kwargs
    assert kwargs["selected_doc_ids"] == {"doc_1"}
    assert kwargs["event_date"] == "2024-12-31"
    assert kwargs["intent"] == "mixed_query"


# --- 8. Followup работает через последовательный мост ---------------------


def test_followup_task_runs_through_sequential_bridge():
    spec = make_query_spec(
        intent="semantic",
        required_slots=["penalty"],
        rewritten_queries_bm25=["оригинальный запрос"],
    )
    tasks, _ = followup(["penalty"], spec)
    assert len(tasks) == 1

    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod,
        "hybrid_search",
        return_value=[make_chunk_dict(chunk_id="c1", score=0.4)],
    ) as mock_search:
        chunks, step = bridge_mod.retrieve_tasks(tasks)

    assert [chunk.chunk_id for chunk in chunks] == ["c1"]
    assert step.extra["execution"] == "sequential"
    assert step.extra["aggregation"] == "none"
    assert step.extra["rrf_levels"] == 1
    assert mock_search.call_args.kwargs["intent"] == "semantic_query"
