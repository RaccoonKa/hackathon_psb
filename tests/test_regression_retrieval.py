"""Регрессионные тесты поискового слоя (пункты 1–8, 14, 15).

Все проверки выполняются на поведении кода, а не на заранее подставленном
ответе. Тесты, которым нужен реальный индекс (BM25/chunks.jsonl), корректно
пропускаются, если артефакты отсутствуют.
"""

import json
from pathlib import Path
from unittest import SkipTest
from unittest.mock import patch

from rag_core.generation.analysis.entity_resolver import (
    extract_doc_routing_sets,
    resolve_single_entity,
)
from rag_core.generation.retrieval import bridge as bridge_mod
from rag_core.generation.retrieval import evidence as evidence_mod
from rag_core.generation.types import (
    Ambiguity,
    QuerySpec,
    RetrievalTask,
    Sufficiency,
    TaskType,
)
from rag_core.retrieval import hybrid as hybrid_mod
from rag_core.retrieval.hybrid import (
    bm25_search,
    hybrid_search,
    load_canonical_chunks_map,
    resolve_allowed_versions_by_date,
)
from tests._common import make_chunk_dict, make_query_spec

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
BM25_PATH = ROOT / "storage" / "bm25_index" / "bm25.pkl"

CONTRACT_DOC_ID = "contract_а-08-2024"
RENT_CHUNK_ID = "contract_а-08-2024_v1_0009_65851a01"
DEADLINE_CHUNK_ID = "contract_а-08-2024_v1_0010_3b16c866"


def _require_real_index() -> None:
    if not CHUNKS_PATH.exists() or not BM25_PATH.exists():
        raise SkipTest(
            "Нет реального индекса для регрессионного теста: "
            f"{CHUNKS_PATH} / {BM25_PATH}"
        )


def _deadline_question() -> str:
    return (
        "Какая арендная плата по договору с ООО «Северный Склад» "
        "и до какого числа её нужно вносить?"
    )


def _chunk_dict(chunk_id: str) -> dict:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("chunk_id") == chunk_id:
                return chunk
    raise AssertionError(f"Чанк {chunk_id} не найден в {CHUNKS_PATH}")


# --- 1. Поиск договора по контрагенту -------------------------------------


def test_entity_resolver_finds_rental_contract_by_counterparty():
    _require_real_index()
    ref = resolve_single_entity("ООО «Северный Склад»")
    assert ref.resolved is True
    assert ref.document_id == CONTRACT_DOC_ID
    assert ref.candidates == [CONTRACT_DOC_ID]

    selected, boost = extract_doc_routing_sets([ref])
    assert selected == {CONTRACT_DOC_ID}
    assert boost is None

    chunks_map = load_canonical_chunks_map()
    index_doc_ids = {chunk["document_id"] for chunk in chunks_map.values()}
    assert selected <= index_doc_ids
    assert CONTRACT_DOC_ID in index_doc_ids


# --- 2–3. Содержимое найденных чанков --------------------------------------


def test_search_returns_chunk_with_rent_amount():
    _require_real_index()
    results = bm25_search(
        "арендная плата 450 000 рублей в месяц включая НДС Северный Склад",
        top_k=10,
        selected_doc_ids={CONTRACT_DOC_ID},
        rerank=False,
    )
    hits = [
        chunk
        for chunk in results
        if chunk["document_id"] == CONTRACT_DOC_ID
        and "450 000" in chunk["text"]
        and "рублей в месяц" in chunk["text"]
        and "НДС" in chunk["text"]
    ]
    assert hits, "BM25 не нашёл чанк про 450 000 руб./мес. с НДС"
    assert any(
        (chunk.get("structure") or {}).get("provision_id") == "3.1"
        for chunk in hits
    )


def test_search_returns_chunk_with_payment_deadline():
    _require_real_index()
    results = bm25_search(
        "срок внесения арендной платы не позднее 5-го числа текущего месяца",
        top_k=10,
        selected_doc_ids={CONTRACT_DOC_ID},
        rerank=False,
    )
    hits = [
        chunk
        for chunk in results
        if chunk["document_id"] == CONTRACT_DOC_ID
        and "5-го числа текущего месяца" in chunk["text"]
        and "не позднее" in chunk["text"]
    ]
    assert hits, "BM25 не нашёл чанк про срок оплаты"
    assert any(
        (chunk.get("structure") or {}).get("provision_id") == "3.2"
        for chunk in hits
    )


# --- 4–5. Трёхсостояние дат ------------------------------------------------


def test_unknown_effective_dates_are_not_excluded_without_event_date():
    chunks_map = {
        "u_unknown": make_chunk_dict(
            chunk_id="u_unknown",
            document_id="doc_unknown",
            effective_from=None,
            effective_to=None,
            effective_dates_known=False,
        ),
        "u_empty": make_chunk_dict(
            chunk_id="u_empty",
            document_id="doc_empty",
            effective_from=None,
            effective_to=None,
            effective_dates_known=True,
        ),
        "known": make_chunk_dict(
            chunk_id="known",
            document_id="doc_known",
            effective_from="2020-01-01",
            effective_to=None,
        ),
    }
    unknown: set[str] = set()
    allowed = resolve_allowed_versions_by_date(
        None, chunks_map, validity_unknown=unknown
    )
    assert "u_unknown" in allowed
    assert "u_unknown" in unknown
    assert "u_empty" in allowed
    assert "u_empty" in unknown
    assert "known" in allowed


def test_expired_and_repealed_document_is_excluded_by_check_date():
    _require_real_index()
    chunks_map = load_canonical_chunks_map()
    expired_doc_id = (
        "doc_регламент-согласования-договоров-ред-2023--утратил-силу"
    )
    expired_cids = [
        cid
        for cid, chunk in chunks_map.items()
        if chunk["document_id"] == expired_doc_id
    ]
    assert expired_cids

    expired_chunk = chunks_map[expired_cids[0]]
    expired_text = expired_chunk["text"].lower()
    assert "утратил" in expired_text and "силу" in expired_text
    assert expired_chunk["effective_to"] == "2025-03-01"

    allowed_after = resolve_allowed_versions_by_date("2025-06-01", chunks_map)
    assert all(cid not in allowed_after for cid in expired_cids)

    allowed_before = resolve_allowed_versions_by_date("2024-06-01", chunks_map)
    assert all(cid in allowed_before for cid in expired_cids)

    # Граница effective_to исключительна.
    boundary = resolve_allowed_versions_by_date("2025-03-01", chunks_map)
    assert all(cid not in boundary for cid in expired_cids)


# --- 6–7. Раздельные каналы ------------------------------------------------


def _task(mode: str, top_k: int = 5) -> RetrievalTask:
    return RetrievalTask(
        query=_deadline_question(),
        mode=mode,
        purpose=f"{mode}_regression",
        selected_doc_ids={CONTRACT_DOC_ID},
        boost_doc_ids=None,
        event_date=None,
        top_k=top_k,
    )


def test_bm25_mode_does_not_touch_dense_channel():
    _require_real_index()
    diagnostics: dict = {}
    fake_bm25 = [(RENT_CHUNK_ID, 5.0), (DEADLINE_CHUNK_ID, 4.0)]

    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        hybrid_mod,
        "get_query_embedding",
        side_effect=AssertionError("dense-канал не должен вызываться в mode=bm25"),
    ), patch.object(
        hybrid_mod,
        "get_chroma_collection",
        side_effect=AssertionError("Chroma не должна вызываться в mode=bm25"),
    ), patch.object(
        hybrid_mod.bm25_store, "search", return_value=fake_bm25
    ):
        chunks, _step = bridge_mod.retrieve_tasks(
            [_task("bm25")], diagnostics_out=diagnostics
        )

    assert chunks
    assert {chunk.chunk_id for chunk in chunks} == {
        RENT_CHUNK_ID,
        DEADLINE_CHUNK_ID,
    }
    entry = diagnostics["tasks"][0]
    assert entry["status"] == "ok"
    assert entry["embeddings"]["provider"] == "none"
    assert entry["search"]["dense_candidates_count"] == 0
    assert diagnostics["executor"]["completed"] == 1


def test_dense_mode_does_not_touch_bm25_channel():
    _require_real_index()

    class FakeCollection:
        def query(self, **kwargs):
            return {
                "ids": [[RENT_CHUNK_ID, DEADLINE_CHUNK_ID]],
                "documents": [["", ""]],
                "metadatas": [[{}, {}]],
            }

    diagnostics: dict = {}
    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        hybrid_mod,
        "bm25_store",
    ) as fake_bm25_store, patch.object(
        hybrid_mod,
        "get_query_embedding",
        return_value=[0.0] * 1024,
    ), patch.object(
        hybrid_mod,
        "get_chroma_collection",
        return_value=FakeCollection(),
    ), patch.object(
        hybrid_mod,
        "validate_query_embedding_dimension",
        return_value=None,
    ):
        fake_bm25_store.search.side_effect = AssertionError(
            "BM25 не должен вызываться в mode=dense"
        )
        chunks, _step = bridge_mod.retrieve_tasks(
            [_task("dense")], diagnostics_out=diagnostics
        )

    assert chunks
    entry = diagnostics["tasks"][0]
    assert entry["status"] == "ok"
    assert entry["search"]["bm25_candidates_count"] == 0
    assert entry["search"]["dense_candidates_count"] >= 1
    assert diagnostics["executor"]["completed"] == 1


# --- 8. RRF, дедупликация, буст соглашений, лимит пула --------------------


def _rrf_chunks_map() -> dict:
    return {
        "base": make_chunk_dict(
            chunk_id="base",
            document_id="base_doc",
            effective_from="2020-01-01",
        ),
        "amend": make_chunk_dict(
            chunk_id="amend",
            document_id="amend_1",
            amends_document_id="base_doc",
            effective_from="2020-01-01",
        ),
        "bm25_only": make_chunk_dict(
            chunk_id="bm25_only",
            document_id="doc_bm25",
            effective_from="2020-01-01",
        ),
        "dense_only": make_chunk_dict(
            chunk_id="dense_only",
            document_id="doc_dense",
            effective_from="2020-01-01",
        ),
    }


class _FakeCollectionIds:
    def __init__(self, ids):
        self._ids = list(ids)

    def query(self, **kwargs):
        return {
            "ids": [list(self._ids)],
            "documents": [["" for _ in self._ids]],
            "metadatas": [[{} for _ in self._ids]],
        }


def test_hybrid_search_rrf_dedup_and_amendment_priority():
    chunks_map = _rrf_chunks_map()
    bm25_results = [("base", 9.0), ("amend", 8.0), ("bm25_only", 7.0)]
    dense_ids = ["amend", "base", "dense_only"]

    with patch.object(
        hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map
    ), patch.object(
        hybrid_mod.bm25_store, "search", return_value=bm25_results
    ), patch.object(
        hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024
    ), patch.object(
        hybrid_mod,
        "get_chroma_collection",
        return_value=_FakeCollectionIds(dense_ids),
    ), patch.object(
        hybrid_mod, "validate_query_embedding_dimension", return_value=None
    ):
        results = hybrid_search(
            "срок оплаты аренды",
            top_k=10,
            intent="mixed_query",
            rerank=False,
        )

    ids = [chunk["chunk_id"] for chunk in results]
    assert len(ids) == len(set(ids)), "RRF не должен дублировать chunk_id"
    assert set(ids) == {"base", "amend", "bm25_only", "dense_only"}
    # Кандидат, найденный обоими каналами, выше одноканальных.
    assert ids.index("base") < ids.index("bm25_only")
    assert ids.index("amend") < ids.index("dense_only")
    # Доп. соглашение получает приоритет над базовым договором.
    assert ids[0] == "amend"
    assert ids[1] == "base"
    scores = {chunk["chunk_id"]: chunk["score"] for chunk in results}
    assert scores["amend"] > scores["base"] > scores["bm25_only"]
    assert scores["amend"] > scores["dense_only"]
    for chunk in results:
        assert chunk["score"] > 0


def test_hybrid_search_candidate_pool_is_limited_to_40():
    chunks_map = {
        f"c{idx}": make_chunk_dict(
            chunk_id=f"c{idx}",
            document_id=f"doc{idx}",
            effective_from="2020-01-01",
        )
        for idx in range(60)
    }
    bm25_results = [(f"c{idx}", 10.0 - idx * 0.01) for idx in range(60)]

    with patch.object(
        hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map
    ), patch.object(
        hybrid_mod.bm25_store, "search", return_value=bm25_results
    ), patch.object(
        hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024
    ), patch.object(
        hybrid_mod,
        "get_chroma_collection",
        return_value=_FakeCollectionIds([]),
    ), patch.object(
        hybrid_mod, "validate_query_embedding_dimension", return_value=None
    ):
        results = hybrid_search(
            "срок оплаты",
            top_k=60,
            intent="mixed_query",
            rerank=False,
        )

    assert len(results) == 40
    ids = [chunk["chunk_id"] for chunk in results]
    assert len(ids) == len(set(ids))


# --- 14. Entity resolver: негативные и неоднозначные случаи ----------------


def test_entity_resolver_unknown_entity_returns_none_routing():
    _require_real_index()
    ref = resolve_single_entity("ООО «Совершенно Неизвестная Организация 999»")
    assert ref.resolved is False
    assert ref.document_id is None
    assert ref.candidates == []

    selected, boost = extract_doc_routing_sets([ref])
    assert selected is None
    assert boost is None
    assert selected != set()


def test_entity_resolver_resolves_contract_by_number_and_title():
    _require_real_index()
    by_number = resolve_single_entity("А-08/2024")
    assert by_number.resolved is True
    assert by_number.document_id == CONTRACT_DOC_ID

    by_title = resolve_single_entity("Договор аренды А-08/2024")
    assert by_title.resolved is True
    assert by_title.document_id == CONTRACT_DOC_ID


def test_entity_resolver_ambiguous_entity_does_not_narrow_to_empty():
    _require_real_index()
    # «Альфа-Трейд» неоднозначен: Альфа-Трейд / Альфа-Трейдинг + соглашение.
    ref = resolve_single_entity("Альфа-Трейд")
    assert ref.resolved is False
    assert ref.document_id is None
    assert len(ref.candidates) >= 2

    selected, boost = extract_doc_routing_sets([ref])
    assert selected is None, "Неоднозначность не должна сужать поиск до пустого"
    assert boost is not None
    assert len(boost) >= 2


# --- 15. Evidence assessment на реальных чанках ---------------------------


def test_evidence_assessment_covers_both_rent_slots_on_real_chunks():
    _require_real_index()

    raw_rent = _chunk_dict(RENT_CHUNK_ID)
    raw_deadline = _chunk_dict(DEADLINE_CHUNK_ID)
    raw_rent["score"] = 0.9
    raw_deadline["score"] = 0.8
    chunks = [
        bridge_mod._dict_to_chunk(raw_rent),
        bridge_mod._dict_to_chunk(raw_deadline),
    ]

    spec = make_query_spec(
        intent="mixed",
        required_slots=["rent_amount", "payment_deadline"],
        event_date=None,
        entities_raw=["ООО «Северный Склад»"],
    )
    bundle, assessment, step = evidence_mod.assess(spec, chunks)

    assert "rent_amount" in assessment.covered_slots
    assert "payment_deadline" in assessment.covered_slots
    assert assessment.missing_slots == []
    assert assessment.reason != "core_slot_not_covered"
    assert assessment.sufficiency != Sufficiency.INSUFFICIENT
    assert bundle.slot_map["rent_amount"]
    assert bundle.slot_map["payment_deadline"]
    assert step.output["slot_match_counts"]["rent_amount"] >= 1
    assert step.output["slot_match_counts"]["payment_deadline"] >= 1
    assert step.output["checked_chunks_count"] == 2


def test_query_spec_helper_is_constructible():
    """Страховка от расхождения типов: используем реальный QuerySpec."""
    spec = QuerySpec(
        intent="mixed",
        task_type=TaskType.LOOKUP,
        entities_raw=[],
        entities_resolved=[],
        article_refs=[],
        dates=[],
        event_date=None,
        premises=[],
        required_slots=["general"],
        assumptions=[],
        ambiguity=Ambiguity.LOW,
        needs_clarification=False,
        decomposition=[],
        rewritten_queries_bm25=["запрос"],
        rewritten_queries_dense=["запрос"],
        history_resolved=False,
        degradation_reason=None,
    )
    assert spec.required_slots == ["general"]
