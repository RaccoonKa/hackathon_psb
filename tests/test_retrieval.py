from src import config
from src.retrieval.hybrid import detect_query_intent, resolve_allowed_versions_by_date
from src.retrieval.keyword_store import BM25KeywordStore, tokenize_ru
from src.retrieval.vector_store import _fallback_embedding, prepare_flat_metadata
from src.generation.retrieval.bridge import _dict_to_chunk, filter_chunks_by_threshold
from src.generation.types import RetrievedChunk
from tests._common import make_chunk_dict, make_retrieved_chunk


def test_tokenize_ru_basic():
    toks = tokenize_ru("Статья 13.11 КоАП РФ — штраф 300 000 руб.")
    assert "статья" in toks
    assert "13" in toks
    assert "11" in toks
    assert "коап" in toks
    assert "300" in toks


def test_tokenize_ru_empty():
    assert tokenize_ru("") == []
    assert tokenize_ru("!!! ??? ...") == []


def test_bm25_build_and_search():
    store = BM25KeywordStore()
    store.build_index([
        make_chunk_dict(chunk_id="c1", text="Срок оплаты 30 дней",
                        embedding_text="Договор поставки. Срок оплаты 30 дней"),
        make_chunk_dict(chunk_id="c2", text="Арендная плата 450000",
                        embedding_text="Договор аренды. Арендная плата 450000"),
        make_chunk_dict(chunk_id="c3", text="Пени 0,1% за просрочку",
                        embedding_text="Договор аренды. Пени 0,1% за просрочку"),
        make_chunk_dict(chunk_id="c4", text="Совершенно другой текст",
                        embedding_text="ТК РФ. Испытательный срок"),
        make_chunk_dict(chunk_id="c5", text="Ещё один документ",
                        embedding_text="НК РФ. Налоговые ставки"),
    ])
    results = store.search("срок", top_k=10)
    assert results
    ids = [cid for cid, _ in results]
    assert "c1" in ids
    assert "c4" in ids


def test_bm25_scores_are_positive():
    store = BM25KeywordStore()
    store.build_index([
        make_chunk_dict(chunk_id="c1", text="срок оплаты",
                        embedding_text="срок оплаты"),
        make_chunk_dict(chunk_id="d1", text="нечто", embedding_text="нечто первое"),
        make_chunk_dict(chunk_id="d2", text="нечто", embedding_text="нечто второе"),
        make_chunk_dict(chunk_id="d3", text="нечто", embedding_text="нечто третье"),
        make_chunk_dict(chunk_id="d4", text="нечто", embedding_text="нечто четвертое"),
    ])
    results = store.search("срок", top_k=10)
    assert results
    for _cid, score in results:
        assert score > 0


def test_bm25_uses_embedding_text_not_raw_text():
    store = BM25KeywordStore()
    store.build_index([
        make_chunk_dict(
            chunk_id="c_target",
            text="совершенно другой текст",
            embedding_text="Статья 112 НК РФ обстоятельства",
        ),
        make_chunk_dict(chunk_id="d1", text="раз", embedding_text="документ один"),
        make_chunk_dict(chunk_id="d2", text="два", embedding_text="документ два"),
        make_chunk_dict(chunk_id="d3", text="три", embedding_text="документ три"),
        make_chunk_dict(chunk_id="d4", text="четыре", embedding_text="документ четыре"),
    ])
    results = store.search("обстоятельства", top_k=10)
    assert results
    assert results[0][0] == "c_target"


def test_bm25_respects_allowed_chunk_ids():
    store = BM25KeywordStore()
    store.build_index([
        make_chunk_dict(chunk_id="c1", text="срок оплаты", embedding_text="срок оплаты"),
        make_chunk_dict(chunk_id="c2", text="срок оплаты", embedding_text="срок оплаты"),
        make_chunk_dict(chunk_id="d1", text="нечто", embedding_text="нечто первое"),
        make_chunk_dict(chunk_id="d2", text="нечто", embedding_text="нечто второе"),
        make_chunk_dict(chunk_id="d3", text="нечто", embedding_text="нечто третье"),
    ])
    results = store.search("срок", top_k=10, allowed_chunk_ids={"c2"})
    assert results
    assert all(cid == "c2" for cid, _ in results)


def test_bm25_empty_query_returns_empty():
    store = BM25KeywordStore()
    store.build_index([make_chunk_dict(chunk_id="c1")])
    assert store.search("", top_k=10) == []


def test_bm25_drops_non_positive_scores_known_behavior():
    store = BM25KeywordStore()
    store.build_index([
        make_chunk_dict(chunk_id="c1", text="срок", embedding_text="срок"),
        make_chunk_dict(chunk_id="c2", text="срок", embedding_text="срок"),
        make_chunk_dict(chunk_id="c3", text="срок", embedding_text="срок"),
    ])
    results = store.search("срок", top_k=10)
    assert results == []


def test_detect_query_intent_exact():
    cfg = detect_query_intent("Статья 112 НК РФ")
    assert cfg["intent"] == "exact_lookup"
    assert cfg["w_bm25"] > cfg["w_dense"]


def test_detect_query_intent_semantic():
    cfg = detect_query_intent("Что означает понятие сверхурочной работы?")
    assert cfg["intent"] == "semantic_query"
    assert cfg["w_dense"] > cfg["w_bm25"]


def test_detect_query_intent_mixed():
    cfg = detect_query_intent(
        "Статья 112 НК РФ: что означает понятие смягчающих обстоятельств?"
    )
    assert cfg["intent"] == "mixed_query"
    assert cfg["w_bm25"] == cfg["w_dense"]


def test_detect_query_intent_uses_precomputed():
    cfg = detect_query_intent("неважно", precomputed_intent="exact_lookup")
    assert cfg["intent"] == "exact_lookup"


def _build_date_map():
    return {
        "c_law": make_chunk_dict(
            chunk_id="c_law", document_id="doc_law", document_type="law",
            effective_from="2025-01-01", effective_to=None,
        ),
        "c_old": make_chunk_dict(
            chunk_id="c_old", document_id="doc_old", document_type="internal",
            effective_from="2023-01-01", effective_to="2025-03-01",
        ),
        "c_new": make_chunk_dict(
            chunk_id="c_new", document_id="doc_new", document_type="internal",
            effective_from="2025-03-01", effective_to=None,
        ),
        "c_amend": make_chunk_dict(
            chunk_id="c_amend", document_id="amend_1_contract_x",
            document_type="contract",
            effective_from="2024-09-03", effective_to=None,
            amends_document_id="contract_x",
        ),
        "c_unknown": make_chunk_dict(
            chunk_id="c_unknown", document_id="doc_unknown",
            effective_from=None, effective_to=None,
            effective_dates_known=False,
        ),
    }


def test_resolve_versions_for_2024():
    unknown = set()
    allowed = resolve_allowed_versions_by_date(
        "2024-06-01", _build_date_map(), validity_unknown=unknown,
    )
    assert "c_law" in allowed
    assert "c_old" in allowed
    assert "c_new" not in allowed
    assert "c_amend" not in allowed
    # Третье состояние: неизвестные даты не отсекаются, но помечаются.
    assert "c_unknown" in allowed
    assert "c_unknown" in unknown


def test_resolve_versions_for_2025():
    allowed = resolve_allowed_versions_by_date("2025-06-01", _build_date_map())
    assert "c_law" in allowed
    assert "c_old" not in allowed
    assert "c_new" in allowed
    assert "c_amend" in allowed


def test_resolve_versions_uses_year_from_query():
    allowed = resolve_allowed_versions_by_date(
        None, _build_date_map(), query="Что было в 2023 году?"
    )
    assert "c_old" in allowed
    assert "c_new" not in allowed


def test_resolve_versions_keeps_laws_even_before_eff_from():
    allowed = resolve_allowed_versions_by_date("2024-01-01", _build_date_map())
    assert "c_law" in allowed


def test_resolve_versions_expands_amendments_of_selected():
    allowed = resolve_allowed_versions_by_date(
        "2025-01-01", _build_date_map(), selected_doc_ids={"contract_x"},
    )
    assert "c_amend" in allowed
    assert "c_law" not in allowed


def test_resolve_versions_selected_excludes_unselected():
    allowed = resolve_allowed_versions_by_date(
        "2025-01-01", _build_date_map(), selected_doc_ids={"doc_law"},
    )
    assert "c_law" in allowed
    assert "c_old" not in allowed
    assert "c_new" not in allowed


def test_prepare_flat_metadata_drops_none_and_flattens():
    chunk = {
        "document_id": "d1",
        "version_id": "v1",
        "chunk_index": 0,
        "document_title": "T",
        "document_type": "contract",
        "scope": "global",
        "chat_id": None,
        "effective_from": "2024-01-01",
        "effective_to": None,
        "effective_dates_known": True,
        "source_hash": "sha256:abc",
        "amends_document_id": None,
        "chunker_version": "1.0",
        "structure": {"provision_id": "1.1"},
    }
    meta = prepare_flat_metadata(chunk)
    assert "chat_id" not in meta
    assert "effective_to" not in meta
    assert "amends_document_id" not in meta
    assert meta["document_id"] == "d1"
    assert meta["effective_dates_known"] is True
    assert meta["provision_id"] == "1.1"


def test_fallback_embedding_deterministic():
    a = _fallback_embedding("привет")
    b = _fallback_embedding("привет")
    c = _fallback_embedding("другой")
    assert a == b
    assert a != c
    assert len(a) == 1024


def test_dict_to_chunk_full():
    raw = make_chunk_dict(chunk_id="x", score=0.9)
    ch = _dict_to_chunk(raw)
    assert isinstance(ch, RetrievedChunk)
    assert ch.chunk_id == "x"
    assert ch.score == 0.9
    assert ch.provision_id == "1.1"
    assert ch.structure_path == ["Раздел 1"]


def test_dict_to_chunk_missing_structure_and_score():
    raw = {
        "chunk_id": "x", "text": "t", "document_id": "d", "version_id": "v",
        "document_title": "T",
    }
    ch = _dict_to_chunk(raw)
    assert ch.provision_id is None
    assert ch.structure_path == []
    assert ch.score == config.REAL_RAG_DEFAULT_SCORE


def test_filter_passes_above_threshold():
    chunks = [
        make_retrieved_chunk(chunk_id="a", score=0.9),
        make_retrieved_chunk(chunk_id="b", score=0.2),
    ]
    passed, out, step = filter_chunks_by_threshold(chunks, threshold=0.5)
    assert passed
    assert len(out) == 2
    assert step.step == "threshold_filter"


def test_filter_fails_when_empty():
    passed, out, _ = filter_chunks_by_threshold([], threshold=0.0)
    assert not passed
    assert out == []


def test_filter_adaptive_threshold_for_rrf_scores():
    chunks = [make_retrieved_chunk(chunk_id="a", score=0.03)]
    passed, out, step = filter_chunks_by_threshold(chunks, threshold=0.1)
    assert passed
    assert step.input["effective_threshold"] <= 0.01


def test_filter_below_threshold():
    chunks = [make_retrieved_chunk(chunk_id="a", score=0.4)]
    passed, out, _ = filter_chunks_by_threshold(chunks, threshold=0.6)
    assert not passed
    assert out == []