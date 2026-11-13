import tempfile
from pathlib import Path
from unittest.mock import patch

from rag_core.retrieval import keyword_store as ks_mod
from rag_core.retrieval.keyword_store import BM25KeywordStore, tokenize_ru
from tests._common import filler_corpus, make_chunk_dict


def test_tokenize_ru_lowercases():
    toks = tokenize_ru("Привет МИР")
    assert "привет" in toks
    assert "мир" in toks


def test_tokenize_ru_keeps_numbers():
    toks = tokenize_ru("Статья 112")
    assert "112" in toks


def test_tokenize_ru_keeps_latin():
    toks = tokenize_ru("Python код")
    assert "python" in toks
    assert "код" in toks


def test_tokenize_ru_ignores_punctuation():
    toks = tokenize_ru("Привет, мир! Как дела?")
    assert "привет" in toks
    assert "мир" in toks
    assert "как" in toks
    assert "дела" in toks


def test_tokenize_ru_empty():
    assert tokenize_ru("") == []


def test_tokenize_ru_only_punctuation():
    assert tokenize_ru("!@#$%^&*()") == []


def test_tokenize_ru_numbers():
    toks = tokenize_ru("13.11")
    assert "13" in toks
    assert "11" in toks


def test_bm25_empty_index_search_empty():
    store = BM25KeywordStore()
    store.build_index([])
    assert store.search("привет", top_k=10) == []


def test_bm25_single_document_returns_empty():
    store = BM25KeywordStore()
    store.build_index([
        make_chunk_dict(chunk_id="c1", embedding_text="уникальный термин"),
    ])
    results = store.search("уникальный", top_k=10)
    assert results == []


def test_bm25_two_docs_same_term_returns_empty():
    store = BM25KeywordStore()
    store.build_index([
        make_chunk_dict(chunk_id="c1", embedding_text="общий термин"),
        make_chunk_dict(chunk_id="c2", embedding_text="общий термин"),
    ])
    results = store.search("термин", top_k=10)
    assert results == []


def test_bm25_ten_docs_finds_rare_term():
    store = BM25KeywordStore()
    store.build_index(filler_corpus(n=10, term="уникальный"))
    results = store.search("уникальный", top_k=10)
    assert results
    assert results[0][0] == "target"


def test_bm25_ten_docs_all_scores_positive():
    store = BM25KeywordStore()
    store.build_index(filler_corpus(n=10, term="уникальный"))
    results = store.search("уникальный", top_k=10)
    for _cid, score in results:
        assert score > 0


def test_bm25_uses_embedding_text():
    store = BM25KeywordStore()
    chunks = filler_corpus(n=10, term="специфика")
    chunks.append(make_chunk_dict(
        chunk_id="raw_only",
        text="совершенно другой текст",
        embedding_text="совершенно другой текст",
    ))
    store.build_index(chunks)
    results = store.search("специфика", top_k=10)
    assert results
    assert results[0][0] == "target"


def test_bm25_allowed_chunk_ids_filter():
    store = BM25KeywordStore()
    store.build_index(filler_corpus(n=10, term="целевой"))
    results = store.search("целевой", top_k=10, allowed_chunk_ids={"target"})
    assert results
    assert all(cid == "target" for cid, _ in results)


def test_bm25_allowed_chunk_ids_empty_set():
    store = BM25KeywordStore()
    store.build_index(filler_corpus(n=10, term="целевой"))
    results = store.search("целевой", top_k=10, allowed_chunk_ids=set())
    assert results == []


def test_bm25_top_k_limits():
    store = BM25KeywordStore()
    store.build_index(filler_corpus(n=10, term="целевой"))
    results = store.search("целевой фоновый", top_k=2)
    assert len(results) <= 2


def test_bm25_empty_query_returns_empty():
    store = BM25KeywordStore()
    store.build_index(filler_corpus(n=10, term="целевой"))
    assert store.search("", top_k=10) == []


def test_bm25_only_punctuation_returns_empty():
    store = BM25KeywordStore()
    store.build_index(filler_corpus(n=10, term="целевой"))
    assert store.search("!!! ???", top_k=10) == []


def test_bm25_no_stemming_query_oplata_vs_oplaty():
    store = BM25KeywordStore()
    chunks = filler_corpus(n=10, term="другое")
    chunks.append(make_chunk_dict(
        chunk_id="oplaty",
        embedding_text="Срок оплаты 30 дней",
    ))
    store.build_index(chunks)
    results = store.search("оплата", top_k=10)
    assert results == []


def test_bm25_exact_form_matches():
    store = BM25KeywordStore()
    chunks = filler_corpus(n=10, term="другое")
    chunks.append(make_chunk_dict(
        chunk_id="oplaty",
        embedding_text="Срок оплаты 30 дней",
    ))
    store.build_index(chunks)
    results = store.search("оплаты", top_k=10)
    assert results
    assert results[0][0] == "oplaty"


def test_bm25_save_and_load():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with patch.object(ks_mod, "BM25_DIR", tmp):
            store = BM25KeywordStore()
            store.build_index(filler_corpus(n=10, term="уникальный"))
            store.save()
            assert (tmp / "bm25.pkl").exists()

            store2 = BM25KeywordStore()
            store2.load()
            results = store2.search("уникальный", top_k=10)
            assert results
            assert results[0][0] == "target"


def test_bm25_save_creates_dir():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "new_dir"
        with patch.object(ks_mod, "BM25_DIR", tmp):
            store = BM25KeywordStore()
            store.build_index(filler_corpus(n=10, term="уникальный"))
            store.save()
            assert tmp.exists()


def test_bm25_search_auto_loads_when_none():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with patch.object(ks_mod, "BM25_DIR", tmp):
            store = BM25KeywordStore()
            store.build_index(filler_corpus(n=10, term="уникальный"))
            store.save()

            fresh = BM25KeywordStore()
            results = fresh.search("уникальный", top_k=10)
            assert results


# --- M-1: search_with_stats (один вызов BM25) ------------------------------


def test_search_with_stats_calls_get_scores_once():
    store = BM25KeywordStore()
    store.build_index(filler_corpus(n=10, term="уникальный"))

    with patch.object(
        store.bm25, "get_scores", wraps=store.bm25.get_scores
    ) as mock_scores:
        results, stats = store.search_with_stats("уникальный", top_k=10)

    mock_scores.assert_called_once()
    assert results
    assert stats["returned_count"] == len(results)
    assert stats["candidates_before_filters"] >= len(results)
    # Без фильтров счётчики совпадают с числом кандидатов до фильтрации.
    assert stats["candidates_after_date_filter"] == stats["candidates_before_filters"]
    assert (
        stats["candidates_after_document_filter"]
        == stats["candidates_before_filters"]
    )


def test_search_with_stats_allowed_filter_counts():
    store = BM25KeywordStore()
    store.build_index(filler_corpus(n=10, term="уникальный"))

    results, stats = store.search_with_stats(
        "уникальный", top_k=10, allowed_chunk_ids={"target"}
    )

    assert results
    assert all(cid == "target" for cid, _score in results)
    assert stats["candidates_before_filters"] >= len(results)
    assert stats["candidates_after_document_filter"] == len(results)
    assert stats["returned_count"] == len(results)


def test_search_with_stats_empty_allowed_yields_no_results():
    store = BM25KeywordStore()
    store.build_index(filler_corpus(n=10, term="уникальный"))

    results, stats = store.search_with_stats(
        "уникальный", top_k=10, allowed_chunk_ids=set()
    )

    assert results == []
    assert stats["candidates_before_filters"] >= 1
    assert stats["candidates_after_document_filter"] == 0
    assert stats["returned_count"] == 0


def test_search_with_stats_date_filter_is_diagnostic_only():
    store = BM25KeywordStore()
    store.build_index(filler_corpus(n=10, term="уникальный"))

    results, stats = store.search_with_stats(
        "уникальный", top_k=10, date_allowed_chunk_ids=set()
    )

    # Фильтр по датам в search_with_stats влияет на диагностику, а не на выдачу:
    # итоговый search() исторически фильтрует только allowed_chunk_ids.
    assert results
    assert stats["candidates_before_filters"] >= 1
    assert stats["candidates_after_date_filter"] == 0
    assert stats["returned_count"] == len(results)


def test_search_with_stats_matches_search_results():
    store = BM25KeywordStore()
    store.build_index(filler_corpus(n=10, term="уникальный"))

    with_stats, _ = store.search_with_stats(
        "уникальный", top_k=10, allowed_chunk_ids={"target"}
    )
    wrapper = store.search("уникальный", top_k=10, allowed_chunk_ids={"target"})

    assert with_stats == wrapper


def test_search_with_stats_empty_index_returns_zero_counters():
    store = BM25KeywordStore()
    store.build_index([])

    results, stats = store.search_with_stats("привет", top_k=10)

    assert results == []
    assert stats["candidates_before_filters"] == 0
    assert stats["candidates_after_date_filter"] == 0
    assert stats["candidates_after_document_filter"] == 0
    assert stats["returned_count"] == 0