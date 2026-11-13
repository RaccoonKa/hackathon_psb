import tempfile
from pathlib import Path
from unittest.mock import patch

from rag_core.retrieval import keyword_store as ks_mod
from rag_core.retrieval.keyword_store import BM25KeywordStore
from tests._common import make_chunk_dict


def test_results_sorted_descending():
    chunks = [
        make_chunk_dict(chunk_id=f"d{i}", embedding_text=f"общий текст {i}")
        for i in range(10)
    ]
    chunks.append(make_chunk_dict(
        chunk_id="single", embedding_text="редкий термин один раз",
    ))
    chunks.append(make_chunk_dict(
        chunk_id="multi", embedding_text="редкий термин термин термин",
    ))
    store = BM25KeywordStore()
    store.build_index(chunks)
    results = store.search("редкий", top_k=10)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_multi_occurrence_ranks_higher():
    chunks = [
        make_chunk_dict(chunk_id=f"d{i}", embedding_text=f"общий текст {i}")
        for i in range(10)
    ]
    chunks.append(make_chunk_dict(
        chunk_id="single", embedding_text="редкий термин",
    ))
    chunks.append(make_chunk_dict(
        chunk_id="multi", embedding_text="редкий редкий редкий термин",
    ))
    store = BM25KeywordStore()
    store.build_index(chunks)
    results = store.search("редкий", top_k=10)
    ids = [cid for cid, _ in results]
    assert ids.index("multi") < ids.index("single")


def test_save_load_preserves_chunk_ids_order():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with patch.object(ks_mod, "BM25_DIR", tmp):
            chunks = [
                make_chunk_dict(chunk_id=f"c{i}", embedding_text=f"уникальный терм{i}")
                for i in range(10)
            ]
            store = BM25KeywordStore()
            store.build_index(chunks)
            store.save()

            fresh = BM25KeywordStore()
            fresh.load()
            assert fresh.chunk_ids == store.chunk_ids


def test_save_load_preserves_scores():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with patch.object(ks_mod, "BM25_DIR", tmp):
            chunks = [
                make_chunk_dict(chunk_id=f"d{i}", embedding_text=f"фон {i}")
                for i in range(10)
            ]
            chunks.append(make_chunk_dict(chunk_id="target", embedding_text="специфика"))
            store = BM25KeywordStore()
            store.build_index(chunks)
            store.save()

            fresh = BM25KeywordStore()
            fresh.load()
            assert fresh.search("специфика", top_k=10) == store.search("специфика", top_k=10)


def test_bm25_save_creates_directory():
    with tempfile.TemporaryDirectory() as td:
        nested = Path(td) / "deep" / "nested"
        with patch.object(ks_mod, "BM25_DIR", nested):
            store = BM25KeywordStore()
            store.build_index([make_chunk_dict(chunk_id="c", embedding_text="текст")])
            store.save()
        assert nested.exists()
        assert (nested / "bm25.pkl").exists()