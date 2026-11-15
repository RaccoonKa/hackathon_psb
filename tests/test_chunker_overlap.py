from unittest.mock import patch

from rag_core.indexing import chunker as chunker_mod
from rag_core.indexing.chunker import (
    HARD_MAX_TOKENS,
    OVERLAP_TOKENS,
    SUB_CHUNK_OVERLAP_CHARS,
    SUB_CHUNK_SIZE_CHARS,
    chunk_document_structural,
)
from rag_core.indexing.version import CHUNKER_VERSION


def _make_doc(content):
    return {
        "content": content,
        "metadata": {
            "document_id": "test_doc",
            "version_id": "test_doc_v1",
            "document_title": "Тестовый документ",
            "document_type": "contract",
            "source_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "chunker_version": CHUNKER_VERSION,
            "effective_from": None,
            "effective_to": None,
            "effective_dates_known": False,
            "amends_document_id": None,
        },
    }


def _oversized_content():
    body = "Слово " * 3000
    return f"1. Раздел\n1.1. {body}\n"


def test_sub_chunk_overlap_is_tokens_times_3():
    assert SUB_CHUNK_OVERLAP_CHARS == OVERLAP_TOKENS * 3


def test_sub_chunk_size_is_hard_max_times_3():
    assert SUB_CHUNK_SIZE_CHARS == HARD_MAX_TOKENS * 3


def test_oversized_uses_splitter():
    parts = ["part-a", "part-b", "part-c"]
    with patch.object(chunker_mod, "_oversized_splitter") as mock_splitter:
        mock_splitter.split_text.return_value = parts
        chunk_document_structural(_make_doc(_oversized_content()))
    mock_splitter.split_text.assert_called_once()


def test_oversized_char_start_formula():
    parts = ["A" * 500, "B" * 500, "C" * 500]
    with patch.object(chunker_mod, "_oversized_splitter") as mock_splitter:
        mock_splitter.split_text.return_value = parts
        chunks = chunk_document_structural(_make_doc(_oversized_content()))
    p11 = [c for c in chunks if c["structure"]["provision_id"] == "1.1"]
    assert len(p11) == 3
    first_start = p11[0]["location"]["char_start"]
    assert p11[0]["location"]["char_end"] == first_start + 500
    assert p11[1]["location"]["char_start"] == first_start + 500 - SUB_CHUNK_OVERLAP_CHARS
    assert p11[1]["location"]["char_end"] == p11[1]["location"]["char_start"] + 500
    assert p11[2]["location"]["char_start"] == p11[1]["location"]["char_start"] + 500 - SUB_CHUNK_OVERLAP_CHARS
    assert p11[2]["location"]["char_end"] == p11[2]["location"]["char_start"] + 500


def test_oversized_char_start_with_exact_overlap():
    parts = ["X" * SUB_CHUNK_OVERLAP_CHARS, "Y" * SUB_CHUNK_OVERLAP_CHARS]
    with patch.object(chunker_mod, "_oversized_splitter") as mock_splitter:
        mock_splitter.split_text.return_value = parts
        chunks = chunk_document_structural(_make_doc(_oversized_content()))
    p11 = [c for c in chunks if c["structure"]["provision_id"] == "1.1"]
    assert len(p11) == 2
    first_start = p11[0]["location"]["char_start"]
    assert p11[0]["location"]["char_end"] == first_start + SUB_CHUNK_OVERLAP_CHARS
    assert p11[1]["location"]["char_start"] == first_start + 1
    assert p11[1]["location"]["char_end"] == p11[1]["location"]["char_start"] + SUB_CHUNK_OVERLAP_CHARS


def test_oversized_short_part_uses_max_one():
    parts = ["Z" * 10, "W" * 10, "V" * 10]
    with patch.object(chunker_mod, "_oversized_splitter") as mock_splitter:
        mock_splitter.split_text.return_value = parts
        chunks = chunk_document_structural(_make_doc(_oversized_content()))
    p11 = [c for c in chunks if c["structure"]["provision_id"] == "1.1"]
    assert len(p11) == 3
    step = p11[1]["location"]["char_start"] - p11[0]["location"]["char_start"]
    assert step == 1
    step2 = p11[2]["location"]["char_start"] - p11[1]["location"]["char_start"]
    assert step2 == 1


def test_oversized_char_end_equals_start_plus_len_part():
    parts = ["A" * 1000, "B" * 750, "C" * 250]
    with patch.object(chunker_mod, "_oversized_splitter") as mock_splitter:
        mock_splitter.split_text.return_value = parts
        chunks = chunk_document_structural(_make_doc(_oversized_content()))
    p11 = [c for c in chunks if c["structure"]["provision_id"] == "1.1"]
    for i, part in enumerate(parts):
        assert p11[i]["location"]["char_end"] - p11[i]["location"]["char_start"] == len(part)


def test_oversized_char_start_monotonic_for_long_parts():
    parts = ["A" * 500, "B" * 500, "C" * 500, "D" * 500]
    with patch.object(chunker_mod, "_oversized_splitter") as mock_splitter:
        mock_splitter.split_text.return_value = parts
        chunks = chunk_document_structural(_make_doc(_oversized_content()))
    p11 = [c for c in chunks if c["structure"]["provision_id"] == "1.1"]
    starts = [c["location"]["char_start"] for c in p11]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_oversized_short_part_may_repeat_start():
    parts = ["Z" * 10, "W" * 10]
    with patch.object(chunker_mod, "_oversized_splitter") as mock_splitter:
        mock_splitter.split_text.return_value = parts
        chunks = chunk_document_structural(_make_doc(_oversized_content()))
    p11 = [c for c in chunks if c["structure"]["provision_id"] == "1.1"]
    assert p11[0]["location"]["char_start"] == p11[1]["location"]["char_start"] - 1


def test_oversized_preserves_provision_and_path():
    parts = ["A" * 500, "B" * 500]
    with patch.object(chunker_mod, "_oversized_splitter") as mock_splitter:
        mock_splitter.split_text.return_value = parts
        chunks = chunk_document_structural(_make_doc(_oversized_content()))
    p11 = [c for c in chunks if c["structure"]["provision_id"] == "1.1"]
    for c in p11:
        assert c["structure"]["provision_id"] == "1.1"
        assert any("Раздел" in p for p in c["structure"]["path"])


def test_oversized_text_equals_part_stripped():
    parts = ["A" * 500, "B" * 500]
    with patch.object(chunker_mod, "_oversized_splitter") as mock_splitter:
        mock_splitter.split_text.return_value = parts
        chunks = chunk_document_structural(_make_doc(_oversized_content()))
    p11 = [c for c in chunks if c["structure"]["provision_id"] == "1.1"]
    for i, part in enumerate(parts):
        assert p11[i]["text"] == part.strip()