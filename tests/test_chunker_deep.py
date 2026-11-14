from rag_core.indexing.chunker import (
    HARD_MAX_TOKENS,
    TARGET_MAX_TOKENS,
    chunk_document_structural,
)
from rag_core.indexing.tokenizer_utils import count_tokens
from tests._common import all_docs, require_data


def _make_doc(content, document_type="contract", amends_document_id=None):
    return {
        "content": content,
        "metadata": {
            "document_id": "test_doc",
            "version_id": "test_doc_v1",
            "document_title": "Тестовый документ",
            "document_type": document_type,
            "source_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "chunker_version": "structural-1.0",
            "effective_from": None,
            "effective_to": None,
            "effective_dates_known": False,
            "amends_document_id": amends_document_id,
        },
    }


def test_oversized_block_is_split_into_multiple_chunks():
    long_body = "слово " * 2000
    content = f"1. Предмет договора\n1.1. {long_body}\n"
    chunks = chunk_document_structural(_make_doc(content))
    p11 = [c for c in chunks if c["structure"]["provision_id"] == "1.1"]
    assert len(p11) >= 2


def test_oversized_chunks_have_increasing_char_offsets():
    long_body = "слово " * 2000
    content = f"1. Предмет договора\n1.1. {long_body}\n"
    chunks = chunk_document_structural(_make_doc(content))
    p11 = [c for c in chunks if c["structure"]["provision_id"] == "1.1"]
    for i in range(len(p11) - 1):
        assert p11[i]["location"]["char_start"] < p11[i + 1]["location"]["char_start"]


def test_oversized_chunks_have_overlap():
    long_body = "слово " * 2000
    content = f"1. Предмет договора\n1.1. {long_body}\n"
    chunks = chunk_document_structural(_make_doc(content))
    p11 = [c for c in chunks if c["structure"]["provision_id"] == "1.1"]
    for i in range(len(p11) - 1):
        end_prev = p11[i]["location"]["char_end"]
        start_next = p11[i + 1]["location"]["char_start"]
        assert start_next < end_prev, "Соседние подчанки должны перекрываться"


def test_oversized_split_preserves_provision_id():
    long_body = "слово " * 2000
    content = f"1. Предмет договора\n1.1. {long_body}\n"
    chunks = chunk_document_structural(_make_doc(content))
    p11 = [c for c in chunks if c["structure"]["provision_id"] == "1.1"]
    for c in p11:
        assert c["structure"]["provision_id"] == "1.1"


def test_preamble_blocks_are_merged():
    content = (
        "Первый абзац преамбулы.\n"
        "Раздел I. Общие\n"
        "Второй абзац преамбулы.\n"
        "Раздел II. Особые\n"
        "Третий абзац преамбулы.\n"
    )
    chunks = chunk_document_structural(_make_doc(content))
    none_chunks = [c for c in chunks if c["structure"]["provision_id"] is None]
    assert len(none_chunks) == 1


def test_preamble_merge_respects_target_max():
    large_preamble = ("текст " * 200) + "\n"
    content = (
        large_preamble +
        "Раздел I\n" +
        large_preamble +
        "Раздел II\n" +
        large_preamble
    )
    chunks = chunk_document_structural(_make_doc(content))
    for c in chunks:
        assert count_tokens(c["text"]) <= TARGET_MAX_TOKENS + 50


def test_short_contract_not_bypassed():
    content = "1. Раздел\n1.1. Первый пункт\n1.2. Второй пункт\n"
    chunks = chunk_document_structural(_make_doc(content, document_type="contract"))
    assert len(chunks) >= 2


def test_short_internal_is_bypassed():
    content = "Короткий текст."
    chunks = chunk_document_structural(_make_doc(content, document_type="internal"))
    assert len(chunks) == 1
    assert chunks[0]["structure"]["path"] == ["Полный текст документа"]


def test_short_amendment_bypassed():
    content = "Короткий допник."
    chunks = chunk_document_structural(_make_doc(
        content, document_type="contract", amends_document_id="contract_x",
    ))
    assert len(chunks) == 1


def test_long_internal_not_bypassed():
    long_text = "слово " * 500
    content = long_text
    chunks = chunk_document_structural(_make_doc(content, document_type="internal"))
    assert len(chunks) >= 1
    assert chunks[0]["structure"]["path"] != ["Полный текст документа"]


def test_empty_contract_returns_no_chunks():
    chunks = chunk_document_structural(_make_doc("", document_type="contract"))
    assert chunks == []


def test_chunk_index_monotonic():
    require_data()
    for d in all_docs():
        chunks = chunk_document_structural(d)
        for i, c in enumerate(chunks):
            assert c["chunk_index"] == i, d["metadata"]["source_filename"]


def test_no_chunk_exceeds_hard_max():
    require_data()
    for d in all_docs():
        for c in chunk_document_structural(d):
            toks = count_tokens(c["text"])
            assert toks <= HARD_MAX_TOKENS + 100, f"{c['chunk_id']}: {toks}"


def test_char_offsets_within_document():
    require_data()
    for d in all_docs():
        doc_len = len(d["content"])
        for c in chunk_document_structural(d):
            assert 0 <= c["location"]["char_start"] <= doc_len
            assert 0 <= c["location"]["char_end"] <= doc_len + 10
            assert c["location"]["char_start"] <= c["location"]["char_end"]


def test_chunk_text_slice_matches_content():
    require_data()
    for d in all_docs():
        for c in chunk_document_structural(d):
            start = c["location"]["char_start"]
            end = c["location"]["char_end"]
            sliced = d["content"][start:end].strip()
            if sliced and c["text"]:
                head = c["text"].split("\n")[0][:30]
                assert head in sliced or sliced.startswith(head[:15])