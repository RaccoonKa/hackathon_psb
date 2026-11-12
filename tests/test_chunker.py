import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.indexing import chunker as chunker_mod
from src.indexing.chunker import (
    HARD_MAX_TOKENS,
    chunk_document_structural,
    generate_chunk_id,
    save_canonical_chunks,
)
from src.indexing.tokenizer_utils import count_tokens
from tests._common import all_docs, doc_by_filename, make_chunk_dict, require_data


def test_chunk_id_stable():
    a = generate_chunk_id("doc_1", "doc_1_v1", "sha256:abc", 0)
    b = generate_chunk_id("doc_1", "doc_1_v1", "sha256:abc", 0)
    assert a == b


def test_chunk_id_index_changes():
    a = generate_chunk_id("doc_1", "doc_1_v1", "sha256:abc", 0)
    b = generate_chunk_id("doc_1", "doc_1_v1", "sha256:abc", 1)
    assert a != b


def test_chunk_id_starts_with_version():
    cid = generate_chunk_id("d", "d_v1", "sha256:abc", 5)
    assert cid.startswith("d_v1_")


def test_chunk_id_contains_index_padded():
    cid = generate_chunk_id("d", "d_v1", "sha256:abc", 5)
    assert "_0005_" in cid


def test_chunk_id_missing_hash():
    cid = generate_chunk_id("d", "d_v1", None, 0)
    assert cid.startswith("d_v1_0000_")


def test_all_chunks_have_required_fields():
    require_data()
    for d in all_docs():
        for ch in chunk_document_structural(d):
            for key in (
                "chunk_id", "document_id", "version_id", "text",
                "embedding_text", "structure", "location",
                "document_title", "document_type",
            ):
                assert key in ch


def test_all_chunks_location_fields():
    require_data()
    for d in all_docs():
        for ch in chunk_document_structural(d):
            loc = ch["location"]
            for key in ("start_line", "end_line", "char_start", "char_end", "page"):
                assert key in loc


def test_all_chunks_structure_has_path_and_provision():
    require_data()
    for d in all_docs():
        for ch in chunk_document_structural(d):
            assert "path" in ch["structure"]
            assert "provision_id" in ch["structure"]
            assert isinstance(ch["structure"]["path"], list)


def test_embedding_text_starts_with_title():
    require_data()
    for d in all_docs():
        title = d["metadata"]["document_title"]
        for ch in chunk_document_structural(d):
            assert ch["embedding_text"].startswith(title)


def test_chunk_ids_unique_per_document():
    require_data()
    for d in all_docs():
        ids = [c["chunk_id"] for c in chunk_document_structural(d)]
        assert len(ids) == len(set(ids))


def test_chunk_ids_unique_globally():
    require_data()
    seen = set()
    for d in all_docs():
        for c in chunk_document_structural(d):
            assert c["chunk_id"] not in seen
            seen.add(c["chunk_id"])


def test_contract_p17_p32_chunk_exists():
    require_data()
    d = doc_by_filename("Договор поставки П-17")
    chunks = chunk_document_structural(d)
    p32 = [c for c in chunks if c["structure"]["provision_id"] == "3.2"]
    assert len(p32) == 1
    assert "Складская" in p32[0]["text"]


def test_contract_p17_p32_has_path_3():
    require_data()
    d = doc_by_filename("Договор поставки П-17")
    chunks = chunk_document_structural(d)
    p32 = [c for c in chunks if c["structure"]["provision_id"] == "3.2"][0]
    assert any("3." in p for p in p32["structure"]["path"])


def test_contract_a08_p51_chunk():
    require_data()
    d = doc_by_filename("Договор аренды А-08")
    chunks = chunk_document_structural(d)
    p51 = [c for c in chunks if c["structure"]["provision_id"] == "5.1"]
    assert len(p51) == 1
    assert "0,1%" in p51[0]["text"]


def test_amendment_becomes_single_full_document_chunk():
    require_data()
    d = doc_by_filename("Доп соглашение 1")
    chunks = chunk_document_structural(d)
    assert len(chunks) == 1
    c = chunks[0]
    assert c["structure"]["provision_id"] is None
    assert c["structure"]["path"] == ["Полный текст документа"]


def test_amendment_has_amends_field():
    require_data()
    d = doc_by_filename("Доп соглашение 1")
    chunks = chunk_document_structural(d)
    assert chunks[0]["amends_document_id"] == "contract_п-17-2024"


def test_no_chunk_exceeds_hard_max():
    require_data()
    for d in all_docs():
        for ch in chunk_document_structural(d):
            toks = count_tokens(ch["text"])
            assert toks <= HARD_MAX_TOKENS + 100


def test_law_152fz_chunk_9_1_exists():
    require_data()
    d = doc_by_filename("152ФЗ_ст9")
    chunks = chunk_document_structural(d)
    ids = {c["structure"]["provision_id"] for c in chunks if c["structure"]["provision_id"]}
    assert "9" in ids
    assert "9.1" in ids


def test_law_koap_chunks_have_parts():
    require_data()
    d = doc_by_filename("КоАП_ст13")
    chunks = chunk_document_structural(d)
    ids = {c["structure"]["provision_id"] for c in chunks if c["structure"]["provision_id"]}
    assert "13.11.2" in ids


def test_chunks_contain_text_body():
    require_data()
    for d in all_docs():
        for ch in chunk_document_structural(d):
            assert ch["text"].strip()
            assert len(ch["text"]) > 5


def test_chunks_embedding_text_contains_body():
    require_data()
    for d in all_docs():
        for ch in chunk_document_structural(d):
            assert ch["text"].strip() in ch["embedding_text"]


def test_chunk_document_title_matches_metadata():
    require_data()
    for d in all_docs():
        for ch in chunk_document_structural(d):
            assert ch["document_title"] == d["metadata"]["document_title"]


def test_chunk_document_id_matches_metadata():
    require_data()
    for d in all_docs():
        for ch in chunk_document_structural(d):
            assert ch["document_id"] == d["metadata"]["document_id"]


def test_chunk_version_id_matches_metadata():
    require_data()
    for d in all_docs():
        for ch in chunk_document_structural(d):
            assert ch["version_id"] == d["metadata"]["version_id"]


def test_save_canonical_chunks_writes_file():
    chunks = [make_chunk_dict(chunk_id="c1"), make_chunk_dict(chunk_id="c2")]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with patch.object(chunker_mod, "PROCESSED_DATA_DIR", tmp):
            path = save_canonical_chunks(chunks)
            assert path == tmp / "chunks.jsonl"
            assert path.exists()


def test_save_canonical_chunks_line_count():
    chunks = [make_chunk_dict(chunk_id=f"c{i}") for i in range(5)]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with patch.object(chunker_mod, "PROCESSED_DATA_DIR", tmp):
            path = save_canonical_chunks(chunks)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 5


def test_save_canonical_chunks_json_valid():
    chunks = [make_chunk_dict(chunk_id="c1", text="привет")]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with patch.object(chunker_mod, "PROCESSED_DATA_DIR", tmp):
            path = save_canonical_chunks(chunks)
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                rec = json.loads(line)
                assert rec["chunk_id"] == "c1"
                assert rec["text"] == "привет"


def test_save_canonical_chunks_preserves_cyrillic():
    chunks = [make_chunk_dict(chunk_id="c1", text="привет мир")]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with patch.object(chunker_mod, "PROCESSED_DATA_DIR", tmp):
            path = save_canonical_chunks(chunks)
            content = path.read_text(encoding="utf-8")
            assert "привет мир" in content