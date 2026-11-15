"""Тесты структурно-адаптивного чанкинга (ТЗ: разделение и склейка)."""

import re

from rag_core.indexing.chunker import (
    HARD_MAX_TOKENS,
    chunk_document_structural,
)
from rag_core.indexing.tokenizer_utils import count_tokens
from tests._common import all_docs, doc_by_filename, require_data


def _normalized(text: str) -> str:
    """Нормализация пробелов/переносов/NBSP для сверки с исходным content."""
    return "".join(text.split())


def _clause_section_numbers(text: str):
    """Номера крупных разделов по началам строк вида ``3.1.``."""
    return {m.group(1) for m in re.finditer(r"(?m)^\s*(\d+)\.\d+\.", text)}


def test_large_law_clause_split_into_multiple_chunks():
    require_data()
    d = doc_by_filename("НК_ст145")
    chunks = chunk_document_structural(d)
    p1 = [c for c in chunks if c["structure"]["provision_id"] == "145.1"]
    assert len(p1) >= 2, "Пункт 1 статьи 145 НК должен делиться на несколько фрагментов"
    assert "Организации и индивидуальные предприниматели" in p1[0]["text"]
    starts = [c["location"]["char_start"] for c in p1]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_split_fragments_preserve_provision_and_path():
    require_data()
    d = doc_by_filename("НК_ст145")
    chunks = chunk_document_structural(d)
    p1 = [c for c in chunks if c["structure"]["provision_id"] == "145.1"]
    for c in p1:
        assert c["structure"]["provision_id"] == "145.1"
        path = " ".join(c["structure"]["path"])
        assert "Статья 145" in path
        assert "Часть 1" in path
        start = c["location"]["char_start"]
        end = c["location"]["char_end"]
        assert _normalized(c["text"]) == _normalized(d["content"][start:end])


def test_group_of_clauses_gets_null_provision_and_section_path():
    require_data()
    d = doc_by_filename("Договор поставки П-17")
    chunks = chunk_document_structural(d)
    target = [c for c in chunks if "Складская" in c["text"]]
    assert len(target) == 1
    chunk = target[0]
    assert chunk["structure"]["provision_id"] is None
    assert chunk["structure"]["path"] == ["3. Порядок поставки"]
    for clause in ("3.1.", "3.2.", "3.3."):
        assert clause in chunk["text"], f"Группа должна сохранять текст {clause}"


def test_single_clause_section_keeps_provision_id():
    require_data()
    d = doc_by_filename("Договор аренды А-08")
    chunks = chunk_document_structural(d)
    p51 = [c for c in chunks if c["structure"]["provision_id"] == "5.1"]
    assert len(p51) == 1
    assert "0,1%" in p51[0]["text"]


def test_groups_do_not_cross_major_section_boundaries():
    require_data()
    d = doc_by_filename("Договор поставки П-17")
    for chunk in chunk_document_structural(d):
        path = chunk["structure"]["path"]
        top_sections = [p for p in path if re.match(r"^\d+\.\s", p)]
        assert len(top_sections) <= 1
        clause_sections = _clause_section_numbers(chunk["text"])
        assert len(clause_sections) <= 1
        if clause_sections and top_sections:
            section_no = re.match(r"^(\d+)\.", top_sections[0]).group(1)
            assert clause_sections == {section_no}


def test_law_chunks_do_not_mix_articles():
    require_data()
    d = doc_by_filename("152ФЗ_ст9")
    for chunk in chunk_document_structural(d):
        articles = [p for p in chunk["structure"]["path"] if p.startswith("Статья ")]
        assert len(articles) == 1


def test_small_preamble_merged_with_first_section():
    require_data()
    d = doc_by_filename("Договор поставки П-17")
    chunk = chunk_document_structural(d)[0]
    assert "ДОГОВОР ПОСТАВКИ" in chunk["text"]
    assert "1.1." in chunk["text"]
    assert chunk["structure"]["provision_id"] is None
    assert any("1. Предмет договора" in p for p in chunk["structure"]["path"])


def test_embedding_text_within_hard_max():
    require_data()
    for d in all_docs():
        for chunk in chunk_document_structural(d):
            tokens = count_tokens(chunk["embedding_text"])
            assert tokens <= HARD_MAX_TOKENS, (
                f"{d['metadata']['source_filename']}: чанк {chunk['chunk_index']} "
                f"превышает HARD_MAX_TOKENS ({tokens})"
            )


def test_char_offsets_match_original_content():
    require_data()
    for d in all_docs():
        content = d["content"]
        for chunk in chunk_document_structural(d):
            start = chunk["location"]["char_start"]
            end = chunk["location"]["char_end"]
            assert 0 <= start <= end <= len(content)
            assert _normalized(chunk["text"]) == _normalized(content[start:end])


def test_chunks_follow_document_order():
    require_data()
    for d in all_docs():
        chunks = chunk_document_structural(d)
        starts = [c["location"]["char_start"] for c in chunks]
        assert starts == sorted(starts)
        for index, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == index


def test_embedding_text_contains_document_title_and_path():
    require_data()
    d = doc_by_filename("НК_ст145")
    title = d["metadata"]["document_title"]
    for chunk in chunk_document_structural(d):
        assert chunk["embedding_text"].startswith(title)
        for part in chunk["structure"]["path"]:
            assert part in chunk["embedding_text"]
        assert chunk["text"] in chunk["embedding_text"]
