from pathlib import Path

from rag_core.indexing.parser import (
    classify_heading,
    extract_document_structure,
    extract_metadata,
    normalize_contract_number,
)


def test_pri_mecaniya_heading_level_3():
    level, title, pid = classify_heading(
        "Примечания:", in_article=True, current_art_num="13.11"
    )
    assert level == 3
    assert pid == "13.11.прим"


def test_pri_mecaniya_item_level_4():
    level, title, pid = classify_heading(
        "1. За административные правонарушения",
        in_article=True, current_art_num="13.11", in_notes=True,
    )
    assert level == 4
    assert pid == "13.11.прим.1"


def test_pri_mecaniya_item_2():
    _, _, pid = classify_heading(
        "2. В частях 10 - 18 настоящей статьи",
        in_article=True, current_art_num="13.11", in_notes=True,
    )
    assert pid == "13.11.прим.2"


def test_punkt_word_form():
    level, _, pid = classify_heading("Пункт 3.2. Товар доставляется")
    assert level == 3
    assert pid == "3.2"


def test_punkt_word_form_three_levels():
    _, _, pid = classify_heading("Пункт 1.2.3. Глубокий")
    assert pid == "1.2.3"


def test_paren_multi_dot_inside_article():
    res = classify_heading(
        "15.1) какой-то текст",
        in_article=True, current_art_num="1", current_part_num="1.5",
    )
    assert res is None


def test_notes_synthetic_full_structure():
    content = (
        "Статья 1. Предмет\n"
        "1. Часть первая\n"
        "Примечания:\n"
        "1. За что-то\n"
        "2. За что-то ещё\n"
    )
    ids = {b["provision_id"] for b in extract_document_structure(content) if b["provision_id"]}
    assert "1" in ids
    assert "1.1" in ids
    assert "1.прим" in ids
    assert "1.прим.1" in ids
    assert "1.прим.2" in ids


def test_nested_path_contains_parent_section():
    content = (
        "1. Крупный раздел\n"
        "1.1. Первый\n"
        "1.2. Второй\n"
    )
    blocks = extract_document_structure(content)
    p12 = [b for b in blocks if b["provision_id"] == "1.2"]
    assert len(p12) == 1
    assert any("Крупный раздел" in p for p in p12[0]["path"])


def test_nested_path_grows_with_depth():
    content = (
        "1. Раздел\n"
        "1.1. Подраздел\n"
        "1.1.1. Пункт\n"
    )
    blocks = extract_document_structure(content)
    deepest = [b for b in blocks if b["provision_id"] == "1.1.1"][0]
    assert len(deepest["path"]) >= 2


def test_duplicate_section_headings_isolated():
    content = (
        "1. Предмет договора\n"
        "1.1. Текст\n"
        "2. Цена\n"
        "2.1. Текст\n"
    )
    blocks = extract_document_structure(content)
    p11 = [b for b in blocks if b["provision_id"] == "1.1"]
    p21 = [b for b in blocks if b["provision_id"] == "2.1"]
    assert len(p11) == 1
    assert len(p21) == 1


def test_empty_content_no_blocks():
    assert extract_document_structure("") == []


def test_content_without_headings_single_block():
    content = "Просто текст.\nБез заголовков.\n"
    blocks = extract_document_structure(content)
    assert len(blocks) == 1
    assert blocks[0]["provision_id"] is None


def test_utratil_silu_parsed_as_effective_to():
    raw = (
        "РЕДАКЦИЯ УТРАТИЛА СИЛУ с 01.03.2025\n"
        "ДОГОВОР № 1 от 01.01.2024\n"
        "текст"
    )
    meta, _ = extract_metadata(Path("test.txt"), raw)
    assert meta["effective_to"] == "2025-03-01"


def test_utratil_silu_text_form():
    raw = (
        "Редакция утратила силу с 1 сентября 2025\n"
        "ДОГОВОР № 1\n"
        "текст"
    )
    meta, _ = extract_metadata(Path("test.txt"), raw)
    assert meta["effective_to"] == "2025-09-01"


def test_future_vstuplenie_goes_to_effective_to():
    raw = (
        "ДОГОВОР № 1 от 01.01.2024\n"
        "вступает в силу с 01.01.2030\n"
        "текст"
    )
    meta, _ = extract_metadata(Path("test.txt"), raw)
    assert meta["effective_from"] == "2024-01-01"
    assert meta["effective_to"] == "2030-01-01"


def test_normalize_contract_number_empty():
    assert normalize_contract_number("") == ""


def test_normalize_contract_number_only_dashes():
    assert normalize_contract_number("---") == ""


def test_normalize_contract_number_only_symbols():
    assert normalize_contract_number("!!!") == ""


def test_metadata_empty_content():
    meta, cleaned = extract_metadata(Path("empty.txt"), "")
    assert cleaned == ""
    assert meta["document_id"]
    assert meta["source_hash"].startswith("sha256:")


def test_koap_has_pri_mecaniya():
    from tests._common import doc_by_filename, require_data
    require_data()
    d = doc_by_filename("КоАП_ст13")
    blocks = extract_document_structure(d["content"])
    ids = {b["provision_id"] for b in blocks if b["provision_id"]}
    assert any(i and "прим" in i for i in ids), f"Нет примечаний среди {ids}"


def test_structure_blocks_have_hierarchy_fields():
    content = (
        "Первый абзац преамбулы.\n"
        "1. Крупный раздел\n"
        "1.1. Первый пункт\n"
        "1.1.1. Глубокий подпункт\n"
    )
    blocks = extract_document_structure(content)
    preamble = blocks[0]
    assert preamble["level"] == 0
    assert preamble["path_levels"] == [0]
    assert preamble["heading_title"] is None
    assert preamble["section_provision_id"] is None

    deep = [b for b in blocks if b["provision_id"] == "1.1.1"][0]
    # backward-compatible поля не удаляют существующие ключи
    for key in ("provision_id", "path", "start_line", "end_line", "char_start", "char_end", "text"):
        assert key in deep
    assert deep["level"] >= 2
    assert len(deep["path_levels"]) == len(deep["path"])
    assert deep["heading_title"]
    # Ближайший уровень <= 2 — крупный раздел "1."
    assert deep["section_provision_id"] == "1"
    assert deep["path_levels"][-1] == deep["level"]