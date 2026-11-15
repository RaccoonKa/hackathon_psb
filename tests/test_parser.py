import re
from rag_core.indexing.parser import (
    calculate_source_hash,
    classify_heading,
    clean_garant_headers,
    detect_doc_type,
    extract_document_structure,
    extract_metadata,
    load_documents_with_metadata,
    normalize_contract_number,
    parse_ru_text_date,
)
from rag_core.indexing.version import CHUNKER_VERSION
from tests._common import all_docs, doc_by_filename, raw_files, require_data


def test_source_hash_prefix():
    h = calculate_source_hash("abc")
    assert h.startswith("sha256:")


def test_source_hash_length():
    h = calculate_source_hash("abc")
    assert len(h) == len("sha256:") + 64


def test_source_hash_deterministic():
    assert calculate_source_hash("abc") == calculate_source_hash("abc")


def test_source_hash_different_for_different_input():
    assert calculate_source_hash("abc") != calculate_source_hash("abd")


def test_source_hash_hex_lowercase():
    h = calculate_source_hash("abc").replace("sha256:", "")
    assert re.match(r"^[0-9a-f]{64}$", h)


def test_source_hash_unicode():
    h = calculate_source_hash("привет")
    assert h.startswith("sha256:")


def test_parse_date_iso():
    assert parse_ru_text_date("2024-08-01") == "2024-08-01"


def test_parse_date_dot_format():
    assert parse_ru_text_date("01.08.2024") == "2024-08-01"


def test_parse_date_dot_single_digit():
    assert parse_ru_text_date("1.8.2024") == "2024-08-01"


def test_parse_date_ru_text():
    assert parse_ru_text_date("1 августа 2024") == "2024-08-01"


def test_parse_date_ru_text_with_g():
    assert parse_ru_text_date("15 апреля 2025 г.") == "2025-04-15"


def test_parse_date_ru_text_all_months():
    months = {
        "января": "01", "февраля": "02", "марта": "03", "апреля": "04",
        "мая": "05", "июня": "06", "июля": "07", "августа": "08",
        "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12",
    }
    for name, num in months.items():
        assert parse_ru_text_date(f"10 {name} 2025") == f"2025-{num}-10"


def test_parse_date_empty():
    assert parse_ru_text_date("") is None


def test_parse_date_none_string():
    assert parse_ru_text_date("не дата") is None


def test_parse_date_garbage():
    assert parse_ru_text_date("абвгд") is None


def test_normalize_contract_number_dash():
    assert normalize_contract_number("П-17/2024") == "п-17-2024"


def test_normalize_contract_number_with_symbol():
    assert normalize_contract_number("№ П-17/2024") == "п-17-2024"


def test_normalize_contract_number_em_dash():
    assert normalize_contract_number("П—17/2024") == "п-17-2024"


def test_clean_garant_strips_source():
    raw = (
        "Источник: Статья 9. Согласие. (с изменениями и дополнениями)\n"
        "URL: https://base.garant.ru/xyz\n"
        "Дата выгрузки: 10.10.2026\n"
        "\n"
        "Статья 9. Тело текста.\n"
    )
    cleaned, title, export_date = clean_garant_headers(raw)
    assert "Источник:" not in cleaned
    assert "URL:" not in cleaned
    assert "Дата выгрузки:" not in cleaned
    assert "Статья 9. Тело текста." in cleaned
    assert title == "Статья 9. Согласие."
    assert export_date == "2026-10-10"


def test_clean_garant_no_header():
    raw = "ДОГОВОР ПОСТАВКИ\nТекст договора."
    cleaned, title, export_date = clean_garant_headers(raw)
    assert title is None
    assert export_date is None
    assert "ДОГОВОР ПОСТАВКИ" in cleaned


def test_clean_garant_multiple_blank_lines_preserved():
    raw = "Источник: X\nURL: Y\nДата выгрузки: 10.10.2026\n\nПервый\n\nВторой\n"
    cleaned, _, _ = clean_garant_headers(raw)
    assert "Первый" in cleaned
    assert "Второй" in cleaned


def test_clean_garant_calculator_line_dropped():
    raw = (
        "Источник: X\nURL: Y\nДата выгрузки: 10.10.2026\n"
        "воспользуйтесь калькулятором, разработанным экспертами компании Гарант\n"
        "Основной текст\n"
    )
    cleaned, _, _ = clean_garant_headers(raw)
    assert "калькулятором" not in cleaned
    assert "Основной текст" in cleaned


def test_detect_law_by_prefix():
    assert detect_doc_type("законы__152ФЗ.txt", "текст") == "law"


def test_detect_law_by_content():
    assert detect_doc_type("file.txt", "Федеральный закон от 27.07.2006") == "law"


def test_detect_internal_by_keyword():
    assert detect_doc_type("reglament.txt", "РЕГЛАМЕНТ согласования договоров") == "internal"


def test_detect_internal_by_prikaz():
    assert detect_doc_type("prikaz.txt", "Приказ № 14-П") == "internal"


def test_detect_contract_by_keyword():
    assert detect_doc_type("file.txt", "ДОГОВОР ПОСТАВКИ № 1") == "contract"


def test_detect_contract_by_dop_soglashenie():
    assert detect_doc_type("file.txt", "Дополнительное соглашение № 1 к договору") == "contract"


def test_detect_default_internal():
    assert detect_doc_type("file.txt", "Какой-то текст без ключевых слов") == "internal"


def test_metadata_p17_has_correct_id():
    require_data()
    d = doc_by_filename("Договор поставки П-17")
    assert d is not None
    assert d["metadata"]["document_id"] == "contract_п-17-2024"


def test_metadata_a08_has_correct_id():
    require_data()
    d = doc_by_filename("Договор аренды А-08")
    assert d is not None
    assert d["metadata"]["document_id"] == "contract_а-08-2024"


def test_metadata_pt05_has_correct_id():
    require_data()
    d = doc_by_filename("Договор поставки ПТ-05")
    assert d is not None
    assert d["metadata"]["document_id"] == "contract_пт-05-2023"


def test_metadata_amendment_links_parent():
    require_data()
    d = doc_by_filename("Доп соглашение 1")
    assert d is not None
    m = d["metadata"]
    assert m["amends_document_id"] == "contract_п-17-2024"
    assert m["document_id"].startswith("amend_")


def test_metadata_reglament_2023_expired():
    require_data()
    d = doc_by_filename("Регламент согласования договоров ред 2023")
    m = d["metadata"]
    assert m["effective_to"] == "2025-03-01"


def test_metadata_reglament_2025_active():
    require_data()
    d = doc_by_filename("Регламент согласования договоров ред 2025")
    m = d["metadata"]
    assert m["effective_from"] == "2025-03-01"
    assert m["effective_to"] is None


def test_metadata_compliance_letter_from():
    require_data()
    d = doc_by_filename("Письмо комплаенс")
    assert d["metadata"]["effective_from"] == "2025-04-21"


def test_metadata_all_have_source_hash():
    require_data()
    for d in all_docs():
        assert d["metadata"]["source_hash"].startswith("sha256:")


def test_metadata_all_have_version_id():
    require_data()
    for d in all_docs():
        assert d["metadata"]["version_id"] == f"{d['metadata']['document_id']}_v1"


def test_metadata_all_have_title():
    require_data()
    for d in all_docs():
        assert d["metadata"]["document_title"]


def test_metadata_all_have_type():
    require_data()
    for d in all_docs():
        assert d["metadata"]["document_type"] in {"law", "internal", "contract"}


def test_metadata_all_have_parser_and_chunker_version():
    require_data()
    for d in all_docs():
        assert d["metadata"]["parser_version"]
        assert d["metadata"]["chunker_version"] == CHUNKER_VERSION


def test_classify_heading_section():
    level, title, pid = classify_heading("Раздел I. Общие положения")
    assert level == 0
    assert "Раздел I" in title
    assert pid is None


def test_classify_heading_chapter():
    level, title, pid = classify_heading("Глава 4. Налоговые ставки")
    assert level == 1
    assert pid is None


def test_classify_heading_article():
    level, title, pid = classify_heading("Статья 112. Обстоятельства")
    assert level == 2
    assert pid == "112"


def test_classify_heading_article_with_dots():
    level, title, pid = classify_heading("Статья 13.11. Нарушение")
    assert level == 2
    assert pid == "13.11"


def test_classify_heading_contract_major():
    level, title, pid = classify_heading("1. Предмет договора")
    assert level == 2
    assert pid == "1"


def test_classify_heading_contract_sub():
    level, title, pid = classify_heading("3.2. Товар доставляется")
    assert level == 3
    assert pid == "3.2"


def test_classify_heading_deep_sub():
    level, title, pid = classify_heading("3.2.1. Глубокий пункт")
    assert pid == "3.2.1"


def test_classify_heading_part_inside_article():
    level, title, pid = classify_heading(
        "1. Субъект персональных данных принимает решение",
        in_article=True, current_art_num="9",
    )
    assert level == 3
    assert pid == "9.1"


def test_classify_heading_paren_subpoint():
    # Элементы перечислений со скобкой ("1)") не должны отрываться от родительской нормы
    res = classify_heading(
        "1) фамилию, имя, отчество",
        in_article=True, current_art_num="9", current_part_num="9.4",
    )
    assert res is None


def test_classify_heading_ignores_modified_note():
    assert classify_heading("Часть 1 изменена с 1 сентября 2025 г.") is None


def test_classify_heading_ignores_utratil_silu():
    assert classify_heading("Пункт 3 утратил силу с 1 января 2026 г.") is None


def test_classify_heading_ignores_attention():
    assert classify_heading("Обратите внимание, что применительно") is None


def test_classify_heading_ignores_cрок_давности():
    assert classify_heading("Cрок давности привлечения к ответственности") is None


def test_classify_heading_ignores_empty():
    assert classify_heading("") is None
    assert classify_heading("   ") is None


def test_extract_structure_contract_p17_has_p32():
    require_data()
    d = doc_by_filename("Договор поставки П-17")
    blocks = extract_document_structure(d["content"])
    p32 = [b for b in blocks if b["provision_id"] == "3.2"]
    assert len(p32) == 1
    assert "Складская" in p32[0]["text"]


def test_extract_structure_contract_p17_has_p75():
    require_data()
    d = doc_by_filename("Договор поставки П-17")
    blocks = extract_document_structure(d["content"])
    p75 = [b for b in blocks if b["provision_id"] == "7.5"]
    assert len(p75) == 1
    assert "50%" in p75[0]["text"]


def test_extract_structure_contract_a08_has_p51():
    require_data()
    d = doc_by_filename("Договор аренды А-08")
    blocks = extract_document_structure(d["content"])
    p51 = [b for b in blocks if b["provision_id"] == "5.1"]
    assert len(p51) == 1
    assert "0,1%" in p51[0]["text"]


def test_extract_structure_law_article_9():
    require_data()
    d = doc_by_filename("152ФЗ_ст9")
    blocks = extract_document_structure(d["content"])
    ids = {b["provision_id"] for b in blocks if b["provision_id"]}
    assert "9" in ids
    assert "9.1" in ids
    assert "9.2" in ids


def test_extract_structure_koap_13_11():
    require_data()
    d = doc_by_filename("КоАП_ст13")
    blocks = extract_document_structure(d["content"])
    ids = {b["provision_id"] for b in blocks if b["provision_id"]}
    assert "13.11" in ids
    assert "13.11.2" in ids
    assert "13.11.2.1" in ids


def test_extract_structure_all_have_path():
    require_data()
    for d in all_docs():
        for b in extract_document_structure(d["content"]):
            assert b["path"]
            assert isinstance(b["path"], list)


def test_extract_structure_all_have_line_numbers():
    require_data()
    for d in all_docs():
        for b in extract_document_structure(d["content"]):
            assert b["start_line"] >= 1
            assert b["end_line"] >= b["start_line"]


def test_extract_structure_all_char_offsets_ordered():
    require_data()
    for d in all_docs():
        for b in extract_document_structure(d["content"]):
            assert b["char_start"] >= 0
            assert b["char_end"] >= b["char_start"]


def test_load_documents_returns_list():
    require_data()
    docs = all_docs()
    assert isinstance(docs, list)
    assert len(docs) >= 15


def test_load_documents_all_have_content():
    require_data()
    for d in all_docs():
        assert d["content"]
        assert isinstance(d["content"], str)


def test_load_documents_no_duplicates_by_filename():
    require_data()
    names = [d["metadata"]["source_filename"] for d in all_docs()]
    assert len(names) == len(set(names))


def test_extract_metadata_from_text():
    from pathlib import Path
    raw = "ДОГОВОР № 1\nот 01.01.2024\nтекст"
    meta, cleaned = extract_metadata(Path("test.txt"), raw)
    assert meta["document_id"]
    assert meta["version_id"].endswith("_v1")
    assert meta["source_hash"].startswith("sha256:")
    assert cleaned