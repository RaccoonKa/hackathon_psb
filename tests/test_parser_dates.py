from pathlib import Path

from src.indexing.parser import extract_metadata, parse_dates_and_version


def test_prolongation_skips_effective_to():
    content = (
        "ДОГОВОР № 1 от 01.01.2024\n"
        "Договор действует до 31.12.2025 и автоматически продлевается.\n"
    )
    dates, _ = parse_dates_and_version(content, "contract.txt", doc_type="contract")
    assert dates["effective_to"] is None


def test_prolongiruemoe_skips_effective_to():
    content = (
        "ДОГОВОР № 1 от 01.01.2024\n"
        "Договор действует до 31.12.2025, пролонгируется автоматически.\n"
    )
    dates, _ = parse_dates_and_version(content, "contract.txt", doc_type="contract")
    assert dates["effective_to"] is None


def test_bez_prolongacii_sets_effective_to():
    content = (
        "ДОГОВОР № 1 от 01.01.2024\n"
        "Договор действует до 31.12.2025.\n"
    )
    dates, _ = parse_dates_and_version(content, "contract.txt", doc_type="contract")
    assert dates["effective_to"] == "2025-12-31"


def test_dva_deystvuet_do_pervyy_prodlevaetsya_vtoroy_net():
    content = (
        "ДОГОВОР № 1 от 01.01.2024\n"
        "Предыдущая редакция действует до 01.01.2025 и продлевается.\n"
        "Текущая редакция действует до 01.01.2027.\n"
    )
    dates, _ = parse_dates_and_version(content, "contract.txt", doc_type="contract")
    assert dates["effective_to"] == "2027-01-01"


def test_dva_deystvuet_do_pervyy_bez_prodleniya():
    content = (
        "ДОГОВОР № 1 от 01.01.2024\n"
        "Редакция действует до 01.06.2025.\n"
        "Редакция действует до 01.09.2026 и продлевается.\n"
    )
    dates, _ = parse_dates_and_version(content, "contract.txt", doc_type="contract")
    assert dates["effective_to"] == "2025-06-01"


def test_utratil_silu_sets_effective_to():
    content = (
        "РЕДАКЦИЯ УТРАТИЛА СИЛУ с 01.03.2025\n"
        "ДОГОВОР № 1 от 01.01.2024\n"
        "текст"
    )
    dates, _ = parse_dates_and_version(content, "contract.txt", doc_type="contract")
    assert dates["effective_to"] == "2025-03-01"


def test_utratil_silu_has_priority_over_deystvuet_do():
    content = (
        "ДОГОВОР № 1 от 01.01.2024\n"
        "Редакция утратила силу с 01.03.2025.\n"
        "Ранее действовала до 01.06.2026.\n"
    )
    dates, _ = parse_dates_and_version(content, "contract.txt", doc_type="contract")
    assert dates["effective_to"] == "2025-03-01"


def test_utratil_silu_text_form():
    content = (
        "Редакция утратила силу с 1 сентября 2025\n"
        "ДОГОВОР № 1 от 01.01.2024\n"
    )
    dates, _ = parse_dates_and_version(content, "contract.txt", doc_type="contract")
    assert dates["effective_to"] == "2025-09-01"


def test_nachinaya_s_sets_effective_from():
    content = (
        "ДОГОВОР № 1\n"
        "Начиная с 01.04.2025 применяется в новой редакции.\n"
    )
    dates, _ = parse_dates_and_version(content, "contract.txt", doc_type="contract")
    assert dates["effective_from"] == "2025-04-01"


def test_redakciya_ot_sets_effective_from():
    content = (
        "ДОГОВОР № 1\n"
        "Редакция от 15.06.2025\n"
    )
    dates, _ = parse_dates_and_version(content, "contract.txt", doc_type="contract")
    assert dates["effective_from"] == "2025-06-15"


def test_deystvuet_s_sets_effective_from():
    content = (
        "ДОГОВОР № 1\n"
        "Действует с 01.02.2025 в новой редакции.\n"
    )
    dates, _ = parse_dates_and_version(content, "contract.txt", doc_type="contract")
    assert dates["effective_from"] == "2025-02-01"


def test_vstupaet_v_silu_s_sets_effective_from():
    content = (
        "ДОГОВОР № 1\n"
        "Новая редакция вступает в силу с 01.09.2025.\n"
    )
    dates, _ = parse_dates_and_version(content, "contract.txt", doc_type="contract")
    assert dates["effective_from"] == "2025-09-01"


def test_future_date_goes_to_effective_to_when_cutoff_before():
    content = (
        "ДОГОВОР № 1 от 01.01.2024\n"
        "Новая редакция вступает в силу с 01.01.2030.\n"
    )
    dates, _ = parse_dates_and_version(
        content, "contract.txt", doc_type="contract",
        export_date="2026-10-10",
    )
    assert dates["effective_from"] == "2024-01-01"
    assert dates["effective_to"] == "2030-01-01"


def test_two_revisions_uses_latest_before_cutoff():
    content = (
        "ДОГОВОР № 1\n"
        "Редакция от 01.01.2023\n"
        "Редакция от 01.03.2025\n"
    )
    dates, _ = parse_dates_and_version(
        content, "contract.txt", doc_type="contract",
        export_date="2026-10-10",
    )
    assert dates["effective_from"] == "2025-03-01"


def test_export_date_passed_via_metadata():
    raw = (
        "ДОГОВОР № 1 от 01.01.2024\n"
        "Новая редакция вступает в силу с 01.01.2030.\n"
    )
    meta, _ = extract_metadata(Path("contract.txt"), raw)
    assert meta["effective_from"] == "2024-01-01"
    assert meta["effective_to"] == "2030-01-01"


def test_utratil_silu_via_metadata():
    raw = (
        "РЕДАКЦИЯ УТРАТИЛА СИЛУ с 01.03.2025\n"
        "ДОГОВОР № 1 от 01.01.2024\n"
        "текст"
    )
    meta, _ = extract_metadata(Path("contract.txt"), raw)
    assert meta["effective_to"] == "2025-03-01"


def test_prolongation_via_metadata():
    raw = (
        "ДОГОВОР № 1 от 01.01.2024\n"
        "Договор действует до 31.12.2025 и автоматически продлевается.\n"
    )
    meta, _ = extract_metadata(Path("contract.txt"), raw)
    assert meta["effective_to"] is None


def test_effective_dates_known_false_when_no_date():
    raw = "ДОГОВОР без указания дат\n"
    dates, _ = parse_dates_and_version(raw, "contract.txt", doc_type="contract")
    assert dates["effective_dates_known"] is False


def test_effective_dates_known_true_when_date_found():
    raw = (
        "ДОГОВОР № 1 от 15.03.2024\n"
        "текст\n"
    )
    dates, _ = parse_dates_and_version(raw, "contract.txt", doc_type="contract")
    assert dates["effective_dates_known"] is True
    assert dates["effective_from"] == "2024-03-15"


def test_amends_soglashenie_detected():
    raw = (
        "ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ № 1\n"
        "к договору поставки № П-17/2024 от 12.02.2024\n"
    )
    _, amends = parse_dates_and_version(
        raw, "доп соглашение.txt", doc_type="contract",
    )
    assert amends == "contract_п-17-2024"