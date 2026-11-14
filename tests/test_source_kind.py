"""Технические тесты определения назначения источника (P1, п. 1–3 ТЗ).

Проверяется только консервативная логика модуля
``rag_core.generation.analysis.source_kind``: слоты, текстовые маркеры,
маппинг ``document_type`` ↔ ``source_kind`` и подписи. Данные синтетические,
без обращения к индексу и сети.
"""

from rag_core.generation.analysis.source_kind import (
    NORMATIVE_SLOTS,
    SOURCE_KIND_NORMATIVE,
    SOURCE_KIND_SPECIFIC,
    SOURCE_KIND_UNKNOWN,
    SPECIFIC_SLOTS,
    STRICT_SLOT_SOURCE_KIND,
    detect_required_source_kinds,
    document_types_for_source_kind,
    document_types_for_source_kinds,
    expected_source_kind_for_slot,
    source_kind_for_document_type,
    source_kind_label,
)
from tests._common import make_query_spec


def _spec(**kwargs):
    # Пустые переформулировки убирают влияние текстовых маркеров.
    kwargs.setdefault("rewritten_queries_bm25", [])
    kwargs.setdefault("rewritten_queries_dense", [])
    return make_query_spec(**kwargs)


# --- 1. Слоты --------------------------------------------------------------


def test_normative_slot_approval_process():
    assert detect_required_source_kinds(
        _spec(required_slots=["approval_process"])
    ) == ["normative"]


def test_normative_slot_overtime_limit():
    assert detect_required_source_kinds(
        _spec(required_slots=["overtime_limit"])
    ) == ["normative"]


def test_specific_slot_rent_amount():
    assert detect_required_source_kinds(
        _spec(required_slots=["rent_amount"])
    ) == ["specific"]


def test_specific_slot_payment_deadline():
    assert detect_required_source_kinds(
        _spec(required_slots=["payment_deadline"])
    ) == ["specific"]


def test_both_kinds_have_fixed_order():
    result = detect_required_source_kinds(
        _spec(required_slots=["payment_deadline", "approval_process"])
    )
    assert result == ["normative", "specific"]


def test_no_signals_returns_empty():
    assert detect_required_source_kinds(_spec(required_slots=["general"])) == []


def test_unknown_slot_returns_empty():
    assert detect_required_source_kinds(
        _spec(required_slots=["some_unknown_slot"])
    ) == []


def test_slot_normalized_case_and_spaces():
    assert detect_required_source_kinds(
        _spec(required_slots=["  Approval_Process "])
    ) == ["normative"]


# --- 2. Текстовые маркеры --------------------------------------------------


def test_normative_text_marker_in_bm25():
    spec = _spec(
        required_slots=["general"],
        rewritten_queries_bm25=["какой установлен порядок согласования"],
    )
    assert detect_required_source_kinds(spec) == ["normative"]


def test_normative_text_marker_in_dense_case_insensitive():
    spec = _spec(
        required_slots=["general"],
        rewritten_queries_dense=["ТРЕБОВАНИЯ РЕГЛАМЕНТА"],
    )
    assert detect_required_source_kinds(spec) == ["normative"]


def test_specific_text_marker_in_bm25():
    spec = _spec(
        required_slots=["general"],
        rewritten_queries_bm25=["какие условия договора согласованы"],
    )
    assert detect_required_source_kinds(spec) == ["specific"]


def test_specific_text_marker_in_dense():
    spec = _spec(
        required_slots=["general"],
        rewritten_queries_dense=["поставщик и покупатель подписали"],
    )
    assert detect_required_source_kinds(spec) == ["specific"]


def test_both_markers_combined_order():
    spec = _spec(
        required_slots=["general"],
        rewritten_queries_bm25=["порядок одобрения"],
        rewritten_queries_dense=["условия поставки"],
    )
    assert detect_required_source_kinds(spec) == ["normative", "specific"]


def test_slot_and_text_marker_dedup():
    spec = _spec(
        required_slots=["approval_process"],
        rewritten_queries_dense=["требования регламента"],
    )
    assert detect_required_source_kinds(spec) == ["normative"]


# --- 3. Маппинг document_type → source_kind --------------------------------


def test_document_type_law_is_normative():
    assert source_kind_for_document_type("law") == SOURCE_KIND_NORMATIVE


def test_document_type_internal_is_normative():
    assert source_kind_for_document_type("internal") == SOURCE_KIND_NORMATIVE


def test_document_type_contract_is_specific():
    assert source_kind_for_document_type("contract") == SOURCE_KIND_SPECIFIC


def test_document_type_unknown_values():
    assert source_kind_for_document_type(None) == SOURCE_KIND_UNKNOWN
    assert source_kind_for_document_type("") == SOURCE_KIND_UNKNOWN
    assert source_kind_for_document_type("note") == SOURCE_KIND_UNKNOWN


def test_document_type_normalized():
    assert source_kind_for_document_type(" LAW ") == SOURCE_KIND_NORMATIVE


# --- 4. Обратный маппинг ---------------------------------------------------


def test_document_types_for_normative():
    assert document_types_for_source_kind(SOURCE_KIND_NORMATIVE) == {
        "law",
        "internal",
    }


def test_document_types_for_specific():
    assert document_types_for_source_kind(SOURCE_KIND_SPECIFIC) == {"contract"}


def test_document_types_for_unknown_returns_none():
    assert document_types_for_source_kind(SOURCE_KIND_UNKNOWN) is None
    assert document_types_for_source_kind("") is None
    assert document_types_for_source_kind(None) is None


def test_document_types_for_kinds_union():
    assert document_types_for_source_kinds(
        [SOURCE_KIND_NORMATIVE, SOURCE_KIND_SPECIFIC]
    ) == {"law", "internal", "contract"}


def test_document_types_for_kinds_empty_returns_none():
    assert document_types_for_source_kinds([]) is None
    assert document_types_for_source_kinds(["unknown"]) is None


# --- 5. Слоты и подписи ----------------------------------------------------


def test_expected_source_kind_for_normative_slot():
    assert expected_source_kind_for_slot("overtime_limit") == SOURCE_KIND_NORMATIVE


def test_expected_source_kind_for_specific_slot():
    assert expected_source_kind_for_slot("rent_amount") == SOURCE_KIND_SPECIFIC


def test_expected_source_kind_for_unknown_slot():
    assert expected_source_kind_for_slot("general") is None
    assert expected_source_kind_for_slot(None) is None


def test_source_kind_labels():
    assert source_kind_label(SOURCE_KIND_NORMATIVE) == "нормативное основание"
    assert source_kind_label(SOURCE_KIND_SPECIFIC) == "документ сделки"


def test_source_kind_label_unknown_and_none():
    assert source_kind_label(None) == ""
    assert source_kind_label("") == ""
    assert source_kind_label("неизвестно") == ""


def test_slot_constants_disjoint():
    assert not (NORMATIVE_SLOTS & SPECIFIC_SLOTS)
    assert set(STRICT_SLOT_SOURCE_KIND) == set(NORMATIVE_SLOTS) | set(
        SPECIFIC_SLOTS
    )
