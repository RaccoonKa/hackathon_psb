"""Технические тесты дат и выбора редакций (P1, п. 4–8 ТЗ).

Проверяются ``parse_iso_date``, ``effective_status``,
``select_dominant_versions``, ``resolve_check_date_details`` и
``resolve_allowed_versions_by_date`` на синтетических ``chunks_map``.
Реальный индекс не используется.
"""

import datetime

from rag_core.retrieval.hybrid import (
    EFFECTIVE_STATUS_APPLICABLE,
    EFFECTIVE_STATUS_CURRENT_ONLY,
    EFFECTIVE_STATUS_NOT_APPLICABLE,
    EFFECTIVE_STATUS_UNKNOWN,
    effective_status,
    parse_iso_date,
    resolve_allowed_versions_by_date,
    resolve_check_date_details,
    select_dominant_versions,
)
from tests._common import make_chunk_dict

CHECK_DATE = "2024-06-15"


# --- parse_iso_date --------------------------------------------------------


def test_parse_iso_date_valid():
    assert parse_iso_date("2024-06-15") == datetime.date(2024, 6, 15)


def test_parse_iso_date_truncates_time_suffix():
    assert parse_iso_date("2024-06-15T10:00:00") == datetime.date(2024, 6, 15)


def test_parse_iso_date_invalid_values_return_none():
    for value in (None, "", "not-a-date", "2025-13-40"):
        assert parse_iso_date(value) is None


def test_parse_iso_date_non_string_returns_none():
    assert parse_iso_date(20240615) is None
    assert parse_iso_date(datetime.date(2024, 6, 15)) is None


# --- effective_status ------------------------------------------------------


def _chunk(**kwargs):
    defaults = {"document_type": "contract"}
    defaults.update(kwargs)
    return make_chunk_dict(**defaults)


def test_effective_status_lower_bound_inclusive():
    chunk = _chunk(effective_from="2024-06-15", effective_to=None)
    assert effective_status(chunk, "2024-06-15") == EFFECTIVE_STATUS_APPLICABLE


def test_effective_status_upper_bound_exclusive():
    chunk = _chunk(effective_from="2021-01-01", effective_to="2024-06-15")
    assert (
        effective_status(chunk, "2024-06-15")
        == EFFECTIVE_STATUS_NOT_APPLICABLE
    )


def test_effective_status_before_from_contract_and_internal():
    for document_type in ("contract", "internal"):
        chunk = _chunk(
            document_type=document_type,
            effective_from="2024-06-15",
            effective_to=None,
        )
        assert (
            effective_status(chunk, "2021-01-01")
            == EFFECTIVE_STATUS_NOT_APPLICABLE
        )


def test_effective_status_before_from_law_is_current_only():
    chunk = _chunk(
        document_type="law",
        effective_from="2024-06-15",
        effective_to=None,
    )
    assert (
        effective_status(chunk, "2021-01-01")
        == EFFECTIVE_STATUS_CURRENT_ONLY
    )


def test_effective_status_open_upper_bound():
    chunk = _chunk(effective_from="2021-01-01", effective_to=None)
    assert effective_status(chunk, "2030-12-31") == EFFECTIVE_STATUS_APPLICABLE


def test_effective_status_open_lower_bound():
    chunk = _chunk(effective_from=None, effective_to="2030-12-31")
    assert effective_status(chunk, "2021-01-01") == EFFECTIVE_STATUS_APPLICABLE


def test_effective_status_dates_known_false():
    chunk = _chunk(
        effective_from="2021-01-01",
        effective_to="2030-12-31",
        effective_dates_known=False,
    )
    assert effective_status(chunk, CHECK_DATE) == EFFECTIVE_STATUS_UNKNOWN


def test_effective_status_empty_dates():
    chunk = _chunk(effective_from=None, effective_to=None)
    assert effective_status(chunk, CHECK_DATE) == EFFECTIVE_STATUS_UNKNOWN


def test_effective_status_invalid_effective_date():
    chunk = _chunk(effective_from="not-a-date", effective_to=None)
    assert effective_status(chunk, CHECK_DATE) == EFFECTIVE_STATUS_UNKNOWN


def test_effective_status_invalid_check_date():
    chunk = _chunk(effective_from="2021-01-01", effective_to=None)
    assert effective_status(chunk, "not-a-date") == EFFECTIVE_STATUS_UNKNOWN


# --- select_dominant_versions ---------------------------------------------


def _version(
    document_id: str,
    version_id: str,
    chunk_id: str,
    **kwargs,
):
    return make_chunk_dict(
        chunk_id=chunk_id,
        document_id=document_id,
        version_id=version_id,
        document_type=kwargs.pop("document_type", "contract"),
        **kwargs,
    )


def test_dominant_keeps_only_applicable_version():
    chunks_map = {
        "old": _version(
            "doc-a", "ver-1", "old",
            effective_from="2021-01-01", effective_to="2024-06-15",
        ),
        "new": _version(
            "doc-a", "ver-2", "new",
            effective_from="2024-06-15", effective_to=None,
        ),
    }
    selected = select_dominant_versions(
        set(chunks_map), chunks_map, CHECK_DATE
    )
    assert selected == {"new"}


def test_dominant_keeps_latest_applicable_effective_from():
    chunks_map = {
        "v1": _version(
            "doc-b", "ver-1", "v1",
            effective_from="2021-01-01", effective_to=None,
        ),
        "v2": _version(
            "doc-b", "ver-2", "v2",
            effective_from="2023-01-01", effective_to=None,
        ),
    }
    selected = select_dominant_versions(
        set(chunks_map), chunks_map, CHECK_DATE
    )
    assert selected == {"v2"}


def test_dominant_keeps_current_only_without_applicable():
    chunks_map = {
        "future_law": _version(
            "doc-c", "ver-future", "future_law",
            document_type="law",
            effective_from="2030-12-31", effective_to=None,
        ),
        "expired": _version(
            "doc-c", "ver-old", "expired",
            document_type="law",
            effective_from="2021-01-01", effective_to="2022-01-01",
        ),
    }
    selected = select_dominant_versions(
        set(chunks_map), chunks_map, CHECK_DATE
    )
    assert selected == {"future_law"}


def test_dominant_keeps_unknown_without_applicable():
    chunks_map = {
        "unknown": _version(
            "doc-d", "ver-unknown", "unknown",
            effective_dates_known=False,
        ),
        "expired": _version(
            "doc-d", "ver-old", "expired",
            effective_from="2021-01-01", effective_to="2022-01-01",
        ),
    }
    selected = select_dominant_versions(
        set(chunks_map), chunks_map, CHECK_DATE
    )
    assert selected == {"unknown"}


def test_dominant_keeps_chunks_without_document_id():
    chunks_map = {
        "no_doc": make_chunk_dict(chunk_id="no_doc", document_id=None),
    }
    selected = select_dominant_versions(
        set(chunks_map), chunks_map, CHECK_DATE
    )
    assert selected == {"no_doc"}


def test_dominant_never_returns_mutually_exclusive_editions():
    chunks_map = {
        "applicable": _version(
            "doc-e", "ver-1", "applicable",
            effective_from="2021-01-01", effective_to=None,
        ),
        "applicable_2": _version(
            "doc-e", "ver-2", "applicable_2",
            effective_from="2023-01-01", effective_to=None,
        ),
    }
    selected = select_dominant_versions(
        set(chunks_map), chunks_map, CHECK_DATE
    )
    versions = {chunks_map[cid]["version_id"] for cid in selected}
    assert len(versions) == 1


# --- resolve_check_date_details -------------------------------------------


def test_resolve_check_date_historical_day():
    check_date, mode, precision = resolve_check_date_details(
        "2024-06-15", ""
    )
    assert check_date == "2024-06-15"
    assert mode == "historical"
    assert precision == "day"


def test_resolve_check_date_year_from_query():
    check_date, mode, precision = resolve_check_date_details(
        None, "условия на 2023 год"
    )
    assert check_date == "2023-06-01"
    assert mode == "historical"
    assert precision == "year"


def test_resolve_check_date_current_day():
    today = datetime.date.today().isoformat()
    check_date, mode, precision = resolve_check_date_details(None, "")
    assert check_date == today
    assert mode == "current"
    assert precision == "day"


def test_resolve_check_date_invalid_event_date():
    today = datetime.date.today().isoformat()
    check_date, mode, precision = resolve_check_date_details(
        "not-a-date", ""
    )
    assert check_date == today
    assert mode == "current"
    assert precision == "none"


# --- resolve_allowed_versions_by_date -------------------------------------


def test_historical_date_selects_historical_edition():
    chunks_map = {
        "old": _version(
            "doc-a", "ver-1", "old",
            effective_from="2021-01-01", effective_to="2024-06-15",
        ),
        "new": _version(
            "doc-a", "ver-2", "new",
            effective_from="2024-06-15", effective_to=None,
        ),
    }
    allowed = resolve_allowed_versions_by_date("2021-06-01", chunks_map)
    assert "old" in allowed
    assert "new" not in allowed


def test_current_date_selects_active_edition_by_effective_from():
    chunks_map = {
        "expired": _version(
            "doc-b", "ver-1", "expired",
            effective_from="2021-01-01", effective_to="2022-01-01",
        ),
        "active": _version(
            "doc-b", "ver-2", "active",
            effective_from="2022-01-01", effective_to=None,
        ),
    }
    allowed = resolve_allowed_versions_by_date("2030-12-31", chunks_map)
    assert "active" in allowed
    assert "expired" not in allowed


def test_future_effective_from_version_excluded():
    chunks_map = {
        "future": _version(
            "doc-c", "ver-future", "future",
            effective_from="2030-12-31", effective_to=None,
        ),
    }
    allowed = resolve_allowed_versions_by_date(CHECK_DATE, chunks_map)
    assert "future" not in allowed


def test_unknown_versions_not_excluded_and_reported():
    chunks_map = {
        "unknown": _version(
            "doc-d", "ver-unknown", "unknown",
            effective_dates_known=False,
        ),
    }
    validity_unknown: set[str] = set()
    allowed = resolve_allowed_versions_by_date(
        CHECK_DATE, chunks_map, validity_unknown=validity_unknown
    )
    assert "unknown" in allowed
    assert "unknown" in validity_unknown


def test_law_before_effective_from_is_kept():
    chunks_map = {
        "law_future": _version(
            "doc-e", "ver-law", "law_future",
            document_type="law",
            effective_from="2030-12-31", effective_to=None,
        ),
    }
    allowed = resolve_allowed_versions_by_date(CHECK_DATE, chunks_map)
    assert "law_future" in allowed


def test_transition_boundary_prefers_new_edition():
    chunks_map = {
        "old": _version(
            "doc-f", "ver-1", "old",
            effective_from="2021-01-01", effective_to="2024-06-15",
        ),
        "new": _version(
            "doc-f", "ver-2", "new",
            effective_from="2024-06-15", effective_to=None,
        ),
    }
    allowed = resolve_allowed_versions_by_date(CHECK_DATE, chunks_map)
    assert "new" in allowed
    assert "old" not in allowed
