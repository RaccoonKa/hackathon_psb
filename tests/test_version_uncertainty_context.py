"""Технические тесты неопределённости редакции и контекста (P1, п. 8–10).

Проверяется, что ``assess`` формирует ``generation_warnings``/``
missing_source_kinds`` без изменения ``version_issues``, а
``context_builder`` выводит назначение источника и статус редакции только
при наличии данных и никогда не подтверждает неизвестный период действия.
Все данные синтетические, сеть не используется.
"""

import json
from unittest.mock import patch

from rag_core.generation.answer import generator as gen_mod
from rag_core.generation.answer.context_builder import (
    build_context_and_sources,
    build_source_entry,
    format_single_chunk,
)
from rag_core.generation.prompts import templates as prompts
from rag_core.generation.retrieval.evidence import assess
from tests._common import (
    make_evidence_bundle,
    make_policy_verdict,
    make_query_spec,
    make_retrieved_chunk,
)


def _assess(spec, chunks):
    bundle, _assessment, _step = assess(spec, chunks)
    return bundle


def _warning_kinds(bundle):
    return {warning.get("kind") for warning in bundle.generation_warnings}


# --- assess: предупреждения ------------------------------------------------


def test_unknown_version_adds_warning_not_version_issues():
    spec = make_query_spec()
    chunks = [
        make_retrieved_chunk(
            chunk_id="c1",
            version_status="unknown",
            validity_uncertain=True,
        )
    ]
    bundle = _assess(spec, chunks)

    assert "validity_unknown" in _warning_kinds(bundle)
    assert bundle.version_issues == []
    assert bundle.missing_source_kinds == []


def test_validity_uncertain_flag_alone_adds_warning():
    spec = make_query_spec()
    chunks = [make_retrieved_chunk(chunk_id="c1", validity_uncertain=True)]
    bundle = _assess(spec, chunks)
    assert "validity_unknown" in _warning_kinds(bundle)


def test_current_only_adds_warning():
    spec = make_query_spec()
    chunks = [make_retrieved_chunk(chunk_id="c1", version_status="current_only")]
    bundle = _assess(spec, chunks)
    assert "only_current_revision_available" in _warning_kinds(bundle)


def test_missing_source_kind_detected():
    spec = make_query_spec(required_source_kinds=["normative"])
    chunks = [make_retrieved_chunk(chunk_id="c1", source_kind="specific")]
    bundle = _assess(spec, chunks)

    assert bundle.missing_source_kinds == ["normative"]
    assert "missing_source_kind" in _warning_kinds(bundle)


def test_present_source_kind_not_missing():
    spec = make_query_spec(required_source_kinds=["normative"])
    chunks = [make_retrieved_chunk(chunk_id="c1", source_kind="normative")]
    bundle = _assess(spec, chunks)
    assert bundle.missing_source_kinds == []
    assert "missing_source_kind" not in _warning_kinds(bundle)


def test_approximate_event_date_month():
    spec = make_query_spec(date_precision="month")
    bundle = _assess(spec, [])
    assert "approximate_event_date" in _warning_kinds(bundle)


def test_approximate_event_date_year():
    spec = make_query_spec(date_precision="year")
    bundle = _assess(spec, [])
    assert "approximate_event_date" in _warning_kinds(bundle)


def test_generation_warnings_are_deduplicated():
    spec = make_query_spec()
    chunks = [
        make_retrieved_chunk(chunk_id="c1", version_status="unknown"),
        make_retrieved_chunk(chunk_id="c1", version_status="unknown"),
    ]
    bundle = _assess(spec, chunks)
    validity = [
        warning
        for warning in bundle.generation_warnings
        if warning.get("kind") == "validity_unknown"
    ]
    assert len(validity) == 1


# --- context_builder: назначение источника и статус редакции ---------------


def test_no_source_role_or_status_lines_by_default():
    out = format_single_chunk("S1", make_retrieved_chunk())
    assert prompts.SOURCE_ROLE_LINE_PREFIX not in out
    assert prompts.VERSION_STATUS_LINE_PREFIX not in out


def test_normative_source_role_line():
    chunk = make_retrieved_chunk(source_kind="normative")
    out = format_single_chunk("S1", chunk)
    assert "Назначение источника: нормативное основание" in out


def test_specific_source_role_line():
    chunk = make_retrieved_chunk(source_kind="specific")
    out = format_single_chunk("S1", chunk)
    assert "Назначение источника: документ сделки" in out


def test_unknown_version_status_hides_confirmed_period():
    chunk = make_retrieved_chunk(
        version_status="unknown",
        effective_from="2021-01-01",
        effective_to="2030-12-31",
    )
    out = format_single_chunk("S1", chunk)

    assert "Статус редакции: период действия не определён" in out
    assert prompts.EFFECTIVE_LINE_PREFIX not in out
    assert "2021-01-01" not in out
    assert "2030-12-31" not in out


def test_current_only_status_line():
    chunk = make_retrieved_chunk(version_status="current_only")
    out = format_single_chunk("S1", chunk)
    assert "доступна только текущая редакция" in out


def test_applicable_historical_status_line():
    chunk = make_retrieved_chunk(
        version_status="applicable",
        effective_date_mode="historical",
    )
    out = format_single_chunk("S1", chunk)
    assert "применима на дату события" in out


def test_applicable_current_status_line():
    chunk = make_retrieved_chunk(
        version_status="applicable",
        effective_date_mode="current",
    )
    out = format_single_chunk("S1", chunk)
    assert "действует на текущую дату" in out


def test_validity_uncertain_hides_period_without_confirmed_status():
    chunk = make_retrieved_chunk(
        validity_uncertain=True,
        effective_from="2021-01-01",
        effective_to="2030-12-31",
    )
    out = format_single_chunk("S1", chunk)
    assert "2021-01-01" not in out
    assert "2030-12-31" not in out


def test_build_source_entry_has_no_new_user_fields():
    chunk = make_retrieved_chunk(
        source_kind="normative",
        version_status="unknown",
    )
    entry = build_source_entry(chunk)

    assert "source_kind" not in entry
    assert "version_status" not in entry
    assert "validity_uncertain" not in entry
    assert "document_type" not in entry


def test_sources_dict_has_no_new_user_fields():
    chunks = [
        make_retrieved_chunk(
            chunk_id="c1", source_kind="normative", version_status="unknown"
        )
    ]
    sources, _context, _step = build_context_and_sources(chunks)
    assert set(sources["S1"]) == {
        "document_id",
        "version_id",
        "chunk_id",
        "document_title",
        "provision_id",
        "structure_path",
        "text",
        "location",
        "score",
        "effective_from",
        "effective_to",
        "effective_dates_known",
        "amends_document_id",
    }


# --- generation_warnings в GENERATION_USER --------------------------------


def test_generation_warnings_are_passed_to_generation_user_prompt():
    captured: dict = {}

    def fake_llm(messages, model, temperature, max_tokens, timeout=None):
        captured["messages"] = messages
        return (
            json.dumps(
                {
                    "answer_text": "Ответ <source>S1</source>",
                    "claims": [],
                    "assumptions": [],
                    "missing_info": [],
                    "refusal_reason": None,
                },
                ensure_ascii=False,
            ),
            1,
            1,
        )

    warning = {"kind": "validity_unknown", "chunk_id": "c1"}
    evidence = make_evidence_bundle(
        generation_warnings=[warning],
        version_issues=[{"kind": "no_effective_metadata"}],
    )
    spec = make_query_spec()

    with patch.object(gen_mod, "llm_complete_with_usage", fake_llm):
        draft, _step = gen_mod.generate_draft(
            spec=spec,
            evidence=evidence,
            policy=make_policy_verdict(),
            context="контекст",
            query="вопрос",
        )

    assert draft.answer_text
    user_content = captured["messages"][1]["content"]
    assert "validity_unknown" in user_content
    # Существующий version_issues тоже попадает в тот же аргумент.
    assert "no_effective_metadata" in user_content
    assert "version_issues" in user_content
