"""Технические тесты проверки соответствия роли источника (P1, п. 5 ТЗ).

Проверяется ``_source_role_mismatch`` и его интеграция в ``verify``:
несоответствие назначения источника слоту переводит SUPPORTED-claim в
``NOT_IN_SOURCE`` и фиксируется в ``VerificationReport.source_role_issues``,
не переопределяя UNVERIFIED/CONTRADICTED. LLM замокан, сети нет.
"""

import copy
import json
from unittest.mock import patch

from rag_core import config
from rag_core.generation.answer import verification as verif_mod
from rag_core.generation.answer.verification import _source_role_mismatch
from rag_core.generation.retrieval.evidence import assess
from rag_core.generation.types import (
    Claim,
    ClaimStatus,
    DraftAnswer,
    VerificationVerdict,
)
from tests._common import make_query_spec, make_retrieved_chunk


def _claim(slot, sid="S1", text="утверждение без дат"):
    return Claim(text=text, source_ids=[sid], slot=slot, original_idx=0)


def _draft(claim):
    return DraftAnswer(
        answer_text="утверждение <source>S1</source>",
        claims=[claim],
        assumptions=[],
        missing_info=[],
        refusal_reason=None,
    )


def _llm_json(status="supported", sid="S1", a2q="match"):
    return json.dumps(
        {
            "verdicts": [
                {
                    "claim_idx": 0,
                    "status": status,
                    "quote": "цитата",
                    "source_id": sid,
                }
            ],
            "answer_to_question": {
                "verdict": a2q,
                "reason": "ok",
                "quote": None,
            },
        },
        ensure_ascii=False,
    )


def _fake_llm(output=None, error=None):
    def _fn(messages, model, temperature, max_tokens, timeout=None):
        if error is not None:
            raise error
        return output, 1, 1

    return _fn


def _run(draft, spec, chunks, *, llm):
    with patch.object(verif_mod, "llm_complete_with_usage", llm), patch.object(
        config, "ENABLE_LLM_VERIFIER", True
    ), patch.object(config, "USE_STUB_LLM", False), patch.object(
        config, "LLM_MODE", config.LLM_MODE_REAL
    ), patch.object(
        config, "MAX_REGENERATE_CYCLES", 0
    ):
        return verif_mod.verify(
            draft=draft,
            chunks=chunks,
            spec=spec,
            can_regenerate=False,
            question="вопрос",
        )


# --- Чистая проверка -------------------------------------------------------


def test_normative_slot_with_specific_chunks_is_mismatch():
    claim = _claim("approval_process")
    chunks = [make_retrieved_chunk(chunk_id="c1", source_kind="specific")]
    mismatch, reason = _source_role_mismatch(claim, chunks)
    assert mismatch is True
    assert reason == "source_role_mismatch"


def test_specific_slot_with_normative_chunks_is_mismatch():
    claim = _claim("rent_amount")
    chunks = [make_retrieved_chunk(chunk_id="c1", source_kind="normative")]
    mismatch, _reason = _source_role_mismatch(claim, chunks)
    assert mismatch is True


def test_one_matching_chunk_prevents_mismatch():
    claim = _claim("approval_process", sid="S1")
    chunks = [
        make_retrieved_chunk(chunk_id="c1", source_kind="specific"),
        make_retrieved_chunk(chunk_id="c2", source_kind="normative"),
    ]
    mismatch, _reason = _source_role_mismatch(claim, chunks)
    assert mismatch is False


def test_unknown_source_kind_prevents_mismatch():
    claim = _claim("approval_process")
    for kind in (None, "unknown"):
        chunks = [make_retrieved_chunk(chunk_id="c1", source_kind=kind)]
        mismatch, _reason = _source_role_mismatch(claim, chunks)
        assert mismatch is False


def test_slot_without_strict_mapping_prevents_mismatch():
    claim = _claim("general")
    chunks = [make_retrieved_chunk(chunk_id="c1", source_kind="specific")]
    mismatch, _reason = _source_role_mismatch(claim, chunks)
    assert mismatch is False


def test_no_cited_chunks_prevents_mismatch():
    claim = _claim("approval_process")
    mismatch, _reason = _source_role_mismatch(claim, [])
    assert mismatch is False


def test_source_role_mismatch_does_not_mutate_inputs():
    claim = _claim("approval_process")
    chunks = [make_retrieved_chunk(chunk_id="c1", source_kind="specific")]
    claim_before = copy.deepcopy(claim)
    chunks_before = copy.deepcopy(chunks)

    _source_role_mismatch(claim, chunks)

    assert claim == claim_before
    assert chunks == chunks_before


# --- Интеграция в verify ---------------------------------------------------


def test_verify_marks_not_in_source_and_records_issue():
    claim = _claim("approval_process")
    spec = make_query_spec(required_slots=["approval_process"])
    chunks = [make_retrieved_chunk(chunk_id="c1", source_kind="specific")]

    report, _step = _run(
        _draft(claim), spec, chunks, llm=_fake_llm(_llm_json())
    )

    assert report.batch_status[0].status == ClaimStatus.NOT_IN_SOURCE
    assert report.batch_status[0].reason == "source_role_mismatch"
    assert report.source_role_issues
    assert "approval_process" in report.source_role_issues[0]
    assert report.supported_locked == []


def test_verify_specific_claim_with_normative_chunks():
    claim = _claim("rent_amount")
    spec = make_query_spec(required_slots=["rent_amount"])
    chunks = [make_retrieved_chunk(chunk_id="c1", source_kind="normative")]

    report, _step = _run(
        _draft(claim), spec, chunks, llm=_fake_llm(_llm_json())
    )

    assert report.batch_status[0].status == ClaimStatus.NOT_IN_SOURCE
    assert report.source_role_issues


def test_verify_matching_role_keeps_supported():
    claim = _claim("approval_process")
    spec = make_query_spec(required_slots=["approval_process"])
    chunks = [make_retrieved_chunk(chunk_id="c1", source_kind="normative")]

    report, _step = _run(
        _draft(claim), spec, chunks, llm=_fake_llm(_llm_json())
    )

    assert report.batch_status[0].status == ClaimStatus.SUPPORTED
    assert report.source_role_issues == []
    assert report.final_verdict == VerificationVerdict.ACCEPT


def test_verify_unknown_kind_keeps_supported():
    claim = _claim("approval_process")
    spec = make_query_spec(required_slots=["approval_process"])
    chunks = [make_retrieved_chunk(chunk_id="c1", source_kind=None)]

    report, _step = _run(
        _draft(claim), spec, chunks, llm=_fake_llm(_llm_json())
    )

    assert report.batch_status[0].status == ClaimStatus.SUPPORTED
    assert report.source_role_issues == []


def test_verify_does_not_override_unverified():
    claim = _claim("approval_process")
    spec = make_query_spec(required_slots=["approval_process"])
    chunks = [make_retrieved_chunk(chunk_id="c1", source_kind="specific")]

    report, _step = _run(
        _draft(claim),
        spec,
        chunks,
        llm=_fake_llm(error=RuntimeError("boom")),
    )

    assert report.batch_status[0].status == ClaimStatus.UNVERIFIED
    assert report.source_role_issues == []


def test_verify_does_not_override_contradicted():
    claim = _claim("approval_process")
    spec = make_query_spec(required_slots=["approval_process"])
    chunks = [make_retrieved_chunk(chunk_id="c1", source_kind="specific")]

    report, _step = _run(
        _draft(claim),
        spec,
        chunks,
        llm=_fake_llm(_llm_json(status="contradicted")),
    )

    assert report.batch_status[0].status == ClaimStatus.CONTRADICTED
    assert report.source_role_issues == []


def test_verify_does_not_change_sufficiency_or_add_confidence():
    claim = _claim("approval_process")
    spec = make_query_spec(required_slots=["approval_process"])
    chunks = [make_retrieved_chunk(chunk_id="c1", source_kind="specific")]

    _bundle, assessment, _assess_step = assess(spec, chunks)
    sufficiency_before = assessment.sufficiency

    report, _step = _run(
        _draft(claim), spec, chunks, llm=_fake_llm(_llm_json())
    )

    assert assessment.sufficiency == sufficiency_before
    assert not hasattr(report, "confidence")
    assert not hasattr(report, "sufficiency")
