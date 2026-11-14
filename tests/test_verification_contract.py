import json
from unittest.mock import patch

from rag_core import config
from rag_core.generation.answer import verification as verif_mod
from rag_core.generation.types import (
    AnswerToQuestionVerdict,
    Claim,
    ClaimStatus,
    DraftAnswer,
    Premise,
    PremiseStatus,
    VerificationVerdict,
)
from tests._common import (
    make_policy_verdict,
    make_premise_status_entry,
    make_query_spec,
    make_retrieved_chunk,
)


def _claim(text="подтверждённое утверждение", sid="S1", slot=None, idx=0):
    return Claim(text=text, source_ids=[sid] if sid else [], slot=slot, original_idx=idx)


def _draft(claims, answer_text="ответ <source>S1</source>", refusal_reason=None):
    return DraftAnswer(
        answer_text=answer_text,
        claims=list(claims),
        assumptions=[],
        missing_info=[],
        refusal_reason=refusal_reason,
    )


def _chunks():
    return [make_retrieved_chunk(text="фрагмент источника")]


def _llm_json(verdicts, a2q="match", a2q_reason="ok"):
    payload = {"verdicts": verdicts}
    if a2q is not None:
        payload["answer_to_question"] = {
            "verdict": a2q,
            "reason": a2q_reason,
            "quote": None,
        }
    return json.dumps(payload, ensure_ascii=False)


def _supported(claim_idx=0, sid="S1"):
    return {
        "claim_idx": claim_idx,
        "status": "supported",
        "quote": "фрагмент источника",
        "source_id": sid,
    }


def _fake_llm(output=None, error=None, captured=None):
    def _fn(messages, model, temperature, max_tokens, timeout=None):
        if captured is not None:
            captured["messages"] = messages
        if error is not None:
            raise error
        return output, 11, 22

    return _fn


def _run(
    draft,
    spec,
    *,
    llm,
    enabled=True,
    stub=False,
    max_cycles=1,
    can_regenerate=True,
    question="исходный вопрос",
    policy=None,
):
    llm_patch = (
        patch.object(verif_mod, "llm_complete_with_usage", llm)
        if llm is not None
        else patch.object(
            verif_mod,
            "llm_complete_with_usage",
            side_effect=AssertionError("LLM должен быть отключён"),
        )
    )
    with llm_patch, patch.object(config, "ENABLE_LLM_VERIFIER", enabled), patch.object(
        config, "USE_STUB_LLM", stub
    ), patch.object(
        config, "LLM_MODE", config.LLM_MODE_STUB if stub else config.LLM_MODE_REAL
    ), patch.object(
        config, "MAX_REGENERATE_CYCLES", max_cycles
    ):
        return verif_mod.verify(
            draft=draft,
            chunks=_chunks(),
            spec=spec,
            can_regenerate=can_regenerate,
            question=question,
            policy=policy,
        )


def test_a_verifier_receives_original_question():
    captured = {}
    spec = make_query_spec(rewritten_queries_dense=["переписанный запрос"])
    draft = _draft([_claim()])
    report, _ = _run(
        draft,
        spec,
        llm=_fake_llm(_llm_json([_supported()]), captured=captured),
        question="исходный вопрос пользователя",
    )
    user_content = captured["messages"][1]["content"]
    assert "исходный вопрос пользователя" in user_content
    assert "переписанный запрос" not in user_content
    assert report.answer_to_question.verdict == AnswerToQuestionVerdict.MATCH


def test_b_match_and_all_supported_accepts():
    report, _ = _run(
        _draft([_claim()]),
        make_query_spec(),
        llm=_fake_llm(_llm_json([_supported()], a2q="match")),
    )
    assert report.final_verdict == VerificationVerdict.ACCEPT
    assert report.supported_locked == [0]
    assert report.verifier_unavailable is False


def test_c_mismatch_regenerates():
    report, _ = _run(
        _draft([_claim()]),
        make_query_spec(),
        llm=_fake_llm(_llm_json([_supported()], a2q="mismatch")),
        can_regenerate=True,
        max_cycles=1,
    )
    assert report.final_verdict == VerificationVerdict.REGENERATE


def test_d_mismatch_without_cycles_refuses():
    draft = _draft([_claim()])
    report, _ = _run(
        draft,
        make_query_spec(),
        llm=_fake_llm(_llm_json([_supported()], a2q="mismatch")),
        can_regenerate=False,
    )
    assert report.final_verdict == VerificationVerdict.REFUSE
    assert draft.refusal_reason == "answer_question_mismatch"


def test_e_undetermined_with_supported_is_partial_not_accept():
    draft = _draft([_claim()])
    report, _ = _run(
        draft,
        make_query_spec(),
        llm=_fake_llm(_llm_json([_supported()], a2q="undetermined")),
    )
    assert report.final_verdict == VerificationVerdict.PARTIAL
    assert report.final_verdict != VerificationVerdict.ACCEPT
    assert "answer_to_question:undetermined" in draft.missing_info


def test_f_undetermined_when_verifier_unavailable():
    report, _ = _run(
        _draft([_claim()]),
        make_query_spec(),
        llm=_fake_llm(error=RuntimeError("boom")),
    )
    assert report.final_verdict == VerificationVerdict.PARTIAL
    assert report.verifier_unavailable is True
    assert report.answer_to_question.verdict == AnswerToQuestionVerdict.UNDETERMINED


def test_g_prompt_contains_slots_and_premise_statuses():
    captured = {}
    premise = Premise(
        text="предпосылка о ставке",
        kind="rate",
        value="1",
        status=PremiseStatus.UNKNOWN,
        status_reason="not_found",
    )
    spec = make_query_spec(required_slots=["required_param"], premises=[premise])
    _run(
        _draft([_claim()]),
        spec,
        llm=_fake_llm(_llm_json([_supported()]), captured=captured),
    )
    user_content = captured["messages"][1]["content"]
    assert "required_param" in user_content
    assert "предпосылка о ставке" in user_content
    assert '"status": "unknown"' in user_content


def test_h_missing_answer_to_question_is_undetermined_no_accept():
    report, _ = _run(
        _draft([_claim()]),
        make_query_spec(),
        llm=_fake_llm(_llm_json([_supported()], a2q=None)),
    )
    assert report.answer_to_question.verdict == AnswerToQuestionVerdict.UNDETERMINED
    assert report.final_verdict != VerificationVerdict.ACCEPT


def test_i_empty_claims_without_tags_is_partial_blocked():
    draft = _draft([], answer_text="текст без тегов источников")
    report, _ = _run(
        draft,
        make_query_spec(),
        llm=_fake_llm(_llm_json([], a2q="match")),
    )
    assert report.final_verdict == VerificationVerdict.PARTIAL
    assert report.auto_accept_blocked is True
    assert "ответ не содержит проверяемых утверждений" in draft.missing_info


def test_j_empty_claims_with_tags_are_synthesized():
    draft = _draft([], answer_text="факт <source>S1</source>.")
    report, _ = _run(
        draft,
        make_query_spec(),
        llm=_fake_llm(_llm_json([_supported()], a2q="match")),
    )
    assert report.claims_synthesized is True
    assert len(report.batch_status) == 1
    assert report.batch_status[0].status == ClaimStatus.SUPPORTED
    assert report.final_verdict == VerificationVerdict.ACCEPT


def test_k_empty_claims_and_empty_text_refuses():
    report, _ = _run(
        _draft([], answer_text=""),
        make_query_spec(),
        llm=_fake_llm(_llm_json([], a2q=None)),
    )
    assert report.final_verdict == VerificationVerdict.REFUSE


def test_l_llm_exception_makes_all_unverified():
    report, _ = _run(
        _draft([_claim()]),
        make_query_spec(),
        llm=_fake_llm(error=RuntimeError("failure")),
    )
    assert report.verifier_unavailable is True
    assert all(v.status == ClaimStatus.UNVERIFIED for v in report.batch_status)
    assert report.supported_locked == []
    assert report.final_verdict != VerificationVerdict.ACCEPT


def test_m_verifier_disabled_makes_all_unverified():
    report, _ = _run(
        _draft([_claim()]),
        make_query_spec(),
        llm=None,
        enabled=False,
    )
    assert report.verifier_unavailable is True
    assert all(v.status == ClaimStatus.UNVERIFIED for v in report.batch_status)
    assert report.supported_locked == []
    assert report.final_verdict != VerificationVerdict.ACCEPT


def test_n_stub_mode_is_unverified_not_supported():
    report, _ = _run(
        _draft([_claim()]),
        make_query_spec(),
        llm=None,
        stub=True,
    )
    assert all(v.status == ClaimStatus.UNVERIFIED for v in report.batch_status)
    assert not any(v.status == ClaimStatus.SUPPORTED for v in report.batch_status)


def test_o_valid_source_without_verdict_is_unverified():
    report, _ = _run(
        _draft([_claim(sid="S1")]),
        make_query_spec(),
        llm=_fake_llm(_llm_json([], a2q="match")),
    )
    assert report.batch_status[0].status == ClaimStatus.UNVERIFIED
    assert report.batch_status[0].reason == "verdict_missing"
    assert report.supported_locked == []


def test_p_supported_with_invalid_source_becomes_not_in_source():
    report, _ = _run(
        _draft([_claim(sid="S1")]),
        make_query_spec(),
        llm=_fake_llm(_llm_json([_supported(sid="S9")], a2q="match")),
    )
    assert report.batch_status[0].status == ClaimStatus.NOT_IN_SOURCE
    assert report.citation_errors


def test_q_unverified_never_in_supported_locked():
    report, _ = _run(
        _draft([_claim()]),
        make_query_spec(),
        llm=_fake_llm(error=RuntimeError("failure")),
    )
    assert all(
        report.batch_status[i].status == ClaimStatus.SUPPORTED
        for i in report.supported_locked
    )
    assert report.supported_locked == []


def test_r_trace_has_contract_fields():
    report, step = _run(
        _draft([_claim()]),
        make_query_spec(),
        llm=_fake_llm(_llm_json([_supported()], a2q="match")),
    )
    for key in (
        "answer_to_question",
        "verifier_unavailable",
        "verifier_unavailable_reason",
        "auto_accept_blocked",
        "claims_synthesized",
        "supported_claims_count",
        "contradicted_claims_count",
        "not_in_source_claims_count",
        "unverified_claims_count",
        "text_claim_consistency",
        "text_claim_issues_count",
        "pruned_claims_count",
        "question_source",
    ):
        assert key in step.output
    assert step.output["question_source"] == "original"
    # Trace по разделу 10: счётчики статусов claims и сверка текста с claims.
    assert step.output["supported_claims_count"] == 1
    assert step.output["contradicted_claims_count"] == 0
    assert step.output["not_in_source_claims_count"] == 0
    assert step.output["text_claim_consistency"] is True
    assert step.output["text_claim_issues_count"] == 0


def test_answer_text_tag_outside_claims_is_not_accept():
    # answer_text содержит <source>-тег на источник, которого нет в claims.
    draft = _draft(
        [_claim(sid="S1")],
        answer_text="подтверждено <source>S1</source>, лишнее <source>S2</source>.",
    )
    report, step = _run(
        draft,
        make_query_spec(),
        llm=_fake_llm(_llm_json([_supported(sid="S1")], a2q="match")),
        can_regenerate=False,
    )
    assert report.final_verdict != VerificationVerdict.ACCEPT
    assert report.text_claim_consistency is False
    assert report.text_claim_issues
    assert step.output["text_claim_consistency"] is False
    assert step.output["text_claim_issues_count"] >= 1
    # PARTIAL после очистки: неподкреплённый тег удалён, подтверждённый сохранён.
    assert "S2" not in draft.answer_text
    assert "S1" in draft.answer_text


def test_premise_contradicted_blocks_accept():
    policy = make_policy_verdict(
        premise_statuses=[
            make_premise_status_entry(
                text="предпосылка", status=PremiseStatus.CONTRADICTED
            )
        ]
    )
    report, _ = _run(
        _draft([_claim()]),
        make_query_spec(),
        llm=_fake_llm(_llm_json([_supported()], a2q="match")),
        policy=policy,
    )
    assert report.final_verdict != VerificationVerdict.ACCEPT


def test_partial_prunes_answer_to_supported_sources():
    draft = _draft(
        [
            _claim(text="подтверждено", sid="S1", idx=0),
            _claim(text="не подтверждено", sid="S2", idx=1),
        ],
        answer_text=(
            "подтверждено <source>S1</source>. "
            "не подтверждено <source>S2</source>."
        ),
    )
    report, _ = _run(
        draft,
        make_query_spec(),
        llm=_fake_llm(
            _llm_json(
                [
                    _supported(claim_idx=0, sid="S1"),
                    {
                        "claim_idx": 1,
                        "status": "not_in_source",
                        "quote": "",
                        "source_id": "S2",
                    },
                ],
                a2q="match",
            )
        ),
        can_regenerate=False,
    )
    assert report.final_verdict == VerificationVerdict.PARTIAL
    assert "S1" in draft.answer_text
    assert "S2" not in draft.answer_text
    assert len(draft.claims) == 1
