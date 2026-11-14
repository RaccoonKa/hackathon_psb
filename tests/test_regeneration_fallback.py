"""Технические тесты безопасной регенерации (сценарии A/B/C) и инвариантов.

Проверяется только механика согласованности answer_text / claims / source_ids:
никакие юридические формулировки и примеры из ТЗ здесь не оцениваются.
"""

import json
import re
from unittest.mock import MagicMock, patch

from rag_core import config
from rag_core.generation import orchestrator as orch_mod
from rag_core.generation.answer.generator import (
    _validate_draft_consistency,
    regenerate_draft,
)
from rag_core.generation.answer.verification import verify
from rag_core.generation.errors import LLMAPIError
from rag_core.generation.policy.precheck import precheck
from rag_core.generation.retrieval import bridge as bridge_mod
from rag_core.generation.retrieval import evidence as evidence_mod
from rag_core.generation.types import (
    AnswerToQuestionCheck,
    AnswerToQuestionVerdict,
    Claim,
    ClaimStatus,
    ClaimVerdict,
    DraftAnswer,
    EvidenceAssessment,
    PolicyVerdict,
    Premise,
    PreFilterResult,
    Sufficiency,
    TraceStep,
    VerificationReport,
    VerificationVerdict,
)
from tests._common import (
    make_evidence_bundle,
    make_policy_verdict,
    make_query_spec,
    make_retrieved_chunk,
    make_verification_report,
)

LLM_PATH = "rag_core.generation.answer.generator.llm_complete_with_usage"
_SOURCE_TAG_RE = re.compile(r"<source>(.*?)</source>", re.DOTALL)


def _step(name: str) -> TraceStep:
    return TraceStep(step=name, input={}, output={}, latency_ms=0.0)


def _tag_ids(answer_text: str) -> list[str]:
    ids: list[str] = []
    for inner in _SOURCE_TAG_RE.findall(answer_text or ""):
        for sid in re.findall(r"S\d+", inner):
            if sid not in ids:
                ids.append(sid)
    return ids


def _previous_draft() -> DraftAnswer:
    return DraftAnswer(
        answer_text=(
            "Полный старый ответ <source>S1</source> с неподтверждённой частью."
        ),
        claims=[
            Claim(
                text="исходный факт",
                source_ids=["S1"],
                slot="rent_amount",
                original_idx=0,
            ),
            Claim(
                text="неподтверждённое утверждение",
                source_ids=["S1"],
                slot=None,
                original_idx=1,
            ),
        ],
        assumptions=[],
        missing_info=[],
        refusal_reason=None,
    )


def _spec(**kwargs):
    return make_query_spec(intent="semantic", **kwargs)


def _draft_json(answer_text="", claims=None, refusal_reason=None) -> str:
    return json.dumps(
        {
            "answer_text": answer_text,
            "claims": claims or [],
            "assumptions": [],
            "missing_info": [],
            "refusal_reason": refusal_reason,
        },
        ensure_ascii=False,
    )


def _assert_consistency_invariants(draft: DraftAnswer, available: set[str]) -> None:
    claim_ids = {sid for claim in draft.claims for sid in claim.source_ids}
    # каждый <source> в answer_text подкреплён claims
    for sid in _tag_ids(draft.answer_text):
        assert sid in claim_ids, f"tag {sid} не подкреплён claims"
    # claims ссылаются только на доступные источники
    assert claim_ids <= available, f"claims содержат недоступные источники: {claim_ids - available}"
    # обоснованный отказ несовместим с утверждениями
    if draft.refusal_reason:
        assert draft.claims == []
    # claims ⇒ непустой answer_text
    if draft.claims:
        assert draft.answer_text.strip()
    # пустой answer_text ⇒ есть refusal_reason
    if not draft.answer_text.strip():
        assert draft.refusal_reason


def test_scenario_a_rebuilds_answer_from_supported_claims_only():
    previous = _previous_draft()
    with patch(LLM_PATH, side_effect=LLMAPIError("boom")):
        draft, step = regenerate_draft(
            previous=previous,
            report=make_verification_report(supported_locked=[0]),
            chunks=[make_retrieved_chunk(chunk_id="c1")],
            spec=_spec(),
        )

    # Старый answer_text НЕ возвращается: текст пересобран из подтверждённого claim.
    assert draft.answer_text != previous.answer_text
    assert "исходный факт" in draft.answer_text
    assert "неподтверждённая часть" not in draft.answer_text
    assert len(draft.claims) == 1
    assert draft.claims[0].text == "исходный факт"
    assert draft.claims[0].slot == "rent_amount"
    assert draft.refusal_reason is None
    assert any("удалена" in note for note in draft.missing_info)

    _assert_consistency_invariants(draft, {"S1"})

    assert step.step == "regenerate_draft"
    assert step.degradation == "regen_fallback"
    assert step.extra["fallback_scenario"] == "A"
    assert step.output["fallback_used"] is True
    assert step.output["fallback_scenario"] == "A"
    assert step.output["supported_claims_kept"] == 1


def test_scenario_b_returns_empty_answer_and_refusal():
    with patch(LLM_PATH, side_effect=LLMAPIError("boom")):
        draft, step = regenerate_draft(
            previous=_previous_draft(),
            report=make_verification_report(supported_locked=[]),
            chunks=[make_retrieved_chunk(chunk_id="c1")],
            spec=_spec(),
        )

    assert draft.answer_text == ""
    assert draft.claims == []
    assert draft.refusal_reason
    assert any("не подтверждён" in note for note in draft.missing_info)

    _assert_consistency_invariants(draft, {"S1"})

    assert step.degradation == "regen_fallback"
    assert step.extra["fallback_scenario"] == "B"
    assert step.output["supported_claims_kept"] == 0
    assert step.output["answer_rewritten"] is True


def test_scenario_c_flags_tag_without_claim_and_uses_safe_fallback():
    # answer_text ссылается на S2, которого нет ни в claims, ни в контексте.
    payload = _draft_json(answer_text="Ответ <source>S2</source>", claims=[])
    with patch(LLM_PATH, return_value=(payload, 3, 2)):
        draft, step = regenerate_draft(
            previous=_previous_draft(),
            report=make_verification_report(supported_locked=[0]),
            chunks=[make_retrieved_chunk(chunk_id="c1")],
            spec=_spec(),
        )

    assert step.output["consistency_issues_count"] >= 1
    assert step.output["fallback_used"] is True
    assert step.output["fallback_scenario"] == "C"
    assert step.extra["fallback_scenario"] == "C"
    assert step.degradation == "regen_fallback"

    # Неподкреплённый тег удалён, подтверждённый claim сохранён дословно.
    assert "S2" not in draft.answer_text
    assert len(draft.claims) == 1
    assert draft.claims[0].text == "исходный факт"
    assert draft.claims[0].slot == "rent_amount"

    _assert_consistency_invariants(draft, {"S1"})


def test_unverified_verdicts_are_not_actionable_feedback():
    report = make_verification_report(
        supported_locked=[0],
        batch_status=[
            ClaimVerdict(
                claim_idx=0, status=ClaimStatus.CONTRADICTED, quote="q0", source_id="S1"
            ),
            ClaimVerdict(
                claim_idx=1, status=ClaimStatus.UNVERIFIED, quote="", source_id=None
            ),
        ],
    )
    with patch(LLM_PATH, side_effect=LLMAPIError("boom")) as mock_llm:
        regenerate_draft(
            previous=_previous_draft(),
            report=report,
            chunks=[make_retrieved_chunk(chunk_id="c1")],
            spec=_spec(),
        )

    user_prompt = mock_llm.call_args.kwargs["messages"][1]["content"]
    assert '"status": "contradicted"' in user_prompt
    assert '"status": "unverified"' not in user_prompt
    assert "answer_question_feedback" in user_prompt


def test_regenerate_passes_answer_question_feedback_to_prompt_and_trace():
    report = make_verification_report(
        supported_locked=[0],
        answer_to_question=AnswerToQuestionCheck(
            verdict=AnswerToQuestionVerdict.MISMATCH,
            reason="ответ про другой объект",
            quote="ответ относится к другому договору",
        ),
    )
    payload = _draft_json(
        answer_text="исходный факт <source>S1</source>",
        claims=[
            {
                "claim_idx": 0,
                "text": "исходный факт",
                "source_ids": ["S1"],
                "slot": "rent_amount",
            }
        ],
    )
    with patch(LLM_PATH, return_value=(payload, 1, 1)) as mock_llm:
        _, step = regenerate_draft(
            previous=_previous_draft(),
            report=report,
            chunks=[make_retrieved_chunk(chunk_id="c1")],
            spec=_spec(),
        )

    user_prompt = mock_llm.call_args.kwargs["messages"][1]["content"]
    assert "mismatch" in user_prompt
    assert "ответ про другой объект" in user_prompt
    feedback = step.output["answer_question_feedback"]
    assert feedback["verdict"] == "mismatch"
    assert feedback["quote"] == "ответ относится к другому договору"


def test_validate_draft_consistency_detects_violations():
    draft = DraftAnswer(
        answer_text="Текст <source>S2</source>",
        claims=[
            Claim(text="a", source_ids=["S1"], slot="s", original_idx=0),
            Claim(text="b", source_ids=["S1"], slot="s", original_idx=0),
        ],
        assumptions=[],
        missing_info=[],
        refusal_reason="нет данных",
    )
    issues = _validate_draft_consistency(
        draft,
        make_verification_report(supported_locked=[0]),
        {"S1"},
    )
    assert any("tag source_id S2" in issue for issue in issues)
    assert any("refusal_reason present" in issue for issue in issues)
    assert any("duplicate original_idx" in issue for issue in issues)


def _sources_for(count: int) -> set[str]:
    return {f"{config.SOURCE_ID_PREFIX}{i}" for i in range(config.SOURCE_START_INDEX, config.SOURCE_START_INDEX + count)}


def test_regen_claims_source_ids_stay_within_available_sources():
    payload = _draft_json(
        answer_text="факт <source>S1</source>",
        claims=[
            {"claim_idx": 0, "text": "исходный факт", "source_ids": ["S1"], "slot": "rent_amount"},
            {"claim_idx": 5, "text": "лишнее", "source_ids": ["S9"], "slot": None},
        ],
    )
    with patch(LLM_PATH, return_value=(payload, 1, 1)):
        draft, step = regenerate_draft(
            previous=_previous_draft(),
            report=make_verification_report(supported_locked=[0]),
            chunks=[make_retrieved_chunk(chunk_id="c1")],
            spec=_spec(),
        )

    _assert_consistency_invariants(draft, _sources_for(1))
    assert all(sid != "S9" for claim in draft.claims for sid in claim.source_ids)
    assert step.output["answer_rewritten"] is True


# --- Интеграционные проверки trace/policy/orchestrator ---------------------


def test_policy_precheck_trace_has_premise_fields():
    spec = make_query_spec(
        premises=[Premise(text="сумма 100 руб", kind="amount", value="100 руб")],
        required_slots=["general"],
    )
    bundle = make_evidence_bundle(
        chunks=[make_retrieved_chunk(chunk_id="c1", text="плата составляет 100 руб.")]
    )
    assessment = EvidenceAssessment(
        covered_slots=["general"],
        missing_slots=[],
        core_slot_covered=True,
        sufficiency=Sufficiency.SUFFICIENT,
        degraded_intents=[],
        reason="all_required_slots_covered",
    )
    with patch.object(config, "ENABLE_PREMISE_CHECK", True):
        _, step = precheck(spec=spec, bundle=bundle, assessment=assessment)

    assert "premises" in step.output
    assert "premise_mismatch" in step.output
    assert "premise_unknown_count" in step.output


def test_verify_draft_trace_has_answer_to_question_fields():
    draft = DraftAnswer(
        answer_text="факт <source>S1</source>",
        claims=[Claim(text="факт", source_ids=["S1"], slot="rent_amount", original_idx=0)],
        assumptions=[],
        missing_info=[],
        refusal_reason=None,
    )
    with patch.object(config, "ENABLE_LLM_VERIFIER", False):
        _, step = verify(
            draft=draft,
            chunks=[make_retrieved_chunk(chunk_id="c1")],
            spec=make_query_spec(required_slots=["rent_amount"]),
            question="исходный вопрос пользователя",
        )

    assert "answer_to_question" in step.output
    assert "verifier_unavailable" in step.output
    assert "auto_accept_blocked" in step.output
    assert step.input["question_source"] == "original"


def _verification_report_mismatch() -> VerificationReport:
    return make_verification_report(
        batch_status=[
            ClaimVerdict(
                claim_idx=0, status=ClaimStatus.SUPPORTED, quote="q", source_id="S1"
            )
        ],
        supported_locked=[0],
        final_verdict=VerificationVerdict.REGENERATE,
        answer_to_question=AnswerToQuestionCheck(
            verdict=AnswerToQuestionVerdict.MISMATCH,
            reason="ответ не на исходный вопрос",
        ),
    )


def test_orchestrator_passes_original_query_and_regenerates_on_mismatch():
    query = "исходный вопрос про арендную плату"
    spec = make_query_spec(
        intent="semantic",
        required_slots=["rent_amount"],
        rewritten_queries_dense=["переформулировка"],
    )
    chunks = [make_retrieved_chunk(chunk_id="c1", text="плата")]
    assessment = EvidenceAssessment(
        covered_slots=["rent_amount"],
        missing_slots=[],
        core_slot_covered=True,
        sufficiency=Sufficiency.SUFFICIENT,
        degraded_intents=[],
        reason="all_required_slots_covered",
    )
    draft = DraftAnswer(
        answer_text="факт <source>S1</source>",
        claims=[Claim("факт", ["S1"], "rent_amount", 0)],
        assumptions=[],
        missing_info=[],
        refusal_reason=None,
    )
    seen_questions: list[str | None] = []

    def verify_side(**kwargs):
        draft_arg = kwargs.get("draft")
        seen_questions.append(kwargs.get("question"))
        if len(seen_questions) == 1:
            return _verification_report_mismatch(), _step("verify_draft")
        draft_arg.refusal_reason = "answer_question_mismatch"
        return (
            make_verification_report(
                batch_status=[
                    ClaimVerdict(
                        claim_idx=0,
                        status=ClaimStatus.SUPPORTED,
                        quote="q",
                        source_id="S1",
                    )
                ],
                supported_locked=[0],
                final_verdict=VerificationVerdict.REFUSE,
                answer_to_question=AnswerToQuestionCheck(
                    verdict=AnswerToQuestionVerdict.MISMATCH,
                    reason="ответ не на исходный вопрос",
                ),
            ),
            _step("verify_draft"),
        )

    regenerate = MagicMock(return_value=(draft, _step("regenerate_draft")))

    with patch.object(
        orch_mod,
        "run_prefilter",
        return_value=(PreFilterResult(False, None, None), _step("prefilter")),
    ), patch.object(
        orch_mod,
        "analyze_query",
        return_value=(spec, _step("query_analysis"), False),
    ), patch.object(
        bridge_mod,
        "retrieve_tasks",
        return_value=(chunks, _step(config.STEP_RAG_RETRIEVAL)),
    ), patch.object(
        bridge_mod,
        "rerank_chunks",
        return_value=(chunks, _step(config.STEP_RERANKING)),
    ), patch.object(
        evidence_mod,
        "assess",
        return_value=(
            make_evidence_bundle(chunks=chunks, slot_map={"rent_amount": ["c1"]}),
            assessment,
            _step("assess_evidence"),
        ),
    ), patch.object(
        orch_mod,
        "precheck",
        return_value=(make_policy_verdict(), _step("policy_precheck")),
    ), patch.object(
        orch_mod,
        "build_context_and_sources",
        return_value=({"S1": {"chunk_id": "c1"}}, "ctx", _step(config.STEP_CONTEXT_ASSEMBLY)),
    ), patch.object(
        orch_mod,
        "generate_draft",
        return_value=(draft, _step(config.STEP_GENERATION)),
    ), patch.object(
        orch_mod, "verify", side_effect=verify_side
    ), patch.object(
        orch_mod, "regenerate_draft", regenerate
    ), patch.object(
        orch_mod.config, "MAX_REGENERATE_CYCLES", 1
    ), patch.object(
        orch_mod.config, "MAX_FOLLOWUP_RETRIEVAL_CYCLES", 0
    ):
        result = orch_mod.run_agent(query)

    # Исходный query передан в verify (а не переформулировка).
    assert seen_questions == [query, query]
    # MISMATCH инициировал ровно одну регенерацию, затем отказ при исчерпании циклов.
    regenerate.assert_called_once()
    assert result.refused is True
    assert result.refusal_reason


def test_positional_constructors_remain_compatible():
    policy = PolicyVerdict(False, [], False, True)
    assert policy.premise_statuses == []
    report = VerificationReport([], [], [], [0], VerificationVerdict.ACCEPT)
    assert report.answer_to_question is None
    assert report.verifier_unavailable is False
    assert report.auto_accept_blocked is False
    assert report.claims_synthesized is False
