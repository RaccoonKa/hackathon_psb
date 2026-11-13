"""Регрессионные тесты различия технической ошибки и отсутствия документов
(пункт 16).

LLM/analysis и сам retrieval замоканы существующими стабами проекта, но
ветвление orchestrator и содержимое trace проверяются на реальном коде.
"""

from unittest.mock import patch

from rag_core import config
from rag_core.generation import orchestrator as orch_mod
from rag_core.generation import prompts
from rag_core.generation import refusal
from rag_core.generation.errors import RetrievalAPIError
from rag_core.generation.retrieval import bridge as bridge_mod
from rag_core.generation.retrieval import evidence as evidence_mod
from rag_core.generation.types import (
    EvidenceAssessment,
    EvidenceBundle,
    PreFilterResult,
    QuerySpec,
    Sufficiency,
    TraceStep,
)
from tests._common import make_query_spec


def _step(name: str) -> TraceStep:
    return TraceStep(step=name, input={}, output={}, latency_ms=0.0)


def _spec() -> QuerySpec:
    return make_query_spec(
        intent="mixed",
        required_slots=["rent_amount"],
        event_date=None,
        rewritten_queries_bm25=["арендная плата Северный Склад"],
        rewritten_queries_dense=["арендная плата Северный Склад"],
    )


def _empty_assessment() -> EvidenceAssessment:
    return EvidenceAssessment(
        covered_slots=[],
        missing_slots=["rent_amount"],
        core_slot_covered=False,
        sufficiency=Sufficiency.INSUFFICIENT,
        degraded_intents=[],
        reason="core_slot_not_covered",
    )


def _empty_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        chunks=[],
        slot_map={"rent_amount": []},
        contradictions=[],
        version_issues=[],
    )


def _run_agent_with(retrieve_side_effect=None, retrieve_return=None):
    spec = _spec()
    retrieval_step = TraceStep(
        step=config.STEP_RAG_RETRIEVAL,
        input={},
        output={},
        latency_ms=0.0,
    )
    retrieve_kwargs = {}
    if retrieve_side_effect is not None:
        retrieve_kwargs["side_effect"] = retrieve_side_effect
    else:
        retrieve_kwargs["return_value"] = (retrieve_return, retrieval_step)

    with patch.object(
        orch_mod,
        "run_prefilter",
        return_value=(PreFilterResult(False, None, None), _step("prefilter")),
    ), patch.object(
        orch_mod,
        "analyze_query",
        return_value=(spec, _step("query_analysis"), False),
    ), patch.object(
        bridge_mod, "retrieve_tasks", **retrieve_kwargs
    ), patch.object(
        evidence_mod,
        "assess",
        return_value=(
            _empty_bundle(),
            _empty_assessment(),
            _step("assess_evidence"),
        ),
    ), patch.object(
        orch_mod.config, "MAX_FOLLOWUP_RETRIEVAL_CYCLES", 0
    ), patch.object(
        orch_mod.config, "TRACE_DETAIL_LEVEL", "basic"
    ), patch.object(
        orch_mod.config, "TRACE_INCLUDE_DIAGNOSTICS", True
    ):
        return orch_mod.run_agent(
            "Какая арендная плата по договору с ООО «Северный Склад»?"
        )


def test_retrieval_exception_maps_to_retrieval_error_result():
    result = _run_agent_with(
        retrieve_side_effect=RetrievalAPIError("synthetic retrieval failure")
    )

    assert result.refused is True
    assert result.refusal_reason == "retrieval_error"
    assert result.final_answer == refusal.RESPONSE_RETRIEVAL_ERROR
    assert result.final_answer != prompts.RESPONSE_REFUSAL_NO_CONTEXT
    assert "retrieval_error" in result.degradation_reasons

    error_steps = [
        step
        for step in result.trace
        if step.step == config.STEP_RAG_RETRIEVAL
        and step.output.get("status") == "error"
    ]
    assert error_steps, "В trace должен быть шаг rag_retrieval со status=error"
    output = error_steps[0].output
    assert output["error_type"] == "RetrievalAPIError"
    assert "synthetic retrieval failure" in output["error_message"]
    assert output.get("error_traceback")
    assert output["degraded"] is True
    assert output["degradation_reason"] == "retrieval_error"


def test_empty_retrieval_maps_to_no_relevant_documents_refusal():
    result = _run_agent_with(retrieve_return=[])

    assert result.refused is True
    assert result.refusal_reason == "no_relevant_documents"
    assert result.final_answer == prompts.RESPONSE_REFUSAL_NO_CONTEXT
    assert "retrieval_error" not in result.degradation_reasons

    retrieval_steps = [
        step for step in result.trace if step.step == config.STEP_RAG_RETRIEVAL
    ]
    assert retrieval_steps
    assert all(
        step.output.get("status") != "error" for step in retrieval_steps
    )


def test_empty_retrieval_does_not_use_retrieval_error_message():
    result = _run_agent_with(retrieve_return=[])

    assert result.final_answer != refusal.RESPONSE_RETRIEVAL_ERROR
    assert result.refusal_reason not in {"retrieval_error"}
