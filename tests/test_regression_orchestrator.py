"""Регрессионные тесты различия технической ошибки и отсутствия документов
(пункт 16), а также разделения состояния A/B тайм-аутов retrieval (Этап 2).

LLM/analysis и сам retrieval замоканы существующими стабами проекта, но
ветвление orchestrator и содержимое trace проверяются на реальном коде.
"""

from unittest.mock import MagicMock, patch

from rag_core import config
from rag_core.generation import orchestrator as orch_mod
from rag_core.generation import prompts
from rag_core.generation import refusal
from rag_core.generation.errors import (
    RetrievalAPIError,
    RetrievalBusyError,
    RetrievalTimeoutError,
)
from rag_core.generation.retrieval import bridge as bridge_mod
from rag_core.generation.retrieval import evidence as evidence_mod
from rag_core.generation.types import (
    DraftAnswer,
    EvidenceAssessment,
    EvidenceBundle,
    PreFilterResult,
    QuerySpec,
    RetrievalTask,
    Sufficiency,
    TraceStep,
)
from tests._common import (
    make_policy_verdict,
    make_query_spec,
    make_retrieved_chunk,
    make_verification_report,
)


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


# --- Этап 2: состояние A (поиск завершился) vs B (технический сбой) --------


class BatchDeadlineExceeded(Exception):
    """Имитация типа из diagnostics.errors (мост: «дедлайн батча исчерпан»)."""


def _busy_error() -> Exception:
    exc = RetrievalBusyError("Предыдущий батч всё ещё удерживает потоки")
    exc.threads_alive = 2
    return exc


def _run_technical_failure(exc: Exception, followup_cycles: int = 2):
    """Запускает run_agent при технической ошибке retrieval (состояние B)."""
    spec = _spec()
    retrieve = MagicMock(side_effect=exc)
    followup = MagicMock()
    generate = MagicMock()

    with patch.object(
        orch_mod,
        "run_prefilter",
        return_value=(PreFilterResult(False, None, None), _step("prefilter")),
    ), patch.object(
        orch_mod,
        "analyze_query",
        return_value=(spec, _step("query_analysis"), False),
    ), patch.object(
        bridge_mod, "retrieve_tasks", retrieve
    ), patch.object(
        evidence_mod,
        "assess",
        return_value=(
            _empty_bundle(),
            _empty_assessment(),
            _step("assess_evidence"),
        ),
    ), patch.object(
        orch_mod.planner, "followup", followup
    ), patch.object(
        orch_mod, "generate_draft", generate
    ), patch.object(
        orch_mod.config, "MAX_FOLLOWUP_RETRIEVAL_CYCLES", followup_cycles
    ), patch.object(
        orch_mod.config, "TRACE_DETAIL_LEVEL", "basic"
    ), patch.object(
        orch_mod.config, "TRACE_INCLUDE_DIAGNOSTICS", True
    ):
        result = orch_mod.run_agent(
            "Какая арендная плата по договору с ООО «Северный Склад»?"
        )
    return result, retrieve, followup, generate


def _assert_technical_refusal(
    result,
    retrieve: MagicMock,
    followup: MagicMock,
    generate: MagicMock,
    expected_error_type: str,
    message_fragment: str,
) -> None:
    # Контролируемый технический отказ, а не «недостаточно сведений».
    assert result.refused is True
    assert result.refusal_reason == "retrieval_error"
    assert result.final_answer == refusal.RESPONSE_RETRIEVAL_ERROR
    assert result.final_answer != prompts.RESPONSE_REFUSAL_NO_CONTEXT
    assert "retrieval_error" in result.degradation_reasons
    assert "technical_retrieval_failure" in result.degradation_reasons

    # Followup по технической ошибке не запускается, повторного retrieval
    # нет, генерация с пустым контекстом не вызывается.
    assert followup.call_count == 0
    assert generate.call_count == 0
    assert retrieve.call_count == 1

    error_steps = [
        step
        for step in result.trace
        if step.step == config.STEP_RAG_RETRIEVAL
        and step.output.get("status") == "error"
    ]
    assert error_steps, "В trace должен быть шаг rag_retrieval со status=error"
    output = error_steps[0].output
    assert output["error_type"] == expected_error_type
    assert message_fragment in output["error_message"]
    assert output["technical_retrieval_failure"] is True

    refusal_steps = [
        step for step in result.trace if step.step == config.STEP_REFUSAL
    ]
    assert refusal_steps, "В trace должен быть шаг отказа"
    refusal_step = refusal_steps[-1]
    assert refusal_step.input.get("reason") == "retrieval_error"
    assert refusal_step.output.get("technical_retrieval_failure") is True
    assert expected_error_type in (refusal_step.output.get("error_types") or [])


def test_retrieval_timeout_does_not_trigger_followup():
    result, retrieve, followup, generate = _run_technical_failure(
        RetrievalTimeoutError("Все retrieval-задачи превысили дедлайн")
    )
    _assert_technical_refusal(
        result,
        retrieve,
        followup,
        generate,
        "RetrievalTimeoutError",
        "превысили дедлайн",
    )


def test_retrieval_busy_does_not_trigger_followup():
    result, retrieve, followup, generate = _run_technical_failure(_busy_error())
    _assert_technical_refusal(
        result,
        retrieve,
        followup,
        generate,
        "RetrievalBusyError",
        "удерживает потоки",
    )


def test_retrieval_api_error_does_not_trigger_followup():
    result, retrieve, followup, generate = _run_technical_failure(
        RetrievalAPIError("synthetic retrieval failure")
    )
    _assert_technical_refusal(
        result,
        retrieve,
        followup,
        generate,
        "RetrievalAPIError",
        "synthetic retrieval failure",
    )


def test_batch_deadline_exceeded_does_not_trigger_followup():
    result, retrieve, followup, generate = _run_technical_failure(
        BatchDeadlineExceeded("Дедлайн батча 20.0 с исчерпан")
    )
    _assert_technical_refusal(
        result,
        retrieve,
        followup,
        generate,
        "BatchDeadlineExceeded",
        "Дедлайн батча",
    )


def test_state_a_successful_retrieval_keeps_followup():
    """Состояние A: поиск завершился, слоты не покрыты — followup работает."""
    spec = _spec()
    chunks = [
        make_retrieved_chunk(
            chunk_id="chunk_1", text="арендная плата 450 000", score=0.5
        )
    ]

    def _retrieval_step() -> TraceStep:
        return TraceStep(
            step=config.STEP_RAG_RETRIEVAL, input={}, output={}, latency_ms=0.0
        )

    retrieve = MagicMock(
        side_effect=[(chunks, _retrieval_step()), (chunks, _retrieval_step())]
    )
    followup_tasks = [
        RetrievalTask(
            query="второй запрос про арендную плату",
            mode="bm25",
            purpose="lookup",
            selected_doc_ids=None,
            boost_doc_ids=None,
            event_date=None,
            top_k=5,
        )
    ]
    followup = MagicMock(
        return_value=(followup_tasks, _step("plan_followup"))
    )
    assess_calls = {"n": 0}

    def assess_side_effect(
        spec,
        chunks,
        on_failure=None,
        on_llm_success=None,
        timeout=None,
    ):
        assess_calls["n"] += 1
        if assess_calls["n"] == 1:
            return _empty_bundle(), _empty_assessment(), _step("assess_evidence")
        bundle = EvidenceBundle(
            chunks=list(chunks),
            slot_map={"rent_amount": ["chunk_1"]},
            contradictions=[],
            version_issues=[],
        )
        assessment = EvidenceAssessment(
            covered_slots=["rent_amount"],
            missing_slots=[],
            core_slot_covered=True,
            sufficiency=Sufficiency.SUFFICIENT,
            degraded_intents=[],
            reason="all_required_slots_covered",
        )
        return bundle, assessment, _step("assess_evidence")

    def rerank_side_effect(chunks, query, top_n):
        return list(chunks[:top_n]), _step(config.STEP_RERANKING)

    generate = MagicMock(
        return_value=(
            DraftAnswer(
                "Арендная плата 450 000 <source>S1</source>",
                [],
                [],
                [],
                None,
            ),
            _step(config.STEP_GENERATION),
        )
    )

    with patch.object(
        orch_mod,
        "run_prefilter",
        return_value=(PreFilterResult(False, None, None), _step("prefilter")),
    ), patch.object(
        orch_mod,
        "analyze_query",
        return_value=(spec, _step("query_analysis"), False),
    ), patch.object(
        bridge_mod, "retrieve_tasks", retrieve
    ), patch.object(
        bridge_mod, "rerank_chunks", side_effect=rerank_side_effect
    ), patch.object(
        evidence_mod, "assess", side_effect=assess_side_effect
    ), patch.object(
        orch_mod.planner, "followup", followup
    ), patch.object(
        orch_mod,
        "precheck",
        return_value=(make_policy_verdict(), _step("policy_precheck")),
    ), patch.object(
        orch_mod,
        "build_context_and_sources",
        return_value=(
            {"S1": {"chunk_id": "chunk_1"}},
            "[S1] арендная плата 450 000",
            _step(config.STEP_CONTEXT_ASSEMBLY),
        ),
    ), patch.object(
        orch_mod, "generate_draft", generate
    ), patch.object(
        orch_mod, "verify", return_value=(make_verification_report(), _step("verify_draft"))
    ), patch.object(
        orch_mod.config, "MAX_FOLLOWUP_RETRIEVAL_CYCLES", 2
    ), patch.object(
        orch_mod.config, "TRACE_DETAIL_LEVEL", "basic"
    ), patch.object(
        orch_mod.config, "TRACE_INCLUDE_DIAGNOSTICS", True
    ):
        result = orch_mod.run_agent(
            "Какая арендная плата по договору с ООО «Северный Склад»?"
        )

    # Followup запущен ровно один раз, retrieval выполнен дважды.
    assert followup.call_count == 1
    assert retrieve.call_count == 2
    # Доказательств достаточно — генерация вызвана с реальным контекстом.
    assert generate.call_count == 1
    assert result.refused is False
    assert result.refusal_reason is None
    assert "technical_retrieval_failure" not in result.degradation_reasons


def test_degraded_partial_retrieval_does_not_trigger_followup():
    """Частичный сбой (degraded) — техническая ошибка без followup."""
    spec = _spec()
    chunks = [
        make_retrieved_chunk(chunk_id="chunk_1", text="текст", score=0.5)
    ]

    def retrieve_degraded(tasks, diagnostics_out=None):
        if diagnostics_out is not None:
            diagnostics_out["degraded"] = True
            diagnostics_out["executor"] = {
                "submitted": len(tasks),
                "completed": 1,
                "failed": 1,
            }
            diagnostics_out["errors"] = [
                {"type": "TimeoutError", "message": "task timeout"}
            ]
            diagnostics_out["tasks"] = []
        return chunks, TraceStep(
            step=config.STEP_RAG_RETRIEVAL, input={}, output={}, latency_ms=0.0
        )

    retrieve = MagicMock(side_effect=retrieve_degraded)
    followup = MagicMock()
    generate = MagicMock()

    def rerank_side_effect(chunks, query, top_n):
        return list(chunks[:top_n]), _step(config.STEP_RERANKING)

    with patch.object(
        orch_mod,
        "run_prefilter",
        return_value=(PreFilterResult(False, None, None), _step("prefilter")),
    ), patch.object(
        orch_mod,
        "analyze_query",
        return_value=(spec, _step("query_analysis"), False),
    ), patch.object(
        bridge_mod, "retrieve_tasks", retrieve
    ), patch.object(
        bridge_mod, "rerank_chunks", side_effect=rerank_side_effect
    ), patch.object(
        evidence_mod,
        "assess",
        return_value=(
            _empty_bundle(),
            _empty_assessment(),
            _step("assess_evidence"),
        ),
    ), patch.object(
        orch_mod.planner, "followup", followup
    ), patch.object(
        orch_mod, "generate_draft", generate
    ), patch.object(
        orch_mod.config, "MAX_FOLLOWUP_RETRIEVAL_CYCLES", 2
    ), patch.object(
        orch_mod.config, "TRACE_DETAIL_LEVEL", "basic"
    ), patch.object(
        orch_mod.config, "TRACE_INCLUDE_DIAGNOSTICS", True
    ):
        result = orch_mod.run_agent(
            "Какая арендная плата по договору с ООО «Северный Склад»?"
        )

    assert followup.call_count == 0
    assert generate.call_count == 0
    assert retrieve.call_count == 1
    assert result.refused is True
    assert result.refusal_reason == "retrieval_error"
    assert result.final_answer == refusal.RESPONSE_RETRIEVAL_ERROR

    partial_steps = [
        step for step in result.trace if step.step == config.STEP_RAG_RETRIEVAL
    ]
    assert partial_steps
    output = partial_steps[0].output
    assert output["degraded"] is True
    assert output["technical_retrieval_failure"] is True
    assert output["degradation_reason"] == "retrieval_partial_failure"
    assert any(
        error.get("type") == "TimeoutError"
        for error in (output.get("errors") or [])
    )

    refusal_steps = [
        step for step in result.trace if step.step == config.STEP_REFUSAL
    ]
    assert refusal_steps
    assert "TimeoutError" in (refusal_steps[-1].output.get("error_types") or [])
