from unittest import SkipTest

from rag_core import config
from rag_core.generation import prompts
from rag_core.generation.refusal import (
    build_no_context_refusal_result,
    build_template_intent_result,
)
from rag_core.generation.types import AgentResult, Intent


def test_template_key_out_of_scope():
    result = build_template_intent_result(
        query="?",
        intent=Intent.OUT_OF_SCOPE,
        prior_trace=[],
        template_key=config.RESPONSE_OUT_OF_SCOPE,
    )
    assert result.final_answer == prompts.RESPONSE_OUT_OF_SCOPE


def test_template_key_about_assistant():
    result = build_template_intent_result(
        query="?",
        intent=Intent.ABOUT_ASSISTANT,
        prior_trace=[],
        template_key=config.RESPONSE_ABOUT_ASSISTANT,
    )
    assert result.final_answer == prompts.RESPONSE_ABOUT_ASSISTANT


def test_template_key_no_context():
    result = build_template_intent_result(
        query="?",
        intent=Intent.SEMANTIC_QUERY,
        prior_trace=[],
        template_key=config.RESPONSE_REFUSAL_NO_CONTEXT,
    )
    assert result.final_answer == prompts.RESPONSE_REFUSAL_NO_CONTEXT


def test_unknown_template_key_falls_back():
    result = build_template_intent_result(
        query="?",
        intent=Intent.OUT_OF_SCOPE,
        prior_trace=[],
        template_key="UNKNOWN_KEY",
    )
    assert result.final_answer == prompts.RESPONSE_OUT_OF_SCOPE


def test_resolve_template_unknown_raises_removed_api():
    raise SkipTest(
        "resolve_template() удалён вместе с src/generation/generator.py: "
        "новая архитектура резолвит ключ шаблона через "
        "build_template_intent_result(template_key=...) и при неизвестном ключе "
        "делает graceful fallback на RESPONSE_OUT_OF_SCOPE вместо KeyError. "
        "Поведение fallback проверено в test_unknown_template_key_falls_back."
    )


def test_template_out_of_scope_refused():
    result = build_template_intent_result(
        query="?", intent=Intent.OUT_OF_SCOPE, prior_trace=[]
    )
    assert result.refused is True
    assert result.final_answer == prompts.RESPONSE_OUT_OF_SCOPE
    assert result.sources == {}
    assert result.intent == "out_of_scope"


def test_template_out_of_scope_trace():
    result = build_template_intent_result(
        query="?", intent=Intent.OUT_OF_SCOPE, prior_trace=[]
    )
    assert result.trace[-1].step == config.STEP_TEMPLATE_RESPONSE


def test_template_about_assistant_not_refused():
    result = build_template_intent_result(
        query="?", intent=Intent.ABOUT_ASSISTANT, prior_trace=[]
    )
    assert result.refused is False
    assert result.final_answer == prompts.RESPONSE_ABOUT_ASSISTANT
    assert result.intent == "about_assistant"


def test_template_preserves_prior_trace():
    from rag_core.generation.types import TraceStep
    prior = [TraceStep(step="x", input={}, output={}, latency_ms=0.0)]
    result = build_template_intent_result(
        query="?", intent=Intent.OUT_OF_SCOPE, prior_trace=prior
    )
    assert len(result.trace) == 2
    assert result.trace[0].step == "x"


def test_template_step_output():
    result = build_template_intent_result(
        query="?", intent=Intent.OUT_OF_SCOPE, prior_trace=[]
    )
    step = result.trace[-1]
    assert step.input["intent"] == "out_of_scope"
    assert step.output["refused"] is True


def test_no_context_refused():
    result = build_no_context_refusal_result(
        query="?", intent=Intent.SEMANTIC_QUERY, prior_trace=[]
    )
    assert result.refused is True
    assert result.final_answer == prompts.RESPONSE_REFUSAL_NO_CONTEXT
    assert result.sources == {}


def test_no_context_trace():
    result = build_no_context_refusal_result(
        query="?", intent=Intent.SEMANTIC_QUERY, prior_trace=[]
    )
    assert result.trace[-1].step == config.STEP_REFUSAL


def test_no_context_intent_kept():
    result = build_no_context_refusal_result(
        query="?", intent=Intent.MIXED_QUERY, prior_trace=[]
    )
    assert result.intent == "mixed_query"


def test_no_context_preserves_prior():
    from rag_core.generation.types import TraceStep
    prior = [TraceStep(step="x", input={}, output={}, latency_ms=0.0)]
    result = build_no_context_refusal_result(
        query="?", intent=Intent.SEMANTIC_QUERY, prior_trace=prior
    )
    assert len(result.trace) == 2


def test_template_out_of_scope_is_agent_result():
    result = build_template_intent_result(
        query="?", intent=Intent.OUT_OF_SCOPE, prior_trace=[]
    )
    assert isinstance(result, AgentResult)


def test_no_context_is_agent_result():
    result = build_no_context_refusal_result(
        query="?", intent=Intent.SEMANTIC_QUERY, prior_trace=[]
    )
    assert isinstance(result, AgentResult)
