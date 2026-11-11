import pytest

from src import config
from src.generation import prompts
from src.generation.refusal import (
    build_no_context_refusal_result,
    build_template_intent_result,
    resolve_template,
)
from src.generation.types import AgentResult, Intent


def test_resolve_out_of_scope():
    assert resolve_template(config.RESPONSE_OUT_OF_SCOPE) == prompts.RESPONSE_OUT_OF_SCOPE


def test_resolve_about_assistant():
    assert resolve_template(config.RESPONSE_ABOUT_ASSISTANT) == prompts.RESPONSE_ABOUT_ASSISTANT


def test_resolve_no_context():
    assert resolve_template(config.RESPONSE_REFUSAL_NO_CONTEXT) == prompts.RESPONSE_REFUSAL_NO_CONTEXT


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        resolve_template("UNKNOWN_KEY")


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
    from src.generation.types import TraceStep
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
    from src.generation.types import TraceStep
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