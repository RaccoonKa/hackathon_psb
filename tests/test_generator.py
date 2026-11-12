from unittest.mock import patch

from src import config
from src.generation import prompts
from src.generation.generator import (
    _sanitize_and_extract_citations,
    assemble_final_response,
    build_generation_messages,
    generate_answer,
)
from src.generation.types import AgentResult, Intent


def test_build_messages_count():
    msgs = build_generation_messages(query="q", context="c")
    assert len(msgs) == 2


def test_build_messages_roles():
    msgs = build_generation_messages(query="q", context="c")
    assert msgs[0]["role"] == config.ROLE_SYSTEM
    assert msgs[1]["role"] == config.ROLE_USER


def test_build_messages_system_equals_prompt():
    msgs = build_generation_messages(query="q", context="c")
    assert msgs[0]["content"] == prompts.GENERATION_SYSTEM


def test_build_messages_includes_query():
    msgs = build_generation_messages(query="Что делать?", context="c")
    assert "Что делать?" in msgs[1]["content"]


def test_build_messages_includes_context():
    msgs = build_generation_messages(query="q", context="Уникальный контекст 123")
    assert "Уникальный контекст 123" in msgs[1]["content"]


def test_generate_answer_mocked():
    with patch(
        "src.generation.generator.llm_complete_with_usage",
        return_value=("Ответ <source>S1</source>", 200, 20),
    ):
        ans, step = generate_answer("Вопрос?", "Контекст")
    assert "S1" in ans
    assert step.step == config.STEP_GENERATION
    assert step.model == config.LLM_MODEL
    assert step.prompt_tokens == 200
    assert step.completion_tokens == 20


def test_generate_answer_trace_input():
    with patch(
        "src.generation.generator.llm_complete_with_usage",
        return_value=("a", 1, 1),
    ):
        _, step = generate_answer("Какой-то вопрос?", "контекст длиной 10")
    assert step.input["query"] == "Какой-то вопрос?"
    assert step.input["context_length"] == len("контекст длиной 10")
    assert step.input["model"] == config.LLM_MODEL


def test_generate_answer_empty_response():
    with patch(
        "src.generation.generator.llm_complete_with_usage",
        return_value=("", 0, 0),
    ):
        ans, _ = generate_answer("q", "c")
    assert ans == ""


def test_sanitize_single_valid():
    cleaned, cited = _sanitize_and_extract_citations(
        "Текст <source>S1</source>.", {"S1": {}}
    )
    assert cited == ["S1"]
    assert "<source>S1</source>" in cleaned


def test_sanitize_multiple_valid():
    cleaned, cited = _sanitize_and_extract_citations(
        "A <source>S1, S2</source> B", {"S1": {}, "S2": {}}
    )
    assert cited == ["S1", "S2"]


def test_sanitize_dedup():
    _, cited = _sanitize_and_extract_citations(
        "A <source>S1</source> B <source>S1</source> C", {"S1": {}}
    )
    assert cited == ["S1"]


def test_sanitize_invalid_removed():
    cleaned, cited = _sanitize_and_extract_citations(
        "A <source>X9</source> B", {"S1": {}}
    )
    assert cited == []
    assert "<source>" not in cleaned


def test_sanitize_mixed():
    cleaned, cited = _sanitize_and_extract_citations(
        "A <source>S1, X9</source> B", {"S1": {}}
    )
    assert cited == ["S1"]
    assert "X9" not in cleaned
    assert "S1" in cleaned


def test_sanitize_empty_tag():
    cleaned, cited = _sanitize_and_extract_citations(
        "A <source></source> B", {"S1": {}}
    )
    assert cited == []
    assert "<source>" not in cleaned


def test_sanitize_whitespace_only():
    cleaned, cited = _sanitize_and_extract_citations(
        "A <source>  ,  ,  </source> B", {"S1": {}}
    )
    assert cited == []


def test_sanitize_no_tags():
    cleaned, cited = _sanitize_and_extract_citations("Просто текст.", {"S1": {}})
    assert cited == []
    assert cleaned == "Просто текст."


def test_sanitize_punctuation_space_cleaned():
    cleaned, _ = _sanitize_and_extract_citations(
        "Текст <source>X9</source> .", {}
    )
    assert " ." not in cleaned


def test_sanitize_preserves_multiline():
    src = {"S1": {}}
    cleaned, _ = _sanitize_and_extract_citations(
        "Первая строка <source>S1</source>\nВторая строка", src
    )
    assert "\n" in cleaned


def test_assemble_basic():
    sources = {"S1": {"chunk_id": "c1"}}
    result = assemble_final_response(
        final_answer="Текст <source>S1</source>",
        sources=sources,
        intent=Intent.SEMANTIC_QUERY,
        trace=[],
    )
    assert isinstance(result, AgentResult)
    assert result.refused is False
    assert list(result.sources.keys()) == ["S1"]
    assert result.intent == "semantic_query"


def test_assemble_no_citations_keeps_all():
    sources = {"S1": {}, "S2": {}}
    result = assemble_final_response(
        final_answer="Без цитат.",
        sources=sources,
        intent=Intent.SEMANTIC_QUERY,
        trace=[],
    )
    assert set(result.sources.keys()) == {"S1", "S2"}


def test_assemble_appends_assembly_step():
    sources = {}
    result = assemble_final_response(
        final_answer="x",
        sources=sources,
        intent=Intent.SEMANTIC_QUERY,
        trace=[],
    )
    assert result.trace[-1].step == config.STEP_RESPONSE_ASSEMBLY


def test_assemble_preserves_prior_trace():
    from src.generation.types import TraceStep
    prior = [TraceStep(step="x", input={}, output={}, latency_ms=0.0)]
    result = assemble_final_response(
        final_answer="x", sources={}, intent=Intent.SEMANTIC_QUERY, trace=prior,
    )
    assert len(result.trace) == 2
    assert result.trace[0].step == "x"


def test_assemble_step_output():
    result = assemble_final_response(
        final_answer="x", sources={"S1": {}}, intent=Intent.SEMANTIC_QUERY, trace=[],
    )
    step = result.trace[-1]
    assert step.output["refused"] is False
    assert step.output["sources_count"] == 1


def test_assemble_exact_intent_kept():
    result = assemble_final_response(
        final_answer="x", sources={}, intent=Intent.EXACT_LOOKUP, trace=[],
    )
    assert result.intent == "exact_lookup"


def test_assemble_mixed_intent_kept():
    result = assemble_final_response(
        final_answer="x", sources={}, intent=Intent.MIXED_QUERY, trace=[],
    )
    assert result.intent == "mixed_query"