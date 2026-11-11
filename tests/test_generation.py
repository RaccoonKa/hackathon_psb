from unittest.mock import MagicMock, patch

from src import config
from src.generation import prompts
from src.generation.context_builder import (
    build_context_and_sources,
    build_source_entry,
    format_single_chunk,
    _format_effective_dates,
)
from src.generation.generator import (
    _sanitize_and_extract_citations,
    assemble_final_response,
    build_generation_messages,
    generate_answer,
)
from src.generation.intent import (
    build_intent_messages,
    detect_intent,
    parse_intent_label,
)
from src.generation.llm_client import (
    _extract_token_usage,
    llm_complete,
    llm_complete_with_usage,
)
from src.generation.refusal import (
    build_no_context_refusal_result,
    build_template_intent_result,
    resolve_template,
)
from src.generation.types import AgentResult, Intent
from tests._common import make_retrieved_chunk


def test_generation_system_has_anti_injection():
    s = prompts.GENERATION_SYSTEM
    assert "PROMPT INJECTION" in s.upper()
    assert "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО" in s.upper()
    assert "<source>" in s


def test_intent_classifier_lists_all_intents():
    s = prompts.INTENT_CLASSIFIER_SYSTEM
    for intent in Intent:
        assert intent.value in s


def test_templates_non_empty():
    for attr in (
        "RESPONSE_OUT_OF_SCOPE",
        "RESPONSE_ABOUT_ASSISTANT",
        "RESPONSE_REFUSAL_NO_CONTEXT",
    ):
        assert getattr(prompts, attr).strip()


def test_parse_intent_label_exact():
    assert parse_intent_label("semantic_query") == Intent.SEMANTIC_QUERY
    assert parse_intent_label("EXACT_LOOKUP") == Intent.EXACT_LOOKUP


def test_parse_intent_label_with_noise():
    assert parse_intent_label("  mixed_query  ") == Intent.MIXED_QUERY
    assert parse_intent_label("Класс: semantic_query.") == Intent.SEMANTIC_QUERY


def test_parse_intent_label_garbage_falls_back():
    assert parse_intent_label("какой-то мусор") == config.INTENT_FALLBACK


def test_build_intent_messages_structure():
    msgs = build_intent_messages(query="Что такое сверхурочная работа?")
    assert len(msgs) == 2
    assert msgs[0]["role"] == config.ROLE_SYSTEM
    assert msgs[0]["content"] == prompts.INTENT_CLASSIFIER_SYSTEM
    assert msgs[1]["role"] == config.ROLE_USER
    assert "Что такое сверхурочная работа?" in msgs[1]["content"]


def test_build_generation_messages_structure():
    msgs = build_generation_messages(query="Вопрос?", context="Контекст")
    assert len(msgs) == 2
    assert msgs[0]["role"] == config.ROLE_SYSTEM
    assert msgs[0]["content"] == prompts.GENERATION_SYSTEM
    assert msgs[1]["role"] == config.ROLE_USER
    assert "Вопрос?" in msgs[1]["content"]
    assert "Контекст" in msgs[1]["content"]


def test_detect_intent_mocked():
    with patch(
        "src.generation.intent.llm_complete_with_usage",
        return_value=("exact_lookup", 12, 3),
    ):
        intent, step = detect_intent("Статья 112")
    assert intent == Intent.EXACT_LOOKUP
    assert step.step == config.STEP_INTENT_DETECTION
    assert step.prompt_tokens == 12
    assert step.completion_tokens == 3
    assert step.output["intent"] == "exact_lookup"


def test_detect_intent_fallback():
    with patch(
        "src.generation.intent.llm_complete_with_usage",
        return_value=("мусор", 0, 0),
    ):
        intent, _ = detect_intent("???")
    assert intent == config.INTENT_FALLBACK


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


def test_format_effective_dates_both():
    ch = make_retrieved_chunk(effective_from="2024-01-01", effective_to="2025-01-01")
    assert _format_effective_dates(ch) == "действует с 2024-01-01 по 2025-01-01"


def test_format_effective_dates_only_from():
    ch = make_retrieved_chunk(effective_from="2024-01-01", effective_to=None)
    assert "по настоящее время" in _format_effective_dates(ch)


def test_format_effective_dates_only_to():
    ch = make_retrieved_chunk(effective_from=None, effective_to="2025-01-01")
    assert "действует по 2025-01-01" in _format_effective_dates(ch)


def test_format_effective_dates_unknown():
    ch = make_retrieved_chunk(effective_dates_known=False)
    assert _format_effective_dates(ch) == ""


def test_format_single_chunk_full():
    ch = make_retrieved_chunk(
        provision_id="3.2",
        structure_path=["3. Порядок поставки"],
        effective_from="2024-02-12",
        effective_to="2026-12-31",
    )
    out = format_single_chunk("S1", ch)
    assert "[S1]" in out
    assert "Пункт: 3.2" in out
    assert "Редакция:" in out
    assert "2024-02-12" in out


def test_format_single_chunk_no_provision():
    ch = make_retrieved_chunk(provision_id=None)
    out = format_single_chunk("S1", ch)
    assert "Пункт:" not in out


def test_format_single_chunk_no_effective():
    ch = make_retrieved_chunk(effective_dates_known=False)
    out = format_single_chunk("S1", ch)
    assert "Редакция:" not in out


def test_format_single_chunk_no_amends():
    ch = make_retrieved_chunk(amends_document_id=None)
    out = format_single_chunk("S1", ch)
    assert "Изменяет документ:" not in out


def test_build_source_entry_full():
    ch = make_retrieved_chunk(
        chunk_id="c1",
        document_id="d1",
        version_id="v1",
        document_title="Договор",
        provision_id="3.2",
        structure_path=["3. Порядок", "3.2. Адрес"],
        score=0.9,
        effective_from="2024-01-01",
        amends_document_id="parent_doc",
    )
    entry = build_source_entry(ch)
    assert entry["chunk_id"] == "c1"
    assert entry["document_id"] == "d1"
    assert entry["provision_id"] == "3.2"
    assert entry["structure_path"] == ["3. Порядок", "3.2. Адрес"]
    assert entry["score"] == 0.9
    assert entry["amends_document_id"] == "parent_doc"


def test_build_context_empty():
    sources, ctx, step = build_context_and_sources([])
    assert sources == {}
    assert ctx == ""
    assert step.step == config.STEP_CONTEXT_ASSEMBLY


def test_build_context_two_chunks():
    chunks = [
        make_retrieved_chunk(chunk_id="a", text="A"),
        make_retrieved_chunk(chunk_id="b", text="B"),
    ]
    sources, ctx, step = build_context_and_sources(chunks)
    assert list(sources.keys()) == ["S1", "S2"]
    assert sources["S1"]["chunk_id"] == "a"
    assert sources["S2"]["chunk_id"] == "b"
    assert "A" in ctx
    assert "B" in ctx
    assert step.output["source_ids"] == ["S1", "S2"]


def test_citations_single_valid():
    sources = {"S1": {}, "S2": {}}
    cleaned, cited = _sanitize_and_extract_citations(
        "Текст <source>S1</source>.", sources
    )
    assert cited == ["S1"]
    assert "<source>S1</source>" in cleaned


def test_citations_multiple_valid():
    sources = {"S1": {}, "S2": {}}
    _, cited = _sanitize_and_extract_citations("A <source>S1, S2</source> B", sources)
    assert cited == ["S1", "S2"]


def test_citations_invalid_removed():
    sources = {"S1": {}}
    cleaned, cited = _sanitize_and_extract_citations(
        "A <source>X9</source> B", sources
    )
    assert cited == []
    assert "<source>" not in cleaned


def test_citations_mixed_keeps_valid():
    sources = {"S1": {}}
    cleaned, cited = _sanitize_and_extract_citations(
        "A <source>S1, X9</source> B", sources
    )
    assert cited == ["S1"]
    assert "<source>S1</source>" in cleaned
    assert "X9" not in cleaned


def test_citations_deduplicates():
    sources = {"S1": {}}
    _, cited = _sanitize_and_extract_citations(
        "A <source>S1</source> B <source>S1</source> C", sources
    )
    assert cited == ["S1"]


def test_citations_space_before_punct_cleaned():
    sources = {}
    cleaned, _ = _sanitize_and_extract_citations(
        "Текст <source>X9</source> .", sources
    )
    assert " ." not in cleaned
    assert "." in cleaned


def test_citations_none_present():
    sources = {"S1": {}}
    cleaned, cited = _sanitize_and_extract_citations("Просто текст.", sources)
    assert cited == []
    assert cleaned == "Просто текст."


def test_assemble_filters_by_citations():
    sources = {"S1": {"chunk_id": "c1"}, "S2": {"chunk_id": "c2"}}
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
    assert result.trace[-1].step == config.STEP_RESPONSE_ASSEMBLY


def test_assemble_keeps_all_when_no_citations():
    sources = {"S1": {}, "S2": {}}
    result = assemble_final_response(
        final_answer="Без цитат.",
        sources=sources,
        intent=Intent.SEMANTIC_QUERY,
        trace=[],
    )
    assert set(result.sources.keys()) == {"S1", "S2"}


def test_resolve_template_keys():
    assert resolve_template(config.RESPONSE_OUT_OF_SCOPE) == prompts.RESPONSE_OUT_OF_SCOPE
    assert resolve_template(config.RESPONSE_ABOUT_ASSISTANT) == prompts.RESPONSE_ABOUT_ASSISTANT
    assert resolve_template(config.RESPONSE_REFUSAL_NO_CONTEXT) == prompts.RESPONSE_REFUSAL_NO_CONTEXT


def test_resolve_template_unknown_raises():
    try:
        resolve_template("UNKNOWN_KEY")
    except KeyError:
        return
    raise AssertionError("Ожидался KeyError")


def test_template_out_of_scope():
    result = build_template_intent_result(
        query="?", intent=Intent.OUT_OF_SCOPE, prior_trace=[]
    )
    assert result.refused is True
    assert result.final_answer == prompts.RESPONSE_OUT_OF_SCOPE
    assert result.sources == {}
    assert result.intent == "out_of_scope"


def test_template_about_assistant():
    result = build_template_intent_result(
        query="?", intent=Intent.ABOUT_ASSISTANT, prior_trace=[]
    )
    assert result.refused is False
    assert result.final_answer == prompts.RESPONSE_ABOUT_ASSISTANT


def test_no_context_refusal():
    result = build_no_context_refusal_result(
        query="?", intent=Intent.SEMANTIC_QUERY, prior_trace=[]
    )
    assert result.refused is True
    assert result.final_answer == prompts.RESPONSE_REFUSAL_NO_CONTEXT
    assert result.sources == {}
    assert result.trace[-1].step == config.STEP_REFUSAL


def test_extract_token_usage_from_metadata():
    resp = MagicMock()
    resp.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
    resp.response_metadata = None
    p, c = _extract_token_usage(resp)
    assert p == 100
    assert c == 50


def test_extract_token_usage_from_response_metadata():
    resp = MagicMock()
    resp.usage_metadata = None
    resp.response_metadata = {
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 5}
    }
    p, c = _extract_token_usage(resp)
    assert p == 10
    assert c == 5


def test_extract_token_usage_fallback():
    resp = MagicMock()
    resp.usage_metadata = None
    resp.response_metadata = None
    p, c = _extract_token_usage(resp)
    assert p == config.STUB_PROMPT_TOKENS
    assert c == config.STUB_COMPLETION_TOKENS


def test_llm_complete_with_usage_mocked():
    with patch(
        "src.generation.llm_client._invoke_real_llm",
        return_value=("Ответ", 10, 5),
    ):
        text, p, c = llm_complete_with_usage(
            messages=[{"role": "user", "content": "hi"}],
            model="test",
            temperature=0.0,
            max_tokens=100,
        )
    assert text == "Ответ"
    assert p == 10
    assert c == 5


def test_llm_complete_mocked():
    with patch(
        "src.generation.llm_client._invoke_real_llm",
        return_value=("Ответ", 10, 5),
    ):
        text = llm_complete(
            messages=[{"role": "user", "content": "hi"}],
            model="test",
            temperature=0.0,
            max_tokens=100,
        )
    assert text == "Ответ"