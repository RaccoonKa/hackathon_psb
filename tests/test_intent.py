from unittest.mock import patch

from src import config
from src.generation import prompts
from src.generation.intent import (
    build_intent_messages,
    detect_intent,
    parse_intent_label,
)
from src.generation.types import Intent


def test_parse_exact_short():
    assert parse_intent_label("exact_lookup") == Intent.EXACT_LOOKUP


def test_parse_semantic_short():
    assert parse_intent_label("semantic_query") == Intent.SEMANTIC_QUERY


def test_parse_mixed_short():
    assert parse_intent_label("mixed_query") == Intent.MIXED_QUERY


def test_parse_uppercase():
    assert parse_intent_label("SEMANTIC_QUERY") == Intent.SEMANTIC_QUERY


def test_parse_with_whitespace():
    assert parse_intent_label("  semantic_query  ") == Intent.SEMANTIC_QUERY


def test_parse_with_trailing_newline():
    assert parse_intent_label("semantic_query\n") == Intent.SEMANTIC_QUERY


def test_parse_with_period():
    assert parse_intent_label("exact_lookup.") == Intent.EXACT_LOOKUP


def test_parse_embedded_sentence():
    assert parse_intent_label("Ответ: mixed_query.") == Intent.MIXED_QUERY


def test_parse_garbage_returns_fallback():
    assert parse_intent_label("что-то бессвязное") == config.INTENT_FALLBACK


def test_parse_empty_string():
    assert parse_intent_label("") == config.INTENT_FALLBACK


def test_parse_only_whitespace():
    assert parse_intent_label("   ") == config.INTENT_FALLBACK


def test_parse_wrapper_word_matches():
    assert parse_intent_label("это semantic_query точно") == Intent.SEMANTIC_QUERY


def test_parse_out_of_scope():
    assert parse_intent_label("out_of_scope") == Intent.OUT_OF_SCOPE


def test_parse_about_assistant():
    assert parse_intent_label("about_assistant") == Intent.ABOUT_ASSISTANT


def test_build_intent_messages_count():
    msgs = build_intent_messages("q")
    assert len(msgs) == 2


def test_build_intent_messages_roles():
    msgs = build_intent_messages("q")
    assert msgs[0]["role"] == config.ROLE_SYSTEM
    assert msgs[1]["role"] == config.ROLE_USER


def test_build_intent_messages_system_content():
    msgs = build_intent_messages("q")
    assert msgs[0]["content"] == prompts.INTENT_CLASSIFIER_SYSTEM


def test_build_intent_messages_includes_query():
    msgs = build_intent_messages("Что такое сверхурочная работа?")
    assert "Что такое сверхурочная работа?" in msgs[1]["content"]


def test_detect_intent_exact():
    with patch(
        "src.generation.intent.llm_complete_with_usage",
        return_value=("exact_lookup", 50, 5),
    ):
        intent, step = detect_intent("Статья 112")
    assert intent == Intent.EXACT_LOOKUP
    assert step.step == config.STEP_INTENT_DETECTION
    assert step.prompt_tokens == 50
    assert step.completion_tokens == 5
    assert step.model == config.LLM_MODEL
    assert step.output["intent"] == "exact_lookup"


def test_detect_intent_semantic():
    with patch(
        "src.generation.intent.llm_complete_with_usage",
        return_value=("semantic_query", 10, 2),
    ):
        intent, _ = detect_intent("Как оплачивается сверхурочная?")
    assert intent == Intent.SEMANTIC_QUERY


def test_detect_intent_llm_returns_garbage():
    with patch(
        "src.generation.intent.llm_complete_with_usage",
        return_value=("abcdef", 0, 0),
    ):
        intent, _ = detect_intent("q")
    assert intent == config.INTENT_FALLBACK


def test_detect_intent_llm_returns_empty():
    with patch(
        "src.generation.intent.llm_complete_with_usage",
        return_value=("", 0, 0),
    ):
        intent, _ = detect_intent("q")
    assert intent == config.INTENT_FALLBACK


def test_detect_intent_input_recorded():
    with patch(
        "src.generation.intent.llm_complete_with_usage",
        return_value=("semantic_query", 1, 1),
    ):
        _, step = detect_intent("Какой-то вопрос?")
    assert step.input["query"] == "Какой-то вопрос?"
    assert step.input["temperature"] == config.INTENT_CLASSIFIER_TEMPERATURE
    assert step.input["max_tokens"] == config.INTENT_CLASSIFIER_MAX_TOKENS


def test_detect_intent_latency_measured():
    with patch(
        "src.generation.intent.llm_complete_with_usage",
        return_value=("semantic_query", 1, 1),
    ):
        _, step = detect_intent("q")
    assert step.latency_ms >= 0