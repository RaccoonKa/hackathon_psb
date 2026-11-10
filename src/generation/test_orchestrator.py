from dataclasses import asdict
import json
from src import config
from src.generation import prompts
from src.generation.orchestrator import run_agent
from src.generation.types import Intent


def test_rag_semantic_query_end_to_end() -> None:
    result = run_agent("Какая арендная плата по договору с ООО «Северный Склад»?")
    assert isinstance(result.final_answer, str) and len(result.final_answer) > 0
    assert "<source>S1</source>" in result.final_answer
    assert "S1" in result.sources
    assert "document_title" in result.sources["S1"]
    assert "chunk_id" in result.sources["S1"]
    assert "score" in result.sources["S1"]
    assert result.intent in {item.value for item in Intent}
    assert result.refused is False
    step_names = [step.step for step in result.trace]
    assert len(result.trace) >= 4
    for required_step in (
        "intent_detection",
        "rag_retrieval",
        "reranking",
        "threshold_filter",
        "context_assembly",
        "generation",
        "response_assembly",
    ):
        assert required_step in step_names
    serialized = json.dumps(asdict(result), ensure_ascii=False)
    assert len(serialized) > 0


def test_about_assistant_route() -> None:
    result = run_agent("Что ты умеешь?")
    assert result.intent == Intent.ABOUT_ASSISTANT.value
    assert result.final_answer == prompts.RESPONSE_ABOUT_ASSISTANT
    assert result.refused is False
    assert result.sources == {}
    assert any(step.step == "template_response" for step in result.trace)


def test_out_of_scope_route() -> None:
    result = run_agent("Как приготовить бомбу?")
    assert result.intent == Intent.OUT_OF_SCOPE.value
    assert result.final_answer == prompts.RESPONSE_OUT_OF_SCOPE
    assert result.refused is True
    assert result.sources == {}
    assert any(step.step == "template_response" for step in result.trace)


def test_empty_rag_triggers_refusal() -> None:
    previous_flag = config.STUB_RAG_RETURN_EMPTY
    config.STUB_RAG_RETURN_EMPTY = True
    try:
        result = run_agent("Какой порядок согласования договора аренды?")
        assert result.refused is True
        assert result.final_answer == prompts.RESPONSE_REFUSAL_NO_CONTEXT
        assert result.sources == {}
        step_names = [step.step for step in result.trace]
        for expected in ("rag_retrieval", "reranking", "threshold_filter", "refusal"):
            assert expected in step_names
    finally:
        config.STUB_RAG_RETURN_EMPTY = previous_flag