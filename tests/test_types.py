from src import config
from src.generation.types import AgentResult, Intent, RetrievedChunk, TraceStep


def test_intent_enum_values():
    # Легаси-значения сохраняются: ими пользуются solution.py/run_live.py
    # и INTENT_FALLBACK в config.py.
    assert Intent.EXACT_LOOKUP.value == "exact_lookup"
    assert Intent.SEMANTIC_QUERY.value == "semantic_query"
    assert Intent.MIXED_QUERY.value == "mixed_query"
    assert Intent.OUT_OF_SCOPE.value == "out_of_scope"
    assert Intent.ABOUT_ASSISTANT.value == "about_assistant"
    assert config.INTENT_FALLBACK in set(Intent)

    # Новый analysis-слой использует внутренние канонические значения.
    assert Intent.SEMANTIC.value == "semantic"
    assert Intent.MIXED.value == "mixed"
    assert Intent.COMPARE.value == "compare"
    assert Intent.DATE_VERSION.value == "date_version"
    assert Intent.CHECKLIST.value == "checklist"
    assert Intent.ILLEGAL.value == "illegal"
    assert Intent.PROMPT_INJECTION.value == "prompt_injection"
    assert len(list(Intent)) == 12


def test_intent_values_unique():
    assert len({i.value for i in Intent}) == len(list(Intent))


def test_retrieved_chunk_frozen():
    chunk = RetrievedChunk(
        chunk_id="c", text="t", document_id="d", version_id="v",
        document_title="T", provision_id=None, structure_path=[],
        score=0.1,
    )
    try:
        chunk.score = 999
    except Exception:
        return
    raise AssertionError("RetrievedChunk должен быть frozen")


def test_trace_step_defaults():
    step = TraceStep(step="x", input={}, output={}, latency_ms=1.0)
    assert step.model is None
    assert step.prompt_tokens is None
    assert step.completion_tokens is None
    assert step.extra == {}


def test_agent_result_construction():
    res = AgentResult(
        final_answer="ok", sources={}, intent="semantic_query",
        refused=False, trace=[],
    )
    assert res.refused is False
    assert res.intent == "semantic_query"
    assert res.sources == {}
    assert res.trace == []


def test_config_sanity():
    assert config.LLM_MODEL
    assert config.EMBEDDING_MODEL
    assert config.RERANKER_MODEL
    assert config.COLLECTION_NAME == "document_chunks"
    assert isinstance(config.RAG_TOP_K, int) and config.RAG_TOP_K > 0
    assert isinstance(config.RERANKER_TOP_N, int) and config.RERANKER_TOP_N > 0


def test_step_constants_unique():
    steps = [
        config.STEP_INTENT_DETECTION,
        config.STEP_TEMPLATE_RESPONSE,
        config.STEP_RAG_RETRIEVAL,
        config.STEP_RERANKING,
        config.STEP_THRESHOLD_FILTER,
        config.STEP_REFUSAL,
        config.STEP_CONTEXT_ASSEMBLY,
        config.STEP_GENERATION,
        config.STEP_RESPONSE_ASSEMBLY,
    ]
    assert len(set(steps)) == len(steps)