from rag_core import config


def test_llm_model_is_set():
    assert isinstance(config.LLM_MODEL, str) and config.LLM_MODEL


def test_embedding_model_is_set():
    assert isinstance(config.EMBEDDING_MODEL, str) and config.EMBEDDING_MODEL


def test_reranker_model_is_set():
    assert isinstance(config.RERANKER_MODEL, str) and config.RERANKER_MODEL


def test_collection_name_fixed():
    assert config.COLLECTION_NAME == "document_chunks"


def test_rag_top_k_positive():
    assert isinstance(config.RAG_TOP_K, int)
    assert config.RAG_TOP_K > 0


def test_reranker_top_n_positive():
    assert isinstance(config.RERANKER_TOP_N, int)
    assert config.RERANKER_TOP_N > 0


def test_reranker_top_n_le_rag_top_k():
    assert config.RERANKER_TOP_N <= config.RAG_TOP_K


def test_chunk_params_positive():
    assert config.CHUNK_SIZE > 0
    assert config.CHUNK_OVERLAP >= 0
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_llm_temperature_range():
    assert 0.0 <= config.LLM_TEMPERATURE <= 2.0


def test_llm_max_tokens_positive():
    assert config.LLM_MAX_TOKENS > 0


def test_intent_classifier_params():
    assert config.INTENT_CLASSIFIER_TEMPERATURE >= 0.0
    assert config.INTENT_CLASSIFIER_MAX_TOKENS > 0


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


def test_step_constants_non_empty():
    for attr in (
        "STEP_INTENT_DETECTION",
        "STEP_TEMPLATE_RESPONSE",
        "STEP_RAG_RETRIEVAL",
        "STEP_RERANKING",
        "STEP_THRESHOLD_FILTER",
        "STEP_REFUSAL",
        "STEP_CONTEXT_ASSEMBLY",
        "STEP_GENERATION",
        "STEP_RESPONSE_ASSEMBLY",
    ):
        val = getattr(config, attr)
        assert isinstance(val, str) and val


def test_response_keys_unique():
    keys = [
        config.RESPONSE_REFUSAL_NO_CONTEXT,
        config.RESPONSE_OUT_OF_SCOPE,
        config.RESPONSE_ABOUT_ASSISTANT,
    ]
    assert len(set(keys)) == len(keys)


def test_role_constants():
    assert config.ROLE_SYSTEM == "system"
    assert config.ROLE_USER == "user"


def test_source_id_constants():
    assert config.SOURCE_ID_PREFIX == "S"
    assert config.SOURCE_START_INDEX >= 1


def test_latency_round_digits_positive():
    assert isinstance(config.LATENCY_ROUND_DIGITS, int)
    assert config.LATENCY_ROUND_DIGITS >= 0


def test_ms_per_second():
    assert config.MS_PER_SECOND == 1000.0


def test_default_min_score():
    assert config.DEFAULT_MIN_SCORE == 0.0


def test_use_stub_flags_are_bool():
    assert isinstance(config.USE_STUB_LLM, bool)
    assert isinstance(config.USE_STUB_RAG, bool)
    assert isinstance(config.USE_STUB_RERANKER, bool)
    assert isinstance(config.STUB_RAG_RETURN_EMPTY, bool)


def test_max_retries_non_negative():
    assert isinstance(config.MAX_RETRIES, int)
    assert config.MAX_RETRIES >= 0


def test_real_rag_default_score():
    assert 0.0 <= config.REAL_RAG_DEFAULT_SCORE <= 1.0