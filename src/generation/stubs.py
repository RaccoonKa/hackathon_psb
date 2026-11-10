from src.generation.types import Intent, RetrievedChunk


def stub_intent_classifier(query: str) -> Intent:
    raise NotImplementedError(
        f"STUB stub_intent_classifier is not implemented. "
        f"Replace with real intent classification. "
        f"Received query={query!r}"
    )


def stub_rag_retrieve(query: str, top_k: int) -> list[RetrievedChunk]:
    raise NotImplementedError(
        f"STUB stub_rag_retrieve is not implemented. "
        f"Connect real RAG pipeline. "
        f"Received query={query!r}, top_k={top_k}"
    )


def stub_reranker(
    chunks: list[RetrievedChunk], query: str, top_n: int
) -> list[RetrievedChunk]:
    raise NotImplementedError(
        f"STUB stub_reranker is not implemented. "
        f"Connect real reranker. "
        f"Received chunks_count={len(chunks)}, query={query!r}, top_n={top_n}"
    )


def stub_llm_complete(
    messages: list[dict], model: str, temperature: float, max_tokens: int
) -> str:
    raise NotImplementedError(
        f"STUB stub_llm_complete is not implemented. "
        f"Connect real LLM endpoint. "
        f"Received model={model!r}, messages_count={len(messages)}, "
        f"temperature={temperature}, max_tokens={max_tokens}"
    )