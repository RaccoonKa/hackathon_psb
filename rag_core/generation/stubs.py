from rag_core.generation.types import RetrievedChunk


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