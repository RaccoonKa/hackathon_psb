from typing import Any, Dict, List, Optional
from openai import OpenAI
from src.config import LLM_API_BASE_URL, LLM_API_KEY, LLM_MODEL
from src.generation.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.retrieval.hybrid import hybrid_search

client_llm = OpenAI(base_url=LLM_API_BASE_URL, api_key=LLM_API_KEY)


def format_context_chunks(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "В базе знаний не найдено релевантных положений."

    formatted = []
    for idx, ch in enumerate(chunks, start=1):
        doc_title = ch.get("document_title", "Документ")
        path_str = " > ".join(ch.get("structure", {}).get("path", []))
        eff_from = ch.get("effective_from") or "не указана"
        eff_to = ch.get("effective_to") or "бессрочно"
        prov_id = ch.get("structure", {}).get("provision_id")
        prov_tag = f", пункт/статья {prov_id}" if prov_id else ""

        header = f"[{idx}] {doc_title} (Путь: {path_str}{prov_tag} | Действует: с {eff_from} по {eff_to})"
        formatted.append(f"{header}\n{ch['text']}\n")

    return "\n".join(formatted)


def ask_rag(
    query: str,
    event_date: Optional[str] = None,
    top_k: int = 5,
    selected_doc_ids: Optional[set] = None,
) -> Dict[str, Any]:
    # 1. Получение чанков через BM25 + bge-m3 + Weighted RRF + Reranker
    evidence_chunks = hybrid_search(
        query=query,
        top_k=top_k,
        event_date=event_date,
        selected_doc_ids=selected_doc_ids,
    )

    context_text = format_context_chunks(evidence_chunks)
    prompt_text = USER_PROMPT_TEMPLATE.format(
        event_date=event_date or "Не указана (актуальное состояние)",
        context=context_text,
        query=query,
    )

    # 2. Вызов DeepSeek
    response = client_llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.0,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "evidence_chunks": evidence_chunks,
        "sources_used": [
            {
                "document_id": c["document_id"],
                "version_id": c["version_id"],
                "chunk_id": c["chunk_id"],
                "title": c["document_title"],
                "path": c["structure"]["path"],
            }
            for c in evidence_chunks
        ],
    }