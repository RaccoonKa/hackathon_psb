"""Публичный фасад RAG-ядра для интеграции с внешним бэкендом."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

if TYPE_CHECKING:
    from src.generation.orchestrator import run_agent as run_agent
    from src.generation.types import AgentResult as AgentResult

__all__ = ["ask", "run_agent", "AgentResult"]


def ask(
    query: str,
    *,
    chat_history: Optional[list[dict[str, str]]] = None,
    event_date: Optional[str] = None,
    selected_doc_ids: Optional[Iterable[str]] = None,
    as_dict: bool = False,
    include_trace: bool = True,
) -> Union["AgentResult", dict[str, Any]]:
    """Единая точка вызова RAG-ядра для внешнего бэкенда."""
    from src.generation.orchestrator import run_agent as _run_agent

    doc_ids_set = (
        {str(doc_id).strip() for doc_id in selected_doc_ids if str(doc_id).strip()}
        if selected_doc_ids is not None
        else None
    )
    result = _run_agent(
        query=query,
        event_date=event_date,
        selected_doc_ids=doc_ids_set,
        chat_history=chat_history,
    )
    if as_dict:
        return result.to_dict(include_trace=include_trace)
    return result


def __getattr__(name: str) -> Any:
    if name == "run_agent":
        from src.generation.orchestrator import run_agent as _run_agent

        return _run_agent
    if name == "AgentResult":
        from src.generation.types import AgentResult as _AgentResult

        return _AgentResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")