from src.generation.types import AgentResult, Intent, RetrievedChunk, TraceStep

__all__ = [
    "AgentResult",
    "Intent",
    "RetrievedChunk",
    "TraceStep",
    "run_agent",
]


def __getattr__(name: str):
    """Ленивый доступ к run_agent, чтобы не провоцировать циклический импорт.

    `src.config` импортирует `src.generation.types`; при eager-импорте
    orchestrator подтягивает retrieval.bridge -> src.retrieval.hybrid ->
    src.config, который ещё не завершил инициализацию. Ленивый __getattr__
    сохраняет публичный API `from src.generation import run_agent`.
    """
    if name == "run_agent":
        from src.generation.orchestrator import run_agent

        return run_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
