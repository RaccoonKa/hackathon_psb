"""Локальный CLI-инструмент для быстрой диагностики RAG-ядра."""

from __future__ import annotations

import json
import sys

from src.generation.orchestrator import run_agent


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print('Использование: python main.py "Ваш вопрос по базе знаний"')
        sys.exit(1)

    result = run_agent(query=query)
    print(
        json.dumps(
            {
                "final_answer": result.final_answer,
                "intent": result.intent,
                "refused": result.refused,
                "confidence": result.confidence,
                "used_history": result.used_history,
                "sources_count": len(result.sources),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()