from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_core import config
from rag_core.generation.llm.client import llm_complete_with_usage
from rag_core.generation.errors import (
    LLMAPIError,
    LLMFormatError,
    LLMTimeoutError,
)


def _mask(value: str | None) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 10:
        return "***"
    return f"{value[:4]}…{value[-2:]} (len={len(value)})"


def main() -> None:
    print("═" * 70)
    print("LLM healthcheck")
    print("═" * 70)
    print(f"LLM_BASE_URL : {config.LLM_BASE_URL or '(empty)'}")
    print(f"LLM_API_KEY  : {_mask(config.LLM_API_KEY)}")
    print(f"LLM_MODEL    : {config.LLM_MODEL}")
    print(f"TIMEOUT_S    : {config.LLM_REQUEST_TIMEOUT_S}")
    print(f"MAX_RETRIES  : {config.MAX_RETRIES}")
    print(f"LLM_MODE     : {config.LLM_MODE}")
    print(f"USE_STUB_LLM : {config.USE_STUB_LLM}")
    print(f"USE_SUBMIT   : {getattr(config, 'USE_SUBMIT', None)}")
    print()

    messages = [
        {"role": "system", "content": "Отвечай одним словом."},
        {"role": "user", "content": "Ответь одним словом: ping"},
    ]

    started = time.perf_counter()
    try:
        text, p, c = llm_complete_with_usage(
            messages=messages,
            model=config.LLM_MODEL,
            temperature=0.0,
            max_tokens=32,
            timeout=config.LLM_REQUEST_TIMEOUT_S,
        )
    except LLMTimeoutError as exc:
        elapsed = round(time.perf_counter() - started, 2)
        print(f"❌ TIMEOUT через {elapsed}s: {exc}")
        print("   Проверьте сеть/VPN и LLM_REQUEST_TIMEOUT_S.")
        sys.exit(1)
    except LLMAPIError as exc:
        elapsed = round(time.perf_counter() - started, 2)
        print(f"❌ API ERROR через {elapsed}s: {exc}")
        print("   Проверьте LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.")
        sys.exit(1)
    except LLMFormatError as exc:
        print(f"❌ FORMAT ERROR (маловероятно для ping): {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"❌ {type(exc).__name__}: {exc}")
        sys.exit(1)

    elapsed = round(time.perf_counter() - started, 2)
    print(f"✅ OK за {elapsed}s")
    print(f"   prompt_tokens={p}, completion_tokens={c}")
    print(f"   ответ: {text[:200]!r}")


if __name__ == "__main__":
    main()