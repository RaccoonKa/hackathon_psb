"""Централизованная конфигурация проекта для режимов LOCAL и SUBMIT.

Единственный источник переключателя — корневой ``settings_submit.py``.
Все остальные модули читают пути, URL, ключи и параметры только отсюда.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from src.generation.types import Intent
from settings_submit import USE_SUBMIT


class ConfigError(RuntimeError):
    """Ошибка конфигурации без раскрытия значений секретов."""


_CONFIG_FILE = Path(__file__).resolve()
_CONFIG_DIR = _CONFIG_FILE.parent
# src/config.py -> корень проекта; submit/config.py -> корень flat-сборки.
BASE_DIR = _CONFIG_DIR.parent if _CONFIG_DIR.name == "src" else _CONFIG_DIR


def _first_env(*names: str) -> Optional[str]:
    """Возвращает первое непустое значение из перечисленных переменных."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default
    if minimum is not None and value < minimum:
        value = minimum
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# .env загружается только в локальном режиме и только если установлен python-dotenv.
if not USE_SUBMIT:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - локальный runtime без python-dotenv
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv()


if USE_SUBMIT:
    # --- входные и рабочие каталоги платформы ---
    DATA_DIR: Path = Path(os.getenv("DATA_DIR") or "data").expanduser().resolve()
    WORK_DIR: Path = Path(
        os.getenv("SUBMIT_WORK_DIR") or tempfile.mkdtemp(prefix="psb_submit_")
    ).expanduser().resolve()

    RAW_DATA_DIR: Path = WORK_DIR / "knowledge_base"
    PROCESSED_DATA_DIR: Path = WORK_DIR / "processed"
    CHROMA_DIR: Path = WORK_DIR / "chroma_db"
    BM25_DIR: Path = WORK_DIR / "bm25_index"
    MANIFESTS_DIR: Path = WORK_DIR / "manifests"

    # --- единый OpenAI-совместимый прокси платформы ---
    LLM_BASE_URL: str = _first_env(
        "LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "API_BASE_URL"
    ) or ""
    LLM_API_KEY: str = _first_env(
        "LLM_API_KEY", "OPENAI_API_KEY", "API_KEY"
    ) or ""

    API_BASE_URL: str = LLM_BASE_URL
    API_KEY: str = LLM_API_KEY

    LLM_MODEL: str = (
        _first_env("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL")
        or "deepseek-ai/DeepSeek-V4.1-Flash"
    )
    EMBEDDING_MODEL: str = (
        _first_env("EMB_MODEL", "EMBEDDING_MODEL") or "BAAI/bge-m3"
    )
    RERANKER_MODEL: str = (
        _first_env("RERANKER_MODEL") or "Qwen/Qwen3-Reranker-8B"
    )

    RERANKER_BACKEND: str = "api"

    EMBEDDING_API_BASE_URL: str = LLM_BASE_URL
    EMBEDDING_API_KEY: str = LLM_API_KEY
    RERANKER_API_BASE_URL: str = LLM_BASE_URL
    RERANKER_API_KEY: str = LLM_API_KEY

    LLM_TEMPERATURE: float = _env_float(
        "TEMPERATURE", _env_float("LLM_TEMPERATURE", 0.0)
    )
    LLM_MAX_TOKENS: int = _env_int(
        "MAX_TOKENS", _env_int("LLM_MAX_TOKENS", 1536), minimum=1
    )
    RAG_TOP_K: int = _env_int(
        "TOP_K", _env_int("RAG_TOP_K", 10), minimum=1
    )
    RERANKER_TOP_N: int = _env_int("RERANKER_TOP_N", 5, minimum=1)
    MAX_WORKERS: int = min(_env_int("MAX_WORKERS", 2, minimum=1), 8)
    CHUNKER: str = _first_env("CHUNKER") or "structural"
    EMBEDDING_DIMENSION: int = _env_int(
        "EMBEDDING_DIMENSION", 1024, minimum=0
    )

    # В SUBMIT заглушки, локальная модель и hash-fallback запрещены.
    USE_STUB_RAG: bool = False
    USE_STUB_RERANKER: bool = False
    USE_STUB_LLM: bool = False
    STUB_RAG_RETURN_EMPTY: bool = False
    LLM_MODE: str = "real"
    USE_LOCAL_EMBEDDING_MODEL: bool = False
    ALLOW_HASH_FALLBACK: bool = False
    USE_DOTENV: bool = False
else:
    # --- локальные каталоги проекта (сохранены без изменений) ---
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    CHROMA_DIR = BASE_DIR / "storage" / "chroma_db"
    BM25_DIR = BASE_DIR / "storage" / "bm25_index"
    MANIFESTS_DIR = BASE_DIR / "data" / "manifests"

    DATA_DIR: Path = BASE_DIR / "data"
    WORK_DIR: Path = BASE_DIR

    # --- параметры API и ключи по умолчанию (локальное поведение) ---
    API_BASE_URL: str = os.getenv(
        "API_BASE_URL", "https://api.platform.local/v1"
    )
    API_KEY: str = os.getenv("API_KEY", "EMPTY")

    EMBEDDING_API_BASE_URL: str = os.getenv(
        "EMBEDDING_API_BASE_URL",
        os.getenv("API_BASE_URL", "https://tokenharbor.ai/v1"),
    )
    EMBEDDING_API_KEY: str = os.getenv(
        "EMBEDDING_API_KEY", os.getenv("API_KEY", "")
    )
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    LLM_MODEL: str = os.getenv(
        "LLM_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash"
    )
    LLM_BASE_URL: str = os.getenv(
        "LLM_BASE_URL",
        os.getenv(
            "LLM_API_BASE_URL",
            os.getenv("API_BASE_URL", "https://tokenharbor.ai/v1"),
        ),
    )
    LLM_API_KEY: str = os.getenv(
        "LLM_API_KEY", os.getenv("API_KEY", "stub-api-key")
    )

    RERANKER_API_BASE_URL: str = os.getenv(
        "RERANKER_API_BASE_URL", API_BASE_URL
    )
    RERANKER_API_KEY: str = os.getenv("RERANKER_API_KEY", API_KEY)
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "Qwen/Qwen3-Reranker-8B"
    )

    RERANKER_BACKEND: str = "api"

    LLM_TEMPERATURE: float = _env_float("LLM_TEMPERATURE", 0.0)
    LLM_MAX_TOKENS: int = _env_int("LLM_MAX_TOKENS", 1536, minimum=1)
    RAG_TOP_K: int = _env_int("RAG_TOP_K", 10, minimum=1)
    RERANKER_TOP_N: int = _env_int("RERANKER_TOP_N", 5, minimum=1)
    MAX_WORKERS: int = min(_env_int("MAX_WORKERS", 1, minimum=1), 8)
    CHUNKER: str = _first_env("CHUNKER") or "structural"
    EMBEDDING_DIMENSION: int = _env_int(
        "EMBEDDING_DIMENSION", 0, minimum=0
    )

    USE_STUB_RAG = os.getenv("USE_STUB_RAG", "false").lower() == "true"
    USE_STUB_RERANKER = (
        os.getenv("USE_STUB_RERANKER", "false").lower() == "true"
    )
    USE_STUB_LLM = os.getenv("USE_STUB_LLM", "false").lower() == "true"
    STUB_RAG_RETURN_EMPTY = (
        os.getenv("STUB_RAG_RETURN_EMPTY", "false").lower() == "true"
    )
    LLM_MODE: str = os.getenv("LLM_MODE", "real")
    USE_LOCAL_EMBEDDING_MODEL: bool = _env_bool(
        "USE_LOCAL_EMBEDDING_MODEL", True
    )
    ALLOW_HASH_FALLBACK: bool = _env_bool("ALLOW_HASH_FALLBACK", True)
    USE_DOTENV: bool = True


# --- общие константы проекта ---
COLLECTION_NAME = "document_chunks"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

LLM_MODE_STUB: str = "stub"
LLM_MODE_REAL: str = "real"

# Унифицированные алиасы параметров платформы.
TEMPERATURE: float = LLM_TEMPERATURE
MAX_TOKENS: int = LLM_MAX_TOKENS
TOP_K: int = RAG_TOP_K

RAG_SCORE_THRESHOLD: float = _env_float("RAG_SCORE_THRESHOLD", 0.0)
REAL_RAG_DEFAULT_SCORE: float = 0.5
DEFAULT_MIN_SCORE: float = 0.0

RRF_FALLBACK_SCORE_CEILING: float = float(
    os.getenv("RRF_FALLBACK_SCORE_CEILING", "0.1")
)
RRF_FALLBACK_THRESHOLD: float = float(
    os.getenv("RRF_FALLBACK_THRESHOLD", "0.005")
)

INTENT_FALLBACK: Intent = Intent(
    os.getenv("INTENT_FALLBACK", Intent.SEMANTIC_QUERY.value)
)
INTENT_CLASSIFIER_TEMPERATURE: float = _env_float(
    "INTENT_CLASSIFIER_TEMPERATURE", 0.0
)
INTENT_CLASSIFIER_MAX_TOKENS: int = _env_int(
    "INTENT_CLASSIFIER_MAX_TOKENS", 32, minimum=1
)

MAX_RETRIES: int = _env_int("MAX_RETRIES", 1, minimum=0)

# Таймауты сетевых вызовов и параметры локального реранкера.
EMBED_BATCH_SIZE: int = _env_int("EMBED_BATCH_SIZE", 16, minimum=1)
RERANKER_TIMEOUT: float = _env_float("RERANKER_TIMEOUT", 30.0)
RERANKER_LOCAL_PATH: Path = Path(
    _first_env("RERANKER_LOCAL_PATH")
    or (BASE_DIR / "models" / "Qwen3-Reranker-8B")
).expanduser().resolve()
RERANKER_MAX_LENGTH: int = _env_int(
    "RERANKER_MAX_LENGTH", 8192, minimum=512
)
RERANKER_BATCH_SIZE: int = _env_int(
    "RERANKER_BATCH_SIZE", 2, minimum=1
)

# Контракт выходного файла.
SUBMISSION_FILENAME: str = "submission.csv"
MAX_ANSWER_CHARS: int = 10000
KB_ARCHIVE_NAMES: tuple[str, ...] = (
    "knowledge-base.zip",
    "kb.zip",
    "knowledge_base.zip",
)

# Константы шаблонов ответов.
RESPONSE_REFUSAL_NO_CONTEXT: str = "RESPONSE_REFUSAL_NO_CONTEXT"
RESPONSE_OUT_OF_SCOPE: str = "RESPONSE_OUT_OF_SCOPE"
RESPONSE_ABOUT_ASSISTANT: str = "RESPONSE_ABOUT_ASSISTANT"

# Шаги трейсинга.
STEP_INTENT_DETECTION: str = "intent_detection"
STEP_TEMPLATE_RESPONSE: str = "template_response"
STEP_RAG_RETRIEVAL: str = "rag_retrieval"
STEP_RERANKING: str = "reranking"
STEP_THRESHOLD_FILTER: str = "threshold_filter"
STEP_REFUSAL: str = "refusal"
STEP_CONTEXT_ASSEMBLY: str = "context_assembly"
STEP_GENERATION: str = "generation"
STEP_RESPONSE_ASSEMBLY: str = "response_assembly"

ROLE_SYSTEM: str = "system"
ROLE_USER: str = "user"

# Префиксы и разделители цитирования.
SOURCE_ID_PREFIX: str = "S"
SOURCE_START_INDEX: int = 1
STRUCTURE_PATH_SEPARATOR: str = " -> "
CONTEXT_CHUNK_SEPARATOR: str = "\n\n"

STUB_PROMPT_TOKENS: int = _env_int("STUB_PROMPT_TOKENS", 128, minimum=0)
STUB_COMPLETION_TOKENS: int = _env_int(
    "STUB_COMPLETION_TOKENS", 64, minimum=0
)

REAL_RAG_DEFAULT_SCORE: float = float(
    os.getenv("REAL_RAG_DEFAULT_SCORE", "1.0")
)

MS_PER_SECOND: float = 1000.0
LATENCY_ROUND_DIGITS: int = 3
def validate_submit_config() -> None:
    """Проверяет обязательные параметры SUBMIT, не раскрывая секреты."""
    if not USE_SUBMIT:
        return

    missing: list[str] = []
    if not LLM_BASE_URL:
        missing.append("LLM_BASE_URL (или OPENAI_BASE_URL)")
    if not LLM_API_KEY:
        missing.append("LLM_API_KEY (или OPENAI_API_KEY)")

    if missing:
        raise ConfigError(
            "Не заданы обязательные переменные платформенного прокси: "
            + ", ".join(missing)
            + ". Значения ключей в лог не выводятся."
        )


# --- бюджеты, таймауты и флаги агентного пайплайна (портировано из feat/rag-agent-logic) ---
MAX_LLM_FAILURES_PER_REQUEST: int = _env_int(
    "MAX_LLM_FAILURES_PER_REQUEST", 12, minimum=1
)
MAX_LLM_FAILURES_PER_STEP: int = _env_int(
    "MAX_LLM_FAILURES_PER_STEP", 3, minimum=1
)

MAX_ATTEMPTS_QUERY_ANALYSIS: int = _env_int(
    "MAX_ATTEMPTS_QUERY_ANALYSIS", 2, minimum=1
)
MAX_ATTEMPTS_GENERATE_INITIAL: int = _env_int(
    "MAX_ATTEMPTS_GENERATE_INITIAL", 2, minimum=1
)
MAX_REGENERATE_CYCLES: int = _env_int("MAX_REGENERATE_CYCLES", 1, minimum=0)
MAX_FOLLOWUP_RETRIEVAL_CYCLES: int = _env_int(
    "MAX_FOLLOWUP_RETRIEVAL_CYCLES", 1, minimum=0
)

LLM_REQUEST_TIMEOUT_S: float = _env_float("LLM_REQUEST_TIMEOUT_S", 30.0)
RETRIEVAL_TIMEOUT_S: float = _env_float("RETRIEVAL_TIMEOUT_S", 20.0)
REQUEST_TOTAL_TIMEOUT_S: float = _env_float("REQUEST_TOTAL_TIMEOUT_S", 90.0)

ENABLE_MEMORY: bool = _env_bool("ENABLE_MEMORY", False)
ENABLE_LLM_VERIFIER: bool = _env_bool("ENABLE_LLM_VERIFIER", True)
ENABLE_PREMISE_CHECK: bool = _env_bool("ENABLE_PREMISE_CHECK", True)
ENABLE_INJECTION_GUARD: bool = _env_bool("ENABLE_INJECTION_GUARD", True)
ENABLE_SANITY_CHECK_IN_ANALYSIS: bool = _env_bool(
    "ENABLE_SANITY_CHECK_IN_ANALYSIS", False
)
