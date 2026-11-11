import os
from pathlib import Path
from dotenv import load_dotenv
from src.generation.types import Intent

load_dotenv()

# Базовые директории проекта
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
CHROMA_DIR = BASE_DIR / "storage" / "chroma_db"
BM25_DIR = BASE_DIR / "storage" / "bm25_index"
MANIFESTS_DIR = BASE_DIR / "data" / "manifests"

# Параметры API платформы и ключи по умолчанию
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.platform.local/v1")
API_KEY = os.getenv("API_KEY", "EMPTY")

# Эмбеддинги (BAAI/bge-m3)
EMBEDDING_API_BASE_URL = os.getenv(
    "EMBEDDING_API_BASE_URL", 
    os.getenv("API_BASE_URL", "https://tokenharbor.ai/v1")
)
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("API_KEY", ""))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# LLM Генератор (DeepSeek V4.1 Flash)
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL", 
    os.getenv("LLM_API_BASE_URL", os.getenv("API_BASE_URL", "https://tokenharbor.ai/v1"))
)
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("API_KEY", "stub-api-key"))

# Реранкер (Qwen3 Reranker 8B)
RERANKER_API_BASE_URL = os.getenv("RERANKER_API_BASE_URL", API_BASE_URL)
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", API_KEY)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-8B")

# Имя коллекции строго по спецификации rag.md
COLLECTION_NAME = "document_chunks"

# Параметры чанкинга
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Режимы LLM
LLM_MODE_STUB: str = "stub"
LLM_MODE_REAL: str = "real"

LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1536"))
LLM_MODE: str = os.getenv("LLM_MODE", LLM_MODE_REAL)

# Параметры поиска и фильтрации:
# RAG_TOP_K = 10 гарантирует попадание коротких писем комплаенса и допников
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "10"))
RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "5"))
RAG_SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.0"))
REAL_RAG_DEFAULT_SCORE: float = 0.5

# Классификатор намерений
INTENT_FALLBACK: Intent = Intent(
    os.getenv("INTENT_FALLBACK", Intent.SEMANTIC_QUERY.value)
)
INTENT_CLASSIFIER_TEMPERATURE: float = float(
    os.getenv("INTENT_CLASSIFIER_TEMPERATURE", "0.0")
)
INTENT_CLASSIFIER_MAX_TOKENS: int = int(
    os.getenv("INTENT_CLASSIFIER_MAX_TOKENS", "32")
)

# Флаги заглушек (по умолчанию выключены для реального пайплайна)
USE_STUB_RAG: bool = os.getenv("USE_STUB_RAG", "false").lower() == "true"
USE_STUB_RERANKER: bool = os.getenv("USE_STUB_RERANKER", "false").lower() == "true"
USE_STUB_LLM: bool = os.getenv("USE_STUB_LLM", "false").lower() == "true"
STUB_RAG_RETURN_EMPTY: bool = os.getenv("STUB_RAG_RETURN_EMPTY", "false").lower() == "true"

MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "1"))

# Константы шаблонов ответов
RESPONSE_REFUSAL_NO_CONTEXT: str = "RESPONSE_REFUSAL_NO_CONTEXT"
RESPONSE_OUT_OF_SCOPE: str = "RESPONSE_OUT_OF_SCOPE"
RESPONSE_ABOUT_ASSISTANT: str = "RESPONSE_ABOUT_ASSISTANT"

# Шаги трейсинга
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

# Префиксы и разделители цитирования
SOURCE_ID_PREFIX: str = "S"
SOURCE_START_INDEX: int = 1
STRUCTURE_PATH_SEPARATOR: str = " -> "
CONTEXT_CHUNK_SEPARATOR: str = "\n\n"

STUB_PROMPT_TOKENS: int = int(os.getenv("STUB_PROMPT_TOKENS", "128"))
STUB_COMPLETION_TOKENS: int = int(os.getenv("STUB_COMPLETION_TOKENS", "64"))

MS_PER_SECOND: float = 1000.0
LATENCY_ROUND_DIGITS: int = 3
DEFAULT_MIN_SCORE: float = 0.0