import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
CHROMA_DIR = BASE_DIR / "storage" / "chroma_db"
BM25_DIR = BASE_DIR / "storage" / "bm25_index"
MANIFESTS_DIR = BASE_DIR / "data" / "manifests"

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.platform.local/v1")
API_KEY = os.getenv("API_KEY", "EMPTY")

EMBEDDING_API_BASE_URL = os.getenv("EMBEDDING_API_BASE_URL", os.getenv("API_BASE_URL", "https://tokenharbor.ai/v1"))
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("API_KEY", ""))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4.1-flash:free")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("LLM_API_BASE_URL", os.getenv("API_BASE_URL", "https://tokenharbor.ai/v1")))
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("API_KEY", "stub-api-key"))

RERANKER_API_BASE_URL = os.getenv("RERANKER_API_BASE_URL", API_BASE_URL)
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", API_KEY)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-8B")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_chunks")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

from src.generation.types import Intent

LLM_MODE_STUB: str = "stub"
LLM_MODE_REAL: str = "real"

LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_MODE: str = os.getenv("LLM_MODE", LLM_MODE_REAL)

RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "3"))
RAG_SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))

INTENT_FALLBACK: Intent = Intent(
    os.getenv("INTENT_FALLBACK", Intent.SEMANTIC_QUERY.value)
)
INTENT_CLASSIFIER_TEMPERATURE: float = float(
    os.getenv("INTENT_CLASSIFIER_TEMPERATURE", "0.0")
)
INTENT_CLASSIFIER_MAX_TOKENS: int = int(
    os.getenv("INTENT_CLASSIFIER_MAX_TOKENS", "32")
)

USE_STUB_RAG: bool = os.getenv("USE_STUB_RAG", "false").lower() == "true"
USE_STUB_RERANKER: bool = os.getenv("USE_STUB_RERANKER", "false").lower() == "true"
USE_STUB_LLM: bool = os.getenv("USE_STUB_LLM", "false").lower() == "true"
STUB_RAG_RETURN_EMPTY: bool = os.getenv("STUB_RAG_RETURN_EMPTY", "false").lower() == "true"

MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "0"))

RESPONSE_REFUSAL_NO_CONTEXT: str = "RESPONSE_REFUSAL_NO_CONTEXT"
RESPONSE_OUT_OF_SCOPE: str = "RESPONSE_OUT_OF_SCOPE"
RESPONSE_ABOUT_ASSISTANT: str = "RESPONSE_ABOUT_ASSISTANT"

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

SOURCE_ID_PREFIX: str = "S"
SOURCE_START_INDEX: int = 1
STRUCTURE_PATH_SEPARATOR: str = " -> "
CONTEXT_CHUNK_SEPARATOR: str = "\n\n"

STUB_PROMPT_TOKENS: int = int(os.getenv("STUB_PROMPT_TOKENS", "128"))
STUB_COMPLETION_TOKENS: int = int(os.getenv("STUB_COMPLETION_TOKENS", "64"))

REAL_RAG_DEFAULT_SCORE: float = float(os.getenv("REAL_RAG_DEFAULT_SCORE", "1.0"))

MS_PER_SECOND: float = 1000.0
LATENCY_ROUND_DIGITS: int = 3
DEFAULT_MIN_SCORE: float = 0.0