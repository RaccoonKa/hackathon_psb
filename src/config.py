import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Базовые директории
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
CHROMA_DIR = BASE_DIR / "storage" / "chroma_db"
BM25_DIR = BASE_DIR / "storage" / "bm25_index"

# Параметры моделей (по регламенту хакатона)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.platform.local/v1")
API_KEY = os.getenv("API_KEY", "EMPTY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-8B")

COLLECTION_NAME = "legal_docs_collection"

# Параметры нарезки текста
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150