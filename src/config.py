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

# LLM (DeepSeek)
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", os.getenv("API_BASE_URL", "https://tokenharbor.ai/v1"))
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("API_KEY", ""))
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4.1-flash:free")

# Embeddings (bge-m3)
EMBEDDING_API_BASE_URL = os.getenv("EMBEDDING_API_BASE_URL", os.getenv("API_BASE_URL", "https://tokenharbor.ai/v1"))
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("API_KEY", ""))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Reranker (Qwen3 Reranker 8B)
RERANKER_API_BASE_URL = os.getenv("RERANKER_API_BASE_URL", EMBEDDING_API_BASE_URL)
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", EMBEDDING_API_KEY)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-8B")

COLLECTION_NAME = "document_chunks"