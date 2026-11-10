import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
CHROMA_DIR = BASE_DIR / "storage" / "chroma_db"
BM25_DIR = BASE_DIR / "storage" / "bm25_index"

# Конфигурация LLM
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", os.getenv("API_BASE_URL", "https://tokenharbor.ai/v1"))
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("API_KEY", ""))
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4.1-flash:free")

# Конфигурация Embeddings (bge-m3)
EMBEDDING_API_BASE_URL = os.getenv("EMBEDDING_API_BASE_URL", os.getenv("API_BASE_URL", "https://tokenharbor.ai/v1"))
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("API_KEY", ""))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

COLLECTION_NAME = "document_chunks"