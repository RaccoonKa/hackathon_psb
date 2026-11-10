from datetime import datetime, timezone
import json
import sys
from src.config import (
    MANIFESTS_DIR,
    EMBEDDING_MODEL,
    RAW_DATA_DIR,
    RERANKER_MODEL,
)
from src.indexing.chunker import chunk_document_structural, save_canonical_chunks
from src.indexing.parser import load_documents_with_metadata
from src.retrieval.keyword_store import BM25KeywordStore
from src.retrieval.vector_store import save_embeddings_to_chroma


def main():
    print("1. Загрузка документов и извлечение метаданных...")
    docs = load_documents_with_metadata(RAW_DATA_DIR)
    if not docs:
        print(f"Ошибка: Директория {RAW_DATA_DIR} не содержит файлов .txt!")
        sys.exit(1)
    print(f"Успешно прочитано {len(docs)} документов.")

    print("2. Структурно-адаптивный чанкинг...")
    all_chunks = []
    doc_summary = []
    for d in docs:
        ch_list = chunk_document_structural(d)
        all_chunks.extend(ch_list)
        doc_summary.append({
            "document_id": d["metadata"]["document_id"],
            "version_id": d["metadata"]["version_id"],
            "source_hash": d["metadata"]["source_hash"],
            "amends_document_id": d["metadata"].get("amends_document_id"),
            "chunks_count": len(ch_list),
        })

    save_canonical_chunks(all_chunks)
    print(f"Сформировано {len(all_chunks)} чанков в chunks.jsonl")

    print("3. Построение BM25 индекса...")
    bm25 = BM25KeywordStore()
    bm25.build_index(all_chunks)
    bm25.save()

    print("4. Сохранение эмбеддингов в ChromaDB...")
    save_embeddings_to_chroma(all_chunks)

    # 5. Сохранение манифеста индексации (раздел 4.3 rag.md)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "index_build_id": f"index_{datetime.now(timezone.utc).strftime('%Y_%m_%d_%H%M%S')}",
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "chunker_version": "structural-1.0",
        "indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_chunks": len(all_chunks),
        "documents": doc_summary,
    }
    with open(MANIFESTS_DIR / "indexing_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    print("Все задачи базового пайплайна RAG успешно завершены!")


if __name__ == "__main__":
    main()