import sys
from src.config import RAW_DATA_DIR
from src.indexing.parser import load_documents_with_metadata
from src.indexing.chunker import chunk_document_structural, save_canonical_chunks
from src.retrieval.vector_store import save_embeddings_to_chroma
from src.retrieval.keyword_store import BM25KeywordStore

def main():
    print("1. Чтение документов и метаданных...")
    docs = load_documents_with_metadata(RAW_DATA_DIR)
    if not docs:
        print("В data/raw/ нет файлов .txt!")
        sys.exit(1)
        
    print(f"2. Структурный чанкинг {len(docs)} документов...")
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_document_structural(d))
    save_canonical_chunks(all_chunks)
    print(f"Создано {len(all_chunks)} чанков в chunks.jsonl")

    print("3. Построение BM25 индекса...")
    bm25 = BM25KeywordStore()
    bm25.build_index(all_chunks)
    bm25.save()

    print("4. Генерация эмбеддингов и сохранение в ChromaDB...")
    save_embeddings_to_chroma(all_chunks)
    print("Все 4 подзадачи базового RAG выполнены успешно!")

if __name__ == "__main__":
    main()