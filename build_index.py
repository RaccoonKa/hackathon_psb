import sys
from src.config import RAW_DATA_DIR, CHROMA_DIR
from src.indexing.parser import load_raw_documents
from src.indexing.chunker import chunk_documents
from src.retrieval.vector_store import add_chunks_to_vector_store

def main():
    print(f"1. Чтение документов из {RAW_DATA_DIR}...")
    docs = load_raw_documents(RAW_DATA_DIR)
    if not docs:
        print(f"Ошибка: В {RAW_DATA_DIR} не найдено .txt файлов базы знаний!")
        sys.exit(1)
    print(f"Найдено документов: {len(docs)}")

    print("2. Нарезка на юридические чанки с метаданными...")
    chunks = chunk_documents(docs)
    print(f"Сформировано чанков: {len(chunks)}")

    print(f"3. Генерация эмбеддингов и сохранение в ChromaDB ({CHROMA_DIR})...")
    add_chunks_to_vector_store(chunks)
    print("Индексация успешно завершена!")

if __name__ == "__main__":
    main()