import json
from pathlib import Path
from src.config import RAW_DATA_DIR, BASE_DIR
from src.indexing.parser import load_documents_with_metadata, extract_document_structure

DEBUG_DIR = BASE_DIR / "data" / "debug" / "parser_output"

def run_parser_tests():
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"-> Запуск парсера на документах из {RAW_DATA_DIR}...")
    
    docs = load_documents_with_metadata(RAW_DATA_DIR)
    if not docs:
        print(f"[!] В директории {RAW_DATA_DIR} не обнаружено TXT файлов.")
        return

    summary = []

    for doc in docs:
        meta = doc["metadata"]
        # Извлекаем структурные блоки (разделы, статьи, пункты)
        structure_blocks = extract_document_structure(doc["content"])
        
        debug_payload = {
            "metadata": meta,
            "total_characters": len(doc["content"]),
            "total_blocks_found": len(structure_blocks),
            "blocks_preview": [
                {
                    "provision_id": b["provision_id"],
                    "path": b["path"],
                    "start_line": b["start_line"],
                    "end_line": b["end_line"],
                    "char_length": len(b["text"]),
                    "text_sample": b["text"][:160].replace("\n", " ") + "..."
                }
                for b in structure_blocks
            ]
        }
        
        out_file = DEBUG_DIR / f"{meta['document_id']}_parsed.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, ensure_ascii=False, indent=2)
            
        summary.append({
            "doc_id": meta["document_id"],
            "title": meta["document_title"][:60],
            "effective_from": meta["effective_from"],
            "blocks": len(structure_blocks),
            "json_path": str(out_file.relative_to(BASE_DIR))
        })

    # Сохраняем сводный манифест
    with open(DEBUG_DIR / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"-> Успешно обработано {len(docs)} документов.")
    print(f"-> Подробные JSON-отчёты сохранены в: {DEBUG_DIR}")

if __name__ == "__main__":
    run_parser_tests()