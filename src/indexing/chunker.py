import hashlib
from typing import List, Dict, Any

def generate_chunk_id(doc_id: str, version_id: str, chunk_index: int, chunker_ver: str = "structural-1.0") -> str:
    raw = f"{doc_id}_{version_id}_{chunker_ver}_{chunk_index:04d}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def chunk_document_structural(doc_data: Dict[str, Any], chunk_size: int = 600) -> List[Dict[str, Any]]:
    content = doc_data["content"]
    meta = doc_data["metadata"]
    lines = content.splitlines(keepends=True)
    
    chunks = []
    current_path = ["Введение"]
    current_lines = []
    current_start_line = 1
    char_offset = 0
    chunk_index = 0
    
    for line_no, line in enumerate(lines, start=1):
        line_clean = line.strip()
        
        # Обнаружение заголовков статей и разделов
        if any(line_clean.startswith(prefix) for prefix in ["Статья ", "Глава ", "Раздел ", "Пункт "]):
            if current_lines:
                chunk_text = "".join(current_lines).strip()
                if chunk_text:
                    prefix_ctx = f"{meta['document_title']}. {' > '.join(current_path)}. "
                    chunks.append({
                        "chunk_id": f"{meta['document_id']}_{meta['version_id']}_{chunk_index:04d}",
                        "document_id": meta["document_id"],
                        "version_id": meta["version_id"],
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "embedding_text": prefix_ctx + chunk_text,
                        "structure": {"path": list(current_path), "provision_id": current_path[-1]},
                        "location": {
                            "start_line": current_start_line,
                            "end_line": line_no - 1,
                            "char_start": char_offset,
                            "char_end": char_offset + len(chunk_text)
                        },
                        "document_title": meta["document_title"],
                        "document_type": meta["document_type"],
                        "scope": meta.get("scope", "global"),
                        "effective_from": meta.get("effective_from"),
                        "effective_to": meta.get("effective_to"),
                        "effective_dates_known": meta.get("effective_dates_known", True),
                        "chunker_version": "structural-1.0"
                    })
                    chunk_index += 1
                current_lines = []
                current_start_line = line_no
            current_path = [line_clean[:100]]

        current_lines.append(line)
        char_offset += len(line)

    # Хвост документа
    if current_lines:
        chunk_text = "".join(current_lines).strip()
        if chunk_text:
            prefix_ctx = f"{meta['document_title']}. {' > '.join(current_path)}. "
            chunks.append({
                "chunk_id": f"{meta['document_id']}_{meta['version_id']}_{chunk_index:04d}",
                "document_id": meta["document_id"],
                "version_id": meta["version_id"],
                "chunk_index": chunk_index,
                "text": chunk_text,
                "embedding_text": prefix_ctx + chunk_text,
                "structure": {"path": list(current_path), "provision_id": current_path[-1]},
                "location": {"start_line": current_start_line, "end_line": len(lines), "char_start": char_offset, "char_end": char_offset + len(chunk_text)},
                "document_title": meta["document_title"],
                "document_type": meta["document_type"],
                "scope": meta.get("scope", "global"),
                "effective_from": meta.get("effective_from"),
                "effective_to": meta.get("effective_to"),
                "effective_dates_known": meta.get("effective_dates_known", True),
                "chunker_version": "structural-1.0"
            })

    return chunks