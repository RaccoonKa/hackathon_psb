import json
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import PROCESSED_DATA_DIR

def chunk_document_structural(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = doc["content"]
    meta = doc["metadata"]
    lines = content.splitlines(keepends=True)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=70,
        separators=["\n\n", "\n", ";\n", ". ", " "],
        keep_separator=True
    )

    chunks = []
    current_path = ["Введение / Общие положения"]
    current_lines = []
    start_line = 1
    char_offset = 0
    chunk_idx = 0

    def flush_block(lines_block: List[str], s_line: int, e_line: int, c_start: int):
        nonlocal chunk_idx
        raw_text = "".join(lines_block).strip()
        if not raw_text:
            return
        
        splits = splitter.split_text(raw_text)
        for part in splits:
            context_prefix = f"{meta['document_title']}. {' > '.join(current_path)}. "
            cid = f"{meta['document_id']}_{meta['version_id']}_{chunk_idx:04d}"
            chunks.append({
                "chunk_id": cid,
                "document_id": meta["document_id"],
                "version_id": meta["version_id"],
                "chunk_index": chunk_idx,
                "text": part.strip(),
                "embedding_text": context_prefix + part.strip(),
                "structure": {"path": list(current_path)},
                "location": {"start_line": s_line, "end_line": e_line, "char_start": c_start},
                "document_title": meta["document_title"],
                "document_type": meta["document_type"],
                "scope": meta["scope"],
                "effective_from": meta.get("effective_from"),
                "effective_to": meta.get("effective_to"),
                "effective_dates_known": meta.get("effective_dates_known", True),
                "chunker_version": "structural-1.0"
            })
            chunk_idx += 1

    for line_no, line in enumerate(lines, start=1):
        clean = line.strip()
        if any(clean.startswith(p) for p in ["Статья ", "Глава ", "Раздел ", "Пункт "]):
            if current_lines:
                flush_block(current_lines, start_line, line_no - 1, char_offset)
                current_lines = []
                start_line = line_no
            current_path = [clean[:100]]

        current_lines.append(line)
        char_offset += len(line)

    if current_lines:
        flush_block(current_lines, start_line, len(lines), char_offset)

    return chunks

def save_canonical_chunks(all_chunks: List[Dict[str, Any]]) -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = PROCESSED_DATA_DIR / "chunks.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    return out_file