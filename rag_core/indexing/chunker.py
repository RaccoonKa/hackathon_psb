import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from rag_core.config import PROCESSED_DATA_DIR
from rag_core.indexing.parser import extract_document_structure
from rag_core.indexing.tokenizer_utils import count_tokens

TARGET_MIN_TOKENS = 300
TARGET_MAX_TOKENS = 600
HARD_MAX_TOKENS = 750
OVERLAP_TOKENS = 60

# Константы сохранены для обратной совместимости с тестами перекрытия
SUB_CHUNK_SIZE_CHARS = HARD_MAX_TOKENS * 3
SUB_CHUNK_OVERLAP_CHARS = OVERLAP_TOKENS * 3

_oversized_splitter = None


def _get_oversized_splitter():
    """Ленивая инициализация langchain-сплиттера по токенам (не нужна при импорте)."""
    global _oversized_splitter
    if _oversized_splitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        _oversized_splitter = RecursiveCharacterTextSplitter(
            chunk_size=HARD_MAX_TOKENS - 15,
            chunk_overlap=OVERLAP_TOKENS,
            length_function=count_tokens,
            separators=["\n\n", "\n", ";\n", r"(?<!\d)\.\s+", " "],
            is_separator_regex=True,
            keep_separator=True,
        )
    return _oversized_splitter


def generate_chunk_id(
    doc_id: str,
    ver_id: str,
    source_hash: Optional[str],
    chunk_index: int,
    chunker_ver: str = "structural-1.0",
) -> str:
    """Генерирует стабильный chunk_id по разделу 3.4 rag.md."""
    s_hash = source_hash or "nohash"
    raw = f"{doc_id}_{ver_id}_{s_hash}_{chunker_ver}_{chunk_index:04d}"
    hash_suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{ver_id}_{chunk_index:04d}_{hash_suffix}"


def _resolve_page_number(
    content: str,
    c_start: int,
    c_end: int,
    page_spans: Optional[List[Tuple[int, int, int]]],
) -> Optional[int]:
    """Определяет номер страницы (1-based) для PDF по символьному смещению; None для TXT/DOCX."""
    if not page_spans:
        return None

    # Смещаем точку поиска к первому непробельному символу фрагмента,
    # чтобы ведущие переносы строк не относили чанк к предыдущей странице.
    target_pos = c_start
    limit = min(len(content), max(c_start, c_end))
    while target_pos < limit and content[target_pos].isspace():
        target_pos += 1

    for p_start, p_end, page_num in page_spans:
        if p_start <= target_pos < p_end:
            return page_num

    if target_pos < page_spans[0][0]:
        return page_spans[0][2]
    return page_spans[-1][2]


def _is_bare_heading_block(b: Dict[str, Any]) -> bool:
    """Определяет, является ли блок лишь структурным заголовком без содержательного текста нормы.

    Заголовки разделов договоров ('1. Предмет договора') или статей законов ('Статья 112...'),
    за которыми сразу следуют подпункты, должны сохраняться в path дочерних элементов, но не
    порождать пустые чанки-пустышки из 3 слов (п. 2 CHUNKING_TASKS.md).
    """
    path = b.get("path") or []
    if not path:
        return False

    clean_text = b.get("text", "").strip()
    if not clean_text:
        return True

    if len(clean_text) > 300 or len(clean_text.split()) > 40:
        return False

    last_heading = path[-1].strip()
    norm_text = re.sub(r"[\s\.\:\;\-]+", " ", clean_text).strip().lower()
    norm_h = re.sub(r"[\s\.\:\;\-]+", " ", last_heading).strip().lower()

    if norm_text == norm_h:
        return True

    if clean_text.rstrip(".:") == last_heading.rstrip(".:"):
        return True

    return False


def chunk_document_structural(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = doc["content"]
    meta = doc["metadata"]
    page_spans: Optional[List[Tuple[int, int, int]]] = doc.get("page_spans")
    total_doc_tokens = count_tokens(content)

    chunks: List[Dict[str, Any]] = []
    chunk_idx = 0

    def make_chunk_payload(
        text: str,
        path: List[str],
        provision_id: Optional[str],
        start_line: int,
        end_line: int,
        c_start: int,
        c_end: int,
    ) -> Dict[str, Any]:
        nonlocal chunk_idx

        prefix_parts = [meta["document_title"]]
        if meta.get("amends_document_id"):
            prefix_parts.append(f"[ИЗМЕНЕНИЕ ДЛЯ {meta['amends_document_id'].upper()}]")
        prefix_parts.extend(path)

        prefix = ". ".join(prefix_parts) + ". "
        clean_text = text.strip()
        page_num = _resolve_page_number(content, c_start, c_end, page_spans)

        cid = generate_chunk_id(
            doc_id=meta["document_id"],
            ver_id=meta["version_id"],
            source_hash=meta.get("source_hash"),
            chunk_index=chunk_idx,
            chunker_ver=meta.get("chunker_version", "structural-1.0"),
        )

        payload = {
            "chunk_id": cid,
            "document_id": meta["document_id"],
            "version_id": meta["version_id"],
            "chunk_index": chunk_idx,
            "text": clean_text,
            "embedding_text": prefix + clean_text,
            "structure": {
                "path": list(path),
                "provision_id": provision_id,
            },
            "location": {
                "start_line": start_line,
                "end_line": end_line,
                "char_start": c_start,
                "char_end": c_end,
                "page": page_num,
            },
            "document_title": meta["document_title"],
            "document_type": meta["document_type"],
            "scope": meta.get("scope", "global"),
            "chat_id": meta.get("chat_id"),
            "effective_from": meta.get("effective_from"),
            "effective_to": meta.get("effective_to"),
            "effective_dates_known": meta.get("effective_dates_known", True),
            "source_hash": meta.get("source_hash"),
            "amends_document_id": meta.get("amends_document_id"),
            "chunker_version": meta.get("chunker_version", "structural-1.0"),
        }
        chunk_idx += 1
        return payload

    # Защита коротких допников и служебных писем (до HARD_MAX_TOKENS): сохраняем целиком
    if total_doc_tokens <= HARD_MAX_TOKENS and (
        meta.get("amends_document_id") or meta.get("document_type") == "internal"
    ):
        chunks.append(
            make_chunk_payload(
                text=content,
                path=["Полный текст документа"],
                provision_id=None,
                start_line=1,
                end_line=len(content.splitlines()),
                c_start=0,
                c_end=len(content),
            )
        )
        return chunks

    blocks = extract_document_structure(content)
    i = 0
    while i < len(blocks):
        b = blocks[i]

        # Пустые заголовки разделов/статей без собственного тела не порождают чанки
        if len(blocks) > 1 and _is_bare_heading_block(b):
            i += 1
            continue

        b_tokens = count_tokens(b["text"])

        # 1. Положение превышает лимит (> 750 токенов) -> делим по реальным токенам с перекрытием
        if b_tokens > HARD_MAX_TOKENS:
            sub_parts = _get_oversized_splitter().split_text(b["text"])
            running_char = b["char_start"]
            search_cursor = b["char_start"]

            for part in sub_parts:
                clean_part = part.strip()
                found_idx = (
                    content.find(clean_part, search_cursor, b["char_end"] + 50)
                    if len(clean_part) >= 15
                    else -1
                )

                if found_idx != -1:
                    c_start = found_idx
                    c_end = found_idx + len(clean_part)
                    search_cursor = max(c_start + 1, c_end - SUB_CHUNK_OVERLAP_CHARS * 3)
                    part_start_line = content.count("\n", 0, c_start) + 1
                    part_end_line = content.count("\n", 0, max(c_start, c_end - 1)) + 1
                else:
                    c_start = running_char
                    c_end = running_char + len(part)
                    part_start_line = b["start_line"]
                    part_end_line = b["end_line"]

                chunks.append(
                    make_chunk_payload(
                        text=part,
                        path=b["path"],
                        provision_id=b["provision_id"],
                        start_line=part_start_line,
                        end_line=part_end_line,
                        c_start=c_start,
                        c_end=c_end,
                    )
                )
                running_char = c_start + max(1, len(part) - SUB_CHUNK_OVERLAP_CHARS)
            i += 1
            continue

        # 2. Мелкие вводные преамбулы без номера нормы -> объединяем до TARGET_MAX_TOKENS
        if b["provision_id"] is None and b_tokens < TARGET_MIN_TOKENS:
            merged_text = [b["text"]]
            merged_end_line = b["end_line"]
            merged_char_end = b["char_end"]
            j = i + 1
            while j < len(blocks) and blocks[j]["provision_id"] is None:
                next_tokens = count_tokens(blocks[j]["text"])
                if count_tokens("\n\n".join(merged_text)) + next_tokens > TARGET_MAX_TOKENS:
                    break
                merged_text.append(blocks[j]["text"])
                merged_end_line = blocks[j]["end_line"]
                merged_char_end = blocks[j]["char_end"]
                j += 1

            chunks.append(
                make_chunk_payload(
                    text="\n\n".join(merged_text),
                    path=b["path"],
                    provision_id=None,
                    start_line=b["start_line"],
                    end_line=merged_end_line,
                    c_start=b["char_start"],
                    c_end=merged_char_end,
                )
            )
            i = j
            continue

        # 3. Самостоятельное положение с provision_id сохраняется в отдельный адресуемый чанк
        chunks.append(
            make_chunk_payload(
                text=b["text"],
                path=b["path"],
                provision_id=b["provision_id"],
                start_line=b["start_line"],
                end_line=b["end_line"],
                c_start=b["char_start"],
                c_end=b["char_end"],
            )
        )
        i += 1

    return chunks


def save_canonical_chunks(all_chunks: List[Dict[str, Any]]) -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = PROCESSED_DATA_DIR / "chunks.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    return out_file