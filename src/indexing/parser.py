import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RU_MONTHS = {
    "января": "01", "январе": "01",
    "февраля": "02", "феврале": "02",
    "марта": "03", "марте": "03",
    "апреля": "04", "апреле": "04",
    "мая": "05", "мае": "05",
    "июня": "06", "июне": "06",
    "июля": "07", "июле": "07",
    "августа": "08", "августе": "08",
    "сентября": "09", "сентябре": "09",
    "октября": "10", "октябре": "10",
    "ноября": "11", "ноябре": "11",
    "декабря": "12", "декабре": "12",
}


def calculate_source_hash(content: str) -> str:
    """Вычисляет детерминированный SHA-256 хеш исходного файла."""
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


def parse_ru_text_date(date_str: str) -> Optional[str]:
    """Конвертирует строковую дату (цифровую или словесную) в ISO YYYY-MM-DD."""
    if not date_str:
        return None
    date_str = date_str.strip().rstrip("г.").strip()

    # Формат ISO YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    # Формат DD.MM.YYYY
    m_dot = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", date_str)
    if m_dot:
        day, month, year = m_dot.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    # Текстовый русский формат: "12 февраля 2024"
    m_text = re.search(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})", date_str, re.IGNORECASE)
    if m_text:
        day, month_name, year = m_text.groups()
        month_name = month_name.lower()
        if month_name in RU_MONTHS:
            return f"{year}-{RU_MONTHS[month_name]}-{int(day):02d}"

    return None


def detect_doc_type(filename: str, content: str) -> str:
    """Определяет тип документа: закон, внутренний регламент или договор."""
    sample = (filename + " " + content[:2000]).lower()
    if any(k in sample for k in ["договор", "контракт", "соглашение"]):
        return "contract"
    if any(k in sample for k in ["фз", "закон", "кодекс", "постановление", "указ", "регламент"]):
        return "law"
    return "internal"


def normalize_contract_number(raw_num: str) -> str:
    """Приводит номер документа вида 'П-17/2024' к каноническому виду 'p-17-2024'."""
    clean = re.sub(r"[^\w\-]", "-", raw_num).strip("-").lower()
    clean = clean.replace("—", "-")
    return clean


def parse_dates_and_version(content: str, filename: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """Извлекает даты начала/конца действия, связывает доп. соглашения с договором."""
    head = content[:2500]

    # 1. Поиск даты документа в шапке (город — дата, либо "от ...")
    date_found = None
    header_date = re.search(
        r"(?:—|от)\s+(\d{1,2}\s+[а-яё]+\s+\d{4}|\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})",
        head,
        re.IGNORECASE,
    )
    if header_date:
        date_found = parse_ru_text_date(header_date.group(1))

    # 2. Поиск окончания действия ("действует до 31 декабря 2026 г.")
    date_to = None
    tail_sample = content[-3000:] + " " + head
    m_to = re.search(
        r"действует\s+до\s+(\d{1,2}\s+[а-яё]+\s+\d{4}|\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})",
        tail_sample,
        re.IGNORECASE,
    )
    if m_to:
        date_to = parse_ru_text_date(m_to.group(1))

    # 3. Распознавание номера базового договора и связи с допниками
    amends_doc_id = None
    m_amend = re.search(
        r"к\s+договору\s+[^№]*№\s*([A-Za-zА-Яа-я0-9\-\/]+)",
        head,
        re.IGNORECASE,
    )

    # Поиск собственного номера документа в заголовке
    m_self_num = re.search(r"№\s*([A-Za-zА-Яа-я0-9\-\/]+)", head[:1000])

    is_amendment = "доп" in filename.lower() and "соглашение" in filename.lower()

    if is_amendment and m_amend:
        parent_num = normalize_contract_number(m_amend.group(1))
        amends_doc_id = f"contract_{parent_num}"
        amend_num = m_self_num.group(1) if m_self_num else "1"
        doc_id = f"amend_{normalize_contract_number(amend_num)}_{parent_num}"
        ver_id = f"{doc_id}_v1"
    else:
        if m_self_num:
            doc_id = f"contract_{normalize_contract_number(m_self_num.group(1))}"
        else:
            stem_clean = normalize_contract_number(Path(filename).stem)
            doc_id = f"doc_{stem_clean}"
        ver_id = f"{doc_id}_v1"

    dates_dict = {
        "document_id": doc_id,
        "version_id": ver_id,
        "published_at": date_found,
        "effective_from": date_found,
        "effective_to": date_to,
        "effective_dates_known": date_found is not None,
    }
    return dates_dict, amends_doc_id


def classify_heading(line: str) -> Optional[Tuple[int, str, Optional[str]]]:
    """
    Классифицирует строку как структурный заголовок с очисткой от обрамляющих кавычек.
    Возвращает (уровень, заголовок, provision_id).
    """
    # Счищаем русские и английские кавычки в начале и конце строки
    clean = re.sub(r'^[«"\'„“”\s]+|[»"\'„“”\s]+$', '', line.strip())
    if not clean:
        return None

    # Уровень 0: Раздел
    m_sec = re.match(r"^(Раздел\s+[IVXLCDM\d]+[^\n\.\;]*)", clean, re.IGNORECASE)
    if m_sec:
        return (0, m_sec.group(1).strip(), None)

    # Уровень 1: Глава
    m_chap = re.match(r"^(Глава\s+[IVXLCDM\d]+[^\n\.\;]*)", clean, re.IGNORECASE)
    if m_chap:
        return (1, m_chap.group(1).strip(), None)

    # Уровень 2: Статья
    m_art = re.match(r"^(Статья\s+(\d+[\.\d]*)[^\n\.\;]*)", clean, re.IGNORECASE)
    if m_art:
        return (2, m_art.group(1).strip(), m_art.group(2).strip())

    # Уровень 2: Крупный нумерованный раздел договора ("1. Предмет договора")
    m_major = re.match(r"^(\d+)\.\s+([А-Яа-яЁёA-Za-z\s]{3,80})$", clean)
    if m_major:
        return (2, clean[:100], m_major.group(1).strip())

    # Уровень 3: Пункт словом ("Пункт 3.2. ...")
    m_punkt = re.match(r"^(Пункт\s+(\d+[\.\d]*)[^\n\.\;]*)", clean, re.IGNORECASE)
    if m_punkt:
        return (3, m_punkt.group(1).strip(), m_punkt.group(2).strip())

    # Уровень 3: Цифровые пункты договора ("3.2.", "3.2. Товар...", "1.1", "2.2.3")
    m_sub = re.match(r"^(\d+\.\d+(?:\.\d+)*)\.?\s*(.*)$", clean)
    if m_sub:
        num_str = m_sub.group(1)
        depth = num_str.count(".")
        level = 3 + min(depth - 1, 2)
        return (level, f"Пункт {num_str}", num_str)

    return None


def extract_document_structure(content: str) -> List[Dict[str, Any]]:
    """Разбивает текст на блоки с поддержкой стека иерархии structure.path."""
    lines = content.splitlines(keepends=True)
    blocks: List[Dict[str, Any]] = []

    path_stack: List[Tuple[int, str]] = [(0, "Введение / Общие положения")]
    current_provision_id: Optional[str] = None
    current_lines: List[str] = []

    start_line = 1
    char_offset = 0
    block_char_start = 0

    for line_no, line in enumerate(lines, start=1):
        h_info = classify_heading(line)
        if h_info is not None:
            level, h_title, prov_id = h_info
            if current_lines:
                b_text = "".join(current_lines).strip()
                if b_text:
                    blocks.append({
                        "provision_id": current_provision_id,
                        "path": [t for _, t in path_stack],
                        "start_line": start_line,
                        "end_line": line_no - 1,
                        "char_start": block_char_start,
                        "char_end": char_offset,
                        "text": b_text,
                    })
                current_lines = []
                start_line = line_no
                block_char_start = char_offset

            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()

            path_stack.append((level, h_title))
            current_provision_id = prov_id

        current_lines.append(line)
        char_offset += len(line)

    if current_lines:
        b_text = "".join(current_lines).strip()
        if b_text:
            blocks.append({
                "provision_id": current_provision_id,
                "path": [t for _, t in path_stack],
                "start_line": start_line,
                "end_line": len(lines),
                "char_start": block_char_start,
                "char_end": char_offset,
                "text": b_text,
            })

    return blocks


def extract_metadata(file_path: Path, content: str) -> Dict[str, Any]:
    """Формирует канонические метаданные версии документа по разделу 2.3 rag.md."""
    dates_ver, amends_id = parse_dates_and_version(content, file_path.name)
    title_line = file_path.stem
    for line in content.splitlines():
        c_line = line.strip()
        if c_line:
            title_line = c_line[:200]
            break

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "document_id": dates_ver["document_id"],
        "version_id": dates_ver["version_id"],
        "document_title": title_line,
        "document_type": detect_doc_type(file_path.name, content),
        "scope": "global",
        "chat_id": None,
        "created_by": "system",
        "source_filename": file_path.name,
        "source_hash": calculate_source_hash(content),
        "published_at": dates_ver["published_at"],
        "effective_from": dates_ver["effective_from"],
        "effective_to": dates_ver["effective_to"],
        "effective_dates_known": dates_ver["effective_dates_known"],
        "ingested_at": now_iso,
        "amends_document_id": amends_id,
        "parser_version": "txt-parser-1.0",
        "chunker_version": "structural-1.0",
    }


def load_documents_with_metadata(raw_dir: Path) -> List[Dict[str, Any]]:
    docs = []
    for p in sorted(raw_dir.glob("**/*.txt")):
        with open(p, "r", encoding="utf-8") as f:
            c = f.read()
        docs.append({"content": c, "metadata": extract_metadata(p, c)})
    return docs