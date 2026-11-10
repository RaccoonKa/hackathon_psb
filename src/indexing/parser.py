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
    """Детерминированный SHA-256 хеш исходного файла."""
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


def parse_ru_text_date(date_str: str) -> Optional[str]:
    """Конвертирует цифровую или текстовую русскую дату в формат ISO YYYY-MM-DD."""
    if not date_str:
        return None
    date_str = date_str.strip().rstrip("г.").strip()

    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    m_dot = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", date_str)
    if m_dot:
        day, month, year = m_dot.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    m_text = re.search(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})", date_str, re.IGNORECASE)
    if m_text:
        day, month_name, year = m_text.groups()
        month_name = month_name.lower()
        if month_name in RU_MONTHS:
            return f"{year}-{RU_MONTHS[month_name]}-{int(day):02d}"

    return None


def clean_garant_headers(content: str) -> Tuple[str, Optional[str]]:
    """Отсекает служебные мета-строки систем Гарант/Консультант и возвращает чистый текст."""
    lines = content.splitlines(keepends=True)
    cleaned_lines = []
    extracted_title = None

    skip_header = True
    for line in lines:
        stripped = line.strip()
        if skip_header:
            if stripped.startswith("Источник:"):
                # Извлекаем название закона из строки источника
                m_title = re.search(r"Источник:\s*(.+)$", stripped)
                if m_title:
                    extracted_title = m_title.group(1).split(".")[0].strip()
                continue
            if stripped.startswith("URL:") or stripped.startswith("Дата выгрузки:"):
                continue
            if not stripped:
                continue
            skip_header = False

        cleaned_lines.append(line)

    return "".join(cleaned_lines), extracted_title


def detect_doc_type(filename: str, content: str) -> str:
    """Точная типизация документа с приоритетом префиксов и служебных отметок."""
    fn_low = filename.lower()
    head_low = content[:1500].lower()

    if fn_low.startswith("законы__") or any(k in head_low for k in ["федеральный закон", "кодекс российской федерации"]):
        return "law"

    if any(k in fn_low or k in head_low for k in ["служебная записка", "приказ", "положение", "регламент", "инструкция"]):
        return "internal"

    if any(k in fn_low or k in head_low for k in ["договор", "контракт", "дополнительное соглашение", "доп соглашение"]):
        return "contract"

    return "internal"


def normalize_contract_number(raw_num: str) -> str:
    clean = re.sub(r"[^\w\-]", "-", raw_num).strip("-").lower()
    return clean.replace("—", "-")


def parse_dates_and_version(content: str, filename: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """Извлекает даты вступления редакции в силу, окончания действия и связи документов."""
    head = content[:3500]

    # 1. Поиск даты актуальной редакции (приоритет для законов)
    date_found = None
    m_revision = re.search(
        r"(?:изменена\s+с|в\s+редакции\s+от|действует\s+с|вступает\s+в\s+силу\s+с)\s+(\d{1,2}\s+[а-яё]+\s+\d{4}|\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})",
        head,
        re.IGNORECASE,
    )
    if m_revision:
        date_found = parse_ru_text_date(m_revision.group(1))

    # Если точной даты редакции нет, ищем дату принятия документа
    if not date_found:
        m_doc_date = re.search(
            r"(?:—|от)\s+(\d{1,2}\s+[а-яё]+\s+\d{4}|\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})",
            head,
            re.IGNORECASE,
        )
        if m_doc_date:
            date_found = parse_ru_text_date(m_doc_date.group(1))

    # 2. Дата окончания действия
    date_to = None
    m_to = re.search(
        r"действует\s+до\s+(\d{1,2}\s+[а-яё]+\s+\d{4}|\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})",
        content[-3000:] + " " + head,
        re.IGNORECASE,
    )
    if m_to:
        date_to = parse_ru_text_date(m_to.group(1))

    # 3. Идентификаторы и связь с доп. соглашениями
    amends_doc_id = None
    m_amend = re.search(r"к\s+договору\s+[^№]*№\s*([A-Za-zА-Яа-я0-9\-\/]+)", head, re.IGNORECASE)
    m_self_num = re.search(r"№\s*([A-Za-zА-Яа-я0-9\-\/]+)", head[:1000])

    is_amendment = "доп" in filename.lower() and "соглашение" in filename.lower()
    fn_stem = Path(filename).stem.replace("документы компании__", "").replace("законы__", "")

    if is_amendment and m_amend:
        parent_num = normalize_contract_number(m_amend.group(1))
        amends_doc_id = f"contract_{parent_num}"
        amend_num = m_self_num.group(1) if m_self_num else "1"
        doc_id = f"amend_{normalize_contract_number(amend_num)}_{parent_num}"
    elif m_self_num and ("договор" in fn_stem.lower() or "контракт" in fn_stem.lower()):
        doc_id = f"contract_{normalize_contract_number(m_self_num.group(1))}"
    else:
        doc_id = f"doc_{normalize_contract_number(fn_stem)}"

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


def classify_heading(line: str, in_article: bool = False, current_art_num: Optional[str] = None) -> Optional[Tuple[int, str, Optional[str]]]:
    """
    Классифицирует заголовок с поддержкой частей законов РФ и очисткой от кавычек.
    Возвращает (уровень, заголовок, provision_id).
    """
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

    # Уровень 2: Статья закона
    m_art = re.match(r"^(Статья\s+(\d+[\.\d]*)[^\n\.\;]*)", clean, re.IGNORECASE)
    if m_art:
        return (2, m_art.group(1).strip(), m_art.group(2).strip())

    # Уровень 2: Крупный раздел договора ("1. Предмет договора")
    m_major = re.match(r"^(\d+)\.\s+([А-Яа-яЁёA-Za-z\s]{3,80})$", clean)
    if m_major and not in_article:
        return (2, clean[:100], m_major.group(1).strip())

    # Уровень 3: Часть статьи закона РФ (например, "1. Субъект персональных данных...")
    m_part = re.match(r"^(\d+)\.\s+(.+)$", clean)
    if m_part and in_article:
        part_num = m_part.group(1)
        prov_id = f"{current_art_num}.{part_num}" if current_art_num else part_num
        return (3, f"Часть {part_num}", prov_id)

    # Уровень 3: Пункт словом ("Пункт 3.2. ...")
    m_punkt = re.match(r"^(Пункт\s+(\d+[\.\d]*)[^\n\.\;]*)", clean, re.IGNORECASE)
    if m_punkt:
        return (3, m_punkt.group(1).strip(), m_punkt.group(2).strip())

    # Уровень 3+: Цифровые подпункты ("3.2.", "1.1", "2.2.3")
    m_sub = re.match(r"^(\d+\.\d+(?:\.\d+)*)\.?\s*(.*)$", clean)
    if m_sub:
        num_str = m_sub.group(1)
        depth = num_str.count(".")
        level = 3 + min(depth - 1, 2)
        return (level, f"Пункт {num_str}", num_str)

    return None


def extract_document_structure(content: str) -> List[Dict[str, Any]]:
    """Разбивает документ на блоки с непрерывным ведением стека иерархии."""
    lines = content.splitlines(keepends=True)
    blocks: List[Dict[str, Any]] = []

    path_stack: List[Tuple[int, str]] = [(0, "Введение / Общие положения")]
    current_provision_id: Optional[str] = None
    current_art_num: Optional[str] = None
    in_article = False
    current_lines: List[str] = []

    start_line = 1
    char_offset = 0
    block_char_start = 0

    for line_no, line in enumerate(lines, start=1):
        h_info = classify_heading(line, in_article=in_article, current_art_num=current_art_num)
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
                popped_level, _ = path_stack.pop()
                if popped_level == 2:
                    in_article = False
                    current_art_num = None

            path_stack.append((level, h_title))
            current_provision_id = prov_id

            if level == 2 and "Статья" in h_title:
                in_article = True
                current_art_num = prov_id

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


def extract_metadata(file_path: Path, raw_content: str) -> Tuple[Dict[str, Any], str]:
    """Формирует канонические метаданные версии документа и возвращает очищенный текст."""
    cleaned_content, extracted_title = clean_garant_headers(raw_content)
    dates_ver, amends_id = parse_dates_and_version(cleaned_content, file_path.name)
    doc_type = detect_doc_type(file_path.name, cleaned_content)

    title = extracted_title
    if not title:
        for line in cleaned_content.splitlines():
            c_line = line.strip()
            if c_line and not c_line.startswith("г.") and not c_line.startswith("ООО"):
                title = c_line[:200]
                break
    if not title:
        title = file_path.stem

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    metadata = {
        "document_id": dates_ver["document_id"],
        "version_id": dates_ver["version_id"],
        "document_title": title,
        "document_type": doc_type,
        "scope": "global",
        "chat_id": None,
        "created_by": "system",
        "source_filename": file_path.name,
        "source_hash": calculate_source_hash(raw_content),
        "published_at": dates_ver["published_at"],
        "effective_from": dates_ver["effective_from"],
        "effective_to": dates_ver["effective_to"],
        "effective_dates_known": dates_ver["effective_dates_known"],
        "ingested_at": now_iso,
        "amends_document_id": amends_id,
        "parser_version": "txt-parser-2.0",
        "chunker_version": "structural-1.0",
    }
    return metadata, cleaned_content


def load_documents_with_metadata(raw_dir: Path) -> List[Dict[str, Any]]:
    docs = []
    for p in sorted(raw_dir.glob("**/*.txt")):
        with open(p, "r", encoding="utf-8") as f:
            raw_c = f.read()
        meta, clean_c = extract_metadata(p, raw_c)
        docs.append({"content": clean_c, "metadata": meta})
    return docs