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


def clean_garant_headers(content: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Отсекает служебные мета-строки систем Гарант/Консультант.
    Возвращает (очищенный текст, полное название документа, дату выгрузки).
    """
    lines = content.splitlines(keepends=True)
    cleaned_lines = []
    extracted_title = None
    export_date = None

    skip_header = True
    for line in lines:
        stripped = line.strip()
        if skip_header:
            if stripped.startswith("Источник:"):
                m_title = re.search(r"Источник:\s*(.+)$", stripped)
                if m_title:
                    raw_title = m_title.group(1).strip()
                    extracted_title = re.sub(
                        r"\s*\(с\s+изменениями\s+и\s+дополнениями\).*", "", raw_title, flags=re.IGNORECASE
                    ).strip()
                continue
            if stripped.startswith("Дата выгрузки:"):
                m_exp = re.search(r"Дата выгрузки:\s*(\d{1,2}\s+[а-яё]+\s+\d{4}|\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})", stripped)
                if m_exp:
                    export_date = parse_ru_text_date(m_exp.group(1))
                continue
            if stripped.startswith("URL:"):
                continue
            if not stripped:
                continue
            skip_header = False

        if re.search(r"воспользуйтесь\s+калькулятором.*?(?:гарант|консультант)", stripped, re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    return "".join(cleaned_lines), extracted_title, export_date


def detect_doc_type(filename: str, content: str) -> str:
    """Точная типизация документа с поддержкой писем комплаенса и служебных записок."""
    fn_low = filename.lower()
    head_low = content[:1500].lower()

    if fn_low.startswith("законы__") or any(k in head_low for k in ["федеральный закон", "кодекс российской федерации"]):
        return "law"

    if any(k in fn_low or k in head_low for k in ["служебная записка", "приказ", "положение", "регламент", "инструкция", "письмо"]):
        return "internal"

    if any(k in fn_low or k in head_low for k in ["договор", "контракт", "дополнительное соглашение", "доп соглашение"]):
        return "contract"

    return "internal"


def normalize_contract_number(raw_num: str) -> str:
    clean = re.sub(r"[^\w\-]", "-", raw_num).strip("-").lower()
    return clean.replace("—", "-")


def parse_dates_and_version(
    content: str, filename: str, doc_type: str = "internal", export_date: Optional[str] = None
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Извлекает даты вступления редакции в силу, окончания действия и связи документов."""
    head = content[:3500]
    cutoff_date = export_date or "2026-10-10"

    # 1. Поиск даты окончания действия или утраты силы
    effective_to = None

    m_expired = re.search(
        r"(?:утратил(?:а|о)?\s+силу\s+с|утрачивает\s+силу\s+с)\s+(\d{1,2}\s+[а-яё]+\s+\d{4}|\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})",
        content,
        re.IGNORECASE,
    )
    if m_expired:
        effective_to = parse_ru_text_date(m_expired.group(1))

    if not effective_to:
        for m_until in re.finditer(
            r"действует\s+до\s+(\d{1,2}\s+[а-яё]+\s+\d{4}|\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})",
            content,
            re.IGNORECASE,
        ):
            line_end = content.find("\n", m_until.end())
            if line_end == -1:
                line_end = len(content)
            window = content[m_until.start(): line_end].lower()
            if any(k in window for k in ["продлева", "пролонгир"]):
                continue
            effective_to = parse_ru_text_date(m_until.group(1))
            if effective_to:
                break

    clean_head = re.sub(r"^РЕДАКЦИЯ\s+УТРАТИЛА\s+СИЛУ[^\n]*\n?", "", head, flags=re.IGNORECASE)

    # 2. Поиск даты актуальной редакции или начала применения
    effective_from = None

    m_action_start = re.search(
        r"(?:начиная\s+с|[Сс])\s+(\d{1,2}\s+[а-яё]+\s+\d{4}|\d{2}\.\d{2}\.\d{4})(?:\s*г\.?)?\s+(?:до\s+подписания|применяется|вступает)",
        clean_head,
        re.IGNORECASE,
    )
    if m_action_start:
        effective_from = parse_ru_text_date(m_action_start.group(1))

    if not effective_from:
        rev_matches = re.findall(
            r"(?:изменена\s+с|(?:в\s+)?редакци[яие]\s+от|действует\s+с|вступает\s+в\s+силу\s+с|вступающие\s+в\s+силу\s+с)\s+(\d{1,2}\s+[а-яё]+\s+\d{4}|\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})",
            clean_head,
            re.IGNORECASE,
        )
        parsed_dates = [parse_ru_text_date(d) for d in rev_matches if parse_ru_text_date(d)]

        if parsed_dates:
            past_or_current = [d for d in parsed_dates if d <= cutoff_date]
            future_dates = [d for d in parsed_dates if d > cutoff_date]

            if past_or_current:
                effective_from = max(past_or_current)
            if future_dates and not effective_to:
                effective_to = min(future_dates)

    if not effective_from:
        m_doc_date = re.search(
            r"(?:—|от)\s+(\d{1,2}\s+[а-яё]+\s+\d{4}|\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})",
            clean_head,
            re.IGNORECASE,
        )
        if m_doc_date:
            effective_from = parse_ru_text_date(m_doc_date.group(1))

    # 3. Идентификаторы и связи документов
    amends_doc_id = None
    m_amend = re.search(r"к\s+договору\s+[^№]*№\s*([A-Za-zА-Яа-я0-9\-\/]+)", clean_head, re.IGNORECASE)
    m_self_num = re.search(r"№\s*([A-Za-zА-Яа-я0-9\-\/]+)", clean_head[:1000])

    is_amendment = "доп" in filename.lower() and "соглашение" in filename.lower()
    fn_stem = Path(filename).stem.replace("документы компании__", "").replace("законы__", "")

    if is_amendment and m_amend:
        parent_num = normalize_contract_number(m_amend.group(1))
        amends_doc_id = f"contract_{parent_num}"
        amend_num = m_self_num.group(1) if m_self_num else "1"
        doc_id = f"amend_{normalize_contract_number(amend_num)}_{parent_num}"
    elif doc_type == "contract" and m_self_num:
        doc_id = f"contract_{normalize_contract_number(m_self_num.group(1))}"
    else:
        doc_id = f"doc_{normalize_contract_number(fn_stem)}"

    ver_id = f"{doc_id}_v1"

    dates_dict = {
        "document_id": doc_id,
        "version_id": ver_id,
        "published_at": effective_from,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "effective_dates_known": effective_from is not None,
    }
    return dates_dict, amends_doc_id


def classify_heading(
    line: str,
    in_article: bool = False,
    current_art_num: Optional[str] = None,
    current_part_num: Optional[str] = None,
    is_nk: bool = False,
    in_notes: bool = False,
) -> Optional[Tuple[int, str, Optional[str]]]:
    """
    Классифицирует заголовок по уровням:
    Уровень 0: Раздел
    Уровень 1: Глава
    Уровень 2: Статья закона / Крупный раздел договора
    Уровень 3: Пункт договора / Часть статьи закона / Примечания
    Уровень 4: Подпункт со скобкой / Пункт примечания
    """
    clean = re.sub(r'^[«"\'„“”\s]+|[»"\'„“”\s]+$', "", line.strip())
    if not clean:
        return None

    # Игнорируем технические пометки справочных систем
    if re.search(
        r"^(?:(?:Пункт|Подпункт|Статья|Часть|Абзац|Раздел|Глава)\s+\d+(?:[\.\-]\d+)*.*?\s+(?:изменен|дополнен|утратил|внесены)|"
        r"Примечания\s+изменены\s+|"
        r"(?:Положения|Действие положений|Действие)\s+|"
        r"О\s+(?:проверке\s+конституционности|конституционно-правовом\s+смысле)|"
        r"Согласно\s+Определению|"
        r"Федеральным\s+законом\s+от|"
        r"Обратите\s+внимание,\s+что\s+применительно|"
        r"Cрок\s+давности\s+привлечения)",
        clean,
        re.IGNORECASE,
    ):
        return None

    # Уровень 0: Раздел
    m_sec = re.match(r"^(Раздел\s+[IVXLCDM\d]+[^\n\.\;]*)", clean, re.IGNORECASE)
    if m_sec:
        return (0, m_sec.group(1).strip(), None)

    # Уровень 1: Глава
    m_chap = re.match(r"^(Глава\s+[IVXLCDM\d]+[^\n\.\;]*)", clean, re.IGNORECASE)
    if m_chap:
        return (1, m_chap.group(1).strip(), None)

    # Уровень 2: Статья закона (например, "Статья 112.", "Статья 13.11.")
    m_art = re.match(r"^Статья\s+(\d+(?:\.\d+)*)\.?(.*)$", clean, re.IGNORECASE)
    if m_art:
        art_num = m_art.group(1).rstrip(".")
        rest = m_art.group(2).strip()
        title = f"Статья {art_num}. {rest}".strip().rstrip(".")
        return (2, title[:120], art_num)

    # Уровень 3: Раздел примечаний к статье (КоАП)
    if re.match(r"^Примечания:?$", clean, re.IGNORECASE):
        art_clean = current_art_num.rstrip(".") if current_art_num else ""
        prov_id = f"{art_clean}.прим" if art_clean else "прим"
        return (3, "Примечания", prov_id)

    # Уровень 2: Крупный раздел договора ("1. Предмет договора", "3. Порядок поставки")
    m_major = re.match(r"^(\d+)\.\s+([А-Яа-яЁёA-Za-z\s]{3,80})$", clean)
    if m_major and not in_article:
        return (2, clean[:100], m_major.group(1).strip())

    # Уровень 4: Пункты внутри блока примечаний: "1. За административные..."
    m_note_item = re.match(r"^(\d+)\.\s+(.+)$", clean)
    if m_note_item and in_notes:
        note_num = m_note_item.group(1)
        art_clean = current_art_num.rstrip(".") if current_art_num else ""
        prov_id = f"{art_clean}.прим.{note_num}" if art_clean else f"прим.{note_num}"
        return (4, f"Примечание {note_num}", prov_id)

    # Уровень 4: Подпункты со скобкой ("1)", "1.1)", "2.1)", "15.1)") — проверяем ДО m_sub!
    m_paren = re.match(r"^(\d+(?:[\.\-]\d+)*)\)\s*(.*)$", clean)
    if m_paren and in_article:
        sub_num = m_paren.group(1).rstrip(".")
        parent_id = current_part_num or current_art_num
        parent_clean = parent_id.rstrip(".") if parent_id else ""
        prov_id = f"{parent_clean}.{sub_num}" if parent_clean else sub_num
        return (4, f"Подпункт {sub_num}", prov_id)

    # Уровень 3+: Цифровые пункты договоров ("3.2.", "1.1", "2.2.3") или части закона с дробью (КоАП "1.1.", "2.1.")
    m_sub = re.match(r"^(\d+\.\d+(?:\.\d+)*)\.?\s*(.*)$", clean)
    if m_sub:
        num_str = m_sub.group(1).rstrip(".")
        depth = num_str.count(".")
        level = 3 + min(depth - 1, 2)
        if in_article and current_art_num:
            prov_id = f"{current_art_num}.{num_str}"
            item_title = f"Пункт {num_str}" if is_nk else f"Часть {num_str}"
            return (level, item_title, prov_id)
        else:
            return (level, f"Пункт {num_str}", num_str)

    # Уровень 3: Часть закона (1., 2., 3.) или Пункт НК РФ — строго внутри статьи закона!
    m_part = re.match(r"^(\d+)\.\s+(.+)$", clean)
    if m_part and in_article and not in_notes:
        part_num = m_part.group(1).rstrip(".")
        art_clean = current_art_num.rstrip(".") if current_art_num else ""
        prov_id = f"{art_clean}.{part_num}" if art_clean else part_num
        item_title = f"Пункт {part_num}" if is_nk else f"Часть {part_num}"
        return (3, item_title, prov_id)

    # Уровень 3: Пункт словом ("Пункт 3.2. ...")
    m_punkt = re.match(r"^Пункт\s+(\d+(?:\.\d+)*)\.?(.*)$", clean, re.IGNORECASE)
    if m_punkt:
        p_num = m_punkt.group(1).rstrip(".")
        return (3, f"Пункт {p_num}", p_num)

    return None


def extract_document_structure(content: str) -> List[Dict[str, Any]]:
    """Разбивает документ на блоки с непрерывным ведением стека иерархии."""
    is_nk = "налоговый кодекс" in content[:2000].lower() or "нк рф" in content[:2000].lower()
    lines = content.splitlines(keepends=True)
    blocks: List[Dict[str, Any]] = []

    path_stack: List[Tuple[int, str]] = [(0, "Введение / Общие положения")]
    current_provision_id: Optional[str] = None
    current_art_num: Optional[str] = None
    current_part_num: Optional[str] = None
    in_article = False
    in_notes = False
    current_lines: List[str] = []

    start_line = 1
    char_offset = 0
    block_char_start = 0

    for line_no, line in enumerate(lines, start=1):
        h_info = classify_heading(
            line,
            in_article=in_article,
            current_art_num=current_art_num,
            current_part_num=current_part_num,
            is_nk=is_nk,
            in_notes=in_notes,
        )
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
                popped_level, popped_title = path_stack.pop()
                if popped_level == 2:
                    in_article = False
                    current_art_num = None
                    current_part_num = None
                    in_notes = False
                elif popped_level == 3 and popped_title == "Примечания":
                    in_notes = False
                elif popped_level == 3:
                    current_part_num = None

            path_stack.append((level, h_title))
            current_provision_id = prov_id

            if level == 2:
                if "Статья" in h_title or is_nk:
                    in_article = True
                    in_notes = False
                    current_art_num = prov_id
                    current_part_num = None
                else:
                    in_article = False
                    in_notes = False
                    current_art_num = None
                    current_part_num = None
            elif level == 3:
                if h_title == "Примечания":
                    in_notes = True
                else:
                    current_part_num = prov_id

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
    cleaned_content, extracted_title, export_date = clean_garant_headers(raw_content)
    doc_type = detect_doc_type(file_path.name, cleaned_content)
    dates_ver, amends_id = parse_dates_and_version(
        cleaned_content, file_path.name, doc_type=doc_type, export_date=export_date
    )

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