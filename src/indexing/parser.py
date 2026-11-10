import re
from pathlib import Path
from typing import Dict, Any, List

def detect_doc_type(filename: str, content: str) -> str:
    """Определяет тип нормативного акта: закон или внутренний документ."""
    name_lower = filename.lower()
    text_sample = content[:1000].lower()
    if any(k in name_lower or k in text_sample for k in ["фз", "федеральный закон", "кодекс", "постановление"]):
        return "law"
    return "internal"

def extract_metadata(file_path: Path, content: str) -> Dict[str, Any]:
    """Извлекает ключевые атрибуты: дату редакции, номер и тип акта."""
    filename = file_path.name
    doc_type = detect_doc_type(filename, content)
    
    # Поиск даты редакции / вступления в силу в первых 2000 символах
    date_pattern = r"(?:редакция от|действует с|утвержден(?:о)?|от)\s+(\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})"
    date_match = re.search(date_pattern, content[:2000], re.IGNORECASE)
    doc_date = date_match.group(1) if date_match else None

    # Поиск номера документа (приказ №..., закон №...-ФЗ)
    number_pattern = r"№\s*([\d\w\-]+)"
    num_match = re.search(number_pattern, content[:1500])
    doc_number = num_match.group(1) if num_match else None

    title_line = content.strip().split("\n")[0][:250]

    return {
        "source": filename,
        "doc_id": file_path.stem,
        "doc_title": title_line,
        "doc_type": doc_type,
        "effective_date": doc_date,
        "doc_number": doc_number,
    }

def load_raw_documents(raw_dir: Path) -> List[Dict[str, Any]]:
    """Загружает все TXT файлы из data/raw с метаданными."""
    docs = []
    for file_path in raw_dir.glob("**/*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        meta = extract_metadata(file_path, content)
        docs.append({"content": content, "metadata": meta})
    return docs