import re
from pathlib import Path
from typing import Dict, Any, List

def detect_doc_type(filename: str, content: str) -> str:
    name_lower = filename.lower()
    sample = content[:1500].lower()
    if any(k in name_lower or k in sample for k in ["фз", "закон", "кодекс", "постановление"]):
        return "law"
    return "internal"

def parse_dates_and_version(content: str, filename: str) -> Dict[str, Any]:
    head = content[:2000]
    
    # Поиск даты редакции или действия
    date_match = re.search(r"(?:редакция от|действует с|от)\s+(\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})", head, re.IGNORECASE)
    effective_from = date_match.group(1) if date_match else None
    
    # Определение версии из имени файла или текста
    ver_match = re.search(r"v(\d+)", filename, re.IGNORECASE)
    version_id = f"v{ver_match.group(1)}" if ver_match else "v1"

    return {
        "effective_from": effective_from,
        "effective_to": None,
        "effective_dates_known": effective_from is not None,
        "version_id": version_id
    }

def load_documents_with_metadata(raw_dir: Path) -> List[Dict[str, Any]]:
    documents = []
    for path in sorted(raw_dir.glob("**/*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        title_line = content.strip().splitlines()[0][:200] if content else path.stem
        dates_ver = parse_dates_and_version(content, path.name)
        
        metadata = {
            "document_id": path.stem,
            "version_id": dates_ver["version_id"],
            "document_title": title_line,
            "document_type": detect_doc_type(path.name, content),
            "scope": "global",
            "effective_from": dates_ver["effective_from"],
            "effective_to": dates_ver["effective_to"],
            "effective_dates_known": dates_ver["effective_dates_known"],
            "source_filename": path.name
        }
        documents.append({"content": content, "metadata": metadata})
    return documents