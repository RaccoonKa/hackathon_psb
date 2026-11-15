#!/usr/bin/env python
"""Самостоятельный диагностический скрипт чанкинга.

Назначение (только диагностика, без индексации):
    чтение документов -> извлечение метаданных -> структурный парсинг ->
    разделение/склейка чанков -> подсчёт токенов count_tokens() ->
    запись JSONL -> статистика.

Скрипт НЕ создаёт эмбеддинги, НЕ обращается к ChromaDB/BM25, НЕ трогает
``storage/`` и основной ``data/processed/chunks.jsonl``. Это не pytest и
не тест.

Запуск из корня проекта:
    python -m rag_core.indexing.run_chunking_diagnostics --input data/raw
    python rag_core/indexing/run_chunking_diagnostics.py --input data/raw
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Bootstrap для запуска как обычного скрипта (python path/to/file.py) ---
if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]

from rag_core.indexing.chunker import chunk_document_structural  # noqa: E402
from rag_core.indexing.parser import load_documents_with_metadata  # noqa: E402
from rag_core.indexing.tokenizer_utils import count_tokens  # noqa: E402

DEFAULT_INPUT = "/data/raw"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "test_chunking"
PROJECT_RAW_FALLBACK = _PROJECT_ROOT / "data" / "raw"

# Целевые пороги (из алгоритма чанкера)
SMALL_LIMIT = 300
TARGET_MAX = 600
HARD_MAX = 750

SAFE_NAME_MAX_LEN = 100


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------


def sanitize_document_dir_name(document_id: str) -> str:
    """Делает безопасное для ФС имя директории из document_id."""
    raw = str(document_id or "")
    chars: List[str] = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    name = "".join(chars).strip()
    name = name[:SAFE_NAME_MAX_LEN]
    if name in {"", ".", ".."}:
        name = "document"
    return name


def resolve_input_dir(raw_arg: str) -> Path:
    """Возвращает существующую входную директорию (с fallback на проектную data/raw)."""
    requested = Path(raw_arg)
    if requested.exists() and requested.is_dir():
        return requested

    if raw_arg == DEFAULT_INPUT and PROJECT_RAW_FALLBACK.exists():
        print(
            f"ВНИМАНИЕ: директория по умолчанию '{DEFAULT_INPUT}' не найдена. "
            f"Используется проектная '{PROJECT_RAW_FALLBACK}'.",
            flush=True,
        )
        return PROJECT_RAW_FALLBACK

    raise SystemExit(
        f"ОШИБКА: входная директория не найдена или не является каталогом: {raw_arg}"
    )


def prepare_output_dir(output_dir: Path) -> None:
    """Полностью очищает выходную директорию и создаёт её заново."""
    if output_dir.exists():
        if not output_dir.is_dir():
            output_dir.unlink()
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def compute_metrics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Вычисляет метрики размера чанков по реальным token_count."""
    token_values = [int(ch["token_count"]) for ch in chunks]
    total = len(token_values)

    unresolved = sum(
        1
        for ch in chunks
        if int(ch["token_count"]) > HARD_MAX
        and ((ch.get("structure") or {}).get("provision_id") is not None)
    )

    if total == 0:
        return {
            "chunk_count": 0,
            "mean_tokens": 0,
            "median_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "chunks_lt_300": 0,
            "chunks_300_600": 0,
            "chunks_601_750": 0,
            "chunks_gt_750": 0,
            "unresolved_atomic_gt_750": 0,
            "mean_rounding": "1 decimal",
        }

    return {
        "chunk_count": total,
        "mean_tokens": round(statistics.fmean(token_values), 1),
        "median_tokens": statistics.median(token_values),
        "min_tokens": min(token_values),
        "max_tokens": max(token_values),
        "chunks_lt_300": sum(1 for v in token_values if v < SMALL_LIMIT),
        "chunks_300_600": sum(1 for v in token_values if 300 <= v <= 600),
        "chunks_601_750": sum(1 for v in token_values if 601 <= v <= 750),
        "chunks_gt_750": sum(1 for v in token_values if v > 750),
        "unresolved_atomic_gt_750": unresolved,
        "mean_rounding": "1 decimal",
    }


def _fmt_num(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def render_overall_block(metrics: Dict[str, Any], docs_count: int) -> str:
    lines = [
        "=" * 50,
        "СТАТИСТИКА ЧАНКИНГА",
        "=" * 50,
        "",
        f"Документов обработано: {docs_count}",
        f"Всего чанков: {metrics['chunk_count']}",
        "",
        f"Средний размер: {_fmt_num(metrics['mean_tokens'])} токенов",
        f"Медианный размер: {_fmt_num(metrics['median_tokens'])} токенов",
        f"Минимальный размер: {_fmt_num(metrics['min_tokens'])} токенов",
        f"Максимальный размер: {_fmt_num(metrics['max_tokens'])} токенов",
        "",
        "Распределение:",
        "",
        f"< 300 токенов:       {metrics['chunks_lt_300']}",
        f"300–600 токенов:     {metrics['chunks_300_600']}",
        f"601–750 токенов:     {metrics['chunks_601_750']}",
        f"> 750 токенов:       {metrics['chunks_gt_750']}",
        "",
        f"Неразделённых атомарных положений > 750: {metrics['unresolved_atomic_gt_750']}",
        "",
        "=" * 50,
    ]
    return "\n".join(lines)


def render_document_block(doc_stat: Dict[str, Any]) -> str:
    metrics = doc_stat["metrics"]
    title = doc_stat.get("document_title") or doc_stat.get("document_id") or "?"
    lines = [
        "-" * 50,
        f"Документ: {title}",
        f"document_id: {doc_stat.get('document_id')}",
        f"version_id: {doc_stat.get('version_id')}",
        f"source_filename: {doc_stat.get('source_filename')}",
        f"chunker_version: {doc_stat.get('chunker_version')}",
        "",
        f"Всего чанков: {metrics['chunk_count']}",
        f"Средний размер: {_fmt_num(metrics['mean_tokens'])} токенов",
        f"Медианный размер: {_fmt_num(metrics['median_tokens'])} токенов",
        f"Минимальный размер: {_fmt_num(metrics['min_tokens'])} токенов",
        f"Максимальный размер: {_fmt_num(metrics['max_tokens'])} токенов",
        "",
        "Распределение:",
        f"< 300 токенов:       {metrics['chunks_lt_300']}",
        f"300–600 токенов:     {metrics['chunks_300_600']}",
        f"601–750 токенов:     {metrics['chunks_601_750']}",
        f"> 750 токенов:       {metrics['chunks_gt_750']}",
        "",
        f"Неразделённых атомарных положений > 750: {metrics['unresolved_atomic_gt_750']}",
        "-" * 50,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------


def _doc_sort_key(doc: Dict[str, Any]) -> Tuple[str, str]:
    meta = doc.get("metadata") or {}
    return (
        str(meta.get("source_filename") or ""),
        str(meta.get("document_id") or ""),
    )


def process_documents(
    docs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Чанкует документы. Возвращает (all_chunks, document_stats, errors)."""
    all_chunks: List[Dict[str, Any]] = []
    document_stats: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    used_dirs: Dict[str, int] = {}

    for doc in docs:
        meta = doc.get("metadata") or {}
        document_id = str(meta.get("document_id") or "")
        source_filename = meta.get("source_filename")

        base_dir_name = sanitize_document_dir_name(document_id)
        count = used_dirs.get(base_dir_name, 0) + 1
        used_dirs[base_dir_name] = count
        dir_name = base_dir_name if count == 1 else f"{base_dir_name}__{count}"

        try:
            chunks = chunk_document_structural(doc)
        except Exception as exc:  # noqa: BLE001 - один документ не должен ронять отчёт
            message = f"{type(exc).__name__}: {exc}"
            print(
                f"ОШИБКА обработки документа '{document_id}' ({source_filename}): {message}",
                file=sys.stderr,
                flush=True,
            )
            errors.append(
                {
                    "document_id": document_id,
                    "source_filename": source_filename,
                    "error": message,
                }
            )
            continue

        chunks.sort(key=lambda ch: int(ch.get("chunk_index", 0)))

        doc_chunks: List[Dict[str, Any]] = []
        for chunk in chunks:
            token_count = count_tokens(chunk["embedding_text"])
            enriched = {"token_count": token_count, **chunk}
            doc_chunks.append(enriched)
            all_chunks.append(enriched)

        document_stats.append(
            {
                "document_id": document_id,
                "version_id": meta.get("version_id"),
                "document_title": meta.get("document_title"),
                "source_filename": source_filename,
                "chunker_version": meta.get("chunker_version"),
                "by_document_dir": dir_name,
                "metrics": compute_metrics(doc_chunks),
                "_chunks": doc_chunks,
            }
        )

    return all_chunks, document_stats, errors


def write_jsonl(path: Path, chunks: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def run_self_check(
    chunks_file: Path,
    all_chunks: List[Dict[str, Any]],
    sample_lines: int = 10,
) -> Tuple[bool, List[str]]:
    """Перечитывает JSONL и проверяет валидность, порядок ключей и сходимость токенов."""
    messages: List[str] = []
    ok = True

    with open(chunks_file, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    if len(lines) != len(all_chunks):
        ok = False
        messages.append(
            f"Число строк ({len(lines)}) не совпадает с числом чанков ({len(all_chunks)})."
        )

    # 1) первые строки — валидный JSON с token_count первым ключом
    for idx, line in enumerate(lines[:sample_lines]):
        try:
            pairs = json.loads(line, object_pairs_hook=list)
        except json.JSONDecodeError as exc:
            ok = False
            messages.append(f"Строка {idx + 1}: некорректный JSON: {exc}")
            continue
        if not pairs or pairs[0][0] != "token_count":
            ok = False
            first = pairs[0][0] if pairs else None
            messages.append(
                f"Строка {idx + 1}: первым ключом должен быть 'token_count', найден {first!r}."
            )

    # 2) все строки валидны и токены сходятся
    mismatches = 0
    for idx, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            ok = False
            messages.append(f"Строка {idx + 1}: некорректный JSON: {exc}")
            continue
        expected = count_tokens(obj.get("embedding_text", ""))
        if expected != obj.get("token_count"):
            mismatches += 1
            if mismatches <= 5:
                messages.append(
                    f"Строка {idx + 1} ({obj.get('chunk_id')}): token_count="
                    f"{obj.get('token_count')}, пересчёт={expected}."
                )
    if mismatches:
        ok = False
        messages.append(f"Всего несовпадений token_count: {mismatches}.")

    return ok, messages


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Диагностика структурно-адаптивного чанкинга (без эмбеддингов/индексов).",
    )
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help=f"Директория с документами (по умолчанию: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(DEFAULT_OUTPUT),
        help=f"Директория результатов (по умолчанию: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    input_dir = resolve_input_dir(args.input)
    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    if output_dir.resolve() == input_dir.resolve():
        raise SystemExit("ОШИБКА: выходная директория не может совпадать с входной.")

    prepare_output_dir(output_dir)

    print(f"Входная директория:  {input_dir}")
    print(f"Выходная директория: {output_dir}")

    docs = load_documents_with_metadata(input_dir)
    docs.sort(key=_doc_sort_key)
    print(f"Загружено документов: {len(docs)}")

    all_chunks, document_stats, errors = process_documents(docs)

    # --- Запись результатов ---
    chunks_file = output_dir / "chunks.jsonl"
    write_jsonl(chunks_file, all_chunks)

    by_document_root = output_dir / "by_document"
    by_document_root.mkdir(parents=True, exist_ok=True)
    for doc_stat in document_stats:
        doc_dir = by_document_root / doc_stat["by_document_dir"]
        write_jsonl(doc_dir / "chunks.jsonl", doc_stat["_chunks"])

    overall_metrics = compute_metrics(all_chunks)
    generated_at = datetime.now(timezone.utc).isoformat()

    statistics_payload: Dict[str, Any] = {
        "generated_at": generated_at,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "overall": {"documents_processed": len(document_stats), **overall_metrics},
        "documents": [
            {
                "document_id": ds["document_id"],
                "version_id": ds["version_id"],
                "document_title": ds["document_title"],
                "source_filename": ds["source_filename"],
                "chunker_version": ds["chunker_version"],
                "by_document_dir": ds["by_document_dir"],
                "chunk_count": ds["metrics"]["chunk_count"],
                "metrics": ds["metrics"],
            }
            for ds in document_stats
        ],
        "errors": errors,
    }
    stats_json_path = output_dir / "statistics.json"
    with open(stats_json_path, "w", encoding="utf-8") as fh:
        json.dump(statistics_payload, fh, ensure_ascii=False, indent=2)

    # --- Текстовый отчёт ---
    report_parts: List[str] = [render_overall_block(overall_metrics, len(document_stats)), ""]
    for doc_stat in document_stats:
        report_parts.append(render_document_block(doc_stat))
        report_parts.append("")
    if errors:
        report_parts.append("=" * 50)
        report_parts.append("ОШИБКИ ОБРАБОТКИ")
        report_parts.append("=" * 50)
        for err in errors:
            report_parts.append(
                f"- {err['document_id']} ({err['source_filename']}): {err['error']}"
            )
        report_parts.append("=" * 50)
    stats_txt_path = output_dir / "statistics.txt"
    stats_txt_path.write_text("\n".join(report_parts) + "\n", encoding="utf-8")

    # --- Самопроверка ---
    check_ok, check_messages = run_self_check(chunks_file, all_chunks)

    # --- Сводка в stdout ---
    print()
    print("=" * 50)
    print("КРАТКАЯ СВОДКА (overall)")
    print("=" * 50)
    print(f"Документов обработано: {len(document_stats)}")
    print(f"Всего чанков: {overall_metrics['chunk_count']}")
    print(f"Средний размер: {_fmt_num(overall_metrics['mean_tokens'])} токенов")
    print(f"Медианный размер: {_fmt_num(overall_metrics['median_tokens'])} токенов")
    print(
        f"Min/Max: {_fmt_num(overall_metrics['min_tokens'])} / "
        f"{_fmt_num(overall_metrics['max_tokens'])} токенов"
    )
    print(
        f"Распределение: <300={overall_metrics['chunks_lt_300']}, "
        f"300-600={overall_metrics['chunks_300_600']}, "
        f"601-750={overall_metrics['chunks_601_750']}, "
        f">750={overall_metrics['chunks_gt_750']}"
    )
    print(
        "Неразделённых атомарных положений > 750: "
        f"{overall_metrics['unresolved_atomic_gt_750']}"
    )
    print()
    print("Созданные файлы:")
    print(f"  - {chunks_file} ({chunks_file.stat().st_size} байт)")
    print(f"  - {stats_json_path} ({stats_json_path.stat().st_size} байт)")
    print(f"  - {stats_txt_path} ({stats_txt_path.stat().st_size} байт)")
    print(f"  - {by_document_root}/ (директории по документам)")
    print()

    oversized = [ch for ch in all_chunks if int(ch["token_count"]) > HARD_MAX]
    if oversized:
        print(f"Чанки > {HARD_MAX} токенов ({len(oversized)}):")
        for ch in oversized:
            provision_id = (ch.get("structure") or {}).get("provision_id")
            print(
                f"  - document_id={ch.get('document_id')}, "
                f"provision_id={provision_id}, "
                f"chunk_id={ch.get('chunk_id')}, "
                f"chunk_index={ch.get('chunk_index')}, "
                f"token_count={ch['token_count']}"
            )
    else:
        print(f"Чанков > {HARD_MAX} токенов нет.")
    print()

    print("САМОПРОВЕРКА JSONL:")
    if check_ok:
        print(f"  OK: все строки валидны, token_count первый, значения сходятся ({len(all_chunks)} чанков).")
    else:
        print("  ОШИБКА:")
        for msg in check_messages:
            print(f"    - {msg}")

    if not check_ok:
        print("Проверка не пройдена: результаты некорректны.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
