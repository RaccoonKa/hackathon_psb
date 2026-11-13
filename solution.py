"""Платформенная точка входа автоматического сабмита.

Сценарий запуска:
1. читает `DATA_DIR` (по умолчанию `data`);
2. находит `questions.csv` и архив базы знаний;
3. безопасно распаковывает архив во временный рабочий каталог;
4. один раз строит индексы общим индексатором `build_index.py`;
5. для каждого вопроса вызывает общий `run_agent(...)`;
6. атомарно записывает `submission.csv` в текущий рабочий каталог.

RAG-логика здесь не дублируется: используется `src.generation.orchestrator`.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from src import config
import build_index
from src.generation.orchestrator import run_agent
from src.generation.answer.context_builder import replace_source_tags 
from src.indexing.parser import safe_extract_zip

logger = logging.getLogger(__name__)

Question = Tuple[str, str]
AnswerRow = Tuple[str, str]


class SubmissionError(RuntimeError):
    """Ошибка подготовки или формирования сабмита."""


def _resolve_data_dir(data_dir: Optional[Path | str] = None) -> Path:
    raw = config.DATA_DIR if data_dir is None else data_dir
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def find_kb_archive(data_dir: Path) -> Optional[Path]:
    """Ищет архив базы знаний с приоритетом имён из технического регламента."""
    for name in config.KB_ARCHIVE_NAMES:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    return None


def read_questions(data_dir: Path) -> List[Question]:
    """Читает questions.csv (UTF-8/BOM), сохраняя ID как строки и порядок."""
    path = Path(data_dir) / "questions.csv"
    if not path.is_file():
        raise SubmissionError(f"Не найден файл вопросов: {path}")

    questions: List[Question] = []
    seen_ids: set[str] = set()

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "id" not in fieldnames or "question" not in fieldnames:
            raise SubmissionError(
                "questions.csv должен содержать колонки id и question; "
                f"получено: {fieldnames}"
            )

        for row in reader:
            raw_id = row.get("id")
            if raw_id is None:
                raise SubmissionError(
                    "В questions.csv найдена строка без значения id"
                )

            question_id = str(raw_id).strip()
            if not question_id:
                raise SubmissionError(
                    "В questions.csv найден пустой id"
                )
            if question_id in seen_ids:
                raise SubmissionError(
                    f"Повторяющийся id в questions.csv: {question_id}"
                )

            seen_ids.add(question_id)
            questions.append((question_id, row.get("question") or ""))

    return questions


def _ensure_within_work_dir(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(config.WORK_DIR.resolve())
    except ValueError as exc:
        raise SubmissionError(
            f"Рабочий путь выходит за пределы SUBMIT_WORK_DIR: {resolved}"
        ) from exc
    return resolved


def _reset_dir(path: Path) -> None:
    resolved = _ensure_within_work_dir(path)
    if resolved == config.WORK_DIR.resolve():
        raise SubmissionError(
            "Нельзя очищать корневой рабочий каталог SUBMIT"
        )
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def prepare_inputs(data_dir: Path) -> Path:
    """Проверяет входные файлы и безопасно распаковывает базу знаний."""
    if not config.USE_SUBMIT:
        raise SubmissionError(
            "solution.py предназначен для SUBMIT-режима. "
            "Установите USE_SUBMIT=True в settings_submit.py."
        )

    data_dir = Path(data_dir).resolve()
    questions_path = data_dir / "questions.csv"
    if not questions_path.is_file():
        raise SubmissionError(f"Не найден файл вопросов: {questions_path}")

    archive = find_kb_archive(data_dir)
    if archive is None:
        names = ", ".join(config.KB_ARCHIVE_NAMES)
        raise SubmissionError(
            f"В {data_dir} не найден архив базы знаний ({names})"
        )

    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.CHROMA_DIR,
        config.BM25_DIR,
        config.MANIFESTS_DIR,
    ):
        _reset_dir(directory)

    safe_extract_zip(archive, config.RAW_DATA_DIR)
    return archive


def limit_answer(
    text: str, max_chars: Optional[int] = None
) -> str:
    """Ограничивает ответ, обрезая его по границе предложения/слова.

    Политика явная: при превышении лимита ответ укорачивается до последней
    границы предложения не ранее половины лимита; если такой границы нет —
    до последнего пробела; в крайнем случае — строго по лимиту.
    """
    if not isinstance(text, str):
        raise SubmissionError("Ответ агента не является строкой")

    limit = config.MAX_ANSWER_CHARS if max_chars is None else max_chars
    if limit <= 0:
        raise SubmissionError("Некорректный лимит длины ответа")
    if len(text) <= limit:
        return text

    logger.warning(
        "Ответ длиной %d символов превышает лимит %d; применяется "
        "сокращение по границе предложения/слова",
        len(text),
        limit,
    )

    cut = text[:limit]
    for separator in ("\n\n", "\n", ". ", "! ", "? "):
        position = cut.rfind(separator)
        if position >= limit // 2:
            shortened = cut[: position + len(separator)].rstrip()
            if shortened:
                return shortened

    position = cut.rfind(" ")
    if position >= limit // 2:
        cut = cut[:position]

    shortened = cut.rstrip()
    return shortened or text[:limit]


def _answer_one(item: Question) -> AnswerRow:
    question_id, question = item
    result = run_agent(question)
    answer = getattr(result, "final_answer", None)
    if not isinstance(answer, str) or not answer.strip():
        raise SubmissionError(
            f"Агент вернул пустой ответ для id={question_id}"
        )

    sources = getattr(result, "sources", None) or {}
    answer = replace_source_tags(answer, sources)

    return question_id, limit_answer(answer)


def _answer_one_safe(item: Question):
    try:
        return _answer_one(item), None
    except Exception as exc:  # noqa: BLE001 - ошибки собираются по id
        return None, (item[0], exc)


def process_questions(
    questions: Sequence[Question], max_workers: int
) -> List[AnswerRow]:
    """Обрабатывает вопросы, сохраняя исходный порядок строк CSV."""
    if not questions:
        return []

    workers = max(1, min(max_workers, len(questions)))
    if workers == 1:
        results = [_answer_one_safe(item) for item in questions]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # executor.map сохраняет порядок результатов.
            results = list(executor.map(_answer_one_safe, questions))

    errors = [error for _, error in results if error is not None]
    if errors:
        preview = "; ".join(
            f"id={question_id}: {type(exc).__name__}: {exc}"
            for question_id, exc in errors[:5]
        )
        suffix = (
            ""
            if len(errors) <= 5
            else f"; всего ошибок: {len(errors)}"
        )
        raise SubmissionError(
            f"Обработка вопросов завершилась ошибкой: {preview}{suffix}"
        ) from errors[0][1]

    return [row for row, _ in results if row is not None]


def write_submission(
    rows: Sequence[AnswerRow], out_path: Path | str
) -> Path:
    """Атомарно пишет submission.csv: UTF-8, ровно колонки id,answer."""
    path = Path(out_path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    prepared: List[AnswerRow] = []
    seen_ids: set[str] = set()
    for question_id, answer in rows:
        question_id = str(question_id)
        if not question_id:
            raise SubmissionError("Пустой id в результате сабмита")
        if question_id in seen_ids:
            raise SubmissionError(
                f"Повторяющийся id в результате сабмита: {question_id}"
            )
        seen_ids.add(question_id)

        if not isinstance(answer, str) or not answer.strip():
            raise SubmissionError(
                f"Пустой ответ в результате сабмита для id={question_id}"
            )
        prepared.append((question_id, limit_answer(answer)))

    temp_path = path.with_name(path.name + ".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["id", "answer"])
            for question_id, answer in prepared:
                writer.writerow([question_id, answer])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return path


def run(
    data_dir: Optional[Path | str] = None,
    questions: Optional[Sequence[Question]] = None,
    out_path: Optional[Path | str] = None,
) -> List[AnswerRow]:
    """Полный прогон: входные данные, индексы, вопросы, submission.csv."""
    resolved_data_dir = _resolve_data_dir(data_dir)

    if questions is None:
        questions = read_questions(resolved_data_dir)
    if not questions:
        raise SubmissionError(
            "questions.csv не содержит ни одного вопроса"
        )

    prepare_inputs(resolved_data_dir)
    summary = build_index.build_indexes()
    logger.info(
        "Индекс построен один раз: документов=%s, чанков=%s",
        summary.get("documents"),
        summary.get("chunks"),
    )

    rows = process_questions(questions, config.MAX_WORKERS)

    if out_path is None:
        out_path = Path.cwd() / config.SUBMISSION_FILENAME
    write_submission(rows, out_path)
    return rows


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    started = time.perf_counter()

    if not config.USE_SUBMIT:
        print(
            "[solution] Отказ запуска: solution.py работает в SUBMIT-режиме. "
            "Соберите артефакт командой `python build_submit.py`.",
            file=sys.stderr,
        )
        return 2

    out_path = Path.cwd() / config.SUBMISSION_FILENAME
    temp_path = out_path.with_name(out_path.name + ".tmp")

    try:
        if out_path.exists():
            out_path.unlink()
        if temp_path.exists():
            temp_path.unlink()

        config.validate_submit_config()
        rows = run(out_path=out_path)
    except config.ConfigError as exc:
        print(f"[solution] Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1
    except SubmissionError as exc:
        print(f"[solution] Ошибка сабмита: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - гарантируем понятный выход
        print(
            f"[solution] Непредвиденная ошибка: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        if temp_path.exists():
            temp_path.unlink()

    elapsed = time.perf_counter() - started
    print(
        f"[solution] submission.csv сформирован: строк {len(rows)}, "
        f"время {elapsed:.1f} c.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
