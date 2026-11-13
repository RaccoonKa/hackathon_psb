"""Платформенная точка входа автоматического сабмита.

Сценарий запуска:
1. читает `DATA_DIR` (по умолчанию `data`);
2. находит `questions.csv` и архив базы знаний;
3. безопасно распаковывает архив во временный рабочий каталог;
4. один раз строит индексы общим индексатором `build_index.py`;
5. для каждого вопроса (или диалога) вызывает общий `run_agent(...)`;
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from rag_core import config
import build_index
from rag_core.generation.orchestrator import run_agent
from rag_core.generation.answer.context_builder import replace_source_tags
from rag_core.indexing.parser import safe_extract_zip

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuestionItem:
    """Элемент входного CSV с опциональной привязкой к многоходовому диалогу."""

    question_id: str
    question: str
    dialog_id: Optional[str] = None
    turn: Optional[int] = None


Question = Union[Tuple[str, str], QuestionItem]
AnswerRow = Tuple[str, str]

_DIALOG_ID_COLUMNS: Tuple[str, ...] = (
    "dialog_id",
    "session_id",
    "conversation_id",
    "chat_id",
)
_TURN_COLUMNS: Tuple[str, ...] = (
    "turn",
    "turn_id",
    "step",
    "message_index",
)
_TURN_MARKER_RE = re.compile(
    r"(?im)^\s*(пользователь|user|ассистент|assistant|бот|bot)\s*:\s*"
)


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


def _detect_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {name.strip().lower(): name for name in fieldnames if name}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def parse_dialog_transcript(
    raw_question: str,
) -> Tuple[str, Optional[List[Dict[str, str]]]]:
    """Разбирает стенограмму диалога вида 'Пользователь: ...\\nАссистент: ...\\nПользователь: ...'.

    Если реплик меньше двух или последняя реплика не от пользователя,
    возвращает исходный текст без истории.
    """
    if not raw_question or ":" not in raw_question:
        return raw_question, None

    matches = list(_TURN_MARKER_RE.finditer(raw_question))
    if len(matches) < 2:
        return raw_question, None

    turns: List[Dict[str, str]] = []
    for idx, match in enumerate(matches):
        role_token = match.group(1).strip().lower()
        role = "user" if role_token in {"пользователь", "user"} else "assistant"
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_question)
        content = raw_question[start:end].strip()
        if content:
            turns.append({"role": role, "content": content})

    if len(turns) < 2 or turns[-1]["role"] != "user":
        return raw_question, None

    return turns[-1]["content"], turns[:-1]


def read_questions(data_dir: Path | str) -> List[Question]:
    """Читает questions.csv (UTF-8/BOM), сохраняя ID как строки и порядок."""
    path = Path(data_dir)
    if path.is_dir():
        path = path / "questions.csv"
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

        dialog_col = _detect_column(fieldnames, _DIALOG_ID_COLUMNS)
        turn_col = _detect_column(fieldnames, _TURN_COLUMNS)

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
            question_text = row.get("question") or ""

            dialog_id: Optional[str] = None
            if dialog_col:
                raw_dialog = row.get(dialog_col)
                if raw_dialog is not None and str(raw_dialog).strip():
                    dialog_id = str(raw_dialog).strip()

            turn_num: Optional[int] = None
            if turn_col:
                raw_turn = row.get(turn_col)
                if raw_turn is not None and str(raw_turn).strip():
                    try:
                        turn_num = int(str(raw_turn).strip())
                    except ValueError:
                        turn_num = None

            if dialog_id is not None:
                questions.append(
                    QuestionItem(
                        question_id=question_id,
                        question=question_text,
                        dialog_id=dialog_id,
                        turn=turn_num,
                    )
                )
            else:
                questions.append((question_id, question_text))

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


def _normalize_question(item: Question) -> QuestionItem:
    if isinstance(item, QuestionItem):
        return item
    question_id, question = item
    return QuestionItem(question_id=question_id, question=question)


def _answer_one(
    item: Question,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> AnswerRow:
    q_item = _normalize_question(item)
    effective_query = q_item.question
    effective_history = list(chat_history) if chat_history else None

    if effective_history is None:
        parsed_query, inline_history = parse_dialog_transcript(q_item.question)
        if inline_history:
            effective_query = parsed_query
            effective_history = inline_history

    if effective_history:
        result = run_agent(effective_query, chat_history=effective_history)
    else:
        result = run_agent(effective_query)

    answer = getattr(result, "final_answer", None)
    if not isinstance(answer, str) or not answer.strip():
        raise SubmissionError(
            f"Агент вернул пустой ответ для id={q_item.question_id}"
        )

    sources = getattr(result, "sources", None) or {}
    answer = replace_source_tags(answer, sources)

    return q_item.question_id, limit_answer(answer)


def _answer_one_safe(item: Question):
    q_item = _normalize_question(item)
    try:
        return _answer_one(q_item), None
    except Exception as exc:  # noqa: BLE001 - ошибки собираются по id
        return None, (q_item.question_id, exc)


def _build_execution_groups(
    items: Sequence[QuestionItem],
) -> List[List[Tuple[int, QuestionItem]]]:
    """Группирует вопросы по dialog_id; одиночные вопросы остаются отдельными задачами."""
    groups: List[List[Tuple[int, QuestionItem]]] = []
    dialog_buckets: Dict[str, List[Tuple[int, QuestionItem]]] = {}
    dialog_order: List[str] = []

    for idx, item in enumerate(items):
        if item.dialog_id:
            if item.dialog_id not in dialog_buckets:
                dialog_buckets[item.dialog_id] = []
                dialog_order.append(item.dialog_id)
            dialog_buckets[item.dialog_id].append((idx, item))
        else:
            groups.append([(idx, item)])

    for dialog_id in dialog_order:
        turns = dialog_buckets[dialog_id]
        turns.sort(
            key=lambda pair: (
                pair[1].turn if pair[1].turn is not None else pair[0],
                pair[0],
            )
        )
        groups.append(turns)

    groups.sort(key=lambda grp: grp[0][0])
    return groups


def _process_group_safe(
    group: Sequence[Tuple[int, QuestionItem]],
) -> Tuple[List[Tuple[int, AnswerRow]], Optional[Tuple[str, Exception]]]:
    """Последовательно выполняет реплики одного диалога с накоплением chat_history."""
    is_dialog = len(group) > 1 or bool(group[0][1].dialog_id)
    chat_history: List[Dict[str, str]] = []
    collected: List[Tuple[int, AnswerRow]] = []

    for orig_idx, item in group:
        try:
            row = _answer_one(
                item,
                chat_history=chat_history if (is_dialog and chat_history) else None,
            )
            collected.append((orig_idx, row))
            if is_dialog:
                chat_history.append({"role": "user", "content": item.question})
                chat_history.append({"role": "assistant", "content": row[1]})
        except Exception as exc:  # noqa: BLE001 - ошибки собираются по id
            return collected, (item.question_id, exc)

    return collected, None


def process_questions(
    questions: Sequence[Question], max_workers: int
) -> List[AnswerRow]:
    """Обрабатывает вопросы и диалоги, сохраняя исходный порядок строк CSV."""
    if not questions:
        return []

    normalized = [_normalize_question(item) for item in questions]
    groups = _build_execution_groups(normalized)

    workers = max(1, min(max_workers, len(groups)))
    if workers == 1:
        group_results = [_process_group_safe(grp) for grp in groups]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            group_results = list(executor.map(_process_group_safe, groups))

    errors = [error for _, error in group_results if error is not None]
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

    ordered_rows: List[Optional[AnswerRow]] = [None] * len(normalized)
    for collected, _ in group_results:
        for orig_idx, row in collected:
            ordered_rows[orig_idx] = row

    return [row for row in ordered_rows if row is not None]


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
    input_file: Optional[Path | str] = None,
    skip_index: Optional[bool] = None,
) -> List[AnswerRow]:
    """Полный прогон: входные данные, индексы, вопросы, submission.csv."""
    resolved_data_dir = _resolve_data_dir(data_dir)

    if questions is None:
        target = input_file or resolved_data_dir
        questions = read_questions(target)
    if not questions:
        raise SubmissionError(
            "questions.csv не содержит ни одного вопроса"
        )

    should_skip = skip_index if skip_index is not None else (
        not config.USE_SUBMIT and find_kb_archive(resolved_data_dir) is None
    )

    if not should_skip:
        prepare_inputs(resolved_data_dir)
        summary = build_index.build_indexes()
        logger.info(
            "Индекс построен один раз: документов=%s, чанков=%s",
            summary.get("documents"),
            summary.get("chunks"),
        )
    else:
        logger.info(
            "Пропуск перестроения индекса: используются существующие индексы"
        )

    rows = process_questions(questions, config.MAX_WORKERS)

    if out_path is None:
        out_path = Path.cwd() / config.SUBMISSION_FILENAME
    write_submission(rows, out_path)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Автоматический сабмит RAG-ассистента")
    parser.add_argument("--input", default=None, help="Путь к questions.csv или файлу с вопросами")
    parser.add_argument("--output", default=None, help="Путь для сохранения submission.csv")
    parser.add_argument("--data-dir", default=None, help="Каталог входных данных")
    parser.add_argument("--skip-index", action="store_true", default=None, help="Пропустить построение индекса")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    started = time.perf_counter()

    if not config.USE_SUBMIT and not args.input:
        print(
            "[solution] Отказ запуска: solution.py работает в SUBMIT-режиме. "
            "Соберите артефакт командой `python build_submit.py`.",
            file=sys.stderr,
        )
        return 2

    out_path = Path(args.output) if args.output else Path.cwd() / config.SUBMISSION_FILENAME
    temp_path = out_path.with_name(out_path.name + ".tmp")

    try:
        if out_path.exists():
            out_path.unlink()
        if temp_path.exists():
            temp_path.unlink()

        if config.USE_SUBMIT:
            config.validate_submit_config()

        rows = run(
            data_dir=args.data_dir,
            out_path=out_path,
            input_file=args.input,
            skip_index=args.skip_index,
        )
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