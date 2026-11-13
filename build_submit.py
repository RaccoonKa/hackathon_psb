#!/usr/bin/env python3
"""Сборка самостоятельного каталога `submit/` с плоской структурой.

Скрипт:
1. проверяет единственный переключатель `USE_SUBMIT` в settings_submit.py;
2. полностью пересоздаёт submit/ (только внутри корня проекта);
3. копирует runtime-модули и переписывает импорты `rag_core.*` через AST;
4. формирует submit/settings_submit.py со значением True;
5. формирует requirements.txt и README.md;
6. проверяет сборку и при --zip создаёт ZIP с файлами в корне архива.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
SUBMIT_DIR = ROOT / "submit"
SETTINGS_FILE = ROOT / "settings_submit.py"
VERIFY_SCRIPT = ROOT / "verify_submit.py"
ZIP_PATH = ROOT / "submit.zip"

MAX_ZIP_BYTES = 128 * 1024 * 1024

# source -> target в плоской сборке
MODULE_SOURCES: List[Tuple[str, str]] = [
    ("rag_core/config.py", "config.py"),
    # --- orchestration / shared ---
    ("rag_core/generation/orchestrator.py", "generation_orchestrator.py"),
    ("rag_core/generation/bailout.py", "generation_bailout.py"),
    ("rag_core/generation/errors.py", "generation_errors.py"),
    ("rag_core/generation/refusal.py", "generation_refusal.py"),
    ("rag_core/generation/stubs.py", "generation_stubs.py"),
    ("rag_core/generation/types.py", "generation_types.py"),
    # --- analysis ---
    (
        "rag_core/generation/analysis/entity_resolver.py",
        "generation_analysis_entity_resolver.py",
    ),
    (
        "rag_core/generation/analysis/prefilter.py",
        "generation_analysis_prefilter.py",
    ),
    (
        "rag_core/generation/analysis/query_analysis.py",
        "generation_analysis_query_analysis.py",
    ),
    # --- answer ---
    (
        "rag_core/generation/answer/context_builder.py",
        "generation_answer_context_builder.py",
    ),
    ("rag_core/generation/answer/generator.py", "generation_answer_generator.py"),
    (
        "rag_core/generation/answer/verification.py",
        "generation_answer_verification.py",
    ),
    # --- llm / planning / policy ---
    ("rag_core/generation/llm/client.py", "generation_llm_client.py"),
    (
        "rag_core/generation/planning/planner.py",
        "generation_planning_planner.py",
    ),
    ("rag_core/generation/policy/precheck.py", "generation_policy_precheck.py"),
    # --- prompts package (плоские модули) ---
    (
        "rag_core/generation/prompts/templates.py",
        "generation_prompts_templates.py",
    ),
    ("rag_core/generation/prompts/bailout.py", "generation_prompts_bailout.py"),
    ("rag_core/generation/prompts/evidence.py", "generation_prompts_evidence.py"),
    ("rag_core/generation/prompts/generator.py", "generation_prompts_generator.py"),
    ("rag_core/generation/prompts/history.py", "generation_prompts_history.py"),
    (
        "rag_core/generation/prompts/query_analysis.py",
        "generation_prompts_query_analysis.py",
    ),
    (
        "rag_core/generation/prompts/regenerate.py",
        "generation_prompts_regenerate.py",
    ),
    ("rag_core/generation/prompts/verifier.py", "generation_prompts_verifier.py"),
    # --- retrieval (generation) ---
    ("rag_core/generation/retrieval/bridge.py", "generation_retrieval_bridge.py"),
    (
        "rag_core/generation/retrieval/evidence.py",
        "generation_retrieval_evidence.py",
    ),
    # --- indexing / retrieval ---
    ("rag_core/indexing/chunker.py", "indexing_chunker.py"),
    ("rag_core/indexing/parser.py", "indexing_parser.py"),
    ("rag_core/indexing/tokenizer_utils.py", "indexing_tokenizer_utils.py"),
    ("rag_core/retrieval/hybrid.py", "retrieval_hybrid.py"),
    ("rag_core/retrieval/keyword_store.py", "retrieval_keyword_store.py"),
    ("rag_core/retrieval/reranker.py", "retrieval_reranker.py"),
    ("rag_core/retrieval/vector_store.py", "retrieval_vector_store.py"),
    ("build_index.py", "build_index.py"),
    ("solution.py", "solution.py"),
]

MODULE_MAP: dict[str, str] = {}
for _source, _target in MODULE_SOURCES:
    if not (_source.startswith("rag_core/") or _source.startswith("src/")):
        continue
    prefix = "rag_core/" if _source.startswith("rag_core/") else "src/"
    _rel = _source[len(prefix) : -len(".py")].replace("/", ".")
    MODULE_MAP[f"rag_core.{_rel}"] = _target[: -len(".py")]
    MODULE_MAP[f"src.{_rel}"] = _target[: -len(".py")]

INTERNAL_MODULES = {
    target[: -len(".py")] for _, target in MODULE_SOURCES
}

FORBIDDEN_NAMES = {
    ".env",
    ".git",
    ".gitignore",
    ".DS_Store",
    "__pycache__",
    "data",
    "storage",
    "models",
    "tests",
    "run_live.py",
    "test_parser.py",
    "main.py",
    "submit.zip",
    "questions.csv",
    "submission.csv",
    "knowledge-base.zip",
    "kb.zip",
    "knowledge_base.zip",
}

SUBMIT_REQUIREMENTS = """chromadb==1.5.9
httpx==0.28.1
langchain-openai==1.7.4
langchain-text-splitters==1.1.3
openai==3.45.0
rank-bm25==0.2.2
requests==2.32.5
tiktoken==0.12.0
"""

SUBMIT_README = """# submit — платформенный артефакт

Каталог собран автоматически командой `python build_submit.py`.
Не редактируйте файлы вручную: повторная сборка полностью их перезапишет.

## Запуск

Из каталога `submit/`:

```bash
python solution.py
```

Платформа передаёт входные файлы через `DATA_DIR`:

- `questions.csv` с колонками `id,question`;
- архив базы знаний `knowledge-base.zip` (также поддерживаются
  `kb.zip` и `knowledge_base.zip`).

В текущем рабочем каталоге создаётся `submission.csv` с колонками
`id,answer`.

## Переменные окружения

| Переменная | Назначение |
|---|---|
| `DATA_DIR` | каталог с `questions.csv` и архивом базы знаний; по умолчанию `data` |
| `LLM_BASE_URL` / `OPENAI_BASE_URL` | единый OpenAI-совместимый прокси платформы |
| `LLM_API_KEY` / `OPENAI_API_KEY` | ключ прокси; в логи не выводится |
| `LLM_MODEL` / `OPENAI_MODEL` / `GEN_MODEL` | генеративная модель |
| `EMB_MODEL` | модель эмбеддингов, по умолчанию `BAAI/bge-m3` |
| `RERANKER_MODEL` | модель реранкера, по умолчанию `Qwen/Qwen3-Reranker-8B` |
| `TEMPERATURE` / `MAX_TOKENS` | параметры генерации |
| `MAX_WORKERS` | число параллельных вопросов, не более 8 |
| `TOP_K` / `RAG_TOP_K` | число финальных источников |
| `CHUNKER` | параметр чанкинга, по умолчанию `structural` |
| `SUBMIT_WORK_DIR` | записываемый рабочий каталог; по умолчанию временный |
| `EMBEDDING_DIMENSION` | ожидаемая размерность эмбеддингов, по умолчанию 1024 |

Все три модели (chat, embeddings, rerank) используют единый прокси и
единый ключ платформы. Локальные веса BGE-M3 в SUBMIT-режиме не
загружаются, hash-fallback эмбеддингов запрещён.

## Политика длины ответа

Если ответ превышает 10 000 символов, он сокращается по последней
границе предложения не ранее половины лимита; если границы нет — по
последнему пробелу. Пустые ответы считаются ошибкой и не попадают в
`submission.csv`.

## Зависимости

`requirements.txt` содержит зафиксированные версии только runtime-
зависимостей проекта. Предполагается, что эти пакеты входят в
разрешённый платформой набор (они присутствуют в базовом
`requirements.txt` проекта). Если allowlist платформы отличается,
зависимости нужно согласовать до отправки.
"""


class BuildError(RuntimeError):
    """Ошибка сборки с понятным сообщением."""


def validate_settings() -> bool:
    if not SETTINGS_FILE.is_file():
        raise BuildError(f"Не найден {SETTINGS_FILE.name}")

    source = SETTINGS_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SETTINGS_FILE))

    assignments = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "USE_SUBMIT":
                    assignments.append(node)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "USE_SUBMIT":
                assignments.append(node)

    if len(assignments) != 1:
        raise BuildError(
            "settings_submit.py должен содержать ровно одно присваивание "
            f"USE_SUBMIT; найдено: {len(assignments)}"
        )

    value_node = assignments[0].value
    if not isinstance(value_node, ast.Constant) or not isinstance(
        value_node.value, bool
    ):
        raise BuildError("USE_SUBMIT должен быть булевым литералом True/False")

    return value_node.value


class SrcImportRewriter(ast.NodeTransformer):
    """Переписывает импорты rag_core.* и src.* в плоские имена модулей."""

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level != 0:
            return self.generic_visit(node)

        module = node.module or ""
        if module in {"src", "rag_core"}:
            new_nodes = []
            for alias in node.names:
                full_name = f"{module}.{alias.name}"
                flat_name = MODULE_MAP.get(full_name) or MODULE_MAP.get(f"rag_core.{alias.name}")
                if flat_name is None:
                    raise BuildError(
                        f"Не найден плоский модуль для импорта {full_name}"
                    )
                new_nodes.append(
                    ast.Import(
                        names=[
                            ast.alias(
                                name=flat_name,
                                asname=alias.asname or alias.name,
                            )
                        ]
                    )
                )
            return new_nodes

        if module in MODULE_MAP:
            node.module = MODULE_MAP[module]
            return node

        if module.startswith(("src.", "rag_core.")):
            new_nodes = []
            for alias in node.names:
                full_name = f"{module}.{alias.name}"
                flat_name = MODULE_MAP.get(full_name)
                if flat_name is None:
                    raise BuildError(
                        f"Не найден плоский модуль для импорта {full_name}"
                    )
                new_nodes.append(
                    ast.Import(
                        names=[
                            ast.alias(
                                name=flat_name,
                                asname=alias.asname or alias.name,
                            )
                        ]
                    )
                )
            return new_nodes

        return node

    def visit_Import(self, node: ast.Import):
        new_aliases = []
        for alias in node.names:
            if alias.name in {"src", "rag_core"} or alias.name.startswith(("src.", "rag_core.")):
                flat_name = MODULE_MAP.get(alias.name)
                if flat_name is None:
                    raise BuildError(
                        f"Не найден плоский модуль для импорта {alias.name}"
                    )
                if alias.asname is None:
                    raise BuildError(
                        f"Импорт {alias.name} без alias нельзя безопасно "
                        "переписать в плоскую структуру"
                    )
                new_aliases.append(
                    ast.alias(name=flat_name, asname=alias.asname)
                )
            else:
                new_aliases.append(alias)
        node.names = new_aliases
        return node


def rewrite_source(source: str, source_name: str) -> str:
    tree = ast.parse(source, filename=source_name)
    tree = SrcImportRewriter().visit(tree)
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree)
    header = (
        "# Сгенерировано build_submit.py из "
        f"{source_name}. Не редактировать вручную.\n"
    )
    return header + code + "\n"


def ensure_required_sources() -> None:
    missing = [
        source for source, _ in MODULE_SOURCES if not (ROOT / source).is_file()
    ]
    if missing:
        raise BuildError(
            "Не найдены обязательные исходные файлы: " + ", ".join(missing)
        )
    if not VERIFY_SCRIPT.is_file():
        raise BuildError(f"Не найден {VERIFY_SCRIPT.name}")


def clean_submit_dir() -> None:
    if SUBMIT_DIR.exists():
        resolved = SUBMIT_DIR.resolve()
        expected_parent = ROOT.resolve()
        if resolved.name != "submit" or resolved.parent != expected_parent:
            raise BuildError(
                f"Отказ очистки: {resolved} не является {expected_parent}/submit"
            )
        shutil.rmtree(resolved)
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)


def write_flat_modules() -> None:
    for source_rel, target_rel in MODULE_SOURCES:
        source_path = ROOT / source_rel
        target_path = SUBMIT_DIR / target_rel
        source_text = source_path.read_text(encoding="utf-8")
        target_path.write_text(
            rewrite_source(source_text, source_rel), encoding="utf-8"
        )


def write_auxiliary_files() -> None:
    (SUBMIT_DIR / "settings_submit.py").write_text(
        "USE_SUBMIT = True\n", encoding="utf-8"
    )
    (SUBMIT_DIR / "requirements.txt").write_text(
        SUBMIT_REQUIREMENTS, encoding="utf-8"
    )
    (SUBMIT_DIR / "README.md").write_text(
        SUBMIT_README, encoding="utf-8"
    )


def iter_generated_python() -> Iterable[Path]:
    return sorted(SUBMIT_DIR.glob("*.py"))


def validate_generated() -> None:
    python_files = list(iter_generated_python())
    if not python_files:
        raise BuildError("В submit/ не создано ни одного Python-модуля")

    nested_python = [
        path
        for path in SUBMIT_DIR.rglob("*.py")
        if path.parent != SUBMIT_DIR
    ]
    if nested_python:
        raise BuildError(
            "Найдены Python-модули во вложенных каталогах submit/: "
            + ", ".join(str(path.relative_to(SUBMIT_DIR)) for path in nested_python)
        )

    for path in python_files:
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            raise BuildError(
                f"Синтаксическая ошибка в {path.name}: {exc}"
            ) from exc

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if (
                        alias.name in {"src", "rag_core"}
                        or alias.name.startswith(("src.", "rag_core."))
                    ):
                        raise BuildError(
                            f"В {path.name} остался импорт {alias.name}"
                        )
                    if (
                        alias.name in INTERNAL_MODULES
                        and not (SUBMIT_DIR / f"{alias.name}.py").is_file()
                    ):
                        raise BuildError(
                            f"В {path.name} не разрешается импорт {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if (
                    module in {"src", "rag_core"}
                    or module.startswith(("src.", "rag_core."))
                ):
                    raise BuildError(
                        f"В {path.name} остался импорт from {module}"
                    )
                if module in INTERNAL_MODULES and not (
                    SUBMIT_DIR / f"{module}.py"
                ).is_file():
                    raise BuildError(
                        f"В {path.name} не разрешается импорт {module}"
                    )
                for alias in node.names:
                    if (
                        module in INTERNAL_MODULES
                        and alias.name in INTERNAL_MODULES
                        and not (
                            SUBMIT_DIR / f"{alias.name}.py"
                        ).is_file()
                    ):
                        raise BuildError(
                            f"В {path.name} не разрешается импорт {alias.name}"
                        )

    for path in SUBMIT_DIR.rglob("*"):
        if path.name in FORBIDDEN_NAMES:
            raise BuildError(
                "В submit/ попал запрещённый файл/каталог: "
                f"{path.relative_to(SUBMIT_DIR)}"
            )
        if path.name.startswith(".env"):
            raise BuildError(
                "В submit/ попал .env-файл: "
                f"{path.relative_to(SUBMIT_DIR)}"
            )

    requirements_path = SUBMIT_DIR / "requirements.txt"
    if not requirements_path.is_file():
        raise BuildError("Не создан submit/requirements.txt")
    if requirements_path.read_text(encoding="utf-8") != SUBMIT_REQUIREMENTS:
        raise BuildError("submit/requirements.txt не совпадает с ожидаемым")

    if not (SUBMIT_DIR / "README.md").is_file():
        raise BuildError("Не создан submit/README.md")
    if not (SUBMIT_DIR / "settings_submit.py").is_file():
        raise BuildError("Не создан submit/settings_submit.py")


def make_zip() -> Path:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(
        ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(SUBMIT_DIR.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            arcname = path.relative_to(SUBMIT_DIR).as_posix()
            archive.write(path, arcname)

    size = ZIP_PATH.stat().st_size
    if size > MAX_ZIP_BYTES:
        raise BuildError(
            f"ZIP превышает лимит 128 МБ: {size} байт"
        )

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        names = archive.namelist()
    if any(name.startswith("submit/") for name in names):
        raise BuildError("ZIP содержит внешнюю директорию submit/")
    if "solution.py" not in names or "requirements.txt" not in names:
        raise BuildError("ZIP не содержит solution.py/requirements.txt в корне")

    return ZIP_PATH


def run_verify(zip_path: Optional[Path]) -> None:
    command = [
        sys.executable,
        str(VERIFY_SCRIPT),
        "--submit-dir",
        str(SUBMIT_DIR),
    ]
    if zip_path is not None:
        command.extend(["--zip-path", str(zip_path)])

    result = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise BuildError("Автоматические проверки submit/ не пройдены")


def build(zip_requested: bool = False) -> None:
    original_settings = SETTINGS_FILE.read_text(encoding="utf-8")
    flag_value = validate_settings()
    ensure_required_sources()

    clean_submit_dir()
    write_flat_modules()
    write_auxiliary_files()
    validate_generated()

    if SETTINGS_FILE.read_text(encoding="utf-8") != original_settings:
        raise BuildError("Исходный settings_submit.py был изменён сборкой")

    zip_path = make_zip() if zip_requested else None
    run_verify(zip_path)

    print(
        "Сборка submit/ завершена. "
        f"Модулей: {len(list(iter_generated_python()))}, "
        f"исходный USE_SUBMIT={flag_value}."
    )
    if zip_path is not None:
        print(f"ZIP: {zip_path} ({zip_path.stat().st_size} байт)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Сборка плоского каталога submit/ для платформы"
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="создать submit.zip с файлами в корне архива",
    )
    args = parser.parse_args()

    try:
        build(zip_requested=args.zip)
    except BuildError as exc:
        print(f"build_submit: ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
