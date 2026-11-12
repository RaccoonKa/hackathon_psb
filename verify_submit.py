#!/usr/bin/env python3
"""Автоматические проверки собранного каталога submit/.

Проверки не требуют установленных runtime-зависимостей: модули проекта
используют ленивые импорты, а сетевые вызовы не выполняются.
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent
MAX_ZIP_BYTES = 128 * 1024 * 1024

REQUIRED_FILES = {
    "solution.py",
    "settings_submit.py",
    "config.py",
    "build_index.py",
    "generation_orchestrator.py",
    "generation_llm_client.py",
    "indexing_parser.py",
    "indexing_chunker.py",
    "retrieval_hybrid.py",
    "retrieval_vector_store.py",
    "requirements.txt",
    "README.md",
}

EXPECTED_REQUIREMENTS = {
    "chromadb": "1.5.9",
    "httpx": "0.28.1",
    "langchain-openai": "1.7.4",
    "langchain-text-splitters": "1.1.3",
    "openai": "3.45.0",
    "rank-bm25": "0.2.2",
    "requests": "2.32.5",
    "tiktoken": "0.12.0",
}

FORBIDDEN_NAMES = {
    ".env",
    ".git",
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

TEST_ENV = {
    "LLM_BASE_URL": "http://proxy.local/v1",
    "LLM_API_KEY": "test-key",
    "LLM_MODEL": "test-gen-model",
    "EMB_MODEL": "test-emb-model",
    "RERANKER_MODEL": "test-rerank-model",
    "TOP_K": "7",
    "MAX_TOKENS": "123",
    "TEMPERATURE": "0.2",
    "MAX_WORKERS": "3",
}


class CheckFailure(AssertionError):
    """Проверка не пройдена."""


def _python_env(extra: Optional[dict] = None) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    env.pop("PYTHONHOME", None)
    if extra:
        env.update(extra)
    return env


def _run_python(
    code: str,
    cwd: Path,
    extra_env: Optional[dict] = None,
    pythonpath: Optional[Path] = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    env = _python_env(extra_env)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _expect_ok(result: subprocess.CompletedProcess, label: str) -> None:
    if result.returncode != 0:
        raise CheckFailure(
            f"{label}: код {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def _copy_submit(submit_dir: Path, tmp: Path) -> Path:
    destination = tmp / "submit_copy"
    shutil.copytree(submit_dir, destination)
    return destination


def check_structure(submit_dir: Path) -> None:
    if not submit_dir.is_dir():
        raise CheckFailure(f"Каталог не найден: {submit_dir}")

    names = {path.name for path in submit_dir.iterdir()}
    missing = REQUIRED_FILES - names
    if missing:
        raise CheckFailure(
            "В submit/ отсутствуют файлы: " + ", ".join(sorted(missing))
        )

    nested_python = [
        path
        for path in submit_dir.rglob("*.py")
        if path.parent != submit_dir
    ]
    if nested_python:
        raise CheckFailure(
            "Python-модули должны лежать в корне submit/: "
            + ", ".join(str(path.relative_to(submit_dir)) for path in nested_python)
        )

    for path in submit_dir.rglob("*"):
        if path.name in FORBIDDEN_NAMES or path.name.startswith(".env"):
            raise CheckFailure(
                "Запрещённый файл/каталог в submit/: "
                f"{path.relative_to(submit_dir)}"
            )

    settings = (submit_dir / "settings_submit.py").read_text(
        encoding="utf-8"
    ).strip()
    if settings != "USE_SUBMIT = True":
        raise CheckFailure(
            "submit/settings_submit.py должен содержать USE_SUBMIT = True"
        )


def check_compile(submit_dir: Path) -> None:
    for path in sorted(submit_dir.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        try:
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            raise CheckFailure(
                f"Синтаксическая ошибка в {path.name}: {exc}"
            ) from exc


def check_imports(submit_dir: Path) -> None:
    internal = {path.stem for path in submit_dir.glob("*.py")}
    for path in sorted(submit_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "src" or alias.name.startswith("src."):
                        raise CheckFailure(
                            f"В {path.name} остался импорт {alias.name}"
                        )
                    if (
                        alias.name in internal
                        and not (submit_dir / f"{alias.name}.py").is_file()
                    ):
                        raise CheckFailure(
                            f"В {path.name} не разрешается {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "src" or module.startswith("src."):
                    raise CheckFailure(
                        f"В {path.name} остался импорт from {module}"
                    )
                if module in internal and not (
                    submit_dir / f"{module}.py"
                ).is_file():
                    raise CheckFailure(
                        f"В {path.name} не разрешается {module}"
                    )


def check_requirements(submit_dir: Path) -> None:
    path = submit_dir / "requirements.txt"
    if not path.is_file():
        raise CheckFailure("Не найден requirements.txt")

    seen = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise CheckFailure(f"Зависимость без фиксированной версии: {line}")
        name, version = line.split("==", 1)
        seen[name.strip()] = version.strip()

    if seen != EXPECTED_REQUIREMENTS:
        raise CheckFailure(
            "requirements.txt не совпадает с ожидаемым набором: "
            f"{seen}"
        )

    forbidden = {"torch", "transformers", "faiss-cpu", "sentence-transformers"}
    if forbidden & set(seen):
        raise CheckFailure("В requirements попали тяжёлые/запрещённые пакеты")


def check_isolated_imports(submit_dir: Path) -> None:
    code = """
import config
import solution
import build_index
import generation_orchestrator
import indexing_parser
import retrieval_hybrid
import retrieval_vector_store
print("imports-ok")
"""
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        copy = _copy_submit(submit_dir, tmp)
        env = dict(TEST_ENV)
        env["DATA_DIR"] = str(tmp / "data")
        env["SUBMIT_WORK_DIR"] = str(tmp / "work")
        _expect_ok(
            _run_python(code, cwd=copy, extra_env=env),
            "Изолированный импорт",
        )


def check_submit_config(submit_dir: Path) -> None:
    code = """
import os
from pathlib import Path
import config

assert config.USE_SUBMIT is True
assert config.DATA_DIR == Path(os.environ["DATA_DIR"]).resolve()
assert config.WORK_DIR == Path(os.environ["SUBMIT_WORK_DIR"]).resolve()
for path in (
    config.RAW_DATA_DIR,
    config.PROCESSED_DATA_DIR,
    config.CHROMA_DIR,
    config.BM25_DIR,
    config.MANIFESTS_DIR,
):
    path.resolve().relative_to(config.WORK_DIR.resolve())

assert config.LLM_BASE_URL == "http://proxy.local/v1"
assert config.LLM_API_KEY == "test-key"
assert config.EMBEDDING_API_BASE_URL == config.LLM_BASE_URL
assert config.EMBEDDING_API_KEY == config.LLM_API_KEY
assert config.RERANKER_API_BASE_URL == config.LLM_BASE_URL
assert config.RERANKER_API_KEY == config.LLM_API_KEY
assert config.RERANKER_BACKEND == "api"
assert config.LLM_MODEL == "test-gen-model"
assert config.EMBEDDING_MODEL == "test-emb-model"
assert config.RERANKER_MODEL == "test-rerank-model"
assert config.RAG_TOP_K == 7
assert config.LLM_MAX_TOKENS == 123
assert abs(config.LLM_TEMPERATURE - 0.2) < 1e-9
assert config.MAX_WORKERS == 3
assert config.USE_LOCAL_EMBEDDING_MODEL is False
assert config.ALLOW_HASH_FALLBACK is False
assert config.USE_STUB_LLM is False

import retrieval_vector_store as vector_store
assert vector_store._load_local_model() == (None, None, None)
print("config-ok")
"""
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        copy = _copy_submit(submit_dir, tmp)
        env = dict(TEST_ENV)
        env["DATA_DIR"] = str(tmp / "data")
        env["SUBMIT_WORK_DIR"] = str(tmp / "work")
        _expect_ok(
            _run_python(code, cwd=copy, extra_env=env),
            "Конфигурация SUBMIT",
        )


def check_local_config() -> None:
    code = """
from pathlib import Path
import src.config as config

assert config.USE_SUBMIT is False
root = Path(config.__file__).resolve().parent.parent
assert config.BASE_DIR == root
assert config.RAW_DATA_DIR == root / "data" / "raw"
assert config.PROCESSED_DATA_DIR == root / "data" / "processed"
assert config.CHROMA_DIR == root / "storage" / "chroma_db"
assert config.BM25_DIR == root / "storage" / "bm25_index"
assert config.MANIFESTS_DIR == root / "data" / "manifests"
assert config.USE_LOCAL_EMBEDDING_MODEL is True
assert config.ALLOW_HASH_FALLBACK is True
# LOCAL допускает оба backend-а локального Qwen: HTTP (llama-server) и transformers.
assert config.RERANKER_BACKEND in {"api", "transformers"}
print("local-ok")
"""
    _expect_ok(
        _run_python(code, cwd=ROOT, pythonpath=ROOT),
        "Конфигурация LOCAL",
    )


def check_safe_extract(submit_dir: Path) -> None:
    code = """
import tempfile
import zipfile
from pathlib import Path
import indexing_parser

with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    bad_zip = tmp / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as archive:
        archive.writestr("../escape.txt", "bad")
        archive.writestr("good.txt", "ok")

    destination = tmp / "dest"
    try:
        indexing_parser.safe_extract_zip(bad_zip, destination)
    except indexing_parser.UnsafeArchiveError:
        pass
    else:
        raise AssertionError("path traversal не заблокирован")
    assert not (tmp / "escape.txt").exists()

    good_zip = tmp / "good.zip"
    with zipfile.ZipFile(good_zip, "w") as archive:
        archive.writestr("nested/документ.txt", "текст")
    indexing_parser.safe_extract_zip(good_zip, destination)
    assert (destination / "nested" / "документ.txt").read_text(
        encoding="utf-8"
    ) == "текст"
print("zip-ok")
"""
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        copy = _copy_submit(submit_dir, tmp)
        _expect_ok(
            _run_python(code, cwd=copy),
            "Безопасная распаковка",
        )


def check_document_loading(submit_dir: Path) -> None:
    code = """
import io
import tempfile
import zipfile
from pathlib import Path
import indexing_parser

DOCX_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    "<w:body>"
    "<w:p><w:r><w:t>Договор аренды № А-08/2024</w:t></w:r></w:p>"
    "<w:p><w:r><w:t>Арендная плата вносится ежемесячно.</w:t></w:r></w:p>"
    "</w:body></w:document>"
)

def docx_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", DOCX_XML)
    return buffer.getvalue()

with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    archive_path = tmp / "knowledge-base.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "документы компании__Договор поставки.txt",
            "Договор поставки № П-17/2024\\n1. Предмет договора",
        )
        archive.writestr(
            "законы__НК_ст224.md",
            "Статья 224. Налоговые ставки",
        )
        archive.writestr(
            "документы компании__Договор аренды.docx",
            docx_bytes(),
        )
        archive.writestr("картинка.bin", b"\\x00\\x01")

    destination = tmp / "extracted"
    indexing_parser.safe_extract_zip(archive_path, destination)
    documents = indexing_parser.load_documents_with_metadata(destination)

    assert len(documents) == 3, len(documents)
    docx_document = next(
        item
        for item in documents
        if item["metadata"]["source_filename"].endswith(".docx")
    )
    assert "Арендная плата" in docx_document["content"]
    assert any(
        item["metadata"]["document_type"] == "law"
        for item in documents
    )
print("parser-ok")
"""
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        copy = _copy_submit(submit_dir, tmp)
        _expect_ok(
            _run_python(code, cwd=copy),
            "Загрузка документов",
        )


def check_csv_contract(submit_dir: Path) -> None:
    code = """
import csv
import tempfile
from pathlib import Path
import solution

with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    questions_path = tmp / "questions.csv"
    questions_path.write_text(
        "\\ufeffid,question\\r\\n"
        "001,Первый вопрос?\\r\\n"
        "2,\\"Второй, с запятой\\"\\r\\n",
        encoding="utf-8",
    )
    rows = solution.read_questions(tmp)
    assert rows == [
        ("001", "Первый вопрос?"),
        ("2", "Второй, с запятой"),
    ], rows

    questions_path.write_text(
        "id,question\\n1,первый\\n1,второй\\n",
        encoding="utf-8",
    )
    try:
        solution.read_questions(tmp)
    except solution.SubmissionError:
        pass
    else:
        raise AssertionError("дубликат id не отклонён")

    output = tmp / "submission.csv"
    solution.write_submission(
        [
            ("001", 'ответ, с кавычкой " и переводом\\nстроки'),
            ("2", "x" * 20000),
        ],
        output,
    )
    with open(output, encoding="utf-8", newline="") as handle:
        parsed = list(csv.reader(handle))

    assert parsed[0] == ["id", "answer"]
    assert [row[0] for row in parsed[1:]] == ["001", "2"]
    assert len(parsed[1]) == 2 and len(parsed[2]) == 2
    assert parsed[1][1].startswith("ответ, с кавычкой")
    assert 0 < len(parsed[2][1]) <= solution.config.MAX_ANSWER_CHARS
print("csv-ok")
"""
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        copy = _copy_submit(submit_dir, tmp)
        _expect_ok(
            _run_python(code, cwd=copy),
            "Контракт CSV",
        )


def check_run_contract(submit_dir: Path) -> None:
    code = """
import tempfile
from pathlib import Path
import solution

queries = []
build_calls = {"count": 0}

class Result:
    def __init__(self, text):
        self.final_answer = text

def fake_run_agent(query, **kwargs):
    queries.append(query)
    return Result("ответ на " + query)

def fake_build_indexes():
    build_calls["count"] += 1
    return {"documents": 1, "chunks": 1}

solution.run_agent = fake_run_agent
solution.build_index.build_indexes = fake_build_indexes
solution.prepare_inputs = lambda data_dir: None

with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    output = tmp / "submission.csv"
    rows = solution.run(
        data_dir=tmp,
        questions=[("001", "q1"), ("2", "q2")],
        out_path=output,
    )
    assert build_calls["count"] == 1, build_calls
    assert queries == ["q1", "q2"], queries
    assert rows == [("001", "ответ на q1"), ("2", "ответ на q2")], rows
    assert output.is_file()

    def failing_run_agent(query, **kwargs):
        raise RuntimeError("boom " + query)

    solution.run_agent = failing_run_agent
    error_output = tmp / "error.csv"
    try:
        solution.run(
            data_dir=tmp,
            questions=[("3", "q3")],
            out_path=error_output,
        )
    except solution.SubmissionError as exc:
        assert "3" in str(exc), str(exc)
    else:
        raise AssertionError("ошибка вопроса не привела к SubmissionError")
    assert not error_output.exists()
print("run-ok")
"""
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        copy = _copy_submit(submit_dir, tmp)
        env = dict(TEST_ENV)
        env["MAX_WORKERS"] = "1"
        env["DATA_DIR"] = str(tmp / "data")
        env["SUBMIT_WORK_DIR"] = str(tmp / "work")
        _expect_ok(
            _run_python(code, cwd=copy, extra_env=env),
            "Контракт запуска",
        )


def check_missing_inputs(submit_dir: Path) -> None:
    code = """
import solution
raise SystemExit(solution.main())
"""
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        copy = _copy_submit(submit_dir, tmp)
        data_dir = tmp / "empty_data"
        data_dir.mkdir()
        env = dict(TEST_ENV)
        env["DATA_DIR"] = str(data_dir)
        env["SUBMIT_WORK_DIR"] = str(tmp / "work")
        result = _run_python(code, cwd=copy, extra_env=env)
        if result.returncode == 0:
            raise CheckFailure("solution.main() не вернул ошибку без входных файлов")
        if (copy / "submission.csv").exists():
            raise CheckFailure(
                "При отсутствии входных файлов создан submission.csv"
            )


def check_zip(zip_path: Path, submit_dir: Path) -> None:
    if not zip_path.is_file():
        raise CheckFailure(f"ZIP не найден: {zip_path}")
    if zip_path.stat().st_size > MAX_ZIP_BYTES:
        raise CheckFailure("ZIP превышает лимит 128 МБ")

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()

    if any(name.startswith("submit/") for name in names):
        raise CheckFailure("В ZIP есть внешняя директория submit/")
    for required in ("solution.py", "requirements.txt", "settings_submit.py"):
        if required not in names:
            raise CheckFailure(f"В корне ZIP отсутствует {required}")

    forbidden_prefixes = ("data/", "storage/", "models/", ".env", "__pycache__")
    for name in names:
        if name.startswith(forbidden_prefixes):
            raise CheckFailure(f"В ZIP попал запрещённый путь: {name}")

    expected = {
        path.relative_to(submit_dir).as_posix()
        for path in submit_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if set(names) != expected:
        raise CheckFailure(
            "Содержимое ZIP не совпадает с submit/: "
            f"zip={sorted(set(names) - expected)}, "
            f"submit={sorted(expected - set(names))}"
        )


def run_checks(
    submit_dir: Path, zip_path: Optional[Path]
) -> int:
    checks: list[tuple[str, Callable[[], None]]] = [
        ("структура submit/", lambda: check_structure(submit_dir)),
        ("compile", lambda: check_compile(submit_dir)),
        ("импорты без src.*", lambda: check_imports(submit_dir)),
        ("requirements", lambda: check_requirements(submit_dir)),
        ("изолированный импорт", lambda: check_isolated_imports(submit_dir)),
        ("конфигурация SUBMIT", lambda: check_submit_config(submit_dir)),
        ("конфигурация LOCAL", check_local_config),
        ("безопасная распаковка", lambda: check_safe_extract(submit_dir)),
        ("загрузка документов", lambda: check_document_loading(submit_dir)),
        ("контракт CSV", lambda: check_csv_contract(submit_dir)),
        ("контракт запуска", lambda: check_run_contract(submit_dir)),
        ("отсутствие входных файлов", lambda: check_missing_inputs(submit_dir)),
    ]
    if zip_path is not None:
        checks.append(
            ("структура ZIP", lambda: check_zip(zip_path, submit_dir))
        )

    failures = 0
    for name, check in checks:
        try:
            check()
        except CheckFailure as exc:
            failures += 1
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(
                f"[FAIL] {name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        else:
            print(f"[OK] {name}")

    if failures:
        print(f"Провалено проверок: {failures}", file=sys.stderr)
        return 1
    print("Все проверки submit/ пройдены.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Проверки собранного каталога submit/"
    )
    parser.add_argument(
        "--submit-dir",
        default=str(ROOT / "submit"),
        help="путь к каталогу submit/",
    )
    parser.add_argument(
        "--zip-path",
        default=None,
        help="путь к ZIP для проверки структуры архива",
    )
    args = parser.parse_args()

    submit_dir = Path(args.submit_dir).resolve()
    zip_path = Path(args.zip_path).resolve() if args.zip_path else None
    return run_checks(submit_dir, zip_path)


if __name__ == "__main__":
    sys.exit(main())
