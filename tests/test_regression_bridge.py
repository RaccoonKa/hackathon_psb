"""Регрессионные тесты пула retrieval-задач (пункты 11–13).

Проверяются: ограниченный по времени таймаут, ровно один shutdown пула с
завершением рабочих потоков и отсутствие ошибки
``cannot schedule new futures after interpreter shutdown`` при штатном
завершении процесса.
"""

import concurrent.futures
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from unittest import SkipTest
from unittest.mock import patch

from src import config
from src.generation.errors import RetrievalTimeoutError
from src.generation.retrieval import bridge as bridge_mod
from src.generation.types import RetrievalTask
from tests._common import make_retrieved_chunk


def _task(purpose: str = "regression") -> RetrievalTask:
    return RetrievalTask(
        query="арендная плата 450 000 рублей",
        mode="hybrid",
        purpose=purpose,
        selected_doc_ids=None,
        boost_doc_ids=None,
        event_date=None,
        top_k=5,
    )


# --- 11. Таймаут retrieval -------------------------------------------------


def test_retrieval_task_timeout_raises_and_is_bounded():
    def slow_task(_task):
        time.sleep(0.6)
        return (
            [make_retrieved_chunk(chunk_id="late")],
            {},
            {},
            600,
        )

    diagnostics: dict = {}
    started = time.monotonic()
    raised = False
    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "_run_single_task", side_effect=slow_task
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 0.1
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_SHUTDOWN_GRACE_S", 0.05
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_MAX_WORKERS", 1
    ):
        try:
            bridge_mod.retrieve_tasks([_task()], diagnostics_out=diagnostics)
        except RetrievalTimeoutError:
            raised = True
    elapsed = time.monotonic() - started

    assert raised, "Ожидался RetrievalTimeoutError при превышении дедлайна"
    # Возврат ограничен по времени: не ждём полного сна задачи (0.6 с).
    assert elapsed < 0.5, f"Таймаут не ограничил время возврата: {elapsed:.3f} с"

    executor = diagnostics["executor"]
    assert executor["timed_out"] >= 1
    assert executor["completed"] == 0
    statuses = [task["status"] for task in diagnostics["tasks"]]
    assert "timeout" in statuses


# --- 12. Корректное завершение ThreadPoolExecutor --------------------------


def test_retrieve_tasks_shuts_down_executor_once_and_joins_threads():
    before = {
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("retrieval") and thread.is_alive()
    }

    def fast_task(_task):
        return (
            [make_retrieved_chunk(chunk_id="ok_1")],
            {},
            {},
            5,
        )

    original_shutdown = concurrent.futures.ThreadPoolExecutor.shutdown
    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "_run_single_task", side_effect=fast_task
    ), patch.object(
        concurrent.futures.ThreadPoolExecutor,
        "shutdown",
        autospec=True,
        side_effect=original_shutdown,
    ) as shutdown_spy:
        chunks, _step = bridge_mod.retrieve_tasks([_task()])

    assert [chunk.chunk_id for chunk in chunks] == ["ok_1"]
    assert shutdown_spy.call_count == 1, "shutdown пула должен вызываться ровно один раз"

    still_alive = {
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("retrieval") and thread.is_alive()
    }
    # Сравнение с baseline: потоки других (возможно, ещё завершающихся)
    # тестов не считаются утечкой текущего пула.
    leaked = still_alive - before
    assert not leaked, (
        "После retrieve_tasks остались живые рабочие потоки пула: "
        f"{[thread.name for thread in leaked]}"
    )


def test_executor_double_shutdown_is_safe():
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="regression"
    )
    try:
        future = executor.submit(lambda: 42)
        assert future.result(timeout=5) == 42
        executor.shutdown(wait=True)
        # Повторный shutdown не должен падать.
        executor.shutdown(wait=True)
    finally:
        if not executor._shutdown:
            executor.shutdown(wait=False, cancel_futures=True)


# --- 13. Нет ошибки "cannot schedule new futures after interpreter shutdown" --


def test_normal_shutdown_has_no_interpreter_shutdown_future_error():
    root = Path(__file__).resolve().parent.parent
    if not (root / "storage" / "bm25_index" / "bm25.pkl").exists() or not (
        root / "data" / "processed" / "chunks.jsonl"
    ).exists():
        raise SkipTest("Нет реального индекса для subprocess-проверки завершения")
    script = f"""
import sys
sys.path.insert(0, {str(root)!r})
import src.generation.retrieval.bridge as bridge
# Пропускаем тяжёлую предзагрузку модели: тест про завершение пула, а не
# про инициализацию бэкендов. Сам поиск BM25 остаётся реальным.
bridge.prewarm_retrieval_backends = lambda: {{"errors": []}}
from src.generation.types import RetrievalTask

task = RetrievalTask(
    query="арендная плата 450 000 рублей",
    mode="bm25",
    purpose="shutdown_regression",
    selected_doc_ids={{"contract_а-08-2024"}},
    boost_doc_ids=None,
    event_date=None,
    top_k=5,
)
chunks, step = bridge.retrieve_tasks([task])
assert chunks, "real bm25 retrieval returned no chunks"
print("ok", len(chunks), flush=True)
"""

    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_shutdown_regression.py", delete=False, encoding="utf-8"
    )
    try:
        handle.write(script)
        handle.close()
        completed = subprocess.run(
            [sys.executable, handle.name],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=180,
        )
    finally:
        Path(handle.name).unlink(missing_ok=True)

    assert completed.returncode == 0, completed.stderr
    assert "cannot schedule new futures after interpreter shutdown" not in completed.stderr.lower()
    assert "cannot schedule new futures" not in completed.stderr.lower()
    assert "ok" in completed.stdout
