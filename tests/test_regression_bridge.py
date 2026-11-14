"""Регрессионные тесты последовательного retrieval-моста (пункты 11–13).

Проверяются: post-hoc тайм-аут без пула потоков, отсутствие retrieval-потоков
и ошибки ``cannot schedule new futures after interpreter shutdown`` при
штатном завершении процесса.
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from unittest import SkipTest
from unittest.mock import patch

from rag_core.generation.errors import RetrievalTimeoutError
from rag_core.generation.retrieval import bridge as bridge_mod
from rag_core.generation.types import RetrievalTask
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


def _retrieval_threads() -> set[threading.Thread]:
    return {
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("retrieval") and thread.is_alive()
    }


# --- 11. Post-hoc таймаут retrieval ---------------------------------------


def test_retrieval_task_timeout_raises_after_post_hoc_check():
    def slow_task(_task, state=None):
        return ([make_retrieved_chunk(chunk_id="late")], {}, {}, 600)

    diagnostics: dict = {}
    raised = False
    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "_run_single_task", side_effect=slow_task
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 0.1
    ):
        try:
            bridge_mod.retrieve_tasks([_task()], diagnostics_out=diagnostics)
        except RetrievalTimeoutError:
            raised = True

    assert raised, "Ожидался RetrievalTimeoutError при превышении дедлайна"

    executor = diagnostics["executor"]
    assert executor["timed_out"] >= 1
    assert executor["completed"] == 0
    assert executor["threads_alive_after_grace"] == 0
    statuses = [task["status"] for task in diagnostics["tasks"]]
    assert "timeout" in statuses


# --- 12. Нет пула потоков и нет утечки retrieval-потоков -------------------


def test_retrieve_tasks_does_not_create_threads_or_executor():
    before = _retrieval_threads()

    def fast_task(_task, state=None):
        return ([make_retrieved_chunk(chunk_id="ok_1")], {}, {}, 5)

    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "_run_single_task", side_effect=fast_task
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 5.0
    ):
        chunks, step = bridge_mod.retrieve_tasks([_task()])

    assert [chunk.chunk_id for chunk in chunks] == ["ok_1"]
    assert step.extra["execution"] == "sequential"
    assert not hasattr(bridge_mod, "_LAST_EXECUTOR")
    assert not hasattr(bridge_mod, "_aggregate_task_results")
    assert not (_retrieval_threads() - before), (
        "После retrieve_tasks появились retrieval-потоки"
    )
    source = Path(bridge_mod.__file__).read_text(encoding="utf-8")
    assert "ThreadPoolExecutor" not in source
    assert "concurrent.futures" not in source


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
import rag_core.generation.retrieval.bridge as bridge
# Пропускаем тяжёлую предзагрузку модели: тест про последовательное
# завершение, а не про инициализацию бэкендов. Сам поиск BM25 остаётся реальным.
bridge.prewarm_retrieval_backends = lambda: {{"errors": []}}
from rag_core.generation.types import RetrievalTask

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
