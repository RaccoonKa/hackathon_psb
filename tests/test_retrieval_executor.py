"""Тесты Этапа 1: warm-up, per-task дедлайны, сбор результатов, lifecycle.

Проверяются:
* single dense/bm25/hybrid возвращают ожидаемую структуру;
* параллельные задачи завершаются без дедлока и потери быстрых результатов;
* искусственный тайм-аут одной задачи не выбрасывает результаты остальных;
* уже завершённая задача не помечается timeout (done() раньше дедлайна);
* queue_timeout отличается от task_timeout;
* prewarm делает warm-up forward и первый батч после него «горячий»;
* после тайм-аута следующий батч работает, потоки не растут;
* новый executor не создаётся, пока живы потоки предыдущего;
* диагностика timeout-задач не содержит фиктивных ``0``/``success: true``.
"""

import concurrent.futures
import sys
import threading
import time
from unittest.mock import patch

from src.generation.errors import RetrievalBusyError, RetrievalTimeoutError
from src.generation.retrieval import bridge as bridge_mod
from src.generation.types import RetrievalTask
from src.retrieval import vector_store as vs_mod
from tests._common import make_chunk_dict, make_retrieved_chunk


def _task(purpose: str = "t", mode: str = "hybrid", query: str = "запрос"):
    return RetrievalTask(
        query=query,
        mode=mode,
        purpose=purpose,
        selected_doc_ids=None,
        boost_doc_ids=None,
        event_date=None,
        top_k=5,
    )


def _drain_last_executor(timeout: float = 2.0) -> None:
    """Дожидается потоков последнего пула, чтобы не влиять на следующие тесты."""
    executor = bridge_mod._LAST_EXECUTOR
    if executor is None:
        return
    for thread in list(getattr(executor, "_threads", ()) or ()):
        thread.join(timeout=timeout)


def _retrieval_threads() -> set[threading.Thread]:
    return {
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("retrieval") and thread.is_alive()
    }


# --- 1. Single задачи разных режимов ---------------------------------------


def test_single_modes_return_expected_structure():
    fake = [make_chunk_dict(chunk_id="c1", score=0.7)]
    expected_fields = {
        "task_id",
        "mode",
        "purpose",
        "queued_at",
        "started_at",
        "finished_at",
        "queue_wait_ms",
        "duration_ms",
        "stage",
        "status",
        "timeout_reason",
        "search_type",
        "embedding_provider",
        "embedding_model",
        "model_reused",
        "bm25_candidates_count",
        "dense_candidates_count",
        "candidates_before_filters",
        "candidates_after_filters",
        "final_chunks_count",
    }
    for mode, func_name in (
        ("bm25", "bm25_search"),
        ("dense", "dense_search"),
        ("hybrid", "hybrid_search"),
    ):
        diagnostics: dict = {}
        with patch.object(bridge_mod.config, "USE_STUB_RAG", False), patch.object(
            bridge_mod, func_name, return_value=fake
        ), patch.object(
            bridge_mod,
            "prewarm_retrieval_backends",
            return_value={"errors": []},
        ), patch.object(
            bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 5.0
        ):
            chunks, step = bridge_mod.retrieve_tasks(
                [_task(mode=mode)], diagnostics_out=diagnostics
            )

        assert [chunk.chunk_id for chunk in chunks] == ["c1"]
        assert step.step == bridge_mod.config.STEP_RAG_RETRIEVAL
        entry = diagnostics["tasks"][0]
        assert expected_fields <= set(entry)
        assert entry["status"] == "ok"
        assert entry["stage"] == "done"
        assert entry["search_type"] == mode
        assert entry["final_chunks_count"] == 1
        assert entry["queue_wait_ms"] is not None
        assert entry["duration_ms"] is not None
        assert diagnostics["executor"]["submitted"] == 1
        assert diagnostics["executor"]["started"] == 1
        assert diagnostics["executor"]["completed"] == 1
        assert diagnostics["executor"]["threads_alive_after_grace"] == 0

        if mode == "bm25":
            # bm25 не использует эмбеддинги: provider none, но не success=true.
            assert entry["embeddings"]["provider"] == "none"
            assert entry["embeddings"]["success"] is None
            assert entry["embeddings"]["measured"] is False


# --- 2. Параллельные задачи ------------------------------------------------


def test_parallel_tasks_complete_without_deadlock_or_lost_results():
    def worker(task):
        time.sleep(0.05)
        return (
            [make_retrieved_chunk(chunk_id=f"c_{task.purpose}")],
            {},
            {},
            50,
        )

    tasks = [_task(purpose=f"p{index}") for index in range(5)]
    diagnostics: dict = {}
    started = time.monotonic()
    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "_run_single_task", side_effect=worker
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_MAX_WORKERS", 4
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 5.0
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_QUEUE_TIMEOUT_S", 5.0
    ):
        chunks, _step = bridge_mod.retrieve_tasks(
            tasks, diagnostics_out=diagnostics
        )
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, f"parallel batch занял слишком много: {elapsed:.2f} с"
    assert diagnostics["executor"]["completed"] == 5
    assert diagnostics["executor"]["failed"] == 0
    assert diagnostics["executor"]["timed_out"] == 0
    statuses = {entry["status"] for entry in diagnostics["tasks"]}
    assert statuses == {"ok"}
    assert {chunk.chunk_id for chunk in chunks} == {
        f"c_p{index}" for index in range(5)
    }
    # Задача, ждавшая в очереди, имеет собственный queue_wait_ms.
    queue_waits = [entry["queue_wait_ms"] for entry in diagnostics["tasks"]]
    assert all(value is not None for value in queue_waits)
    assert diagnostics["executor"]["threads_alive_after_grace"] == 0


# --- 3. Искусственный тайм-аут одной задачи --------------------------------


def test_slow_task_times_out_and_fast_results_are_kept():
    def worker(task):
        if task.purpose == "slow":
            time.sleep(0.4)
            return ([make_retrieved_chunk(chunk_id="late")], {}, {}, 400)
        return (
            [make_retrieved_chunk(chunk_id=f"fast_{task.purpose}")],
            {},
            {},
            5,
        )

    tasks = [
        _task(purpose="fast1"),
        _task(purpose="slow"),
        _task(purpose="fast2"),
        _task(purpose="fast3"),
    ]
    diagnostics: dict = {}
    try:
        with patch.object(
            bridge_mod,
            "prewarm_retrieval_backends",
            return_value={"errors": []},
        ), patch.object(
            bridge_mod, "_run_single_task", side_effect=worker
        ), patch.object(
            bridge_mod.config, "RETRIEVAL_MAX_WORKERS", 4
        ), patch.object(
            bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 0.1
        ), patch.object(
            bridge_mod.config, "RETRIEVAL_QUEUE_TIMEOUT_S", 5.0
        ), patch.object(
            bridge_mod.config, "RETRIEVAL_SHUTDOWN_GRACE_S", 0.05
        ):
            chunks, step = bridge_mod.retrieve_tasks(
                tasks, diagnostics_out=diagnostics
            )
    finally:
        _drain_last_executor()

    assert diagnostics["executor"]["completed"] == 3
    assert diagnostics["executor"]["timed_out"] >= 1
    assert {chunk.chunk_id for chunk in chunks} == {
        "fast_fast1",
        "fast_fast2",
        "fast_fast3",
    }
    slow_entry = next(
        entry for entry in diagnostics["tasks"] if entry["purpose"] == "slow"
    )
    assert slow_entry["status"] == "timeout"
    assert slow_entry["timeout_reason"] == "task_timeout"
    assert slow_entry["stage"] == "timeout"
    # Незавершённая задача не отдаёт фиктивные измерения.
    assert slow_entry["embeddings"]["success"] is None
    assert slow_entry["embeddings"]["provider"] == "unknown"
    assert slow_entry["bm25_candidates_count"] is None
    assert slow_entry["dense_candidates_count"] is None
    assert slow_entry["final_chunks_count"] is None
    assert diagnostics["degraded"] is True
    assert step.output["degraded"] is True
    assert any(
        error["type"] == "TimeoutError" for error in diagnostics["errors"]
    )
    assert isinstance(
        diagnostics["executor"]["threads_alive_after_grace"], int
    )
    assert diagnostics["executor"]["batch_duration_ms"] >= 0


# --- 4. Завершённая задача не помечается timeout ---------------------------


def test_completed_future_is_not_marked_timeout():
    def worker(task):
        time.sleep(0.02)
        return ([make_retrieved_chunk(chunk_id="quick")], {}, {}, 20)

    diagnostics: dict = {}
    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "_run_single_task", side_effect=worker
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_MAX_WORKERS", 1
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 5.0
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_QUEUE_TIMEOUT_S", 0.0
    ):
        chunks, _step = bridge_mod.retrieve_tasks(
            [_task(purpose="quick")], diagnostics_out=diagnostics
        )

    entry = diagnostics["tasks"][0]
    assert [chunk.chunk_id for chunk in chunks] == ["quick"]
    # future.done() проверяется раньше дедлайна: результат не выбрасывается,
    # несмотря на нулевой queue-бюджет.
    assert entry["status"] == "ok"
    assert entry["timeout_reason"] is None
    assert diagnostics["executor"]["completed"] == 1


# --- 5. queue_timeout отличается от task_timeout ----------------------------


def test_queue_timeout_differs_from_task_timeout():
    def worker(task):
        if task.purpose == "first":
            time.sleep(0.5)
            return ([make_retrieved_chunk(chunk_id="first")], {}, {}, 500)
        return ([make_retrieved_chunk(chunk_id="second")], {}, {}, 1)

    tasks = [_task(purpose="first"), _task(purpose="second")]
    diagnostics: dict = {}
    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "_run_single_task", side_effect=worker
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_MAX_WORKERS", 1
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 5.0
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_QUEUE_TIMEOUT_S", 0.1
    ):
        chunks, _step = bridge_mod.retrieve_tasks(
            tasks, diagnostics_out=diagnostics
        )

    first = next(
        entry for entry in diagnostics["tasks"] if entry["purpose"] == "first"
    )
    second = next(
        entry for entry in diagnostics["tasks"] if entry["purpose"] == "second"
    )
    assert [chunk.chunk_id for chunk in chunks] == ["first"]
    assert first["status"] == "ok"
    assert first["timeout_reason"] is None
    # Задача из очереди не стартовала и получила собственный queue_timeout.
    assert second["status"] == "queue_timeout"
    assert second["timeout_reason"] == "queue_timeout"
    assert second["started_at"] is None
    assert second["stage"] == "queued"
    assert second["queue_wait_ms"] is not None
    assert second["queue_wait_ms"] >= 50
    assert diagnostics["executor"]["queue_timeout"] == 1
    assert diagnostics["executor"]["timed_out"] == 1
    assert diagnostics["executor"]["started"] == 1
    assert diagnostics["executor"]["queued"] == 1


# --- 6. Cold/warm: prewarm делает warm-up forward --------------------------


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeTensor:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, key):
        if isinstance(key, tuple) and key == (slice(None), 0):
            return _FakeTensor([[r[0]] for r in self._rows])
        return self

    def cpu(self):
        return self

    def to(self, device):
        return self

    def tolist(self):
        return self._rows


class _FakeOutputs:
    def __init__(self, rows):
        self.last_hidden_state = _FakeTensor(rows)


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _SlowFirstTokenizer:
    def __call__(self, texts, **kwargs):
        return _FakeBatchEncoding({"input_ids": [[1, 2, 3]] * len(texts)})


class _SlowFirstModel:
    """Первый forward медленный (имитация холодного bge-m3)."""

    def __init__(self, delay: float = 0.4):
        self.delay = delay
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            time.sleep(self.delay)
        count = len(kwargs.get("input_ids", []))
        return _FakeOutputs([[0.5, 0.5] for _ in range(count)])


class _FakeTorchModule:
    class cuda:
        @staticmethod
        def is_available():
            return False

    class nn:
        class functional:
            @staticmethod
            def normalize(tensor, p, dim):
                return tensor

    @staticmethod
    def no_grad():
        return _NullCtx()


class _FakeCollection:
    def count(self):
        return 0


def test_prewarm_warms_first_forward_and_first_batch_is_warm():
    fake_model = _SlowFirstModel(delay=0.4)
    fake_tokenizer = _SlowFirstTokenizer()
    saved_torch = sys.modules.get("torch")
    sys.modules["torch"] = _FakeTorchModule()
    try:
        vs_mod.reset_local_model_cache()
        with patch.object(
            vs_mod,
            "_load_local_model",
            return_value=(fake_tokenizer, fake_model, "cpu"),
        ), patch.object(
            vs_mod, "get_chroma_collection", return_value=_FakeCollection()
        ), patch.object(
            vs_mod, "get_collection_embedding_dimension", return_value=None
        ):
            report = vs_mod.prewarm_retrieval_backends()

            assert report["warmup"] == "ok"
            assert report["warmup_ms"] is not None
            assert report["warmup_ms"] >= 250
            # Warm-up уже израсходовал медленный первый forward.
            assert fake_model.calls == 1

            def worker(task):
                vs_mod.compute_embeddings([task.query])
                return (
                    [make_retrieved_chunk(chunk_id="warm")],
                    {},
                    vs_mod.get_last_embedding_diagnostics(),
                    1,
                )

            diagnostics: dict = {}
            with patch.object(
                bridge_mod,
                "prewarm_retrieval_backends",
                return_value=report,
            ), patch.object(
                bridge_mod, "_run_single_task", side_effect=worker
            ), patch.object(
                bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 0.2
            ), patch.object(
                bridge_mod.config, "RETRIEVAL_QUEUE_TIMEOUT_S", 5.0
            ):
                chunks, _step = bridge_mod.retrieve_tasks(
                    [_task(purpose="warmup")], diagnostics_out=diagnostics
                )

            assert [chunk.chunk_id for chunk in chunks] == ["warm"]
            assert diagnostics["tasks"][0]["status"] == "ok"
            assert diagnostics["prewarm"]["warmup"] == "ok"
            # Второй forward (уже горячий) уложился в 0.2 с.
            assert fake_model.calls == 2
    finally:
        vs_mod.reset_local_model_cache()
        if saved_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved_torch


# --- 7. Последовательные батчи после тайм-аута -----------------------------


def test_sequential_batches_after_timeout_do_not_block():
    def worker(task):
        if task.purpose == "slow":
            time.sleep(0.25)
            return ([make_retrieved_chunk(chunk_id="late")], {}, {}, 250)
        return ([make_retrieved_chunk(chunk_id="ok")], {}, {}, 1)

    first_diag: dict = {}
    try:
        with patch.object(
            bridge_mod,
            "prewarm_retrieval_backends",
            return_value={"errors": []},
        ), patch.object(
            bridge_mod, "_run_single_task", side_effect=worker
        ), patch.object(
            bridge_mod.config, "RETRIEVAL_MAX_WORKERS", 1
        ), patch.object(
            bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 0.1
        ), patch.object(
            bridge_mod.config, "RETRIEVAL_QUEUE_TIMEOUT_S", 5.0
        ), patch.object(
            bridge_mod.config, "RETRIEVAL_SHUTDOWN_GRACE_S", 0.02
        ):
            try:
                bridge_mod.retrieve_tasks(
                    [_task(purpose="slow")], diagnostics_out=first_diag
                )
            except RetrievalTimeoutError:
                pass
            else:
                raise AssertionError("Ожидался RetrievalTimeoutError")
    finally:
        _drain_last_executor()

    assert first_diag["executor"]["timed_out"] == 1

    before = _retrieval_threads()
    second_diag: dict = {}
    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "_run_single_task", side_effect=worker
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_MAX_WORKERS", 1
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 5.0
    ):
        chunks, _step = bridge_mod.retrieve_tasks(
            [_task(purpose="fast")], diagnostics_out=second_diag
        )

    assert [chunk.chunk_id for chunk in chunks] == ["ok"]
    assert second_diag["tasks"][0]["status"] == "ok"
    # Число рабочих потоков не растёт бесконтрольно (зомби-поток дожидается
    # в пределах grace и не блокирует следующий батч).
    assert not (_retrieval_threads() - before)


# --- 8. Новый executor при живых потоках предыдущего -----------------------


def test_new_executor_not_created_while_previous_threads_alive():
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="retrieval"
    )
    started = threading.Event()
    release = threading.Event()

    def hold():
        started.set()
        release.wait(2.0)
        return 1

    try:
        future = executor.submit(hold)
        assert started.wait(1.0)
        bridge_mod._LAST_EXECUTOR = executor

        diagnostics: dict = {}
        with patch.object(
            bridge_mod,
            "prewarm_retrieval_backends",
            return_value={"errors": []},
        ), patch.object(
            bridge_mod.config, "RETRIEVAL_SHUTDOWN_GRACE_S", 0.02
        ), patch.object(
            bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 1.0
        ):
            try:
                bridge_mod.retrieve_tasks(
                    [_task(purpose="busy")], diagnostics_out=diagnostics
                )
            except RetrievalBusyError as exc:
                assert getattr(exc, "threads_alive", 0) >= 1
            else:
                raise AssertionError(
                    "Ожидался RetrievalBusyError при живых потоках"
                )

        assert diagnostics["executor"]["threads_alive_after_grace"] >= 1
        assert diagnostics["degraded"] is True
        assert diagnostics["errors"][-1]["type"] == "RetrievalBusyError"
        # Новый пул не создавался: последним остаётся ручной executor.
        assert bridge_mod._LAST_EXECUTOR is executor
    finally:
        release.set()
        future.result(timeout=2.0)
        executor.shutdown(wait=True)
        bridge_mod._LAST_EXECUTOR = None
