"""Тесты последовательного retrieval pipeline (Этап 1).

Проверяются:
* single dense/bm25/hybrid возвращают ожидаемую структуру;
* задачи исполняются последовательно, без retrieval-потоков;
* post-hoc тайм-аут отбрасывает результат задачи;
* частичный сбой даёт degraded, полный сбой — контролируемое исключение;
* prewarm делает warm-up forward и первый батч после него «горячий»;
* диагностика timeout-задач не содержит фиктивных ``0``/``success: true``.
"""

import sys
import threading
import time
from unittest.mock import patch

from rag_core.generation.errors import RetrievalAPIError, RetrievalTimeoutError
from rag_core.generation.retrieval import bridge as bridge_mod
from rag_core.generation.types import RetrievalTask
from rag_core.retrieval import vector_store as vs_mod
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
        assert diagnostics["executor"]["timed_out"] == 0
        assert diagnostics["executor"]["queue_timeout"] == 0
        assert diagnostics["executor"]["threads_alive_after_grace"] == 0

        if mode == "bm25":
            assert entry["embeddings"]["provider"] == "none"
            assert entry["embeddings"]["success"] is None
            assert entry["embeddings"]["measured"] is False


# --- 2. Последовательное исполнение без потоков ----------------------------


def test_tasks_execute_sequentially_without_retrieval_threads():
    thread_ids: list[int] = []

    def worker(task, state=None):
        thread_ids.append(threading.get_ident())
        return (
            [make_retrieved_chunk(chunk_id=f"c_{task.purpose}")],
            {},
            {},
            1,
        )

    tasks = [_task(purpose=f"p{index}") for index in range(5)]
    diagnostics: dict = {}
    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "_run_single_task", side_effect=worker
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 5.0
    ):
        chunks, step = bridge_mod.retrieve_tasks(
            tasks, diagnostics_out=diagnostics
        )

    assert len(thread_ids) == 5
    assert len(set(thread_ids)) == 1, "задачи должны идти в одном потоке"
    assert not _retrieval_threads()
    assert step.extra["execution"] == "sequential"
    assert diagnostics["executor"]["completed"] == 5
    assert diagnostics["executor"]["failed"] == 0
    assert diagnostics["executor"]["queued"] == 0
    assert diagnostics["executor"]["threads_alive_after_grace"] == 0
    assert {chunk.chunk_id for chunk in chunks} == {
        f"c_p{index}" for index in range(5)
    }


# --- 3. Post-hoc тайм-аут одной задачи -------------------------------------


def test_slow_task_times_out_post_hoc_and_fast_result_is_kept():
    def worker(task, state=None):
        if task.purpose == "slow":
            return ([make_retrieved_chunk(chunk_id="late")], {}, {}, 500)
        return (
            [make_retrieved_chunk(chunk_id=f"fast_{task.purpose}")],
            {},
            {},
            5,
        )

    tasks = [_task(purpose="fast1"), _task(purpose="slow")]
    diagnostics: dict = {}
    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "_run_single_task", side_effect=worker
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 0.1
    ):
        chunks, step = bridge_mod.retrieve_tasks(
            tasks, diagnostics_out=diagnostics
        )

    assert diagnostics["executor"]["completed"] == 1
    assert diagnostics["executor"]["timed_out"] == 1
    assert {chunk.chunk_id for chunk in chunks} == {"fast_fast1"}
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
    assert diagnostics["executor"]["batch_duration_ms"] >= 0
    assert not _retrieval_threads()


# --- 4. Задача в пределах дедлайна не помечается timeout -------------------


def test_task_under_timeout_is_not_marked_timeout():
    def worker(task, state=None):
        return ([make_retrieved_chunk(chunk_id="quick")], {}, {}, 20)

    diagnostics: dict = {}
    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "_run_single_task", side_effect=worker
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 5.0
    ):
        chunks, _step = bridge_mod.retrieve_tasks(
            [_task(purpose="quick")], diagnostics_out=diagnostics
        )

    entry = diagnostics["tasks"][0]
    assert [chunk.chunk_id for chunk in chunks] == ["quick"]
    assert entry["status"] == "ok"
    assert entry["timeout_reason"] is None
    assert diagnostics["executor"]["completed"] == 1


# --- 5. Частичный и полный сбой --------------------------------------------


def test_partial_failure_marks_degraded_and_keeps_success():
    def worker(task, state=None):
        if task.purpose == "bad":
            raise ValueError("boom")
        return ([make_retrieved_chunk(chunk_id="ok")], {}, {}, 1)

    diagnostics: dict = {}
    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "_run_single_task", side_effect=worker
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 5.0
    ):
        chunks, step = bridge_mod.retrieve_tasks(
            [_task(purpose="bad"), _task(purpose="good")],
            diagnostics_out=diagnostics,
        )

    assert [chunk.chunk_id for chunk in chunks] == ["ok"]
    assert diagnostics["executor"]["completed"] == 1
    assert diagnostics["executor"]["failed"] == 1
    assert diagnostics["degraded"] is True
    assert step.output["degraded"] is True
    assert any(error["type"] == "ValueError" for error in diagnostics["errors"])


def test_all_tasks_failed_raises_api_error():
    def worker(task, state=None):
        raise RuntimeError("boom")

    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "_run_single_task", side_effect=worker
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 5.0
    ):
        try:
            bridge_mod.retrieve_tasks([_task()])
        except RetrievalAPIError:
            pass
        else:
            raise AssertionError("Ожидался RetrievalAPIError")


def test_all_tasks_timed_out_raises_timeout_error():
    def worker(task, state=None):
        return ([make_retrieved_chunk(chunk_id="late")], {}, {}, 500)

    with patch.object(
        bridge_mod,
        "prewarm_retrieval_backends",
        return_value={"errors": []},
    ), patch.object(
        bridge_mod, "_run_single_task", side_effect=worker
    ), patch.object(
        bridge_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 0.1
    ):
        try:
            bridge_mod.retrieve_tasks([_task()])
        except RetrievalTimeoutError:
            pass
        else:
            raise AssertionError("Ожидался RetrievalTimeoutError")


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

            def worker(task, state=None):
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
