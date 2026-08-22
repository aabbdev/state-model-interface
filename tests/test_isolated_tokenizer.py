from __future__ import annotations

import time
from multiprocessing import active_children
from typing import Any

import pytest

from state_model_interface.compiler import CompiledSMITraining, render_smi_plan
from state_model_interface.isolated_tokenizer import (
    IsolatedSMIBatchCompiler,
    IsolatedSMICompilerClosedError,
    IsolatedSMICompilerTimeoutError,
    IsolatedSMIWorkerError,
)


def _success_worker(request_queue: Any, response_queue: Any) -> None:
    while True:
        request = request_queue.get()
        if request is None:
            return
        request_id, plans = request
        results = [CompiledSMITraining([len(plan.fragments)], [-100]) for plan in plans]
        response_queue.put(("result", request_id, results))


def _error_worker(request_queue: Any, response_queue: Any) -> None:
    while True:
        request = request_queue.get()
        if request is None:
            return
        request_id, _ = request
        try:
            raise ValueError("synthetic tokenizer failure")
        except ValueError as error:
            from state_model_interface.isolated_tokenizer import _serialize_error

            response_queue.put(("error", request_id, _serialize_error(error)))


def _timeout_worker(request_queue: Any, response_queue: Any) -> None:
    del response_queue
    while request_queue.get() is not None:
        time.sleep(60)


def test_persistent_spawn_worker_compiles_multiple_batches() -> None:
    plan = render_smi_plan([{"role": "user", "content": "hello"}])
    with IsolatedSMIBatchCompiler(2.0, worker_target=_success_worker) as compiler:
        first = compiler.compile([plan])
        process = compiler._process
        assert process is not None
        worker_pid = process.pid
        second = compiler.compile([plan, plan])

        assert process is compiler._process
        assert process.is_alive()
        assert first == [CompiledSMITraining([len(plan.fragments)], [-100])]
        assert len(second) == 2

    assert compiler._process is None
    assert worker_pid not in {child.pid for child in active_children()}
    compiler.close()
    with pytest.raises(IsolatedSMICompilerClosedError):
        compiler.compile([plan])


def test_worker_error_is_typed_and_worker_remains_alive() -> None:
    plan = render_smi_plan([{"role": "user", "content": "hello"}])
    with IsolatedSMIBatchCompiler(2.0, worker_target=_error_worker) as compiler:
        with pytest.raises(IsolatedSMIWorkerError) as caught:
            compiler.compile([plan])

        process = compiler._process
        assert process is not None and process.is_alive()
        assert caught.value.remote_type == "builtins.ValueError"
        assert caught.value.remote_message == "synthetic tokenizer failure"
        assert "raise ValueError" in caught.value.remote_traceback


def test_timeout_kills_worker_and_next_call_recreates_it() -> None:
    plan = render_smi_plan([{"role": "user", "content": "hello"}])
    with IsolatedSMIBatchCompiler(0.1, worker_target=_timeout_worker) as compiler:
        with pytest.raises(IsolatedSMICompilerTimeoutError):
            compiler.compile([plan])

        timed_out_process = compiler._process
        assert timed_out_process is None

        with pytest.raises(IsolatedSMICompilerTimeoutError):
            compiler.compile([plan])
        assert compiler._process is None


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_timeout_must_be_positive_and_finite(timeout: float) -> None:
    with pytest.raises(ValueError):
        IsolatedSMIBatchCompiler(timeout)
