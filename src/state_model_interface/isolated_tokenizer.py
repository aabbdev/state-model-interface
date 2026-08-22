"""Compile SMI plans in a persistent, spawn-isolated tokenizer process."""

from __future__ import annotations

import math
import multiprocessing
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from queue import Empty
from threading import RLock
from typing import Any

from .compiler import (
    CompiledSMITraining,
    SMICompilationPlan,
    compile_smi_plans_batched,
    install_smi_tokens,
)
from .linear_tokenizer import LinearRWKVTokenizer

TOKENIZER = "aabbdev/RWKV7-1.5B-20260805"
TOKENIZER_REVISION = "5904f9d1cdb05a565e5da9304db0447c8a8eb938"

WorkerTarget = Callable[[Any, Any], None]


class IsolatedSMICompilerError(RuntimeError):
    """Base error raised by the isolated SMI compiler."""


class IsolatedSMICompilerTimeoutError(IsolatedSMICompilerError, TimeoutError):
    """The tokenizer worker did not answer before the hard timeout."""


class IsolatedSMIWorkerError(IsolatedSMICompilerError):
    """An exception raised in the tokenizer worker."""

    def __init__(
        self,
        remote_type: str,
        remote_message: str,
        remote_traceback: str,
    ) -> None:
        self.remote_type = remote_type
        self.remote_message = remote_message
        self.remote_traceback = remote_traceback
        super().__init__(f"tokenizer worker raised {remote_type}: {remote_message}")


class IsolatedSMICompilerClosedError(IsolatedSMICompilerError):
    """The isolated compiler has already been closed."""


@dataclass(frozen=True, slots=True)
class _SerializedError:
    type_name: str
    message: str
    traceback: str


def _serialize_error(error: BaseException) -> _SerializedError:
    error_type = type(error)
    return _SerializedError(
        f"{error_type.__module__}.{error_type.__qualname__}",
        str(error),
        "".join(traceback.format_exception(error)),
    )


def _default_worker_target(request_queue: Any, response_queue: Any) -> None:
    """Load the pinned local tokenizer once, then serve compile requests."""
    try:
        from transformers import AutoTokenizer, PreTrainedConfig

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER,
            revision=TOKENIZER_REVISION,
            config=PreTrainedConfig(),
            local_files_only=True,
            trust_remote_code=False,
        )
        token_ids = install_smi_tokens(tokenizer)
        linear_tokenizer = LinearRWKVTokenizer.from_tokenizer(tokenizer)
    except Exception as error:  # noqa: BLE001 - serialize arbitrary worker failures
        response_queue.put(("startup_error", _serialize_error(error)))
        return

    while True:
        request = request_queue.get()
        if request is None:
            return
        request_id, plans = request
        try:
            compiled = compile_smi_plans_batched(
                linear_tokenizer,
                plans,
                token_ids=token_ids,
            )
        except Exception as error:  # noqa: BLE001 - keep the worker serving
            response_queue.put(("error", request_id, _serialize_error(error)))
        else:
            response_queue.put(("result", request_id, compiled))


class IsolatedSMIBatchCompiler:
    """Context manager for a persistent, spawn-isolated SMI batch compiler.

    ``worker_target`` is an injectable spawn-safe process target intended for
    focused tests. It receives the request and response queues and must use the
    same private queue protocol as :func:`_default_worker_target`.
    """

    def __init__(
        self,
        timeout_seconds: float,
        *,
        worker_target: WorkerTarget | None = None,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and greater than zero")
        self.timeout_seconds = float(timeout_seconds)
        self._worker_target = worker_target or _default_worker_target
        self._context = multiprocessing.get_context("spawn")
        self._request_queue: Any | None = None
        self._response_queue: Any | None = None
        self._process: Any | None = None
        self._next_request_id = 0
        self._closed = False
        self._lock = RLock()

    def __enter__(self) -> IsolatedSMIBatchCompiler:  # noqa: PYI034
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def compile(self, plans: Sequence[SMICompilationPlan]) -> list[CompiledSMITraining]:
        """Compile one batch, restarting lazily after a timed-out worker."""
        with self._lock:
            if self._closed:
                raise IsolatedSMICompilerClosedError(
                    "cannot compile after the isolated compiler is closed"
                )
            self._ensure_worker()
            assert self._request_queue is not None
            assert self._response_queue is not None

            request_id = self._next_request_id
            self._next_request_id += 1
            self._request_queue.put((request_id, list(plans)))
            try:
                response = self._response_queue.get(timeout=self.timeout_seconds)
            except Empty:
                self._stop_worker()
                raise IsolatedSMICompilerTimeoutError(
                    "tokenizer worker did not respond within "
                    f"{self.timeout_seconds:g} seconds"
                ) from None

            kind = response[0]
            if kind == "startup_error":
                self._stop_worker()
                self._raise_worker_error(response[1])
            if response[1] != request_id:
                self._stop_worker()
                raise IsolatedSMICompilerError(
                    "tokenizer worker returned a mismatched request ID"
                )
            if kind == "error":
                self._raise_worker_error(response[2])
            if kind != "result":
                self._stop_worker()
                raise IsolatedSMICompilerError(
                    f"tokenizer worker returned an unknown response: {kind!r}"
                )
            return response[2]

    def close(self) -> None:
        """Stop the worker and release all multiprocessing resources."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop_worker(graceful=True)

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._stop_worker()
        self._request_queue = self._context.Queue()
        self._response_queue = self._context.Queue()
        self._process = self._context.Process(
            target=self._worker_target,
            args=(self._request_queue, self._response_queue),
            daemon=False,
            name="smi-tokenizer-worker",
        )
        try:
            self._process.start()
        except Exception:
            self._stop_worker()
            raise

    def _stop_worker(self, *, graceful: bool = False) -> None:
        process = self._process
        request_queue = self._request_queue
        response_queue = self._response_queue
        self._process = None
        self._request_queue = None
        self._response_queue = None

        if process is not None:
            if process.is_alive() and graceful and request_queue is not None:
                try:
                    request_queue.put(None)
                    process.join(timeout=0.25)
                except (OSError, ValueError):
                    pass
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.25)
            if process.is_alive():
                process.kill()
            process.join()
            process.close()

        for queue in (request_queue, response_queue):
            if queue is not None:
                if not graceful:
                    queue.cancel_join_thread()
                queue.close()
                if graceful:
                    queue.join_thread()

    @staticmethod
    def _raise_worker_error(error: _SerializedError) -> None:
        raise IsolatedSMIWorkerError(
            error.type_name,
            error.message,
            error.traceback,
        )


__all__ = [
    "IsolatedSMIBatchCompiler",
    "IsolatedSMICompilerClosedError",
    "IsolatedSMICompilerError",
    "IsolatedSMICompilerTimeoutError",
    "IsolatedSMIWorkerError",
]
