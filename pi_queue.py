"""In-process Pi-Task queue with bounded concurrency (Issue #183 T2).

Architecture:
  - asyncio.Queue holds (PiJob, asyncio.Future)-tuples
  - N worker-coroutines pull from the queue, dispatch to a registered
    handler, set the future's result/exception.
  - Caller `enqueue(job)` returns the future; await for sync-feeling
    behaviour while concurrency stays bounded.
  - `set_handler(callable)` injects the actual Pi-Agent worker logic
    (decoupled — keeps queue testable without real Ollama).

The handler is async: `Callable[[PiJob], Awaitable[dict]]`.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from src.agents.pi_jobs import PiJob

_log = logging.getLogger(__name__)

_HandlerType = Callable[[PiJob], Awaitable[dict]]
_DEFAULT_CONCURRENCY = 2


class PiQueue:
    """Bounded-concurrency in-process queue for Pi-Tasks."""

    def __init__(self, concurrency: int = _DEFAULT_CONCURRENCY) -> None:
        self._concurrency = concurrency
        self._queue: asyncio.Queue[tuple[PiJob, asyncio.Future]] = asyncio.Queue()
        self._handler: _HandlerType | None = None
        self._workers: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._pending_futures: set[asyncio.Future] = set()

    def set_handler(self, handler: _HandlerType) -> None:
        """Inject the worker logic. Must be called before `start()`."""
        self._handler = handler

    async def start(self) -> None:
        """Spawn N worker-coroutines."""
        if self._workers:
            return
        if self._handler is None:
            raise RuntimeError("PiQueue.start() called without set_handler()")
        self._shutdown_event.clear()
        self._workers = [
            asyncio.create_task(self._worker_loop(idx))
            for idx in range(self._concurrency)
        ]
        _log.info("PiQueue started with concurrency=%d", self._concurrency)

    def enqueue(self, job: PiJob) -> asyncio.Future:
        """Add job to queue, return a future that resolves with the result."""
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_futures.add(fut)
        fut.add_done_callback(lambda f: self._pending_futures.discard(f))
        self._queue.put_nowait((job, fut))
        return fut

    async def _worker_loop(self, worker_id: int) -> None:
        """Pull jobs from queue, dispatch to handler, set future result."""
        while not self._shutdown_event.is_set():
            try:
                job, fut = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1,
                )
            except asyncio.TimeoutError:
                continue
            if fut.cancelled():
                self._queue.task_done()
                continue
            try:
                assert self._handler is not None
                result = await self._handler(job)
                if not fut.done():
                    fut.set_result(result)
            except Exception as exc:
                if not fut.done():
                    fut.set_exception(exc)
            finally:
                self._queue.task_done()

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel pending jobs + worker-coroutines."""
        self._shutdown_event.set()
        for fut in list(self._pending_futures):
            if not fut.done():
                fut.cancel()
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []


_SINGLETON: PiQueue | None = None


def get_pi_queue() -> PiQueue:
    """Return process-singleton PiQueue. Create on first access."""
    global _SINGLETON
    if _SINGLETON is None:
        import os
        concurrency = int(os.getenv("PI_CONCURRENCY", str(_DEFAULT_CONCURRENCY)))
        _SINGLETON = PiQueue(concurrency=concurrency)
    return _SINGLETON
