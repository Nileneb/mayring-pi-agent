"""Background worker thread that drains pi_jobs.

Runs as a daemon thread spawned at import time by `local_mcp.py`. Polls the
`pi_jobs` table for `status='queued'` rows, atomically claims them, and runs
`run_task_with_memory()` directly (no subprocess). Results are persisted via
`complete_job()` / `fail_job()`.

A single workflow.start() guard prevents double-starting the loop when the
module is imported multiple times (e.g. tests vs runtime).
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from src.agents import pi_jobs
from src.agents.pi_jobs import PiJob

logger = logging.getLogger(__name__)


_DEFAULT_POLL_INTERVAL = 1.0
_DEFAULT_MAX_WORKERS = 2

_lock = threading.Lock()
_started: bool = False
_stop_event: threading.Event | None = None
_executor: ThreadPoolExecutor | None = None
_loop_thread: threading.Thread | None = None


def _resolve_executor() -> ThreadPoolExecutor:
    """Lazy-create a ThreadPoolExecutor; capacity comes from PI_ASYNC_WORKERS env."""
    global _executor
    if _executor is None:
        size = max(1, int(os.getenv("PI_ASYNC_WORKERS", str(_DEFAULT_MAX_WORKERS))))
        _executor = ThreadPoolExecutor(max_workers=size, thread_name_prefix="pi-worker")
        logger.info("pi_worker: ThreadPoolExecutor started (workers=%d)", size)
    return _executor


def _execute(job: PiJob) -> None:
    """Run a single job. Persists outcome via complete_job/fail_job."""
    try:
        from src.agents.pi import run_task_with_memory
        ollama = job.ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        result = run_task_with_memory(
            task=job.task_text,
            ollama_url=ollama,
            model=job.model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
            repo_slug=job.repo_slug or None,
            timeout=job.timeout_s,
        )
        pi_jobs.complete_job(job.job_id, {"text": result})
        logger.info("pi_worker: completed %s", job.job_id)
    except Exception as e:
        logger.exception("pi_worker: failed %s", job.job_id)
        pi_jobs.fail_job(job.job_id, repr(e))


def _loop(stop: threading.Event, poll_interval: float) -> None:
    """Polling loop body. Runs in a dedicated thread."""
    executor = _resolve_executor()
    while not stop.is_set():
        try:
            job = pi_jobs.claim_next()
        except Exception:
            logger.exception("pi_worker: claim_next failed")
            stop.wait(poll_interval)
            continue
        if job is None:
            stop.wait(poll_interval)
            continue
        executor.submit(_execute, job)


def start(*, poll_interval: float = _DEFAULT_POLL_INTERVAL) -> bool:
    """Start the worker loop once per process. Returns True on first start.

    Subsequent calls return False without side effects — the loop is a
    process-scoped singleton.
    """
    global _started, _stop_event, _loop_thread
    with _lock:
        if _started:
            return False
        if os.getenv("PI_ASYNC_DISABLED", "").lower() in ("1", "true", "yes"):
            logger.info("pi_worker: disabled via PI_ASYNC_DISABLED")
            _started = True
            return False
        _stop_event = threading.Event()
        _loop_thread = threading.Thread(
            target=_loop,
            args=(_stop_event, poll_interval),
            name="pi-worker-loop",
            daemon=True,
        )
        _loop_thread.start()
        _started = True
        logger.info("pi_worker: loop thread started (poll=%.1fs)", poll_interval)
        return True


def stop(timeout: float = 2.0) -> None:
    """Stop the loop and shut the executor down. Used from tests."""
    global _started, _stop_event, _executor, _loop_thread
    with _lock:
        if not _started:
            return
        if _stop_event is not None:
            _stop_event.set()
        if _loop_thread is not None:
            _loop_thread.join(timeout=timeout)
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=True)
        _stop_event = None
        _executor = None
        _loop_thread = None
        _started = False


def _is_running() -> bool:
    """For tests: peek at internal state."""
    return _started


__all__ = ("start", "stop", "_is_running")
