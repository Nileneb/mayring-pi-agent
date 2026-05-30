"""Background worker thread that drains pi_jobs (local + cloud).

Runs as a daemon thread spawned at import time by `local_mcp.py`. Two modes:

- LOCAL (default): polls `pi_jobs` table directly, claims `scope='local'`
  rows with `claim_next()`, runs `run_task_with_memory()` in the
  ThreadPoolExecutor.
- CLOUD (default ON, opt out via PI_CLOUD_POLLING=false): in addition to
  local, calls the cloud MCP `pi_task_claim_cloud` over HTTP with the
  persistent worker_id and the configured capabilities. Successfully
  claimed cloud jobs run the same way as local; result is reported back
  via `pi_task_complete_cloud`.

A single `start()` guard prevents double-starting the loop when the module
is imported multiple times.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mayring_pi_agent import pi_jobs
from mayring_pi_agent.pi_jobs import PiJob

logger = logging.getLogger(__name__)


_DEFAULT_POLL_INTERVAL = 1.0
_DEFAULT_CLOUD_POLL_INTERVAL = 5.0
_DEFAULT_MAX_WORKERS = 2
_DEFAULT_CAPABILITIES = ["local-gpu"]
_WORKER_ID_FILE = Path.home() / ".config" / "mayring" / "worker_id"

_lock = threading.Lock()
_started: bool = False
_stop_event: threading.Event | None = None
_executor: ThreadPoolExecutor | None = None
_loop_thread: threading.Thread | None = None
_cloud_thread: threading.Thread | None = None


_OLLAMA_CONFIG_FILE = Path.home() / ".config" / "mayring" / "ollama.conf"


def _resolve_ollama_url(job_url: str = "") -> str:
    """Determine the Ollama URL for a single job execution.

    Three layers of override, highest priority first — every layer can be
    changed at runtime (no restart needed):

      1. **Per-job:** `job_url` — caller explicitly chose a backend at submit
         time (`pi_task_start(..., ollama_url="https://three.linn.games")`).
      2. **Runtime-mutable config file:** the first non-empty line of
         `~/.config/mayring/ollama.conf` (or `$MAYRING_OLLAMA_CONFIG`). Read
         fresh on every job, so `echo … > ollama.conf` flips backends mid-
         session without restarting the MCP server.
      3. **Process-start env:** `OLLAMA_URL` — set when the worker booted.
      4. **Hard default:** `http://localhost:11434` (the worker's own GPU).

    Empty / unreachable layers fall through to the next one. Nothing is
    enforced — the worker honours the user's explicit choice. Tenant
    scoping in the queue is the boundary; this resolver does not impose
    a second one.
    """
    if job_url:
        return job_url
    config_path = Path(os.getenv("MAYRING_OLLAMA_CONFIG", str(_OLLAMA_CONFIG_FILE)))
    if config_path.is_file():
        try:
            for line in config_path.read_text().splitlines():
                value = line.strip()
                if value and not value.startswith("#"):
                    return value
        except OSError:
            pass
    return os.getenv("OLLAMA_URL", "http://localhost:11434")


def _resolve_executor() -> ThreadPoolExecutor:
    """Lazy-create a ThreadPoolExecutor; capacity comes from PI_ASYNC_WORKERS env."""
    global _executor
    if _executor is None:
        size = max(1, int(os.getenv("PI_ASYNC_WORKERS", str(_DEFAULT_MAX_WORKERS))))
        _executor = ThreadPoolExecutor(max_workers=size, thread_name_prefix="pi-worker")
        logger.info("pi_worker: ThreadPoolExecutor started (workers=%d)", size)
    return _executor


def _worker_id() -> str:
    """Persist a UUID per-device under ~/.config/mayring/worker_id."""
    try:
        if _WORKER_ID_FILE.is_file():
            wid = _WORKER_ID_FILE.read_text().strip()
            if wid:
                return wid
        _WORKER_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        wid = "wkr_" + uuid.uuid4().hex[:12]
        _WORKER_ID_FILE.write_text(wid)
        return wid
    except OSError:
        return "wkr_" + uuid.uuid4().hex[:12]


def _capabilities() -> list[str]:
    raw = os.getenv("PI_WORKER_CAPABILITIES", "")
    caps = [c.strip() for c in raw.split(",") if c.strip()] if raw.strip() else list(_DEFAULT_CAPABILITIES)
    # WHY(SECURITY): advertised capabilities MUST match the tools the agent
    # actually has, or the cloud routes a write/exec job to a worker that then
    # rejects it. Derive from the SAME env flags that gate the tools in pi.py.
    from mayring_pi_agent.pi import exec_enabled, write_enabled
    if write_enabled() and "write" not in caps:
        caps.append("write")
    if exec_enabled() and "exec" not in caps:
        caps.append("exec")
    return caps


def _execute(job: PiJob, *, on_cloud_complete=None) -> None:
    """Run a single job. Persists outcome via complete_job/fail_job.

    For cloud-scoped jobs `on_cloud_complete(job_id, result, error)` is
    invoked so the worker can also notify the cloud queue (in addition to
    writing to the local mirror).
    """
    try:
        from mayring_pi_agent.pi import run_task_with_memory
        from mayring_core.model_router import ModelRouter
        # Resolve the Ollama URL for THIS execution. The caller can swap
        # backends at runtime via three layers — see _resolve_ollama_url
        # for the precedence rules.
        ollama = _resolve_ollama_url(job.ollama_url)
        # Issue #88: ModelRouter is the only allowed model resolution path.
        # Reading OLLAMA_MODEL from env bypasses /stats/admin/model-routes
        # overrides so users can no longer change the model at runtime.
        chosen_model = (
            job.model
            or ModelRouter(ollama).resolve("text")
            or "qwen2.5-coder:7b"
        )
        result = run_task_with_memory(
            task=job.task_text,
            ollama_url=ollama,
            model=chosen_model,
            repo_slug=job.repo_slug or None,
            timeout=job.timeout_s,
        )
        pi_jobs.complete_job(job.job_id, {"text": result})
        if on_cloud_complete is not None:
            on_cloud_complete(job.job_id, {"text": result}, None)
        logger.info("pi_worker: completed %s (scope=%s)", job.job_id, job.scope)
    except Exception as e:
        logger.exception("pi_worker: failed %s", job.job_id)
        pi_jobs.fail_job(job.job_id, repr(e))
        if on_cloud_complete is not None:
            on_cloud_complete(job.job_id, None, repr(e))


def _loop(stop: threading.Event, poll_interval: float) -> None:
    """LOCAL polling loop. Runs in its own thread."""
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


def _claim_one(
    *,
    slots: threading.Semaphore,
    post_fn,
    submit_fn,
    worker_id: str,
    caps: list[str],
    on_complete,
    idle_wait,
) -> str:
    """One bounded claim attempt. Returns 'busy' | 'empty' | 'dispatched'.

    WHY(prod-outage 2026-05-24): the old loop claimed a job and immediately
    looped to claim the next with NO pacing — so a backlog was drained at full
    HTTP speed, flooding /pi_task_claim_cloud AND firing unbounded concurrent
    job callbacks at the single-core API → CPU pegged, every other request
    (memory hooks) starved. We now gate each claim on a free worker slot:
    concurrency is capped at the executor size and the claim rate can never
    outrun job completion. The slot is released by the wrapped completion
    callback (or eagerly on busy/empty/submit-failure) so it can't leak.
    """
    if not slots.acquire(blocking=False):
        idle_wait()
        return "busy"
    resp = post_fn(
        "/pi_task_claim_cloud",
        {"worker_id": worker_id, "capabilities": caps},
    )
    if resp is None or not resp.get("job"):
        slots.release()
        idle_wait()
        return "empty"
    cj = resp["job"]
    job = PiJob(
        job_id=cj["job_id"],
        task_text=cj["task_text"],
        repo_slug=cj.get("repo_slug", ""),
        ollama_url=cj.get("ollama_url", ""),
        model=cj.get("model", ""),
        timeout_s=float(cj.get("timeout_s", 180.0)),
        scope="cloud",
        capability_required=cj.get("capability_required", ""),
        claimed_by=worker_id,
        claimed_at=cj.get("claimed_at", ""),
        status="running",
    )

    def _release_after(job_id, result, error):
        try:
            on_complete(job_id, result, error)
        finally:
            slots.release()

    try:
        submit_fn(_execute, job, on_cloud_complete=_release_after)
    except Exception:
        slots.release()
        raise
    return "dispatched"


def _cloud_loop(stop: threading.Event, poll_interval: float) -> None:
    """CLOUD polling loop. Calls cloud MCP via HTTP to claim jobs.

    Active by default; opt out via PI_CLOUD_POLLING=false. Reads
    MAYRING_API_URL + hook.jwt for auth — same pattern as the postcompact
    + memory_sync hooks.
    """
    api = os.getenv("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
    jwt_path = os.getenv("MAYRING_HOOK_JWT") or str(
        Path.home() / ".config" / "mayring" / "hook.jwt"
    )
    try:
        token = Path(jwt_path).read_text().strip()
    except OSError:
        logger.warning("pi_worker: cloud-polling enabled but no JWT at %s", jwt_path)
        return
    if not token:
        logger.warning("pi_worker: cloud-polling enabled but JWT empty")
        return

    worker_id = _worker_id()
    caps = _capabilities()
    logger.info(
        "pi_worker: cloud-polling started (worker_id=%s, capabilities=%s)",
        worker_id, caps,
    )
    executor = _resolve_executor()

    def _post(path: str, body: dict, timeout: float = 10.0) -> dict | None:
        # Trusted target: api comes from a known env var, body is module-built.
        req = urllib.request.Request(
            f"{api}{path}",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                return json.loads(resp.read())
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            return None

    # Self-register so the registry's capabilities match what we advertise.
    # WHY(research-relay): effective_capabilities() lets the REGISTRY win over the
    # claim body — a stale registration (e.g. an old run that registered ['local-gpu'])
    # silently gates every claim, so the worker polls forever without ever matching
    # a capability_required job. Re-registering on startup keeps registry == caps.
    if _post("/devices/register", {"device_id": worker_id, "capabilities": caps}) is None:
        logger.warning(
            "pi_worker: device self-register failed — claims may be gated by stale "
            "registry caps until /devices/register succeeds",
        )

    def _on_cloud_complete(job_id: str, result, error) -> None:
        body = {"job_id": job_id}
        if error is not None:
            body["error"] = error
        else:
            body["result"] = result
        _post("/pi_task_complete_cloud", body)

    capacity = max(1, int(os.getenv("PI_ASYNC_WORKERS", str(_DEFAULT_MAX_WORKERS))))
    slots = threading.Semaphore(capacity)
    while not stop.is_set():
        _claim_one(
            slots=slots,
            post_fn=_post,
            submit_fn=executor.submit,
            worker_id=worker_id,
            caps=caps,
            on_complete=_on_cloud_complete,
            idle_wait=lambda: stop.wait(poll_interval),
        )


def _ensure_schema() -> None:
    """Run the idempotent migration on the DB the worker will read.

    pi_jobs.claim_next/get_job default to MEMORY_DB_PATH — distinct from
    MAYRING_LOCAL_DB which local_mcp.py initialises. Without this call,
    the worker thread immediately hits "no such table: pi_jobs" because
    nothing has run init_memory_db() against that path. Idempotent: a
    fully-migrated DB short-circuits in milliseconds.

    The DB path is imported from mayring_core.memory.store as the single source of
    truth — nothing here hardcodes the location.

    Errors are narrowed to sqlite3.Error (corrupt / unwritable DB) and
    OSError (permissions, disk full). We re-raise so a fresh-install
    misconfiguration surfaces at startup, not 1s later as a confusing
    "no such table" inside the loop.
    """
    import sqlite3
    from mayring_core.memory.store import init_memory_db
    # WHY(dual-DB footgun): init the SAME DB pi_jobs reads/writes (honours
    # MAYRING_LOCAL_DB) instead of the bare MEMORY_DB_PATH — otherwise in
    # local-MCP mode the worker migrates one DB while claiming from another.
    db_path = pi_jobs._effective_db_path()
    try:
        init_memory_db(db_path).close()
    except (sqlite3.Error, OSError):
        logger.exception(
            "pi_worker: schema init failed at %s — fix the DB path or "
            "permissions and restart the MCP server", db_path,
        )
        raise


def start(
    *,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
    cloud_poll_interval: float = _DEFAULT_CLOUD_POLL_INTERVAL,
) -> bool:
    """Start the worker loops once per process. Returns True on first start.

    Always starts the LOCAL loop. CLOUD loop is on by default; opt out
    with PI_CLOUD_POLLING=false. Cloud loop also requires a hook JWT.
    """
    global _started, _stop_event, _loop_thread, _cloud_thread
    with _lock:
        if _started:
            return False
        if os.getenv("PI_ASYNC_DISABLED", "").lower() in ("1", "true", "yes"):
            logger.info("pi_worker: disabled via PI_ASYNC_DISABLED")
            _started = True
            return False
        _ensure_schema()
        _stop_event = threading.Event()
        _loop_thread = threading.Thread(
            target=_loop,
            args=(_stop_event, poll_interval),
            name="pi-worker-loop",
            daemon=True,
        )
        _loop_thread.start()
        # Cloud-queue polling default ON. The pi-worker container exists
        # specifically to drain the cloud pi_jobs queue — if we leave it
        # off, every `pi_task_submit_cloud` call enqueues forever. Local
        # dev (no cloud queue / no MAYRING_API_URL) opts out via
        # PI_CLOUD_POLLING=false.
        if os.getenv("PI_CLOUD_POLLING", "true").lower() not in ("0", "false", "no"):
            _cloud_thread = threading.Thread(
                target=_cloud_loop,
                args=(_stop_event, cloud_poll_interval),
                name="pi-worker-cloud",
                daemon=True,
            )
            _cloud_thread.start()
        _started = True
        logger.info(
            "pi_worker: started (local poll=%.1fs, cloud=%s)",
            poll_interval, "on" if _cloud_thread is not None else "off",
        )
        return True


def stop(timeout: float = 2.0) -> None:
    """Stop the loops and shut the executor down. Used from tests."""
    global _started, _stop_event, _executor, _loop_thread, _cloud_thread
    with _lock:
        if not _started:
            return
        if _stop_event is not None:
            _stop_event.set()
        for t in (_loop_thread, _cloud_thread):
            if t is not None:
                t.join(timeout=timeout)
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=True)
        _stop_event = None
        _executor = None
        _loop_thread = None
        _cloud_thread = None
        _started = False


def _is_running() -> bool:
    """For tests: peek at internal state."""
    return _started


def main() -> None:
    """Run the worker as a foreground process (systemd / `python -m`).

    start() spawns daemon threads, so block here until interrupted.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    start()
    logger.info("pi_worker: foreground run — caps=%s, worker_id=%s", _capabilities(), _worker_id())
    forever = threading.Event()
    try:
        forever.wait()
    except KeyboardInterrupt:
        stop()


__all__ = ("start", "stop", "main", "_is_running")


if __name__ == "__main__":
    main()
