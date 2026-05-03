"""Persistent pi-task job lifecycle on top of memory.db.

Lifecycle:
    queued    — `insert_job()` writes the row, status='queued'
    running   — worker calls `claim_next()`: atomically flips one queued row
                to 'running' and returns it
    completed — worker calls `complete_job(job_id, result_json)`
    failed    — worker calls `fail_job(job_id, error)`

The atomic flip in `claim_next()` is implemented as a single
`UPDATE … WHERE status='queued' AND job_id IN (SELECT … LIMIT 1)`. SQLite
serialises writes so two concurrent workers cannot grab the same job.

This module is intentionally independent of FastMCP / LLM logic — keeps the
data layer testable in isolation.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import CACHE_DIR

VALID_STATUSES = ("queued", "running", "completed", "failed")
VALID_PREFER = ("auto", "local", "cloud")
VALID_SCOPE = ("local", "cloud")


@dataclass
class PiJob:
    job_id: str
    task_text: str
    repo_slug: str = ""
    workspace_id: str = "default"
    status: str = "queued"
    prefer: str = "auto"
    ollama_url: str = ""
    model: str = ""
    result_json: str = ""
    error: str = ""
    timeout_s: float = 180.0
    scope: str = "local"
    capability_required: str = ""
    claimed_by: str = ""
    claimed_at: str = ""
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        # Decode result_json into a real object for callers
        d["result"] = _safe_json_load(self.result_json) if self.result_json else None
        return d


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_job_id() -> str:
    raw = f"{time.time_ns()}:{secrets.token_hex(4)}"
    return "pij_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _safe_json_load(s: str) -> Any:
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return s


def _conn(db_path: Path | None = None) -> sqlite3.Connection:
    """Lightweight per-call connection — pi_jobs operations are short.

    Each call uses its own connection with row_factory=Row + WAL so reads
    and writes from the worker thread don't fight a long-lived sqlite
    handle that another module may have opened.
    """
    p = db_path or (CACHE_DIR / "memory.db")
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def _do_insert(
    *,
    task_text: str,
    repo_slug: str,
    workspace_id: str,
    prefer: str,
    ollama_url: str,
    model: str,
    timeout_s: float,
    scope: str,
    capability_required: str,
    db_path: Path | None,
) -> PiJob:
    if prefer not in VALID_PREFER:
        raise ValueError(f"prefer must be one of {VALID_PREFER}, got {prefer!r}")
    if scope not in VALID_SCOPE:
        raise ValueError(f"scope must be one of {VALID_SCOPE}, got {scope!r}")
    job = PiJob(
        job_id=_new_job_id(),
        task_text=task_text,
        repo_slug=repo_slug,
        workspace_id=workspace_id,
        prefer=prefer,
        ollama_url=ollama_url,
        model=model,
        timeout_s=float(timeout_s),
        scope=scope,
        capability_required=capability_required,
        created_at=_now_iso(),
    )
    conn = _conn(db_path)
    try:
        conn.execute(
            "INSERT INTO pi_jobs (job_id, task_text, repo_slug, workspace_id, "
            "status, prefer, ollama_url, model, timeout_s, scope, "
            "capability_required, created_at) "
            "VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)",
            (
                job.job_id, job.task_text, job.repo_slug, job.workspace_id,
                job.prefer, job.ollama_url, job.model, job.timeout_s,
                job.scope, job.capability_required, job.created_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return job


def insert_job(
    task_text: str,
    *,
    repo_slug: str = "",
    workspace_id: str = "default",
    prefer: str = "auto",
    ollama_url: str = "",
    model: str = "",
    timeout_s: float = 180.0,
    db_path: Path | None = None,
) -> PiJob:
    """Insert a LOCAL job (Phase 1 path). Worker drains via claim_next()."""
    return _do_insert(
        task_text=task_text, repo_slug=repo_slug, workspace_id=workspace_id,
        prefer=prefer, ollama_url=ollama_url, model=model, timeout_s=timeout_s,
        scope="local", capability_required="", db_path=db_path,
    )


def insert_cloud_job(
    task_text: str,
    *,
    repo_slug: str = "",
    workspace_id: str = "default",
    prefer: str = "auto",
    ollama_url: str = "",
    model: str = "",
    timeout_s: float = 180.0,
    capability_required: str = "",
    db_path: Path | None = None,
) -> PiJob:
    """Insert a CLOUD-routable job (Phase 2 path).

    `capability_required` is a comma-separated whitelist; an empty string
    matches any worker. Workers claim with `claim_cloud_next()`.
    """
    return _do_insert(
        task_text=task_text, repo_slug=repo_slug, workspace_id=workspace_id,
        prefer=prefer, ollama_url=ollama_url, model=model, timeout_s=timeout_s,
        scope="cloud", capability_required=capability_required, db_path=db_path,
    )


def claim_next(*, db_path: Path | None = None) -> PiJob | None:
    """Atomically take the oldest queued LOCAL job, flip to 'running'."""
    conn = _conn(db_path)
    try:
        # Single UPDATE … WHERE … = atomic flip; RETURNING is sqlite >= 3.35
        row = conn.execute(
            "UPDATE pi_jobs SET status='running', started_at=? "
            "WHERE job_id = (SELECT job_id FROM pi_jobs "
            "                WHERE status='queued' AND scope='local' "
            "                ORDER BY created_at LIMIT 1) "
            "RETURNING *",
            (_now_iso(),),
        ).fetchone()
        conn.commit()
    finally:
        conn.close()
    return _row_to_job(row) if row else None


def _capability_match(capability_required: str, capabilities: list[str]) -> bool:
    if not capability_required:
        return True
    needed = {c.strip() for c in capability_required.split(",") if c.strip()}
    return needed.issubset(set(capabilities))


def claim_cloud_next(
    worker_id: str,
    capabilities: list[str] | None = None,
    *,
    workspace_id: str | None = None,
    db_path: Path | None = None,
) -> PiJob | None:
    """Atomically take the oldest queued CLOUD job whose capability_required
    is satisfied by `capabilities`.

    Strategy: SELECT candidates with the static workspace + scope filters in
    SQL, filter capability in Python, then issue a targeted UPDATE against
    the chosen job_id with `status='queued'` as a guard. Two concurrent
    claimers can't both win the same row.
    """
    if not worker_id:
        raise ValueError("worker_id required for cloud claim")
    caps = list(capabilities or [])
    conn = _conn(db_path)
    try:
        if workspace_id:
            rows = conn.execute(
                "SELECT * FROM pi_jobs WHERE status='queued' AND scope='cloud' "
                "AND workspace_id=? ORDER BY created_at LIMIT 20",
                (workspace_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM pi_jobs WHERE status='queued' AND scope='cloud' "
                "ORDER BY created_at LIMIT 20",
            ).fetchall()
        for r in rows:
            if not _capability_match(r["capability_required"] or "", caps):
                continue
            now = _now_iso()
            updated = conn.execute(
                "UPDATE pi_jobs SET status='running', started_at=?, "
                "claimed_by=?, claimed_at=? "
                "WHERE job_id=? AND status='queued' "
                "RETURNING *",
                (now, worker_id, now, r["job_id"]),
            ).fetchone()
            if updated is None:
                continue  # lost the race, try the next candidate
            conn.commit()
            return _row_to_job(updated)
        return None
    finally:
        conn.close()


def complete_job(job_id: str, result: Any, *, db_path: Path | None = None) -> None:
    """Mark a job completed with the JSON-encodable `result` payload."""
    payload = json.dumps(result) if not isinstance(result, str) else result
    conn = _conn(db_path)
    try:
        conn.execute(
            "UPDATE pi_jobs SET status='completed', result_json=?, finished_at=? "
            "WHERE job_id=? AND status='running'",
            (payload, _now_iso(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def fail_job(job_id: str, error: str, *, db_path: Path | None = None) -> None:
    """Mark a job failed with a short error string."""
    conn = _conn(db_path)
    try:
        conn.execute(
            "UPDATE pi_jobs SET status='failed', error=?, finished_at=? "
            "WHERE job_id=? AND status IN ('queued', 'running')",
            (str(error)[:1000], _now_iso(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_job(
    job_id: str,
    *,
    workspace_id: str | None = None,
    db_path: Path | None = None,
) -> PiJob | None:
    """Read a single job. When `workspace_id` is set, only return the row when
    it belongs to that tenant — otherwise return None (treat foreign jobs as
    "not found" so a caller cannot probe for their existence)."""
    conn = _conn(db_path)
    try:
        if workspace_id:
            row = conn.execute(
                "SELECT * FROM pi_jobs WHERE job_id=? AND workspace_id=?",
                (job_id, workspace_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM pi_jobs WHERE job_id=?", (job_id,),
            ).fetchone()
    finally:
        conn.close()
    return _row_to_job(row) if row else None


def list_recent(
    *,
    only_active: bool = False,
    scope: str | None = None,
    workspace_id: str | None = None,
    limit: int = 10,
    db_path: Path | None = None,
) -> list[PiJob]:
    """Return recent jobs. When `workspace_id` is set, the result is scoped
    to that tenant — every MCP-facing caller MUST pass it. Internal callers
    (worker loops) may omit it deliberately."""
    where = []
    params: list = []
    if only_active:
        where.append("status IN ('queued', 'running')")
    if scope:
        where.append("scope=?")
        params.append(scope)
    if workspace_id:
        where.append("workspace_id=?")
        params.append(workspace_id)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    params.append(limit)
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            f"SELECT * FROM pi_jobs{where_sql} "
            f"ORDER BY created_at DESC LIMIT ?",
            tuple(params),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_job(r) for r in rows]


def _row_to_job(row: sqlite3.Row) -> PiJob:
    return PiJob(
        job_id=row["job_id"],
        task_text=row["task_text"],
        repo_slug=row["repo_slug"] or "",
        workspace_id=row["workspace_id"] or "default",
        status=row["status"],
        prefer=row["prefer"] or "auto",
        ollama_url=row["ollama_url"] or "",
        model=row["model"] or "",
        result_json=row["result_json"] or "",
        error=row["error"] or "",
        timeout_s=float(row["timeout_s"] or 180.0),
        scope=row["scope"] or "local",
        capability_required=row["capability_required"] or "",
        claimed_by=row["claimed_by"] or "",
        claimed_at=row["claimed_at"] or "",
        created_at=row["created_at"],
        started_at=row["started_at"] or "",
        finished_at=row["finished_at"] or "",
    )


__all__ = (
    "PiJob",
    "VALID_PREFER",
    "VALID_SCOPE",
    "VALID_STATUSES",
    "insert_job",
    "insert_cloud_job",
    "claim_next",
    "claim_cloud_next",
    "complete_job",
    "fail_job",
    "get_job",
    "list_recent",
)
