"""In-process Pi-Task queue with 3-lane priority routing (Issue #192 C).

Architecture:
  - 3 separate asyncio.Queues (priority / standard / background), each with
    own N worker-coroutines and concurrency cap.
  - Callers `enqueue(job)` → _lane_for(job) routes to correct queue.
  - `set_handler(callable)` injects the actual Pi-Agent worker logic
    (decoupled — keeps queue testable without real Ollama).
  - `PiQueue(concurrency=N)` backward-compat: all jobs land in standard lane.
  - `PiQueue.with_lanes(priority, standard, background)` is the 3-lane ctor.

The handler is async: `Callable[[PiJob], Awaitable[dict]]`.

# WHY(#192, scaling): 3-lane priority routing so Hook-bursts (mini)
# don't get blocked by long Background-Jobs (ingest/classify) and vice-versa.
# CHANGE WITH CARE — single-queue revert loses the fairness guarantee.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import time as _time
from typing import Awaitable, Callable

from mayring_pi_agent.pi_jobs import PiJob

_log = logging.getLogger(__name__)

_HandlerType = Callable[[PiJob], Awaitable[dict]]
# WHY(#192, asymmetric-routing): default lane sizes vor 2026-05-11 waren
# 2/2/1 (priority/standard/background) — bei drei parallelen subagents
# bottleneckte das. Neue defaults 4/4/2 = 10 slots total. OLLAMA_NUM_PARALLEL
# auf three.linn.games muss ≥4 sein damit das nicht in Ollama-seitige
# sequentialisierung läuft (siehe issue #192).
_DEFAULT_CONCURRENCY = 4
_STATS_BUFFER_SIZE = 200

_BACKGROUND_KINDS = frozenset({"ingest", "classify"})


def _make_lane(concurrency: int) -> dict:
    return {
        "queue": asyncio.Queue(),
        "concurrency": concurrency,
        "workers": [],
        "in_flight": 0,
        "completed": collections.deque(maxlen=_STATS_BUFFER_SIZE),
    }


class PiQueue:
    """Bounded-concurrency in-process queue for Pi-Tasks with 3-lane routing."""

    def __init__(self, concurrency: int = _DEFAULT_CONCURRENCY) -> None:
        # Backward-compat: legacy single-queue mode → all jobs into standard lane.
        # priority and background lanes have concurrency=0 so workers are never
        # spawned; enqueue() falls back to standard for those lanes.
        self._lanes: dict[str, dict] = {
            "priority": _make_lane(0),
            "standard": _make_lane(concurrency),
            "background": _make_lane(0),
        }
        self._handler: _HandlerType | None = None
        self._shutdown_event = asyncio.Event()
        self._pending_futures: set[asyncio.Future] = set()

    @classmethod
    def with_lanes(cls, priority: int, standard: int, background: int) -> "PiQueue":
        """Construct a PiQueue with explicit per-lane concurrency."""
        q = cls.__new__(cls)
        q._lanes = {
            "priority": _make_lane(priority),
            "standard": _make_lane(standard),
            "background": _make_lane(background),
        }
        q._handler = None
        q._shutdown_event = asyncio.Event()
        q._pending_futures = set()
        return q

    @staticmethod
    def _lane_for(job: PiJob) -> str:
        """Route a job to the appropriate lane.

        Explicit kind hint (ingest/classify) → background.
        mini job_class → priority.
        Everything else → standard.
        """
        if job.kind in _BACKGROUND_KINDS:
            return "background"
        if job.job_class == "mini":
            return "priority"
        return "standard"

    def set_handler(self, handler: _HandlerType) -> None:
        """Inject the worker logic. Must be called before `start()`."""
        self._handler = handler

    async def start(self) -> None:
        """Spawn worker-coroutines for all lanes with concurrency > 0."""
        if any(lane["workers"] for lane in self._lanes.values()):
            return
        if self._handler is None:
            raise RuntimeError("PiQueue.start() called without set_handler()")
        self._shutdown_event.clear()
        for lane_name, lane in self._lanes.items():
            lane["workers"] = [
                asyncio.create_task(self._worker_loop(lane_name, idx))
                for idx in range(lane["concurrency"])
            ]
        total = sum(l["concurrency"] for l in self._lanes.values())
        _log.info(
            "PiQueue started: priority=%d standard=%d background=%d (total=%d workers)",
            self._lanes["priority"]["concurrency"],
            self._lanes["standard"]["concurrency"],
            self._lanes["background"]["concurrency"],
            total,
        )

    def enqueue(self, job: PiJob) -> asyncio.Future:
        """Add job to the appropriate lane queue; return a resolving future."""
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_futures.add(fut)
        fut.add_done_callback(lambda f: self._pending_futures.discard(f))
        lane_name = self._lane_for(job)
        lane = self._lanes[lane_name]
        # WHY(#192): if target lane has no workers (e.g. legacy mode), fall back
        # to standard rather than silently dropping the job.
        if lane["concurrency"] == 0:
            lane = self._lanes["standard"]
        lane["queue"].put_nowait((job, fut))
        return fut

    async def _worker_loop(self, lane_name: str, worker_id: int) -> None:
        """Pull jobs from a specific lane queue, dispatch to handler."""
        lane = self._lanes[lane_name]
        while not self._shutdown_event.is_set():
            try:
                job, fut = await asyncio.wait_for(lane["queue"].get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            if fut.cancelled():
                lane["queue"].task_done()
                continue
            lane["in_flight"] += 1
            start = _time.monotonic()
            try:
                assert self._handler is not None
                result = await self._handler(job)
                job.latency_ms = int((_time.monotonic() - start) * 1000)
                if not fut.done():
                    fut.set_result(result)
            except Exception as exc:
                job.latency_ms = int((_time.monotonic() - start) * 1000)
                if not fut.done():
                    fut.set_exception(exc)
            finally:
                lane["in_flight"] -= 1
                lane["completed"].append({
                    "job_class": job.job_class,
                    "kind": job.kind,
                    "latency_ms": job.latency_ms,
                    "model_used": job.model_used,
                    "model_requested": job.model,
                })
                lane["queue"].task_done()

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel pending jobs + worker-coroutines across all lanes."""
        self._shutdown_event.set()
        for fut in list(self._pending_futures):
            if not fut.done():
                fut.cancel()
        all_workers = [w for lane in self._lanes.values() for w in lane["workers"]]
        for w in all_workers:
            w.cancel()
        await asyncio.gather(*all_workers, return_exceptions=True)
        for lane in self._lanes.values():
            lane["workers"] = []

    def stats(self) -> dict:
        """Return combined stats plus per-lane breakdown.

        Top-level keys (backward-compat): in_flight, completed_total, by_class.
        New key: lanes — dict with per-lane in_flight/concurrency/completed/p50/p95.
        """
        lanes_out: dict = {}
        all_entries: list = []
        for lane_name, lane in self._lanes.items():
            entries = list(lane["completed"])
            all_entries.extend(entries)
            latencies = sorted(e["latency_ms"] for e in entries)
            n = len(latencies)
            p50 = latencies[n // 2] if n else 0
            p95 = latencies[min(n - 1, max(0, int(n * 0.95)))] if n else 0
            lanes_out[lane_name] = {
                "in_flight": lane["in_flight"],
                "concurrency": lane["concurrency"],
                "completed": len(entries),
                "p50_ms": p50,
                "p95_ms": p95,
            }

        # Combined by_class (all lanes merged — backward-compat)
        by_class: dict = {}
        for entry in all_entries:
            cls = entry["job_class"]
            bucket = by_class.setdefault(cls, {"count": 0, "latencies": [], "fallbacks": 0})
            bucket["count"] += 1
            bucket["latencies"].append(entry["latency_ms"])
            if entry["model_requested"] and entry["model_used"] != entry["model_requested"]:
                bucket["fallbacks"] += 1
        out: dict = {}
        for cls, b in by_class.items():
            lat = sorted(b["latencies"])
            n = len(lat)
            p50 = lat[n // 2] if n else 0
            p95 = lat[min(n - 1, max(0, int(n * 0.95)))] if n else 0
            out[cls] = {
                "count": b["count"],
                "p50_ms": p50,
                "p95_ms": p95,
                "fallback_rate": round(b["fallbacks"] / b["count"], 3) if b["count"] else 0.0,
            }

        combined_in_flight = sum(lane["in_flight"] for lane in self._lanes.values())
        combined_completed = sum(len(lane["completed"]) for lane in self._lanes.values())
        return {
            "in_flight": combined_in_flight,
            "completed_total": combined_completed,
            "by_class": out,
            "lanes": lanes_out,
        }


_SINGLETON: PiQueue | None = None


def get_pi_queue() -> PiQueue:
    """Return process-singleton PiQueue with 3-lane routing.

    WHY(#192): Defaults bumped (priority 2→4, background 1→2) damit
    parallele subagent-runs nicht hinter dem standard-lane stauen. Override
    per env wenn Ollama-server kleiner ist — bei OLLAMA_NUM_PARALLEL<4
    sinken die defaults auf das ollama-cap herab (siehe issue #192).
    """
    global _SINGLETON
    if _SINGLETON is None:
        prio = int(os.getenv("PI_PRIORITY_CONCURRENCY", "4"))
        std = int(os.getenv("PI_STANDARD_CONCURRENCY",
                            os.getenv("PI_CONCURRENCY", str(_DEFAULT_CONCURRENCY))))
        bg = int(os.getenv("PI_BACKGROUND_CONCURRENCY", "2"))
        _SINGLETON = PiQueue.with_lanes(priority=prio, standard=std, background=bg)
    return _SINGLETON
