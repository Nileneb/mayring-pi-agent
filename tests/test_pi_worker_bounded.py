"""Bounded cloud-claiming invariants for pi_worker._claim_one.

Regression guard for the 2026-05-24 prod outage: the cloud loop used to claim
a job and immediately loop for the next with no pacing, draining a backlog at
full HTTP speed + firing unbounded concurrent job callbacks → single-core API
CPU-pegged, memory hooks starved. _claim_one gates every claim on a free worker
slot.
"""
import threading

from mayring_pi_agent import pi_worker

_JOB = {"job": {"job_id": "j1", "task_text": "do x"}}


def test_busy_capacity_does_not_claim():
    """At capacity → no claim is issued at all (this is the anti-flood guard)."""
    slots = threading.Semaphore(1)
    slots.acquire()  # exhaust capacity
    posts, waited = [], []
    outcome = pi_worker._claim_one(
        slots=slots,
        post_fn=lambda p, b: posts.append(p) or _JOB,
        submit_fn=lambda *a, **k: None,
        worker_id="w", caps=[],
        on_complete=lambda *a: None,
        idle_wait=lambda: waited.append(1),
    )
    assert outcome == "busy"
    assert posts == []          # CRITICAL: no /pi_task_claim_cloud when full
    assert waited == [1]


def test_dispatch_consumes_one_slot():
    slots = threading.Semaphore(2)
    submitted, posts = [], []

    def fake_submit(fn, job, *, on_cloud_complete):
        submitted.append((job, on_cloud_complete))

    outcome = pi_worker._claim_one(
        slots=slots,
        post_fn=lambda p, b: posts.append(p) or _JOB,
        submit_fn=fake_submit,
        worker_id="w", caps=[],
        on_complete=lambda *a: None,
        idle_wait=lambda: None,
    )
    assert outcome == "dispatched"
    assert posts == ["/pi_task_claim_cloud"]
    assert len(submitted) == 1
    assert slots.acquire(blocking=False) is True    # 1 of 2 left
    assert slots.acquire(blocking=False) is False   # in-flight job holds the other


def test_empty_queue_releases_slot():
    slots = threading.Semaphore(1)
    outcome = pi_worker._claim_one(
        slots=slots,
        post_fn=lambda p, b: None,        # empty queue
        submit_fn=lambda *a, **k: None,
        worker_id="w", caps=[],
        on_complete=lambda *a: None,
        idle_wait=lambda: None,
    )
    assert outcome == "empty"
    assert slots.acquire(blocking=False) is True     # slot not leaked


def test_completion_releases_slot_and_calls_through():
    slots = threading.Semaphore(1)
    captured, completed = {}, []

    def fake_submit(fn, job, *, on_cloud_complete):
        captured["cb"] = on_cloud_complete

    pi_worker._claim_one(
        slots=slots,
        post_fn=lambda p, b: _JOB,
        submit_fn=fake_submit,
        worker_id="w", caps=[],
        on_complete=lambda *a: completed.append(a),
        idle_wait=lambda: None,
    )
    assert slots.acquire(blocking=False) is False    # slot held by in-flight job
    captured["cb"]("j1", {"text": "ok"}, None)       # simulate job finish
    assert completed == [("j1", {"text": "ok"}, None)]
    assert slots.acquire(blocking=False) is True      # slot freed by completion


def test_submit_failure_releases_slot():
    slots = threading.Semaphore(1)

    def boom(*a, **k):
        raise RuntimeError("executor down")

    try:
        pi_worker._claim_one(
            slots=slots,
            post_fn=lambda p, b: _JOB,
            submit_fn=boom,
            worker_id="w", caps=[],
            on_complete=lambda *a: None,
            idle_wait=lambda: None,
        )
    except RuntimeError:
        pass
    assert slots.acquire(blocking=False) is True      # slot not leaked on submit failure


def test_next_idle_delay_backoff_and_reset():
    """Empty queue → exponential backoff capped; work/capacity → reset to base."""
    base, cap = 5.0, 30.0
    # consecutive empties escalate, capped at `cap`
    d = base
    d = pi_worker._next_idle_delay("empty", d, base, cap); assert d == 10.0
    d = pi_worker._next_idle_delay("empty", d, base, cap); assert d == 20.0
    d = pi_worker._next_idle_delay("empty", d, base, cap); assert d == 30.0  # capped
    d = pi_worker._next_idle_delay("empty", d, base, cap); assert d == 30.0  # stays capped
    # work resets immediately
    assert pi_worker._next_idle_delay("dispatched", 30.0, base, cap) == base
    # capacity pressure (busy) also resets so a freed slot polls promptly
    assert pi_worker._next_idle_delay("busy", 30.0, base, cap) == base
