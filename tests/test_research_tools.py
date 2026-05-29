import json
import sqlite3
from unittest.mock import patch

from mayring_pi_agent import pi, pi_jobs


def _init_db(path):
    from mayring_core.memory.store import init_memory_db
    init_memory_db(path).close()
    return path


def test_fail_stale_cloud_jobs_marks_old_queued(tmp_path):
    db = _init_db(tmp_path / "jobs.db")
    j = pi_jobs.insert_cloud_job("alt", capability_required="research", db_path=db)
    with sqlite3.connect(db) as c:
        c.execute("UPDATE pi_jobs SET created_at=? WHERE job_id=?",
                  ("2000-01-01T00:00:00+00:00", j.job_id))
    n = pi_jobs.fail_stale_cloud_jobs(max_age_s=600, db_path=db)
    assert n == 1
    assert pi_jobs.get_job(j.job_id, db_path=db).status == "failed"


def test_insert_cloud_job_honours_explicit_job_id(tmp_path):
    db = _init_db(tmp_path / "jobs.db")
    j = pi_jobs.insert_cloud_job("x", capability_required="research",
                                 job_id="a2a-task-123", db_path=db)
    assert j.job_id == "a2a-task-123"
    assert pi_jobs.get_job("a2a-task-123", db_path=db).task_text == "x"


def test_fail_stale_cloud_jobs_keeps_fresh(tmp_path):
    db = _init_db(tmp_path / "jobs.db")
    j = pi_jobs.insert_cloud_job("frisch", capability_required="research", db_path=db)
    n = pi_jobs.fail_stale_cloud_jobs(max_age_s=600, db_path=db)
    assert n == 0
    assert pi_jobs.get_job(j.job_id, db_path=db).status == "queued"


def test_web_search_returns_formatted_results(monkeypatch):
    monkeypatch.setenv("MAYRING_API_URL", "https://mcp.linn.games")
    fake = {"results": [
        {"title": "A2A spec", "url": "https://example.org/a2a", "content": "Agent2Agent protocol"},
        {"title": "SearXNG", "url": "https://example.org/sx", "content": "meta search"},
    ]}

    class _Resp:
        status = 200

        def read(self, *_):
            return json.dumps(fake).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch("mayring_pi_agent.pi.urllib.request.urlopen", return_value=_Resp()):
        out = pi._execute_web_search("a2a protocol")
    assert "A2A spec" in out and "https://example.org/a2a" in out
    assert "Agent2Agent protocol" in out


def test_web_fetch_allowlist_wildcard_allows_any_domain():
    assert pi._domain_allowed("https://anything.example.com/x", ["*"]) is True
    assert pi._domain_allowed("https://anything.example.com/x", ["github.com"]) is False


def test_ingest_posts_to_cloud_and_reports_ok(monkeypatch):
    monkeypatch.setenv("MAYRING_API_URL", "https://mcp.linn.games")
    captured = {}

    class _Resp:
        status = 200

        def read(self, *_):
            return b'{"ingested": 1}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = req.data
        return _Resp()

    with patch("mayring_pi_agent.pi.urllib.request.urlopen", side_effect=_fake_urlopen):
        out = pi._execute_ingest("Research X", "Findings: ...")
    assert "/ingest" in captured["url"]
    assert b"Research X" in captured["body"]
    assert "ok" in out.lower() or "ingest" in out.lower()
