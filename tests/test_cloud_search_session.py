"""Recency-lane wiring: the pi-agent's cloud memory search must forward the
controlling session_id so the server-side recency-lane surfaces the latest
session decisions (not just topically-similar older chunks). num_predict must
be configurable so stronger reasoners (Gemma) don't truncate mid-thought.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

from mayring_pi_agent import pi


class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()
    def read(self):
        return self._b
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_cloud_search_forwards_session_id(tmp_path, monkeypatch):
    jwt = tmp_path / "hook.jwt"
    jwt.write_text("tok")
    monkeypatch.setattr(pi, "_MEMORY_JWT_FILE", str(jwt))
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp({"results": [{"source_id": "conversation:x:sess", "score_final": 0.5,
                                       "text": "the recent fix", "chunk_id": "c1"}]})

    with patch.object(pi.urllib.request, "urlopen", side_effect=_fake_urlopen):
        out = pi._cloud_search("what was fixed", top_k=5, repo="mayringcoder",
                               session_id="8d17c95d-8482-4294")

    assert captured["body"]["session_id"] == "8d17c95d-8482-4294"   # recency-lane activated
    assert captured["body"]["query"] == "what was fixed"
    assert "recent fix" in out


def test_cloud_search_omits_session_id_when_absent(tmp_path, monkeypatch):
    jwt = tmp_path / "hook.jwt"
    jwt.write_text("tok")
    monkeypatch.setattr(pi, "_MEMORY_JWT_FILE", str(jwt))
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp({"results": []})

    with patch.object(pi.urllib.request, "urlopen", side_effect=_fake_urlopen):
        pi._cloud_search("q", top_k=3, repo=None)

    assert "session_id" not in captured["body"]   # back-compat: no key when unset


def test_cloud_search_prioritises_session_thread_in_budget(tmp_path, monkeypatch):
    """A strong semantic match must NOT crowd the session thread out of the
    formatted context: session-recency chunks lead, even with a lower score."""
    jwt = tmp_path / "hook.jwt"
    jwt.write_text("tok")
    monkeypatch.setattr(pi, "_MEMORY_JWT_FILE", str(jwt))

    server_results = {
        "results": [
            {"source_id": "repo:x:strong.py", "score_final": 0.95,
             "text": "A" * 1700, "chunk_id": "c0", "reasons": ["embedding_similarity"]},
            {"source_id": "conversation:ws:sess", "score_final": 0.50,
             "text": "SESSION THREAD what I am doing now", "chunk_id": "c1",
             "reasons": ["recent_chunk", "session-recency"]},
        ]
    }

    def _fake_urlopen(req, timeout=None):
        return _FakeResp(server_results)

    with patch.object(pi.urllib.request, "urlopen", side_effect=_fake_urlopen):
        out = pi._cloud_search("q", top_k=5, repo=None, char_budget=1800,
                               session_id="sess")

    # the session thread must appear (and lead), not be truncated by the 0.95 chunk
    assert "SESSION THREAD" in out
    assert out.index("SESSION THREAD") < (out.index("AAAA") if "AAAA" in out else len(out))


def test_num_predict_env_default(monkeypatch):
    """PI_NUM_PREDICT env raises the reasoning budget without a code change."""
    monkeypatch.setenv("PI_NUM_PREDICT", "6144")
    captured = {}

    def _fake_loop(**kw):
        captured.update(kw)
        return ("done", 0)

    # disable_memory=True → skips ambient/db; we only assert the budget threading
    with patch.object(pi, "_agent_loop", side_effect=_fake_loop), \
         patch.object(pi, "init_memory_db", MagicMock(return_value=MagicMock())), \
         patch.object(pi, "get_or_create_chroma_collection", MagicMock(return_value=MagicMock())):
        pi.run_task_with_memory("task", "http://localhost:11434", "gemma4:e4b",
                                disable_memory=True, session_id="sess-42")

    assert captured["num_predict"] == 6144
    assert captured["session_id"] == "sess-42"
