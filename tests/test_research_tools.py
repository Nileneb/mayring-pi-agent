import json
from unittest.mock import patch

from mayring_pi_agent import pi


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
