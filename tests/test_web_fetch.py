"""web_fetch read-tool tests (#211) — Allow-List ist eine SECURITY-Boundary
(deny-all by default), darum direkt getestet. Kein Ollama, kein Netz (urlopen
gemockt)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from mayring_pi_agent import pi


@pytest.fixture(autouse=True)
def _clear_cache():
    pi._WEB_FETCH_CACHE.clear()
    yield
    pi._WEB_FETCH_CACHE.clear()


def test_domain_allowed_matches_host_and_subdomains():
    allow = ["github.com", "docs.python.org"]
    assert pi._domain_allowed("https://github.com/x", allow)
    assert pi._domain_allowed("https://api.github.com/x", allow)
    assert not pi._domain_allowed("https://notgithub.com/x", allow)


def test_non_http_scheme_rejected():
    assert "nur http(s)" in pi._execute_web_fetch("file:///etc/passwd")


def test_deny_when_no_allowlist(monkeypatch):
    monkeypatch.delenv("PI_WEB_FETCH_ALLOWLIST", raising=False)
    assert "keine Allow-List" in pi._execute_web_fetch("https://github.com/x")


def test_deny_domain_not_in_allowlist(monkeypatch):
    monkeypatch.setenv("PI_WEB_FETCH_ALLOWLIST", "github.com")
    assert "nicht in Allow-List" in pi._execute_web_fetch("https://evil.example.com/x")


class _FakeResp:
    def __init__(self, body: bytes):
        self._b = body

    def read(self, n: int = -1) -> bytes:
        return self._b[:n] if n and n > 0 else self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_allowed_and_cached(monkeypatch):
    monkeypatch.setenv("PI_WEB_FETCH_ALLOWLIST", "example.com")
    with patch("urllib.request.urlopen", return_value=_FakeResp(b"hello")) as m:
        assert pi._execute_web_fetch("https://example.com/p") == "hello"
        assert pi._execute_web_fetch("https://example.com/p") == "hello"  # cache hit
        assert m.call_count == 1


def test_body_truncated_at_cap(monkeypatch):
    monkeypatch.setenv("PI_WEB_FETCH_ALLOWLIST", "example.com")
    big = b"x" * (pi._WEB_FETCH_MAX_BYTES + 500)
    with patch("urllib.request.urlopen", return_value=_FakeResp(big)):
        out = pi._execute_web_fetch("https://example.com/big")
    assert "[abgeschnitten bei 200kB]" in out
