import os
from pathlib import Path
from mayring_pi_agent import pi_worker


def test_prefers_service_token(monkeypatch):
    monkeypatch.setenv("MCP_SERVICE_TOKEN", "svc-abc")
    assert pi_worker._embed_auth_token() == "svc-abc"


def test_falls_back_to_hook_jwt(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_SERVICE_TOKEN", raising=False)
    jwt = tmp_path / "hook.jwt"
    jwt.write_text("user-jwt-xyz\n")
    monkeypatch.setenv("MAYRING_HOOK_JWT", str(jwt))
    assert pi_worker._embed_auth_token() == "user-jwt-xyz"


def test_empty_when_neither(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_SERVICE_TOKEN", raising=False)
    monkeypatch.setenv("MAYRING_HOOK_JWT", str(tmp_path / "nope.jwt"))
    assert pi_worker._embed_auth_token() == ""
