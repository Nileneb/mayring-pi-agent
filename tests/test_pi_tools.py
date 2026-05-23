"""Sandbox + write/exec capability-gate tests for pi.py (and worker capability
derivation). These exercise the security-critical helpers directly — no Ollama."""
from __future__ import annotations

from pathlib import Path

import pytest

from mayring_pi_agent import pi


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("PI_FS_ROOT", "PI_ALLOW_WRITE", "PI_ALLOW_EXEC", "PI_WORKER_CAPABILITIES"):
        monkeypatch.delenv(var, raising=False)
    yield


def _tool_names(tools) -> list[str]:
    return [t["function"]["name"] for t in tools]


# --- read_file sandbox ---------------------------------------------------

def test_read_denied_without_root():
    assert "keine Sandbox-Root" in pi._execute_read_file("/etc/hostname")


def test_read_within_root(monkeypatch, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello-sandbox")
    monkeypatch.setenv("PI_FS_ROOT", str(tmp_path))
    assert pi._execute_read_file(str(f)) == "hello-sandbox"


def test_read_outside_root_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_FS_ROOT", str(tmp_path))
    assert "außerhalb" in pi._execute_read_file("/etc/hostname")


def test_read_traversal_rejected(monkeypatch, tmp_path):
    root = tmp_path / "sub"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("top-secret")
    monkeypatch.setenv("PI_FS_ROOT", str(root))
    # ../secret.txt escapes the configured root → rejected
    assert "außerhalb" in pi._execute_read_file(str(root / ".." / "secret.txt"))


# --- write_file gate + sandbox -------------------------------------------

def test_write_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_FS_ROOT", str(tmp_path))
    out = pi._execute_write_file(str(tmp_path / "x.txt"), "data")
    assert "DEAKTIVIERT" in out
    assert not (tmp_path / "x.txt").exists()


def test_write_enabled_within_root(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_ALLOW_WRITE", "1")
    monkeypatch.setenv("PI_FS_ROOT", str(tmp_path))
    target = tmp_path / "nested" / "x.txt"
    out = pi._execute_write_file(str(target), "payload")
    assert out.startswith("OK:")
    assert target.read_text() == "payload"


def test_write_outside_root_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_ALLOW_WRITE", "1")
    monkeypatch.setenv("PI_FS_ROOT", str(tmp_path))
    assert "außerhalb" in pi._execute_write_file("/tmp/evil.txt", "x")


def test_write_enabled_without_root_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_ALLOW_WRITE", "1")
    assert "keine Sandbox-Root" in pi._execute_write_file(str(tmp_path / "x"), "d")


# --- bash gate -----------------------------------------------------------

def test_bash_disabled_by_default():
    assert "DEAKTIVIERT" in pi._execute_bash("echo hi")


def test_bash_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_ALLOW_EXEC", "1")
    monkeypatch.setenv("PI_FS_ROOT", str(tmp_path))
    out = pi._execute_bash("echo hi")
    assert "exit=0" in out and "hi" in out


# --- tool advertisement matches flags ------------------------------------

def test_build_tools_readonly():
    names = _tool_names(pi._build_tools())
    assert "write_file" not in names and "bash" not in names
    assert "read_file" in names and "search_memory" in names


def test_build_tools_write_only(monkeypatch):
    monkeypatch.setenv("PI_ALLOW_WRITE", "1")
    names = _tool_names(pi._build_tools())
    assert "write_file" in names and "bash" not in names


def test_build_tools_exec(monkeypatch):
    monkeypatch.setenv("PI_ALLOW_EXEC", "1")
    assert "bash" in _tool_names(pi._build_tools())


def test_system_prompt_reflects_flags(monkeypatch):
    assert "write_file" not in pi._task_system_prompt()
    monkeypatch.setenv("PI_ALLOW_WRITE", "1")
    assert "write_file" in pi._task_system_prompt()


# --- worker capability derivation ----------------------------------------

def test_worker_capabilities_default(monkeypatch):
    from mayring_pi_agent import pi_worker
    assert "write" not in pi_worker._capabilities()


def test_worker_capabilities_with_write(monkeypatch):
    from mayring_pi_agent import pi_worker
    monkeypatch.setenv("PI_ALLOW_WRITE", "1")
    monkeypatch.setenv("PI_ALLOW_EXEC", "1")
    caps = pi_worker._capabilities()
    assert "write" in caps and "exec" in caps
