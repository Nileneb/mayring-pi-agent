"""Architektur-Trajektorie pro File — Pi-Agent helper.

WHY: Opus ist teuer für mechanische diff-zusammenfassungen. Dieses Modul
läuft lokal auf qwen3.5:9b (oder vom ModelRouter aufgelöstem text-model)
und liefert Opus die fertige drei-teilige Synthese (Trajektorie / Obsolete
/ Aktiv) ohne dass Opus selbst `git log -p` lesen muss.

Wird aufgerufen von:
- `src/api/mcp_agent_tools.py::diff_history` (MCP-Tool für Claude)
- `tools/diff_history.py` (CLI für Mensch)
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from pathlib import Path


SYSTEM_PROMPT = """Du analysierst die Architektur-Trajektorie einer einzelnen Code-Datei.

Eingabe: eine chronologische Liste von Commits (jeweils Hash, Datum,
Subject, Diff). Letzter Commit zuerst.

Aufgabe: Schreibe eine kompakte (max 250 Wörter) Architektur-Zusammen-
fassung in 3 Abschnitten:

**Trajektorie:** woher kam die Datei (was war ihr ursprünglicher Zweck),
welche Phasen hat sie durchlaufen (z.B. "von Langdock-Integration zu
Claude-API-direkt", "von REST zu MCP-Streamable-HTTP"), wo steht sie
heute.

**Obsolete:** welche Konzepte/Funktionen wurden aus der Datei entfernt
und sollten nirgendwo wieder eingebaut werden (z.B. Langdock-Calls,
veraltete Routen, deprecated APIs). Mit commit-hash als Beleg.

**Aktiv:** welche Konzepte sind heute noch produktiv (mit kurzer
Begründung, warum). Hilft zukünftigen Reviewern zu verstehen was
"intended state" der Datei ist.

KEINE generic Phrasen. KEINE Wiederholung der commit-messages. Wenn
du eine Architektur-Entscheidung erkennst (z.B. Migration), nenn sie
explizit beim Namen.
"""


class DiffHistoryError(Exception):
    """Raised when git or ollama call fails."""


def fetch_history(file_path: str, n: int, *, repo_root: Path | None = None) -> str:
    """Run `git log --follow --patch -nN -- file` and return raw output."""
    if not os.path.exists(file_path):
        raise DiffHistoryError(f"file not found: {file_path}")
    cmd = [
        "git", "log", "--follow", "--patch", f"-n{n}",
        "--pretty=format:%n=== %h | %ad | %s ===%n", "--date=short", "--",
        file_path,
    ]
    result = subprocess.run(
        cmd,
        cwd=str(repo_root) if repo_root else None,
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise DiffHistoryError(f"git log failed: {result.stderr.strip()}")
    return result.stdout


def summarize(
    history: str,
    file_path: str,
    *,
    ollama_url: str,
    model: str,
    timeout: float = 120.0,
) -> str:
    """POST history to Ollama, return generated text."""
    body = {
        "model": model,
        "stream": False,
        "system": SYSTEM_PROMPT,
        "prompt": (
            f"# File: {file_path}\n\n## Commits (chronologisch)\n\n{history[:30000]}"
        ),
        "options": {"num_predict": 600, "temperature": 0.2},
        "think": False,
    }
    req = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
    except Exception as exc:
        raise DiffHistoryError(f"ollama call failed: {exc}") from exc
    return payload.get("response", "").strip() or "(empty response)"


def run(
    file_path: str,
    *,
    commits: int = 15,
    ollama_url: str | None = None,
    model: str | None = None,
    repo_root: Path | None = None,
) -> dict:
    """High-level entrypoint — fetch + summarize. Returns dict for MCP/JSON.

    Defaults pulled from env at call-time so tests can override.
    """
    url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
    if not model:
        try:
            from src.model_router import ModelRouter
            model = ModelRouter(url).resolve("text") or "qwen2.5-coder:7b"
        except Exception:
            model = os.environ.get("MAYRING_DIFF_MODEL", "qwen2.5-coder:7b")

    history = fetch_history(file_path, commits, repo_root=repo_root)
    if not history.strip():
        return {
            "file": file_path,
            "commits": commits,
            "model": model,
            "summary": "",
            "note": "no git history found",
        }
    summary = summarize(history, file_path, ollama_url=url, model=model)
    return {
        "file": file_path,
        "commits": commits,
        "model": model,
        "summary": summary,
    }
