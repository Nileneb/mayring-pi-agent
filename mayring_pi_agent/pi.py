"""Pi-Agent: qwen3.5:2b mit Memory-Tool-Calling für kontextbewusste Code-Analyse."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Cloud memory search endpoint — hybrid pipeline (vector + symbolic + recency
# + source_affinity) runs server-side with ChromaDB. Local SQLite + Chroma
# remain as offline fallbacks only; preferred path is the cloud API.
_MEMORY_API_URL = os.environ.get("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
_MEMORY_JWT_FILE = os.path.expanduser(
    os.environ.get("MAYRING_JWT_FILE", "~/.config/mayring/hook.jwt")
)

if TYPE_CHECKING:
    from mayring_core.llm.endpoint import LLMEndpoint

# web_fetch tool (#211): READ-only GET, allow-listed, size-capped, cached.
_WEB_FETCH_MAX_BYTES = 200_000
_WEB_FETCH_TIMEOUT = 15.0
_WEB_FETCH_CACHE: dict[str, str] = {}


def _web_fetch_allowlist() -> list[str]:
    """Domains the Pi-Agent may fetch. Empty = deny-all (must be configured).

    WHY(#211, SECURITY): kein offener Web-Zugriff vom server-side Agent — nur
    explizit per PI_WEB_FETCH_ALLOWLIST freigegebene Domains (komma-separiert).
    """
    raw = os.environ.get("PI_WEB_FETCH_ALLOWLIST", "")
    return [d.strip().lower() for d in raw.split(",") if d.strip()]


def _domain_allowed(url: str, allowlist: list[str]) -> bool:
    from urllib.parse import urlparse
    if "*" in allowlist:  # local research worker opts into unrestricted fetch
        return url.startswith(("http://", "https://"))
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in allowlist)


def _execute_web_fetch(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "web_fetch Fehler: nur http(s)-URLs erlaubt"
    allow = _web_fetch_allowlist()
    if not allow:
        return (
            "web_fetch Fehler: keine Allow-List konfiguriert — setze "
            "PI_WEB_FETCH_ALLOWLIST (z.B. 'docs.python.org,github.com')"
        )
    if not _domain_allowed(url, allow):
        return f"web_fetch Fehler: Domain nicht in Allow-List {allow}"
    if url in _WEB_FETCH_CACHE:
        return _WEB_FETCH_CACHE[url]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "mayring-pi-agent/1.0"})
        with urllib.request.urlopen(req, timeout=_WEB_FETCH_TIMEOUT) as resp:
            raw = resp.read(_WEB_FETCH_MAX_BYTES + 1)
    except urllib.error.URLError as exc:
        return f"web_fetch Fehler: {exc}"
    except Exception as exc:
        return f"web_fetch Fehler: {exc}"
    body = raw[:_WEB_FETCH_MAX_BYTES].decode("utf-8", errors="replace")
    if len(raw) > _WEB_FETCH_MAX_BYTES:
        body += "\n[abgeschnitten bei 200kB]"
    _WEB_FETCH_CACHE[url] = body
    return body


def _read_jwt() -> str:
    """DRY helper — same JWT as _cloud_search (Path(_MEMORY_JWT_FILE).read_text())."""
    try:
        return Path(_MEMORY_JWT_FILE).read_text().strip()
    except Exception:
        return ""


_SEARXNG_TIMEOUT = float(os.getenv("PI_WEB_SEARCH_TIMEOUT", "20"))
_SEARXNG_MAX_RESULTS = int(os.getenv("PI_WEB_SEARCH_MAX_RESULTS", "8"))


def _searxng_url() -> str:
    api = os.getenv("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
    return f"{api}/searxng/search"


def _execute_web_search(query: str) -> str:
    if not query.strip():
        return "web_search Fehler: leere Query"
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    url = f"{_searxng_url()}?{params}"
    headers = {"User-Agent": "mayring-pi-agent/1.0"}
    token = _read_jwt()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=_SEARXNG_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        return f"web_search Fehler: {exc}"
    results = (data.get("results") or [])[:_SEARXNG_MAX_RESULTS]
    if not results:
        return f"web_search: keine Treffer für {query!r}"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '')}\n   {r.get('url', '')}\n   {r.get('content', '')}")
    return "\n".join(lines)


def _execute_ingest(title: str, text: str) -> str:
    if not text.strip():
        return "ingest Fehler: leerer Text"
    api = os.getenv("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
    body = json.dumps({
        "text": text,
        "source_id": f"research:{title}"[:200],
        "source_type": "knowledge",
    }).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "mayring-pi-agent/1.0"}
    token = _read_jwt()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(f"{api}/ingest", data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except Exception as exc:
        return f"ingest Fehler: {exc}"
    return f"ingest ok: '{title}' ins Memory geschrieben"


# ---------------------------------------------------------------------------
# Filesystem capability gates (#224 follow-up — dual-mode deployment).
#
# WHY(SECURITY): the cloud/server deployment (mcp.linn.games, app.linn.games)
# runs READ-ONLY + SANDBOXED — no env flags there, so write/exec stay off and
# read_file is confined to PI_FS_ROOT. The SAME package run as a LOCAL worker
# (user's own machine + local Ollama) opts in via PI_ALLOW_WRITE / PI_ALLOW_EXEC
# so it can execute write-jobs pulled from the cloud queue. pi_worker derives
# its advertised capabilities from these SAME flags, so the queue never routes
# a write-job to a worker that would then reject it.
_TRUE = ("1", "true", "yes", "on")

_WRITE_DISABLED_MSG = (
    "{name} ist DEAKTIVIERT — dieser Pi-Agent läuft read-only "
    "(PI_ALLOW_WRITE/PI_ALLOW_EXEC nicht gesetzt). Auf Cloud/Server ist das "
    "Absicht. Gib die vorgeschlagene Änderung als text/diff zurück, damit der "
    "Orchestrator sie client-side anwendet."
)


def write_enabled() -> bool:
    return os.environ.get("PI_ALLOW_WRITE", "").strip().lower() in _TRUE


def exec_enabled() -> bool:
    return os.environ.get("PI_ALLOW_EXEC", "").strip().lower() in _TRUE


def _fs_roots() -> list[Path]:
    """Allowed filesystem roots for read_file/write_file (PI_FS_ROOT).

    Empty = deny-all (read_file/write_file refuse) — same fail-closed default
    as the web_fetch allow-list. A local worker that wants unrestricted access
    sets PI_FS_ROOT=/ explicitly.
    """
    roots: list[Path] = []
    for part in os.environ.get("PI_FS_ROOT", "").split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        try:
            roots.append(Path(part).expanduser().resolve())
        except OSError:
            continue
    return roots


def _within_roots(p: Path, roots: list[Path]) -> bool:
    # resolve() collapses .. and follows symlinks, so a path that escapes the
    # root via either trick lands outside and is rejected.
    try:
        rp = p.expanduser().resolve()
    except OSError:
        return False
    return any(rp == r or r in rp.parents for r in roots)


def _execute_read_file(raw_path: str) -> str:
    roots = _fs_roots()
    if not roots:
        return (
            "read_file Fehler: keine Sandbox-Root konfiguriert — setze PI_FS_ROOT "
            "(z.B. '/srv/repos'). Ohne Root ist read_file deaktiviert (fail-closed)."
        )
    if not raw_path:
        return "read_file Fehler: leerer Pfad"
    p = Path(raw_path)
    if not _within_roots(p, roots):
        return f"read_file Fehler: Pfad außerhalb der erlaubten Root(s) {[str(r) for r in roots]}"
    try:
        content = p.expanduser().resolve().read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"read_file Fehler: {exc}"
    return content[:8000] + ("\n[abgeschnitten]" if len(content) > 8000 else "")


def _execute_write_file(raw_path: str, content: str) -> str:
    if not write_enabled():
        return _WRITE_DISABLED_MSG.format(name="write_file")
    roots = _fs_roots()
    if not roots:
        return (
            "write_file Fehler: keine Sandbox-Root konfiguriert — setze PI_FS_ROOT. "
            "Ohne Root ist write_file deaktiviert (fail-closed)."
        )
    if not raw_path:
        return "write_file Fehler: leerer Pfad"
    p = Path(raw_path)
    if not _within_roots(p, roots):
        return f"write_file Fehler: Pfad außerhalb der erlaubten Root(s) {[str(r) for r in roots]}"
    try:
        target = p.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"write_file Fehler: {exc}"
    return f"OK: {target} geschrieben ({len(content)} Zeichen)"


def _execute_bash(command: str, cwd: str = "") -> str:
    if not exec_enabled():
        return _WRITE_DISABLED_MSG.format(name="bash")
    if not command:
        return "bash Fehler: leerer Befehl"
    import subprocess

    # WHY(SECURITY): cwd is NOT a sandbox — a shell command escapes any cwd.
    # exec_enabled() (PI_ALLOW_EXEC, default OFF) is the real gate and is only
    # ever set on a local, user-owned worker — never on the server.
    roots = _fs_roots()
    work = cwd or (str(roots[0]) if roots else None)
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60, cwd=work,
        )
    except subprocess.TimeoutExpired:
        return "bash Timeout (60s)"
    except Exception as exc:
        return f"bash Fehler: {exc}"
    out = (proc.stdout + proc.stderr)[:4000]
    return f"exit={proc.returncode}\n{out}"


def _resolve_ollama_compatible(endpoint: "LLMEndpoint") -> tuple[str, str]:
    """Unpack an LLMEndpoint into (base_url, model) for the Ollama chat loop.

    Only accepts providers that speak the Ollama /api/chat protocol (or a
    compatible subset). Anthropic has a different API shape and is not
    supported here — callers needing anthropic-byo routing must branch off
    to dispatch.generate() before the tool-calling loop.
    """
    if endpoint.provider not in ("ollama", "platform", "openai"):
        raise NotImplementedError(
            f"pi.py tool-calling loop does not support provider={endpoint.provider!r}. "
            "Use mayring_core.llm.dispatch.generate() for anthropic or other non-Ollama providers."
        )
    return endpoint.base_url, endpoint.model

# Tool-Definition (OpenAI-Format, Ollama /api/chat kompatibel)
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Suche im Projekt-Memory nach Konventionen, bekannten Patterns und Kontext",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Suchbegriff (z.B. 'Laravel artisan Konvention' oder 'Policy authorization pattern')",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Anzahl Ergebnisse (default: 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wiki",
            "description": "Finde thematisch verwandte Dateien über funktionale Zusammenhänge (Import, Aufruf, Label) — auch ohne semantische Ähnlichkeit",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Dateiname, Klasse oder Thema (z.B. 'CreditService', 'auth', 'payment')",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repo-Slug (z.B. 'app.linn.games'), leer für alle",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Liest den Inhalt einer Datei vom Dateisystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absoluter oder relativer Pfad zur Datei"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Lädt den Text-Inhalt einer http(s)-URL (READ-only GET). Nur "
                "Domains aus der Allow-List (PI_WEB_FETCH_ALLOWLIST). Body wird "
                "bei 200kB abgeschnitten, Ergebnisse werden gecacht."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Vollständige http(s)-URL"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Durchsucht das Web (SearXNG) und liefert Top-Treffer als "
                "Titel + URL + Snippet. Danach mit web_fetch die beste URL laden."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Suchbegriff"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest",
            "description": (
                "Schreibt ein Recherche-Ergebnis dauerhaft ins Memory (durchsuchbar). "
                "Nutze dies am Ende, um wichtige Findings zu sichern."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Kurzer Titel"},
                    "text": {"type": "string", "description": "Der zu speichernde Inhalt"},
                },
                "required": ["title", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan",
            "description": (
                "Notiere oder revidiere einen mehrstufigen Plan für komplexe "
                "Tasks. Rufe dies zu Beginn auf und erneut, wenn ein Tool-Ergebnis "
                "zeigt, dass der Plan angepasst werden muss (Replan)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Geordnete Schritte",
                    },
                },
                "required": ["steps"],
            },
        },
    },
]

# WHY(#224 follow-up): write_file + bash are NOT in the read-only base above.
# They are appended by _build_tools() ONLY when the env flags opt in, so the
# cloud/server (no flags) never advertises them. Sandbox = PI_FS_ROOT.
_WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Schreibt Inhalt in eine Datei innerhalb der Sandbox-Root (PI_FS_ROOT)",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Pfad innerhalb PI_FS_ROOT"},
                "content": {"type": "string", "description": "Dateiinhalt"},
            },
            "required": ["path", "content"],
        },
    },
}

_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Führt einen Shell-Befehl aus (Tests, git, etc.) — nur lokaler Worker",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell-Befehl"},
                "cwd": {"type": "string", "description": "Arbeitsverzeichnis (optional)"},
            },
            "required": ["command"],
        },
    },
}


def _build_tools() -> list[dict]:
    """Read-only tools always; write/exec tools only when the env flags opt in.

    WHY: advertising a tool the handler would reject just wastes the model's
    tool-budget. Gating the *definition* means cloud/server never even sees
    write_file/bash.
    """
    tools = list(_TOOLS)
    if write_enabled():
        tools.append(_WRITE_FILE_TOOL)
    if exec_enabled():
        tools.append(_BASH_TOOL)
    return tools

_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "pi_system.md"

_TASK_SYSTEM_PROMPT = """\
Du bist Pi, ein READ-ONLY Analyse-Agent mit Zugriff auf Memory + Dateisystem-Lesezugriff.

**Tools:**
- search_memory: Projektkontext, Konventionen, bekannte Patterns abrufen
- search_wiki: Thematisch verwandte Dateien finden
- read_file: Datei lesen (absoluter Pfad bevorzugt)
- web_search: Das Web durchsuchen (SearXNG) → Titel + URL + Snippet. Für aktuelle/externe Infos.
- web_fetch: Text-Inhalt einer http(s)-URL laden (nach web_search die beste URL öffnen)
- ingest: Wichtige Recherche-Findings dauerhaft ins Memory schreiben (am Ende)
- plan: mehrstufigen Plan notieren/revidieren (zu Beginn + bei Replan)

**Wichtig:** Du kannst KEINE Dateien schreiben und KEINE shell-commands ausführen
— du läufst server-side, write/exec wären ein security-risiko. Für Code-Tasks:
gib die vorgeschlagene Änderung als text oder unified-diff zurück. Der Orchestrator
(claude-code, client-side) wendet sie an + führt tests aus.

**Workflow für Recherche-Tasks:** plan → web_search (Treffer finden) → web_fetch
(beste URLs lesen) → search_memory (eigenes Wissen) → Synthese → ingest (Findings sichern).
**Grundsatz:** Analysiere gründlich, nutze web_search aktiv für externe/aktuelle Fragen,
schlage konkret vor."""


def _task_system_prompt() -> str:
    """Read-only base prompt; documents write/exec only when the flags opt in."""
    extra: list[str] = []
    if write_enabled():
        extra.append("- write_file: Datei in der Sandbox-Root (PI_FS_ROOT) schreiben/überschreiben")
    if exec_enabled():
        extra.append("- bash: Shell-Befehl ausführen (Tests, git, etc.)")
    if not extra:
        return _TASK_SYSTEM_PROMPT
    return (
        _TASK_SYSTEM_PROMPT
        + "\n\n**Zusätzlich aktiviert (lokaler Worker):**\n"
        + "\n".join(extra)
        + "\n→ Du DARFST hier Dateien schreiben/Befehle ausführen (Sandbox = PI_FS_ROOT). "
        "Setze Änderungen direkt um und verifiziere mit bash, falls verfügbar."
    )


def _load_system_prompt() -> str:
    if _SYSTEM_PROMPT_PATH.exists():
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return "Du bist Pi, ein präziser Code-Reviewer. Antworte nur mit JSON: {\"file_summary\":\"...\",\"potential_smells\":[]}"


try:
    from mayring_core.memory.retrieval import compress_for_prompt, search  # type: ignore
except ImportError:
    search = None  # type: ignore
    compress_for_prompt = None  # type: ignore

try:
    from mayring_core.memory.store import init_memory_db  # type: ignore
except ImportError:
    init_memory_db = None  # type: ignore

try:
    from mayring_core.memory.ingest import get_or_create_chroma_collection  # type: ignore
except ImportError:
    get_or_create_chroma_collection = None  # type: ignore


def _cloud_search(
    query: str,
    top_k: int,
    repo: str | None,
    char_budget: int = 1800,
    timeout: float = 30.0,
    session_id: str | None = None,
) -> str | None:
    """Hybrid memory search via the cloud API (server-side ChromaDB + SQLite).

    Returns formatted markdown context, "" if no results, or None if the cloud
    is unreachable (signalling the caller to try the local fallback).

    ``session_id`` activates the server-side recency-lane: the controlling
    session's rolling conversation thread is guaranteed into the results even
    when semantic similarity is weak — so the worker sees the latest decisions
    (e.g. a fix made minutes ago), not just topically-similar older chunks.
    """
    try:
        token = Path(_MEMORY_JWT_FILE).read_text().strip()
    except (FileNotFoundError, OSError):
        return None
    if not token:
        return None

    payload: dict[str, Any] = {"query": query, "top_k": top_k}
    if repo:
        payload["repo"] = repo
    if session_id:
        payload["session_id"] = session_id
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{_MEMORY_API_URL}/memory/search",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None

    results = data.get("results") or []
    if not results:
        return ""

    # Prioritise the session-recency thread within the (tight) char budget: the
    # server guarantees it into top_k, but a strong semantic match can otherwise
    # crowd it out of the formatted output before it reaches the model. Surface
    # session-recency chunks first, then by score — so "what I'm doing now" lands.
    results.sort(key=lambda r: (
        0 if "session-recency" in (r.get("reasons") or []) else 1,
        -float(r.get("score_final") or 0.0),
    ))

    out: list[str] = []
    used = 0
    for r in results:
        sid = (r.get("source_id") or "")[:80]
        chunk_id = (r.get("chunk_id") or "")[:24]
        score = float(r.get("score_final") or 0.0)
        text = (r.get("text") or "").strip()
        head = f"### {sid}  (score={score:.2f}, chunk={chunk_id})"
        max_body = max(0, char_budget - used - len(head) - 4)
        if max_body <= 0:
            break
        if len(text) > max_body:
            text = text[:max_body].rstrip() + " […]"
        block = f"{head}\n{text}\n"
        out.append(block)
        used += len(block)
        if used >= char_budget:
            break
    return "\n".join(out)


def _execute_search_memory(
    query: str,
    top_k: int,
    conn: sqlite3.Connection,
    chroma_collection: Any,
    ollama_url: str,
    repo_slug: str | None,
    session_id: str | None = None,
) -> str:
    """Execute search_memory tool call — returns markdown context string.

    Architecture: ChromaDB lives on the cloud server (mcp.linn.games), local
    side keeps only an SQLite replica. Therefore prefer the cloud /memory/search
    endpoint (full 4-stage hybrid pipeline). Fall back to the local
    `retrieval.search()` only if the cloud is unreachable AND a populated
    local Chroma collection is available — the legacy offline path.
    """
    cloud_text = _cloud_search(query, top_k, repo_slug, session_id=session_id)
    if cloud_text is not None:
        return cloud_text or "Keine relevanten Memory-Einträge gefunden."

    _search = search
    _compress = compress_for_prompt
    if _search is None or _compress is None:
        from mayring_core.memory.retrieval import compress_for_prompt as _compress, search as _search  # type: ignore

    opts: dict = {"top_k": top_k, "include_text": True}
    if repo_slug:
        opts["repo"] = repo_slug

    try:
        results = _search(
            query=query,
            conn=conn,
            chroma_collection=chroma_collection,
            ollama_url=ollama_url,
            opts=opts,
        )
        context = _compress(results, char_budget=1500)
        return context if context else "Keine relevanten Memory-Einträge gefunden."
    except Exception as exc:
        return f"Memory-Suche fehlgeschlagen (lokaler Fallback): {exc}"


def _sanitize_repo_slug_for_filename(slug: str) -> str:
    """Return a strictly validated filesystem-safe slug for cache wiki filenames."""
    slug = slug.strip()
    # Allow only simple filename-safe repo slugs.
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?", slug):
        return ""
    if ".." in slug:
        return ""
    return slug


def _execute_search_wiki(args: dict, repo_slug_hint: str = "") -> str:
    """Execute search_wiki tool call — returns markdown context string.

    Tries wiki_v2 DB first (live graph), falls back to legacy *_wiki.md.
    """
    slug = args.get("repo") or repo_slug_hint
    topic = args.get("topic", "").lower()

    # --- wiki_v2 path ---
    safe_slug = _sanitize_repo_slug_for_filename(str(slug)) if slug else ""
    if safe_slug:
        try:
            from mayring_core.config import CACHE_DIR
            try:
                from src.wiki_v2.graph import WikiGraph  # nur im MayringCoder-Monorepo
            except ImportError:
                return "search_wiki nicht verfügbar im standalone Pi-Service"
            db = WikiGraph(safe_slug, safe_slug, CACHE_DIR / "wiki_v2.db")
            if db.node_count() > 0:
                clusters = db.get_clusters()
                nodes = db.all_nodes()
                db.close()

                # Match clusters by name/description containing topic
                matched = [
                    c for c in clusters
                    if topic in c.name.lower() or topic in (c.description or "").lower()
                ] if topic else clusters[:5]

                if not matched and clusters:
                    matched = clusters[:3]

                parts = []
                node_tiers = {n.id: n.turbulence_tier for n in nodes}
                for c in matched[:4]:
                    members_fmt = []
                    for m in c.members[:8]:
                        tier = node_tiers.get(m, "")
                        tier_mark = " 🔥" if tier == "hot" else (" ⚡" if tier == "warm" else "")
                        members_fmt.append(f"  - {m}{tier_mark}")
                    more = f"\n  …+{len(c.members)-8} weitere" if len(c.members) > 8 else ""
                    parts.append(
                        f"### {c.name}\n"
                        + (f"{c.description}\n" if c.description else "")
                        + "\n".join(members_fmt) + more
                    )

                if parts:
                    return f"## Wiki-Cluster für '{topic}'\n\n" + "\n\n".join(parts)
            else:
                db.close()
        except Exception:
            pass

    # --- legacy fallback: *_wiki.md ---
    safe_slug_fs = _sanitize_repo_slug_for_filename(str(slug)) if slug else ""
    cache_dir = Path("cache")
    wiki_files = list(cache_dir.glob("*_wiki.md")) if cache_dir.exists() else []
    if not wiki_files:
        return "Kein Wiki vorhanden. Zuerst --generate-wiki oder --populate-memory ausführen."

    wiki_path: Path | None = None
    if safe_slug_fs:
        expected_name = f"{safe_slug_fs}_wiki.md"
        for candidate in wiki_files:
            if candidate.name == expected_name:
                wiki_path = candidate
                break
    if wiki_path is None:
        wiki_path = wiki_files[0]

    content = wiki_path.read_text(encoding="utf-8")
    sections: list[str] = []
    current: list[str] = []
    for line in content.splitlines():
        if line.startswith("## 🔗"):
            if current and any(topic in l.lower() for l in current):
                sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current and any(topic in l.lower() for l in current):
        sections.append("\n".join(current))

    if not sections:
        return f"Keine Wiki-Einträge für '{topic}' gefunden."
    return "\n\n".join(sections[:3])


def _agent_loop(
    messages: list[dict],
    system_prompt: str,
    model: str,
    ollama_url: str,
    timeout: float,
    max_tool_calls: int,
    conn: Any,
    chroma: Any,
    repo_slug: str | None,
    num_predict: int = 1024,
    session_id: str | None = None,
) -> tuple[str, int]:
    """Shared tool-calling loop.

    Returns:
        (final_content, tool_calls_made)
    Raises:
        Exception on HTTP failure — callers handle this.
    """
    tool_calls_made = 0
    _start = time.perf_counter()
    _base_url = ollama_url.rstrip("/")

    while True:
        request_body: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,  # Qwen3: disable thinking mode so content is not empty
            "system": system_prompt,
            "options": {
                "temperature": 0.3,
                "top_k": 5,
                "num_predict": num_predict,
            },
        }
        if tool_calls_made < max_tool_calls:
            request_body["tools"] = _build_tools()

        from mayring_core.ollama_client import chat as _oc_chat
        data = _oc_chat(
            ollama_url, model, messages,
            system=system_prompt,
            tools=request_body.get("tools"),
            options=request_body["options"],
            stream=False,
            timeout=timeout,
        )

        message = data.get("message", {})
        tool_calls = message.get("tool_calls") or []

        # No tool calls → final response
        if not tool_calls or tool_calls_made >= max_tool_calls:
            try:
                from mayring_core.memory.store import log_llm_call
                _dur = int((time.perf_counter() - _start) * 1000)
                _prompt_text = messages[0].get("content", "") if messages else ""
                log_llm_call(
                    conn, "pi_task", model,
                    prompt=_prompt_text,
                    response=message.get("content", ""),
                    tool_calls=tool_calls_made,
                    duration_ms=_dur,
                )
            except Exception:
                pass
            return message.get("content", "").strip(), tool_calls_made

        # Append assistant message with tool_calls
        messages.append({
            "role": "assistant",
            "content": message.get("content", ""),
            "tool_calls": tool_calls,
        })

        # Execute each tool call
        for tc in tool_calls:
            func = tc.get("function", {})
            func_name = func.get("name", "")
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}

            if func_name == "search_memory":
                query = args.get("query", "")
                top_k = int(args.get("top_k", 5))
                result_text = _execute_search_memory(
                    query=query,
                    top_k=top_k,
                    conn=conn,
                    chroma_collection=chroma,
                    ollama_url=ollama_url,
                    repo_slug=repo_slug,
                    session_id=session_id,
                )
                tool_calls_made += 1
                print(f"    [Pi] search_memory({query!r:.40}) → {len(result_text)} chars", flush=True)
            elif func_name == "search_wiki":
                result_text = _execute_search_wiki(args, repo_slug_hint=repo_slug or "")
                tool_calls_made += 1
                topic = args.get("topic", "")
                print(f"    [Pi] search_wiki({topic!r:.40}) → {len(result_text)} chars", flush=True)
            elif func_name == "read_file":
                result_text = _execute_read_file(args.get("path", ""))
                print(f"    [Pi] read_file({str(args.get('path', '')):.50}) → {len(result_text)} chars", flush=True)
            elif func_name == "web_fetch":
                url = args.get("url", "")
                result_text = _execute_web_fetch(url)
                tool_calls_made += 1
                print(f"    [Pi] web_fetch({url!r:.50}) → {len(result_text)} chars", flush=True)
            elif func_name == "web_search":
                q = args.get("query", "")
                result_text = _execute_web_search(q)
                tool_calls_made += 1
                print(f"    [Pi] web_search({q!r:.50}) → {len(result_text)} chars", flush=True)
            elif func_name == "ingest":
                result_text = _execute_ingest(args.get("title", ""), args.get("text", ""))
                print(f"    [Pi] ingest({str(args.get('title', '')):.40}) → {result_text[:60]}", flush=True)
            elif func_name == "plan":
                steps = args.get("steps", []) or []
                result_text = "Plan notiert:\n" + "\n".join(
                    f"{i + 1}. {s}" for i, s in enumerate(steps)
                )
                tool_calls_made += 1
                print(f"    [Pi] plan({len(steps)} Schritte)", flush=True)
            elif func_name == "write_file":
                # WHY(SECURITY): _execute_write_file self-gates on write_enabled()
                # + PI_FS_ROOT — a stale prompt calling it on a read-only deploy
                # gets the disabled message, never a write.
                result_text = _execute_write_file(args.get("path", ""), args.get("content", ""))
                tool_calls_made += 1
                print(f"    [Pi] write_file({str(args.get('path', '')):.50}) → {result_text[:40]}", flush=True)
            elif func_name == "bash":
                result_text = _execute_bash(args.get("command", ""), args.get("cwd", ""))
                tool_calls_made += 1
                print(f"    [Pi] bash({str(args.get('command', '')):.50})", flush=True)
            else:
                result_text = f"Unbekanntes Tool: {func_name}"

            messages.append({
                "role": "tool",
                "content": result_text,
            })

        # Safety: if we've hit limit after processing, force final response next iteration
        if tool_calls_made >= max_tool_calls:
            continue


def analyze_with_memory(
    file: dict,
    ollama_url: str,
    model: str,
    repo_slug: str | None = None,
    max_tool_calls: int = 3,
    timeout: float = 120.0,
    endpoint: "LLMEndpoint | None" = None,
    wiki_context: str = "",
) -> dict:
    """Analyze a file using Pi agent loop with memory tool-calling.

    Args:
        file: {"filename": str, "content": str, "category": str}
        ollama_url: Default Ollama base URL. Ignored if `endpoint` is set.
        model: Default model name. Ignored if `endpoint` is set.
        repo_slug: Repository slug for memory scope filtering
        max_tool_calls: Maximum number of search_memory calls allowed
        timeout: HTTP timeout per request in seconds
        endpoint: Optional LLMEndpoint overriding ollama_url+model. Callers that
            resolve a per-user/per-workspace endpoint (via get_endpoint_for_request)
            should pass it here. Provider must be ollama/platform/openai —
            anthropic needs dispatch.generate and is not supported in the
            tool-calling loop yet.

    Returns:
        Analysis result dict with "file_summary" and "potential_smells"
    """
    if endpoint is not None:
        ollama_url, model = _resolve_ollama_compatible(endpoint)
    try:
        from src.analysis.analyzer import _parse_llm_json
    except ImportError:
        from mayring_pi_agent.json_utils import parse_llm_json as _parse_llm_json

    _init_db = init_memory_db
    if _init_db is None:
        from mayring_core.memory.store import init_memory_db as _init_db  # type: ignore
    _get_chroma = get_or_create_chroma_collection
    if _get_chroma is None:
        from mayring_core.memory.ingest import get_or_create_chroma_collection as _get_chroma  # type: ignore

    filename = file.get("filename", "?")
    content = file.get("content", "")
    category = file.get("category", "")

    _CONTENT_LIMIT = 3000
    content_truncated = len(content) > _CONTENT_LIMIT
    if content_truncated:
        print(f"  [Pi] WARN: {filename} truncated {len(content)} → {_CONTENT_LIMIT} chars", flush=True)
    content_for_prompt = content[:_CONTENT_LIMIT]

    conn = _init_db()
    chroma = _get_chroma()
    system_prompt = _load_system_prompt()
    if wiki_context:
        system_prompt += f"\n\n## Projekt-Kontext (Wiki)\n{wiki_context}"

    user_content = (
        f"Analysiere diese Datei. Antworte EXAKT in diesem Format (keine anderen Keys):\n"
        f'{{\"file_summary\":\"...\",\"potential_smells\":[]}}\n\n'
        f"Datei: {filename}\nKategorie: {category}\n\n"
        f"```\n{content_for_prompt}\n```"
    )
    messages = [{"role": "user", "content": user_content}]

    try:
        raw_content, tool_calls_made = _agent_loop(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            ollama_url=ollama_url,
            timeout=timeout,
            max_tool_calls=max_tool_calls,
            conn=conn,
            chroma=chroma,
            repo_slug=repo_slug,
        )
    except Exception as exc:
        return {
            "filename": filename,
            "category": category,
            "file_summary": "",
            "potential_smells": [],
            "error": f"Pi-Agent HTTP-Fehler: {exc}",
            "_parse_error": True,
        }
    finally:
        conn.close()

    # Strip markdown fences if model wrapped JSON
    if raw_content.startswith("```"):
        raw_content = raw_content.strip("`").strip()
        if raw_content.startswith("json"):
            raw_content = raw_content[4:].strip()

    parsed = _parse_llm_json(raw_content)
    if parsed:
        parsed.setdefault("filename", filename)
        parsed.setdefault("category", category)
        parsed["truncated"] = content_truncated
        parsed.setdefault("_pi_tool_calls", tool_calls_made)
        smells = parsed.get("potential_smells", [])
        parsed["potential_smells"] = [s for s in smells if isinstance(s, dict)] if isinstance(smells, list) else []
        return parsed

    return {
        "filename": filename,
        "category": category,
        "file_summary": raw_content[:200] if raw_content else "",
        "potential_smells": [],
        "_parse_error": True,
        "_pi_tool_calls": tool_calls_made,
        "truncated": content_truncated,
    }


def run_task_with_memory(
    task: str,
    ollama_url: str,
    model: str,
    repo_slug: str | None = None,
    system_prompt: str | None = None,
    max_tool_calls: int = 10,
    timeout: float = 180.0,
    endpoint: "LLMEndpoint | None" = None,
    disable_memory: bool = False,
    session_id: str | None = None,
    num_predict: int | None = None,
) -> str:
    """Run a free-form task using Pi agent with memory tool-calling.

    Unlike analyze_with_memory(), this function accepts any task prompt and
    returns the model's raw text response — no JSON parsing, no fixed format.

    Args:
        task: Free-form task description, e.g. "Entwickle PICO-Suchterms für..."
        ollama_url: Default Ollama base URL. Ignored if `endpoint` is set.
        model: Default model name. Ignored if `endpoint` is set.
        repo_slug: Repository slug for memory scope filtering
        system_prompt: Custom system prompt (default: _TASK_SYSTEM_PROMPT)
        max_tool_calls: Maximum number of search_memory calls (default: 5)
        timeout: HTTP timeout per request in seconds
        endpoint: Optional LLMEndpoint from get_endpoint_for_request. Overrides
            ollama_url+model when set. Provider must be ollama/platform/openai.
        disable_memory: When True, skip ambient context injection and all memory
            tool calls — useful as a no-memory baseline for benchmarking.

    Returns:
        Model response as plain text (Markdown, lists, prose — whatever the model returns)
    """
    if endpoint is not None:
        ollama_url, model = _resolve_ollama_compatible(endpoint)
    safe_repo_slug = repo_slug or ""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", safe_repo_slug):
        safe_repo_slug = ""

    _init_db = init_memory_db
    if _init_db is None:
        from mayring_core.memory.store import init_memory_db as _init_db  # type: ignore
    _get_chroma = get_or_create_chroma_collection
    if _get_chroma is None:
        from mayring_core.memory.ingest import get_or_create_chroma_collection as _get_chroma  # type: ignore

    conn = _init_db()
    chroma = _get_chroma()
    messages = [{"role": "user", "content": task}]

    # Ambient context — silent skip if no snapshot or memory disabled
    ambient_ctx = ""
    _trigger_ids: list[str] = []
    if not disable_memory:
        try:
            from mayring_core.memory.ambient import build_context
            ambient_ctx = build_context(
                task, conn, ollama_url, safe_repo_slug,
                _out_trigger_ids=_trigger_ids,
                chroma_collection=chroma,
            )
        except Exception:
            pass

    _effective_max_tool_calls = 0 if disable_memory else max_tool_calls
    # Token budget for reasoning: param > env (PI_NUM_PREDICT) > 2048 default.
    # Stronger reasoners (Gemma) need room (≥4k) or they truncate mid-thought.
    if num_predict is None:
        try:
            num_predict = int(os.environ.get("PI_NUM_PREDICT", "2048"))
        except ValueError:
            num_predict = 2048
    _system = (system_prompt or _task_system_prompt()) + (f"\n\n{ambient_ctx}" if ambient_ctx else "")

    content = ""
    tool_calls_made = 0
    try:
        content, tool_calls_made = _agent_loop(
            messages=messages,
            system_prompt=_system,
            model=model,
            ollama_url=ollama_url,
            timeout=timeout,
            max_tool_calls=_effective_max_tool_calls,
            conn=conn,
            chroma=chroma,
            repo_slug=safe_repo_slug,
            num_predict=num_predict,
            session_id=session_id,
        )
    except Exception as exc:
        conn.close()
        return f"[Pi-Agent Fehler] {exc}"

    print(f"  [Pi] Fertig — {tool_calls_made} Memory-Abfragen", flush=True)

    # Implicit feedback — silent fail
    if ambient_ctx and _trigger_ids:
        try:
            from mayring_core.memory.ambient import compute_feedback, update_trigger_stats
            _retrieval_happened = bool(ambient_ctx and "## Relevante Erinnerungen" in ambient_ctx)
            led_to_retrieval = tool_calls_made > 0 or _retrieval_happened
            fb = compute_feedback(ambient_ctx, content, _trigger_ids, led_to_retrieval, conn, ollama_url)
            update_trigger_stats(_trigger_ids, fb.was_referenced, conn)
        except Exception:
            pass

    conn.close()
    return content
