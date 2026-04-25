"""Pi-Agent: qwen3.5:2b mit Memory-Tool-Calling für kontextbewusste Code-Analyse."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.llm.endpoint import LLMEndpoint


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
            "Use src.llm.dispatch.generate() for anthropic or other non-Ollama providers."
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
    }
]

_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "pi_system.md"

_TASK_SYSTEM_PROMPT = """\
Du bist Pi, ein intelligenter Assistent mit Zugriff auf ein Projekt-Memory-System.

**Tool: search_memory**
Nutze es, um relevanten Projektkontext abzurufen, bevor du antwortest.
Maximal 5 Aufrufe pro Auftrag.

**Grundsatz:** Antworte präzise und strukturiert. Nutze das Memory aktiv."""


def _load_system_prompt() -> str:
    if _SYSTEM_PROMPT_PATH.exists():
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return "Du bist Pi, ein präziser Code-Reviewer. Antworte nur mit JSON: {\"file_summary\":\"...\",\"potential_smells\":[]}"


try:
    from src.memory.retrieval import compress_for_prompt, search  # type: ignore
except ImportError:
    search = None  # type: ignore
    compress_for_prompt = None  # type: ignore

try:
    from src.memory.store import init_memory_db  # type: ignore
except ImportError:
    init_memory_db = None  # type: ignore

try:
    from src.memory.ingest import get_or_create_chroma_collection  # type: ignore
except ImportError:
    get_or_create_chroma_collection = None  # type: ignore


def _execute_search_memory(
    query: str,
    top_k: int,
    conn: sqlite3.Connection,
    chroma_collection: Any,
    ollama_url: str,
    repo_slug: str | None,
) -> str:
    """Execute search_memory tool call — returns markdown context string."""
    _search = search
    _compress = compress_for_prompt
    if _search is None or _compress is None:
        from src.memory.retrieval import compress_for_prompt as _compress, search as _search  # type: ignore

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
        return f"Memory-Suche fehlgeschlagen: {exc}"


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
            from src.config import CACHE_DIR
            from src.wiki_v2.graph import WikiGraph
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
            request_body["tools"] = _TOOLS

        from src.ollama_client import chat as _oc_chat
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
                from src.memory.store import log_llm_call
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
                )
                tool_calls_made += 1
                print(f"    [Pi] search_memory({query!r:.40}) → {len(result_text)} chars", flush=True)
            elif func_name == "search_wiki":
                result_text = _execute_search_wiki(args, repo_slug_hint=repo_slug or "")
                tool_calls_made += 1
                topic = args.get("topic", "")
                print(f"    [Pi] search_wiki({topic!r:.40}) → {len(result_text)} chars", flush=True)
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
    from src.analysis.analyzer import _parse_llm_json

    _init_db = init_memory_db
    if _init_db is None:
        from src.memory.store import init_memory_db as _init_db  # type: ignore
    _get_chroma = get_or_create_chroma_collection
    if _get_chroma is None:
        from src.memory.ingest import get_or_create_chroma_collection as _get_chroma  # type: ignore

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
    max_tool_calls: int = 5,
    timeout: float = 180.0,
    endpoint: "LLMEndpoint | None" = None,
    disable_memory: bool = False,
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
        from src.memory.store import init_memory_db as _init_db  # type: ignore
    _get_chroma = get_or_create_chroma_collection
    if _get_chroma is None:
        from src.memory.ingest import get_or_create_chroma_collection as _get_chroma  # type: ignore

    conn = _init_db()
    chroma = _get_chroma()
    messages = [{"role": "user", "content": task}]

    # Ambient context — silent skip if no snapshot or memory disabled
    ambient_ctx = ""
    _trigger_ids: list[str] = []
    if not disable_memory:
        try:
            from src.memory.ambient import build_context
            ambient_ctx = build_context(
                task, conn, ollama_url, safe_repo_slug,
                _out_trigger_ids=_trigger_ids,
                chroma_collection=chroma,
            )
        except Exception:
            pass

    _effective_max_tool_calls = 0 if disable_memory else max_tool_calls
    _system = (system_prompt or _TASK_SYSTEM_PROMPT) + (f"\n\n{ambient_ctx}" if ambient_ctx else "")

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
            num_predict=2048,
        )
    except Exception as exc:
        conn.close()
        return f"[Pi-Agent Fehler] {exc}"

    print(f"  [Pi] Fertig — {tool_calls_made} Memory-Abfragen", flush=True)

    # Implicit feedback — silent fail
    if ambient_ctx and _trigger_ids:
        try:
            from src.memory.ambient import compute_feedback, update_trigger_stats
            _retrieval_happened = bool(ambient_ctx and "## Relevante Erinnerungen" in ambient_ctx)
            led_to_retrieval = tool_calls_made > 0 or _retrieval_happened
            fb = compute_feedback(ambient_ctx, content, _trigger_ids, led_to_retrieval, conn, ollama_url)
            update_trigger_stats(_trigger_ids, fb.was_referenced, conn)
        except Exception:
            pass

    conn.close()
    return content
