# mayring-pi-agent

> **Ökosystem:** Teil des 4-Repo-MayringCoder-Systems (Pi-Agent Microservice, :8091).
> Gesamtkarte: [`MayringCoder/ARCHITECTURE.md`](https://github.com/Nileneb/MayringCoder/blob/master/ARCHITECTURE.md).
> Eingebunden als Git-Submodule `vendor/mayring-pi-agent`.

Read-only Pi-Agent als eigenständiger Microservice — aus MayringCoder ausgelagert
(Stufe 3 der Modularisierung, [#266](https://github.com/Nileneb/MayringCoder/issues/266)).
History via `git subtree split` erhalten.

Der Pi-Agent kategorisiert/analysiert über lokales **Ollama** mit Tool-Calling
(`search_memory`, `search_wiki`, `read_file`, `web_fetch`, `plan`).

**Zwei Deployment-Profile aus einem Paket** (gesteuert über Env-Flags, default-OFF):

- **Cloud/Server** (mcp.linn.games, app.linn.games): **read-only + sandboxed**.
  Keine write/exec-Flags → kein file-write, keine shell-execution; `read_file`
  ist auf `PI_FS_ROOT` beschränkt (Security, MayringCoder PR #224).
- **Lokaler Worker** (eigene Maschine + lokales Ollama): zieht Jobs aus der
  Cloud-Queue und führt sie lokal aus. Schreibt/exekutiert NUR wenn explizit
  per `PI_ALLOW_WRITE` / `PI_ALLOW_EXEC` aktiviert — beide sandboxed auf
  `PI_FS_ROOT`. Die Worker-Capabilities (`write`/`exec`) werden aus denselben
  Flags abgeleitet, damit die Queue keine write-Jobs an einen read-only-Worker
  routet.

## Struktur

```
mayring_pi_agent/
├── pi.py           # Agent-Loop + Tools (web_fetch, plan, …)
├── pi_jobs.py      # Job-Klassen, classify_pi_job
├── pi_queue.py     # In-Process-Queue
├── pi_worker.py    # Worker-Coroutines
├── pi_server.py    # FastAPI-Service (Port 8091): POST /task, GET /health
├── vision.py       # Vision-Captioning (Pillow + Ollama)
├── auth.py         # RS256-JWT-Validierung (Contract: MayringCoder src/api/jwt_auth.py)
└── json_utils.py   # lokaler LLM-JSON-Parser (Fallback für src.analysis)
```

## Dependency: mayring-core

Der Agent importiert `mayring_core.*` (memory/llm/ollama_client/model_router).
`mayring-core` lebt im MayringCoder-Repo unter `core/` und wird als
git-subdirectory-Dependency gezogen (siehe `pyproject.toml`). **Resolvt erst,
wenn MayringCoder `master` das `core/`-Package enthält** (PR #273).

Lokale Entwicklung gegen ein MayringCoder-Checkout:
```bash
pip install -e /pfad/zu/MayringCoder/core
pip install -e .
```

## Run

**Cloud/Server (read-only + sandbox):**
```bash
export OLLAMA_URL=https://three.linn.games   # Reverse-Proxy, KEIN Port (niemals :11434). Lokal: http://localhost:11434
export PI_WEB_FETCH_ALLOWLIST=docs.python.org,github.com   # web_fetch Allow-List
export PI_FS_ROOT=/srv/repos                       # read_file-Sandbox (PFLICHT, sonst read_file deaktiviert)
# JWT-Auth (RS256) — Public Key kommt von app.linn.games (dort liegt der Private Key):
export JWT_PUBLIC_KEY_PATH=/etc/mayring/jwt_public.pem
export JWT_ISSUER=https://app.linn.games           # default, überschreibbar
export JWT_AUDIENCE=mayringcoder                   # default, überschreibbar
mayring-pi-agent            # uvicorn auf :8091  (PI_PORT überschreibbar)
# health: curl localhost:8091/health
```

**Lokaler Worker (write-enabled):** zusätzlich
```bash
export PI_ALLOW_WRITE=1     # write_file in PI_FS_ROOT erlauben
export PI_ALLOW_EXEC=1      # bash/Tests erlauben (separate, höhere Berechtigung)
export PI_FS_ROOT=/         # oder ein Projekt-Root; Sandbox für write/read
# Cloud-Polling nutzt ~/.config/mayring/hook.jwt (MCP-Login) → MAYRING_API_URL
```

## Vor Production zu erledigen (Cutover, siehe MayringCoder docs/pi-agent-extraction-266.md)

- [x] `auth.py`: echte RS256-JWT-Validierung gegen den MayringCoder-Contract
      (`src/api/jwt_auth.py`). Public Key via `JWT_PUBLIC_KEY_PATH`.
- [x] `read_file`-Sandbox (`PI_FS_ROOT`, fail-closed) + write/exec-Capability-Gates.
- [ ] `search_wiki` / `analyze_with_memory`: optionale `src.wiki_v2` / `src.analysis`
      sind im Standalone-Service nicht vorhanden — entweder per HTTP gegen
      MayringCoder lösen oder als „nicht verfügbar" belassen (aktuell graceful).
- [ ] `mayring-core` als installierbare Dependency stabilisieren (publish o. pin).
- [ ] CI: `build-and-push` + Deploy-Target (eigener Container).
- [ ] In MayringCoder: `src/api/mcp_agent_tools.py` voll auf `PI_AGENT_URL`-HTTP
      umstellen, dann `src/agents/` entfernen. Latenz-Delta < 50ms messen.
