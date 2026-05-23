# mayring-pi-agent

Read-only Pi-Agent als eigenständiger Microservice — aus MayringCoder ausgelagert
(Stufe 3 der Modularisierung, [#266](https://github.com/Nileneb/MayringCoder/issues/266)).
History via `git subtree split` erhalten.

Der Pi-Agent kategorisiert/analysiert über lokales **Ollama** mit Tool-Calling
(`search_memory`, `search_wiki`, `read_file`, `web_fetch`, `plan`) — **read-only**,
keine file-writes / shell-execution (Security, MayringCoder PR #224).

## Struktur

```
mayring_pi_agent/
├── pi.py           # Agent-Loop + Tools (web_fetch, plan, …)
├── pi_jobs.py      # Job-Klassen, classify_pi_job
├── pi_queue.py     # In-Process-Queue
├── pi_worker.py    # Worker-Coroutines
├── pi_server.py    # FastAPI-Service (Port 8091): POST /task, GET /health
├── vision.py       # Vision-Captioning (Pillow + Ollama)
├── auth.py         # Bearer-Token-Gate (⚠ vor Prod: echte JWT-Validierung)
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

```bash
export OLLAMA_URL=http://three.linn.games:11434   # niemals Docker-Service ohne GPU
export PI_WEB_FETCH_ALLOWLIST=docs.python.org,github.com   # web_fetch Allow-List
mayring-pi-agent            # uvicorn auf :8091  (PI_PORT überschreibbar)
# health: curl localhost:8091/health
```

## Vor Production zu erledigen (Cutover, siehe MayringCoder docs/pi-agent-extraction-266.md)

- [ ] `auth.py`: Bearer-Gate durch echte JWT-Validierung gegen das gemeinsame
      MayringCoder-Secret ersetzen (Contract: `src/api/jwt_auth.py`).
- [ ] `search_wiki` / `analyze_with_memory`: optionale `src.wiki_v2` / `src.analysis`
      sind im Standalone-Service nicht vorhanden — entweder per HTTP gegen
      MayringCoder lösen oder als „nicht verfügbar" belassen (aktuell graceful).
- [ ] `mayring-core` als installierbare Dependency stabilisieren (publish o. pin).
- [ ] CI: `build-and-push` + Deploy-Target (eigener Container).
- [ ] In MayringCoder: `src/api/mcp_agent_tools.py` voll auf `PI_AGENT_URL`-HTTP
      umstellen, dann `src/agents/` entfernen. Latenz-Delta < 50ms messen.
