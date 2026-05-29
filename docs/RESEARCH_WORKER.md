# Research Worker (A2A cloud-pull)

Ein Laptop-Agent, der tiefergehende Recherchen ausführt und über `mcp.linn.games`
(A2A-Protokoll) aus Langdock oder jedem A2A-Client angestoßen wird. Der Laptop wählt nur
ausgehend (kein NAT-/Port-Problem), zieht Jobs aus der Cloud-Queue und rechnet lokal mit
Ollama + SearXNG + Cloud-Memory.

```
Langdock ──A2A message/send──▶ mcp.linn.games/a2a ──cloud-job(cap=research)──▶ pi_jobs
   ▲                                                                              │
   └──A2A tasks/get (poll)──── result ◀── complete_cloud ◀── dieser Worker ──claim┘
```

## Voraussetzungen

- Lokales Ollama mit `qwen3.5:9b` (`ollama pull qwen3.5:9b`).
- `~/.config/mayring/hook.jwt` — der JWT, dessen Workspace == der des Cloud-Gateways
  (`MAYRING_A2A_WORKSPACE_ID`, default `019e14d6`). Sonst findet der claim die Jobs nie.
- `pip install -e .` (editable) in dieser Umgebung.

## Start (systemd-user, empfohlen)

```bash
cp deploy/mayring-research-worker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now mayring-research-worker
journalctl --user -u mayring-research-worker -f
```

## Start (manuell, zum Testen)

```bash
MAYRING_API_URL=https://mcp.linn.games \
OLLAMA_URL=http://localhost:11434 \
PI_WORKER_CAPABILITIES=research \
PI_NUM_PREDICT=6000 \
PI_WEB_FETCH_ALLOWLIST='*' \
python -m mayring_pi_agent.pi_worker
```

## Wie es scoped (klaut Prod-Pi keine Jobs)

Cloud-Research-Jobs tragen `capability_required="research"`. Nur ein Worker, der `research`
advertised (`PI_WORKER_CAPABILITIES=research`), claimt sie (`_capability_match`: required ⊆ caps).
`research` ist nicht-privilegiert → wird auch ohne Registry-Registrierung honoriert. Der Prod-Pi
(andere/keine caps) lässt Research-Jobs liegen.

## Tools des Worker-Agents

- `search_memory` — Cloud-Memory (live, cloud-first).
- `web_search` — SearXNG via `mcp.linn.games/searxng` (Bearer-JWT).
- `web_fetch` — beliebige URL laden (`PI_WEB_FETCH_ALLOWLIST=*` = unrestricted, nur lokal!).
- `ingest` — Findings dauerhaft ins Memory schreiben.

## Langdock anbinden

In Langdock einen A2A-Agent anlegen mit:
- Agent-Card: `https://mcp.linn.games/.well-known/agent-card.json`
- Auth: `Authorization: Bearer <hook.jwt>`

Langdock entdeckt den `deep-research`-Skill, schickt `message/send` und pollt `tasks/get` bis
`COMPLETED`. Lange Aufträge sind ok (async).
