"""Local JSON-from-LLM parser (standalone fallback for src.analysis.analyzer).

Im MayringCoder-Monorepo nutzt pi.py ``src.analysis.analyzer._parse_llm_json``;
im ausgelagerten Service gibt es src.analysis nicht, daher diese robuste
Variante: extrahiert das erste JSON-Objekt aus einer LLM-Antwort (toleriert
```json-Fences und Prosa drumherum).

WHY(no-silent-fail): ``None`` ist ein legitimes Ergebnis für *leeren* Input,
aber wenn das LLM nicht-leeren Text liefert, der KEIN valides JSON enthält
(z.B. bei API-Ausfällen / verschluckter Antwort), wird das jetzt LAUT auf
stderr gemeldet statt still verschluckt. Der Rückgabe-Contract (dict | None)
bleibt, damit der Caller in pi.py (``_parse_error``-Branch) unverändert greift.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any


def parse_llm_json(raw: str) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None  # empty input is a legitimate None — no warning
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()

    for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            obj = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj

    # Greedy fallback: a single object spanning nested braces (the non-greedy
    # pass above stops at the first '}' and misses nested structures).
    greedy = re.search(r"\{.*\}", text, re.DOTALL)
    if greedy:
        try:
            obj = json.loads(greedy.group(0))
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # WHY(no-silent-fail): non-empty input that yields no JSON object is an
    # actual failure (model returned prose / truncated / API hiccup). Surface
    # it loudly so it isn't mistaken for "model had nothing to say".
    print(
        f"  [json_utils] WARN: kein valides JSON-Objekt in LLM-Antwort "
        f"({len(raw)} Zeichen): {text[:120]!r}",
        file=sys.stderr,
        flush=True,
    )
    return None
