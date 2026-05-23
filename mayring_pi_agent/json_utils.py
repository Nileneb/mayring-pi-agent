"""Local JSON-from-LLM parser (standalone fallback for src.analysis.analyzer).

Im MayringCoder-Monorepo nutzt pi.py ``src.analysis.analyzer._parse_llm_json``;
im ausgelagerten Service gibt es src.analysis nicht, daher diese minimale,
robuste Variante: extrahiert das erste JSON-Objekt aus einer LLM-Antwort
(toleriert ```json-Fences und Prosa drumherum).
"""
from __future__ import annotations

import json
import re
from typing import Any


def parse_llm_json(raw: str) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    for match in re.finditer(r"\{.*\}", text, re.DOTALL):
        try:
            obj = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None
