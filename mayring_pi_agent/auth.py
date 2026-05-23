"""Minimal auth dependency for the standalone Pi-Agent service (#266).

Beim Auslagern aus MayringCoder ersetzt dies das dortige
``src.api.auth.get_token_info``. Diese Variante ist ein **Bearer-Token-Gate**:
sie verlangt einen ``Authorization: Bearer <token>``-Header und gibt den
rohen Token zurück.

⚠ Vor Production: durch echte JWT-Validierung gegen das gemeinsame
MayringCoder-Secret ersetzen (gleicher Contract wie src/api/jwt_auth.py),
sonst akzeptiert der Service jeden nicht-leeren Token. Siehe README → Deploy.
"""
from __future__ import annotations

import os

from fastapi import Header, HTTPException


def get_token_info(authorization: str = Header(default="")) -> str:
    """Require a Bearer token; return the raw token. Dev-Gate, kein voller JWT-Check.

    ``PI_AGENT_ALLOW_ANON=1`` deaktiviert das Gate (nur lokale Entwicklung).
    """
    if os.getenv("PI_AGENT_ALLOW_ANON") == "1":
        return ""
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="empty bearer token")
    return token
