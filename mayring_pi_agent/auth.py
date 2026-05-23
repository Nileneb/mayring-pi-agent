"""RS256 JWT auth for the standalone Pi-Agent service (#266).

Mirrors the MayringCoder contract in ``src/api/jwt_auth.py`` (RS256,
public-key validation): app.linn.games signs tokens with the private key,
this service holds only the matching public key. Required claims:
``exp, iss, aud, email, sub`` and ``scope ∋ "mcp:memory"``.

Required env:
    JWT_PUBLIC_KEY_PATH   Path to the RS256 public-key PEM file
    JWT_ISSUER            Expected "iss"  (default https://app.linn.games)
    JWT_AUDIENCE          Expected "aud"  (default mayringcoder)

WHY(#224/#266, SECURITY): the previous placeholder accepted ANY non-empty
bearer token. Combined with the unsandboxed ``read_file`` tool that means
arbitrary server-file exfiltration to anyone who can reach ``/task``. The
source-of-truth claim contract lives in MayringCoder ``src/api/jwt_auth.py`` —
keep the validated claim set in sync there. This service only needs caller
identity (workspace_id) + scopes, so the membership/llm-routing claims from
the orchestrator's TokenInfo are intentionally dropped (YAGNI).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jwt  # PyJWT — RS256 requires the `cryptography` extra (see pyproject)
from fastapi import Header, HTTPException

from mayring_core.identity.workspace_resolver import email_to_slug


@dataclass(frozen=True)
class TokenInfo:
    """Authenticated caller identity extracted from a valid JWT."""

    workspace_id: str
    scopes: tuple[str, ...] = ()
    sub: str | None = None

    @property
    def is_admin(self) -> bool:
        return "admin" in self.scopes


@lru_cache(maxsize=1)
def _public_key() -> str | None:
    path = os.getenv("JWT_PUBLIC_KEY_PATH", "").strip()
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        # WHY(SECURITY): fail closed (reject all) — but make a *configured*
        # yet unreadable key loud, so a deploy misconfig isn't a silent 401-all.
        print(f"  [Pi-auth] WARN: JWT_PUBLIC_KEY_PATH set but unreadable: {exc}", flush=True)
        return None


def _issuer() -> str:
    return os.getenv("JWT_ISSUER", "https://app.linn.games")


def _audience() -> str:
    return os.getenv("JWT_AUDIENCE", "mayringcoder")


def validate_jwt_token(token: str) -> TokenInfo | None:
    """Return TokenInfo for a valid RS256 JWT, or None.

    Rejects: missing/expired/invalid-sig/wrong-iss/wrong-aud, unknown contract
    version, missing sub/email, or missing ``mcp:memory`` scope.
    """
    if not token:
        return None
    public_key = _public_key()
    if not public_key:
        return None

    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            issuer=_issuer(),
            audience=_audience(),
            options={"require": ["exp", "iss", "aud"]},
        )
    except jwt.InvalidTokenError:
        return None

    # WHY(v2-stufe4-contract): V2-JWTs carry version=2. Unknown future versions
    # are rejected; V1 tokens (no version field) stay accepted for backward-compat.
    contract_version = payload.get("version")
    if contract_version is not None and contract_version != 2:
        return None

    sub_raw = payload.get("sub")
    if sub_raw is None or not str(sub_raw).strip():
        return None
    sub_str = str(sub_raw).strip()

    # workspace_id is the email-slug home bucket (bene@linn.games → "bene"),
    # not the JWT's `workspace_id` claim — same rule as the orchestrator.
    workspace_id = email_to_slug(payload.get("email") or "")
    if not workspace_id:
        return None

    raw_scopes = payload.get("scope", [])
    if isinstance(raw_scopes, str):
        scopes = tuple(s for s in raw_scopes.split() if s)
    elif isinstance(raw_scopes, list):
        scopes = tuple(str(s) for s in raw_scopes if s)
    else:
        scopes = ()

    # WHY(SECURITY): a token minted for another service (e.g. paper-search)
    # must not work here even if iss/aud happen to match.
    if "mcp:memory" not in scopes:
        return None

    return TokenInfo(workspace_id=workspace_id, scopes=scopes, sub=sub_str)


def reset_public_key_cache() -> None:
    """Clear the cached public key (tests / key rotation)."""
    _public_key.cache_clear()


def get_token_info(authorization: str = Header(default="")) -> TokenInfo:
    """FastAPI dependency: require + validate an RS256 Bearer JWT.

    ``PI_AGENT_ALLOW_ANON=1`` bypasses validation — **local development only**.
    """
    if os.getenv("PI_AGENT_ALLOW_ANON") == "1":
        # WHY(SECURITY): dev-only escape hatch. Loud on purpose — if this ever
        # fires on a server it means auth is OFF (env-bleed footgun, cf. #88).
        print("  [Pi-auth] WARN: PI_AGENT_ALLOW_ANON=1 — auth DISABLED (dev only)", flush=True)
        return TokenInfo(workspace_id="anon", scopes=("mcp:memory",))
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization[7:].strip()
    info = validate_jwt_token(token)
    if info is None:
        raise HTTPException(status_code=401, detail="invalid token")
    return info
