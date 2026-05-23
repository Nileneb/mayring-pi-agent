"""RS256 JWT auth tests — mirrors MayringCoder tests/test_jwt_auth.py for the
claim contract, plus the standalone get_token_info() FastAPI dependency."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

import jwt as pyjwt

from mayring_pi_agent import auth


@pytest.fixture(scope="module")
def rsa_keys() -> tuple[str, str]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


@pytest.fixture
def configured_env(rsa_keys, tmp_path: Path, monkeypatch):
    private_pem, public_pem = rsa_keys
    key_file = tmp_path / "jwt_public.pem"
    key_file.write_text(public_pem)
    monkeypatch.setenv("JWT_PUBLIC_KEY_PATH", str(key_file))
    monkeypatch.setenv("JWT_ISSUER", "https://app.linn.games")
    monkeypatch.setenv("JWT_AUDIENCE", "mayringcoder")
    monkeypatch.delenv("PI_AGENT_ALLOW_ANON", raising=False)
    auth.reset_public_key_cache()
    yield private_pem
    auth.reset_public_key_cache()


def _sign(private_pem: str, **claims) -> str:
    claims.setdefault("scope", ["mcp:memory"])
    claims.setdefault("sub", "2")
    claims.setdefault("email", "test@example.com")
    claims.setdefault("iss", "https://app.linn.games")
    claims.setdefault("aud", "mayringcoder")
    claims.setdefault("exp", int(time.time()) + 300)
    return pyjwt.encode(claims, private_pem, algorithm="RS256")


# --- validate_jwt_token --------------------------------------------------

def test_valid_token(configured_env):
    info = auth.validate_jwt_token(_sign(configured_env, sub="2", email="bene@linn.games"))
    assert info is not None
    assert info.workspace_id == "bene"
    assert info.sub == "2"
    assert info.is_admin is False


def test_admin_scope(configured_env):
    info = auth.validate_jwt_token(_sign(configured_env, scope=["mcp:memory", "admin"]))
    assert info is not None and info.is_admin is True


def test_admin_scope_space_string(configured_env):
    info = auth.validate_jwt_token(_sign(configured_env, scope="mcp:memory admin"))
    assert info is not None and info.is_admin is True


def test_missing_mcp_memory_scope_rejected(configured_env):
    assert auth.validate_jwt_token(_sign(configured_env, scope=["paper-search:read"])) is None


def test_expired_token(configured_env):
    assert auth.validate_jwt_token(_sign(configured_env, exp=int(time.time()) - 1)) is None


def test_missing_exp_rejected(configured_env):
    # exp omitted → options require=["exp"] rejects it
    tok = pyjwt.encode(
        {"iss": "https://app.linn.games", "aud": "mayringcoder",
         "sub": "9", "email": "a@b.de", "scope": ["mcp:memory"]},
        configured_env, algorithm="RS256",
    )
    assert auth.validate_jwt_token(tok) is None


def test_wrong_issuer(configured_env):
    assert auth.validate_jwt_token(_sign(configured_env, iss="https://evil.example")) is None


def test_wrong_audience(configured_env):
    assert auth.validate_jwt_token(_sign(configured_env, aud="other-service")) is None


def test_tampered_signature(configured_env):
    tok = _sign(configured_env)
    assert auth.validate_jwt_token(tok[:-4] + "AAAA") is None


def test_missing_email_rejected(configured_env):
    tok = pyjwt.encode(
        {"iss": "https://app.linn.games", "aud": "mayringcoder",
         "exp": int(time.time()) + 300, "sub": "9", "scope": ["mcp:memory"]},
        configured_env, algorithm="RS256",
    )
    assert auth.validate_jwt_token(tok) is None


def test_blank_sub_rejected(configured_env):
    assert auth.validate_jwt_token(_sign(configured_env, sub="   ")) is None


def test_unknown_version_rejected(configured_env):
    assert auth.validate_jwt_token(_sign(configured_env, version=3)) is None


def test_empty_token(configured_env):
    assert auth.validate_jwt_token("") is None


def test_no_public_key_configured(monkeypatch):
    monkeypatch.delenv("JWT_PUBLIC_KEY_PATH", raising=False)
    auth.reset_public_key_cache()
    try:
        assert auth.validate_jwt_token("any.token.here") is None
    finally:
        auth.reset_public_key_cache()


# --- get_token_info (FastAPI dependency) ---------------------------------

def test_get_token_info_valid(configured_env):
    info = auth.get_token_info(f"Bearer {_sign(configured_env, email='x@y.de')}")
    assert info.workspace_id == "x"


def test_get_token_info_missing_bearer(configured_env):
    with pytest.raises(HTTPException) as e:
        auth.get_token_info("")
    assert e.value.status_code == 401


def test_get_token_info_invalid_token(configured_env):
    with pytest.raises(HTTPException) as e:
        auth.get_token_info("Bearer garbage")
    assert e.value.status_code == 401


def test_get_token_info_anon_bypass(monkeypatch):
    monkeypatch.setenv("PI_AGENT_ALLOW_ANON", "1")
    info = auth.get_token_info("")
    assert info.workspace_id == "anon"
