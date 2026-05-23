"""parse_llm_json: contract (dict|None) + no-silent-fail warning behaviour."""
from __future__ import annotations

from mayring_pi_agent.json_utils import parse_llm_json


def test_plain_json():
    assert parse_llm_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert parse_llm_json('```json\n{"file_summary": "ok"}\n```') == {"file_summary": "ok"}


def test_json_with_prose_around():
    out = parse_llm_json('Hier ist das Ergebnis: {"x": true}. Fertig.')
    assert out == {"x": True}


def test_nested_object():
    assert parse_llm_json('{"a": {"b": 1}}') == {"a": {"b": 1}}


def test_empty_input_is_silent_none(capsys):
    assert parse_llm_json("") is None
    assert parse_llm_json("   ") is None
    # empty is legitimate → must NOT warn
    assert capsys.readouterr().err == ""


def test_unparseable_nonempty_warns(capsys):
    # Model returned prose, no JSON → None, but LOUD (no silent fail).
    assert parse_llm_json("Es tut mir leid, ich kann das nicht.") is None
    assert "WARN" in capsys.readouterr().err
