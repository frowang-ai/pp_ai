"""Kimi Coding requests must identify as pi.

Port of the `pi-user-agent` half of upstream `fix(ai): use pi user agent for
Kimi Coding requests`. The generated catalog gives Kimi models a
`User-Agent: KimiCLI/1.5`, which claims to be a different client.
"""

from __future__ import annotations

import re

import pytest
from pi_ai.api.anthropic_messages import merge_client_headers, merge_headers
from pi_ai.model_catalog import load_models
from pi_ai.utils.pi_user_agent import get_pi_user_agent


def _model(provider: str, model_id: str | None = None):
    models = load_models(provider)
    if model_id is None:
        return models[0]
    return next(model for model in models if model.id == model_id)


def test_the_user_agent_names_pi_and_the_platform() -> None:
    assert re.fullmatch(r"pi \(\S+ \S+; \S+\)", get_pi_user_agent())


def test_kimi_requests_replace_the_catalog_user_agent() -> None:
    kimi = _model("kimi-coding", "k3")
    assert kimi.headers is not None
    # The catalog ships the header this replaces; if that ever changes, the
    # replacement is no longer doing anything.
    assert any(name.lower() == "user-agent" for name in kimi.headers)

    merged = merge_client_headers(kimi, kimi.headers, {"accept": "application/json"})

    assert merged["User-Agent"] == get_pi_user_agent()
    assert merged["accept"] == "application/json"


@pytest.mark.parametrize("spelling", ["User-Agent", "user-agent", "USER-AGENT"])
def test_any_spelling_of_the_incoming_header_is_replaced(spelling: str) -> None:
    """Header names are case-insensitive, so a stale one must not survive."""
    kimi = _model("kimi-coding", "k3")

    merged = merge_client_headers(kimi, {spelling: "KimiCLI/1.5"})

    assert [name for name in merged if name.lower() == "user-agent"] == ["User-Agent"]
    assert merged["User-Agent"] == get_pi_user_agent()


def test_other_providers_keep_their_user_agent() -> None:
    anthropic = _model("anthropic")

    merged = merge_client_headers(anthropic, {"User-Agent": "Something/1.0"})

    assert merged["User-Agent"] == "Something/1.0"


def test_merge_client_headers_otherwise_matches_merge_headers() -> None:
    anthropic = _model("anthropic")
    sources = ({"a": "1"}, None, {"b": "2", "a": "3"})

    assert merge_client_headers(anthropic, *sources) == merge_headers(*sources)
