"""Tests for the pi_ai OAuth login CLI.

No test may launch a browser, hit the network, or read the real auth.json:
the interaction and the auth file location are both injectable.

Stubs here mimic the *real* shape: a fake provider exposes `auth.oauth.login`
as a coroutine function, exactly as `lazy_oauth` does. An earlier version of
this file stubbed the provider table with a sync `lambda: flow`, which let
`cli.login` call an `async` loader without awaiting it and still pass --
`pp-ai login github-copilot` crashed with "'coroutine' object has no
attribute 'login'" while the suite stayed green. `test_every_real_provider_*`
below runs against the real registry so that class of bug cannot hide again.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any

import pytest
from pi_ai.auth.types import AuthEvent, AuthInteraction, AuthPrompt
from pi_ai.cli import (
    ConsoleInteraction,
    find_provider,
    load_auth,
    main,
    oauth_providers,
    provider_ids,
    save_auth,
)
from pi_ai.utils.abort import AbortSignal


def capture():
    lines: list[str] = []
    return lines, lines.append


@dataclass
class _FakeAuth:
    oauth: Any


@dataclass
class _FakeProvider:
    """Mirrors the parts of `registry.Provider` the CLI reads."""

    id: str
    name: str
    auth: _FakeAuth


def _fake_provider(provider_id: str, name: str, flow: Any) -> _FakeProvider:
    return _FakeProvider(id=provider_id, name=name, auth=_FakeAuth(oauth=flow))


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def test_help_lists_commands_and_providers():
    lines, write = capture()
    assert main(["--help"], write=write) == 0
    output = "\n".join(lines)
    assert "login [provider]" in output
    assert "list" in output
    for provider in oauth_providers():
        assert provider.id in output


@pytest.mark.parametrize("argv", [[], ["help"], ["-h"], ["--help"]])
def test_all_help_forms(argv):
    lines, write = capture()
    assert main(argv, write=write) == 0
    assert "Usage:" in lines[0]


def test_list_prints_every_provider():
    lines, write = capture()
    assert main(["list"], write=write) == 0
    assert len(lines) == len(oauth_providers())
    assert lines[0].startswith("anthropic")


def test_unknown_command_fails():
    lines, write = capture()
    assert main(["bogus"], write=write) == 1
    assert "Unknown command: bogus" in lines[0]


def test_login_with_unknown_provider_fails():
    lines, write = capture()
    assert main(["login", "nope"], write=write) == 1
    assert "Unknown provider: nope" in lines[0]


# --------------------------------------------------------------------------
# provider registry
# --------------------------------------------------------------------------


def test_provider_ids_are_unique():
    ids = provider_ids()
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("provider_id", [provider.id for provider in oauth_providers()])
def test_every_real_provider_resolves_and_exposes_an_awaitable_login(provider_id):
    """The regression guard for the `pp-ai login` crash.

    `login` must be a coroutine function on the *real* provider: the previous
    bug was that the CLI held a raw `async` loader and called it unawaited, so
    `login` was reached on a coroutine object instead of a flow.
    """
    provider = find_provider(provider_id)
    assert provider is not None
    assert provider.auth.oauth is not None
    assert inspect.iscoroutinefunction(provider.auth.oauth.login)


def test_find_provider_returns_none_for_unknown():
    assert find_provider("nope") is None


# --------------------------------------------------------------------------
# credential storage
# --------------------------------------------------------------------------


def test_load_auth_returns_empty_for_a_missing_file(tmp_path):
    assert load_auth(tmp_path / "absent.json") == {}


def test_load_auth_returns_empty_for_a_corrupt_file(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_auth(path) == {}


def test_load_auth_returns_empty_for_a_non_object(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text("[1, 2]", encoding="utf-8")
    assert load_auth(path) == {}


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "auth.json"
    save_auth({"anthropic": {"access": "x"}}, path)
    assert load_auth(path) == {"anthropic": {"access": "x"}}
    # Written indented, matching the TypeScript.
    assert "\n  " in path.read_text(encoding="utf-8")


def test_save_auth_preserves_other_providers(tmp_path):
    path = tmp_path / "auth.json"
    save_auth({"a": 1}, path)
    existing = load_auth(path)
    existing["b"] = 2
    save_auth(existing, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}


# --------------------------------------------------------------------------
# terminal interaction
# --------------------------------------------------------------------------


def make_interaction(answers: list[str]):
    lines: list[str] = []
    remaining = list(answers)
    interaction = ConsoleInteraction(
        signal=AbortSignal(),
        read_line=lambda _prompt: remaining.pop(0),
        write=lines.append,
    )
    return interaction, lines


async def test_text_prompt_returns_the_typed_value():
    interaction, _lines = make_interaction(["typed"])
    prompt = AuthPrompt(type="text", message="Enter key")
    assert await interaction.prompt(prompt) == "typed"


async def test_text_prompt_shows_the_placeholder():
    interaction, _lines = make_interaction(["v"])
    prompt = AuthPrompt(type="text", message="Enter host", placeholder="company.example")
    captured: list[str] = []
    interaction.read_line = lambda text: (captured.append(text), "v")[1]
    await interaction.prompt(prompt)
    assert "company.example" in captured[0]


async def test_select_prompt_returns_the_chosen_option_id():
    interaction, lines = make_interaction(["2"])
    prompt = AuthPrompt(
        type="select",
        message="Pick one",
        options=({"id": "a", "label": "Alpha"}, {"id": "b", "label": "Beta"}),
    )
    assert await interaction.prompt(prompt) == "b"
    assert any("Alpha" in line for line in lines)


async def test_select_prompt_rejects_an_out_of_range_choice():
    interaction, _lines = make_interaction(["9"])
    prompt = AuthPrompt(type="select", message="Pick", options=({"id": "a", "label": "Alpha"},))
    with pytest.raises(ValueError, match="Invalid selection"):
        await interaction.prompt(prompt)


async def test_select_prompt_rejects_a_non_numeric_choice():
    interaction, _lines = make_interaction(["abc"])
    prompt = AuthPrompt(type="select", message="Pick", options=({"id": "a", "label": "Alpha"},))
    with pytest.raises(ValueError, match="Invalid selection"):
        await interaction.prompt(prompt)


def test_notify_prints_an_auth_url():
    interaction, lines = make_interaction([])
    interaction.notify(AuthEvent(type="auth_url", url="https://example.invalid/auth"))
    assert any("https://example.invalid/auth" in line for line in lines)


def test_notify_prints_a_device_code():
    interaction, lines = make_interaction([])
    interaction.notify(
        AuthEvent(type="device_code", verification_uri="https://example.invalid/device", user_code="ABCD-1234")
    )
    joined = "\n".join(lines)
    assert "https://example.invalid/device" in joined
    assert "ABCD-1234" in joined


def test_notify_prints_info_and_progress_messages():
    interaction, lines = make_interaction([])
    interaction.notify(AuthEvent(type="info", message="hello"))
    interaction.notify(AuthEvent(type="progress", message="working"))
    assert lines == ["hello", "working"]


# --------------------------------------------------------------------------
# login
# --------------------------------------------------------------------------


async def test_login_persists_the_returned_credential(tmp_path, monkeypatch):
    from pi_ai import cli as cli_module

    class FakeCredential:
        def __init__(self) -> None:
            self.data = {"access": "tok", "refresh": "ref"}

    class FakeFlow:
        def __init__(self) -> None:
            self.interactions: list[AuthInteraction] = []

        async def login(self, interaction):
            self.interactions.append(interaction)
            return FakeCredential()

    flow = FakeFlow()
    monkeypatch.setattr(cli_module, "oauth_providers", lambda: [_fake_provider("fake", "Fake Provider", flow)])

    auth_file = tmp_path / "auth.json"
    lines, write = capture()
    await cli_module.login("fake", auth_file=auth_file, write=write)

    assert json.loads(auth_file.read_text(encoding="utf-8")) == {"fake": {"access": "tok", "refresh": "ref"}}
    assert any("Credentials saved" in line for line in lines)
    assert len(flow.interactions) == 1


async def test_login_rejects_an_unknown_provider(tmp_path):
    from pi_ai import cli as cli_module

    with pytest.raises(ValueError, match="Unknown provider"):
        await cli_module.login("nope", auth_file=tmp_path / "auth.json")
