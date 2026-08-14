"""Extra coverage tests for `pi_ai.cli`'s argument handling and error paths.

The provider list is stubbed by replacing `cli.oauth_providers`, and the stub
mirrors the real `registry.Provider` shape (`id`, `name`, `auth.oauth`). Do not
stub it with a differently-shaped object: an earlier version of these tests
used a sync `lambda: flow` in place of the provider table, which is what let
the `pp-ai login` crash ship with the suite green.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pi_ai import cli as cli_module
from pi_ai.cli import main


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


def stub_providers(monkeypatch, *providers: tuple[str, str, Any]) -> None:
    """Replace the CLI's provider list with fakes shaped like real providers."""
    fakes = [_FakeProvider(id=pid, name=name, auth=_FakeAuth(oauth=flow)) for pid, name, flow in providers]
    monkeypatch.setattr(cli_module, "oauth_providers", lambda: fakes)


# --------------------------------------------------------------------------
# login command: no provider arg — interactive provider selection (line 163-169)
# --------------------------------------------------------------------------


def test_login_no_provider_arg_prompts_and_succeeds(monkeypatch):
    """lines 163-169: 'login' with no arg prompts user to pick a number."""

    class _FakeFlow:
        async def login(self, interaction):
            class _C:
                data = {"access": "tok", "refresh": "ref"}  # noqa: RUF012

            return _C()

    stub_providers(monkeypatch, ("prov-a", "Prov A", _FakeFlow()))

    auth_file_holder: list = []

    async def _fake_login(provider_id, auth_file=cli_module.AUTH_FILE, **kw):
        auth_file_holder.append(provider_id)

    monkeypatch.setattr(cli_module, "login", _fake_login, raising=True)

    # Simulate user picking "1" at the interactive prompt
    monkeypatch.setattr("builtins.input", lambda _: "1")
    _lines, write = capture()
    result = main(["login"], write=write)
    assert result == 0
    assert auth_file_holder == ["prov-a"]


def test_login_no_provider_arg_invalid_choice_fails(monkeypatch):
    """lines 163-169: invalid numeric choice -> provider_id becomes None -> error."""
    stub_providers(monkeypatch, ("prov-a", "Prov A", None))
    monkeypatch.setattr("builtins.input", lambda _: "99")  # out of range
    lines, write = capture()
    result = main(["login"], write=write)
    assert result == 1
    assert any("Unknown provider" in line for line in lines)


def test_login_no_provider_arg_non_numeric_choice_fails(monkeypatch):
    """lines 163-169: non-numeric input -> ValueError -> provider_id becomes None."""
    stub_providers(monkeypatch, ("prov-a", "Prov A", None))
    monkeypatch.setattr("builtins.input", lambda _: "abc")
    _lines, write = capture()
    result = main(["login"], write=write)
    assert result == 1


# --------------------------------------------------------------------------
# login command: known provider, asyncio.run path (lines 173-178, 185)
# --------------------------------------------------------------------------


def test_login_provider_runs_flow_and_returns_0(tmp_path, monkeypatch):
    """lines 173-178: valid provider -> asyncio.run(login(...)) -> return 0."""

    class _FakeFlow:
        async def login(self, interaction):
            class _C:
                data = {"access": "tok", "refresh": "ref"}  # noqa: RUF012

            return _C()

    auth_file = tmp_path / "auth.json"

    async def _fake_login(provider_id, auth_file=auth_file, **kw):
        cli_module.save_auth({"prov-b": {"access": "tok"}}, auth_file)
        kw.get("write", print)(f"\nCredentials saved to {auth_file}")

    stub_providers(monkeypatch, ("prov-b", "Prov B", _FakeFlow()))
    monkeypatch.setattr(cli_module, "login", _fake_login, raising=True)

    _lines, write = capture()
    result = main(["login", "prov-b"], write=write)
    assert result == 0
    assert json.loads(auth_file.read_text(encoding="utf-8")) == {"prov-b": {"access": "tok"}}


def test_login_provider_exception_returns_1(monkeypatch):
    """line 185: exception from asyncio.run -> write error -> return 1."""

    async def _bad_login(*_a, **_kw):
        raise RuntimeError("auth server down")

    stub_providers(monkeypatch, ("prov-c", "Prov C", None))
    monkeypatch.setattr(cli_module, "login", _bad_login, raising=True)

    lines, write = capture()
    result = main(["login", "prov-c"], write=write)
    assert result == 1
    assert any("auth server down" in line for line in lines)


# --------------------------------------------------------------------------
# __main__ guard (line 110->exit)
# --------------------------------------------------------------------------


def test_main_returns_zero_for_help():
    _lines, write = capture()
    assert main(["--help"], write=write) == 0


# --------------------------------------------------------------------------
# login command: unknown provider with explicit name (line 91, 106)
# --------------------------------------------------------------------------


def test_login_unknown_provider_returns_error(monkeypatch):
    """line 91,106: find_provider returns None -> write error -> return 1."""
    stub_providers(monkeypatch, ("known", "Known", None))
    lines, write = capture()
    result = main(["login", "unknown-one"], write=write)
    assert result == 1
    assert any("Unknown provider" in line for line in lines)
