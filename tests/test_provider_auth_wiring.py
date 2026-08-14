"""Shape guards for the *real* built-in providers.

Not a port of a TypeScript test file. This pins the wiring that the
`pp-ai login github-copilot` crash (`'coroutine' object has no attribute
'login'`) exposed: the CLI held a raw `async def load_*_oauth` and called it
without awaiting, and the CLI's own tests stubbed the provider table with a
*sync* callable returning a ready flow, so the stub was satisfiable in a way
the real provider was not.

`tests/test_cli.py` guards `provider.auth.oauth.login` on the real providers.
The same latent defect applies to every other lazily loaded hook and to the
loaders themselves, so those are checked here against the real objects rather
than against a fake. Everything below is offline: constructing a provider and
awaiting a loader only builds flow objects, it never opens a connection.
"""

from __future__ import annotations

import inspect
import math

import pytest
from pi_ai.auth import oauth as oauth_barrel
from pi_ai.auth.oauth import load as oauth_load
from pi_ai.auth.types import Credential, OAuthAuth, ResolvedAuth
from pi_ai.providers.all import builtin_providers
from pi_ai.registry import Provider


def oauth_provider_ids() -> list[str]:
    return [provider.id for provider in builtin_providers() if provider.auth.oauth is not None]


def api_key_provider_ids() -> list[str]:
    return [
        provider.id
        for provider in builtin_providers()
        if provider.auth.api_key is not None and provider.auth.api_key.login is not None
    ]


def find_builtin(provider_id: str) -> Provider:
    return next(provider for provider in builtin_providers() if provider.id == provider_id)


@pytest.mark.parametrize("provider_id", oauth_provider_ids())
def test_every_real_oauth_hook_is_a_coroutine_function(provider_id: str) -> None:
    """`login`, `refresh` and `to_auth` are all awaited by their callers.

    `lazy_oauth` wraps each of the three; if any wrapper forgot to await the
    loader, the caller would reach the hook on a coroutine object. `login` is
    the one that crashed in the CLI, but nothing makes it special.
    """
    oauth = find_builtin(provider_id).auth.oauth
    assert oauth is not None
    assert inspect.iscoroutinefunction(oauth.login)
    assert inspect.iscoroutinefunction(oauth.refresh)
    assert inspect.iscoroutinefunction(oauth.to_auth)


@pytest.mark.parametrize("provider_id", oauth_provider_ids())
async def test_every_real_oauth_flow_resolves_a_credential_end_to_end(provider_id: str) -> None:
    """Drives the real lazy wrapper instead of asserting on its shape.

    `iscoroutinefunction` above only proves the hook is `async def`; it would
    still pass if `lazy_oauth` called its loader without awaiting. Awaiting
    `to_auth` for real proves the loader was awaited and a flow -- not a
    coroutine -- reached the hook: the unawaited version raises
    `'coroutine' object has no attribute 'to_auth'` right here. `to_auth` is
    pure credential derivation for every built-in flow, so this stays offline.
    """
    oauth = find_builtin(provider_id).auth.oauth
    assert oauth is not None

    resolved = await oauth.to_auth(Credential(type="oauth", access="tid=a;exp=1;rest", refresh="r", expires=math.inf))

    assert isinstance(resolved, ResolvedAuth)


@pytest.mark.parametrize("provider_id", api_key_provider_ids())
def test_every_real_api_key_login_is_a_coroutine_function(provider_id: str) -> None:
    """`Models.login(..., interaction=...)` awaits this hook.

    Bedrock's AWS-profile flow and Vertex's ADC flow reach the credential store
    only through it, so a sync hook here would be a silent no-op rather than a
    crash.
    """
    api_key = find_builtin(provider_id).auth.api_key
    assert api_key is not None
    assert inspect.iscoroutinefunction(api_key.login)


@pytest.mark.parametrize("loader_name", sorted(name for name in dir(oauth_load) if name.startswith("load_")))
async def test_every_oauth_loader_awaits_into_a_real_flow(loader_name: str) -> None:
    """The exact object the CLI mishandled.

    Each loader is an `async def`; awaiting it must yield an `OAuthAuth` whose
    own hooks are coroutine functions. A sync loader (or one returning a
    coroutine) would reproduce the original crash one level up.
    """
    loader = getattr(oauth_load, loader_name)
    assert inspect.iscoroutinefunction(loader)

    # `load_radius_oauth` is the only loader taking arguments (name, gateway).
    flow = await (loader("Radius", "https://gateway.example") if loader_name == "load_radius_oauth" else loader())

    assert isinstance(flow, OAuthAuth)
    assert inspect.iscoroutinefunction(flow.login)
    assert inspect.iscoroutinefunction(flow.refresh)
    assert inspect.iscoroutinefunction(flow.to_auth)


def test_the_oauth_barrel_exposes_loaders_not_constructed_flows() -> None:
    """Keeps the barrel from handing callers a coroutine that looks like a flow.

    The CLI bug was possible because a `load_*` name reads like a flow object.
    Anything the barrel exports under that name must be a coroutine function,
    never a ready `OAuthAuth`.
    """
    for name in oauth_barrel.__all__:
        exported = getattr(oauth_barrel, name)
        if name.startswith("load_"):
            assert inspect.iscoroutinefunction(exported)
        else:
            assert not isinstance(exported, OAuthAuth)


@pytest.mark.parametrize("provider_id", [provider.id for provider in builtin_providers()])
def test_provider_api_modules_return_streams_synchronously(provider_id: str) -> None:
    """The shape every test fake for an api module has to match.

    `stream`/`stream_simple` return an `AssistantMessageEventStream` directly;
    they are not coroutine functions. Test doubles that make them `async def`
    would be easier to satisfy than production and would hide an unawaited
    call the same way the CLI stub did.
    """
    provider = find_builtin(provider_id)
    modules = provider.api.values() if isinstance(provider.api, dict) else [provider.api]

    for module in modules:
        assert callable(module.stream)
        assert callable(module.stream_simple)
        assert not inspect.iscoroutinefunction(module.stream)
        assert not inspect.iscoroutinefunction(module.stream_simple)
