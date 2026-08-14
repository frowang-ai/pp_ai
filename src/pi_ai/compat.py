"""Legacy global API-provider registry and env-key-aware dispatch.

Python port of `packages/ai/src/compat.ts`. TypeScript's `compat.ts` is a
temporary compatibility entrypoint that lets old code keep calling a global
`stream()`/`complete()` pair that dispatches by `model.api` through a mutable
registry (`registerApiProvider`), on top of a static generated model catalog
and env-var API-key injection (`env-api-keys.ts`).

This port keeps the part of that surface with lasting test value: the mutable
api-id registry (`register_api_provider` / `get_api_provider` /
`unregister_api_providers` / `reset_api_providers`) and the `stream`/`complete`/
`stream_simple`/`complete_simple` dispatch functions that route a request to
whichever api implementation is registered for `model.api`, plus
`register_faux_provider` for tests that want a scripted provider reachable
through that same global registry (as opposed to `providers.faux_provider`,
which builds a `registry.Provider` for explicit `Models` collections).

`stream`/`stream_simple` reproduce the full TypeScript dispatch order:

1. When the model belongs to a built-in provider *and* nothing has overridden
   the api-registry entry for `model.api`, dispatch through that provider's own
   api module. This matters because a provider may wrap the shared module (e.g.
   `cloudflare_streams` substitutes account/gateway placeholders in the base
   URL); the flat registry entry is unwrapped.
2. Cloudflare models whose auth is not already resolved go through the `Models`
   runtime instead, so provider auth can supply the account/gateway env and the
   `cf-aig-authorization` header.
3. Otherwise fill `api_key` from the environment (`get_env_api_key`), skipping
   the ambient-auth marker, and dispatch through the registry.

Deliberately dropped: `bedrock-converse-stream` and `openai-codex-responses`,
which are not ported at all.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .api import (
    anthropic_messages,
    azure_openai_responses,
    google_generative_ai,
    google_vertex,
    mistral_conversations,
    openai_completions,
    openai_responses,
    pi_messages,
)
from .env_api_keys import get_env_api_key
from .providers.faux import (
    FauxProviderState,
    RegisterFauxProviderOptions,
    create_faux_core,
)
from .types import (
    AssistantMessage,
    Context,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StreamOptions,
)
from .utils.event_stream import AssistantMessageEventStream
from .utils.tasks import spawn


class ApiStreamFunction(Protocol):
    def __call__(
        self, model: Model, context: Context, options: StreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream: ...


@dataclass
class ApiProviderEntry:
    api: str
    stream: ApiStreamFunction
    stream_simple: ApiStreamFunction


@dataclass
class _RegisteredApiProvider:
    provider: ApiProviderEntry
    source_id: str | None = None


_api_provider_registry: dict[str, _RegisteredApiProvider] = {}


def _wrap(api: str, fn: ApiStreamFunction) -> ApiStreamFunction:
    def wrapped(
        model: Model, context: Context, options: StreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream:
        if model.api != api:
            raise RuntimeError(f"Mismatched api: {model.api} expected {api}")
        return fn(model, context, options, **kwargs)

    return wrapped


def register_api_provider(
    api: str,
    stream: ApiStreamFunction,
    stream_simple: ApiStreamFunction,
    source_id: str | None = None,
) -> None:
    _api_provider_registry[api] = _RegisteredApiProvider(
        provider=ApiProviderEntry(api=api, stream=_wrap(api, stream), stream_simple=_wrap(api, stream_simple)),
        source_id=source_id,
    )


def get_api_provider(api: str) -> ApiProviderEntry | None:
    entry = _api_provider_registry.get(api)
    return entry.provider if entry else None


def get_api_providers() -> list[ApiProviderEntry]:
    return [entry.provider for entry in _api_provider_registry.values()]


def unregister_api_providers(source_id: str) -> None:
    for api, entry in list(_api_provider_registry.items()):
        if entry.source_id == source_id:
            del _api_provider_registry[api]


def clear_api_providers() -> None:
    _api_provider_registry.clear()


# Every api module this port implements. `bedrock-converse-stream` and
# `openai-codex-responses` are intentionally absent: they are not ported.
_BUILTIN_APIS: list[tuple[str, Any]] = [
    ("anthropic-messages", anthropic_messages),
    ("openai-completions", openai_completions),
    ("openai-responses", openai_responses),
    ("azure-openai-responses", azure_openai_responses),
    ("google-generative-ai", google_generative_ai),
    ("google-vertex", google_vertex),
    ("mistral-conversations", mistral_conversations),
    ("pi-messages", pi_messages),
]

_builtin_api_provider_instances: dict[str, ApiProviderEntry | None] = {}


def register_builtin_api_providers() -> None:
    """Register the builtin API implementations without clobbering existing entries.

    Mirrors `registerBuiltInApiProviders`: this module may load after a test or
    extension already registered an override for a builtin api id.
    """
    for api, module in _BUILTIN_APIS:
        if get_api_provider(api) is None:
            register_api_provider(api, module.stream, module.stream_simple)
        _builtin_api_provider_instances[api] = get_api_provider(api)


def reset_api_providers() -> None:
    clear_api_providers()
    _builtin_api_provider_instances.clear()
    register_builtin_api_providers()


register_builtin_api_providers()


def _resolve_api_provider(api: str) -> ApiProviderEntry:
    provider = get_api_provider(api)
    if provider is None:
        raise RuntimeError(f"No API provider registered for api: {api}")
    return provider


_AMBIENT_AUTH_MARKER = "<authenticated>"

_compat_models: Any = None


def _get_compat_models() -> Any:
    """The built-in `Models` collection, built lazily to avoid an import cycle."""
    global _compat_models
    if _compat_models is None:
        from .providers.all import builtin_models

        _compat_models = builtin_models()
    return _compat_models


def _has_explicit_api_key(api_key: str | None) -> bool:
    return isinstance(api_key, str) and bool(api_key.strip())


def _with_env_api_key(model: Model, options: StreamOptions | None) -> StreamOptions | None:
    """Fill a missing ``api_key`` from the provider's known environment variables."""
    if options is not None and _has_explicit_api_key(options.api_key):
        return options
    api_key = get_env_api_key(model.provider, options.env if options is not None else None)
    if not api_key or api_key == _AMBIENT_AUTH_MARKER:
        return options
    resolved = replace(options) if options is not None else StreamOptions()
    resolved.api_key = api_key
    return resolved


def _has_resolved_cloudflare_auth(options: StreamOptions | None) -> bool:
    if options is None:
        return False
    return _has_explicit_api_key(options.api_key) or isinstance(options.headers.get("cf-aig-authorization"), str)


def _get_builtin_provider_for_model(model: Model) -> Any:
    """The built-in provider owning ``model``, unless its api has been overridden."""
    if get_api_provider(model.api) is not _builtin_api_provider_instances.get(model.api):
        return None
    provider = _get_compat_models().get_provider(model.provider)
    if provider is None:
        return None
    return provider if any(candidate.api == model.api for candidate in provider.get_models()) else None


def stream(
    model: Model, context: Context, options: StreamOptions | None = None, **kwargs: Any
) -> AssistantMessageEventStream:
    builtin_provider = _get_builtin_provider_for_model(model)
    if builtin_provider is not None:
        if model.provider.startswith("cloudflare-") and not _has_resolved_cloudflare_auth(options):
            return _stream_through_models("stream", model, context, options, **kwargs)
        return builtin_provider.stream(model, context, _with_env_api_key(model, options), **kwargs)
    provider = _resolve_api_provider(model.api)
    return provider.stream(model, context, _with_env_api_key(model, options), **kwargs)


async def complete(
    model: Model, context: Context, options: StreamOptions | None = None, **kwargs: Any
) -> AssistantMessage:
    return await stream(model, context, options, **kwargs).result()


def stream_simple(
    model: Model, context: Context, options: SimpleStreamOptions | None = None, **kwargs: Any
) -> AssistantMessageEventStream:
    builtin_provider = _get_builtin_provider_for_model(model)
    if builtin_provider is not None:
        if model.provider.startswith("cloudflare-") and not _has_resolved_cloudflare_auth(options):
            return _stream_through_models("stream_simple", model, context, options, **kwargs)
        return builtin_provider.stream_simple(model, context, _with_env_api_key(model, options), **kwargs)
    provider = _resolve_api_provider(model.api)
    return provider.stream_simple(model, context, _with_env_api_key(model, options), **kwargs)


def _stream_through_models(
    method: str, model: Model, context: Context, options: StreamOptions | None, **kwargs: Any
) -> AssistantMessageEventStream:
    """Bridge `Models.stream`/`Models.stream_simple`, which are coroutines here.

    TypeScript's `Models.stream` returns the event stream synchronously and
    resolves auth inside it; the port awaits auth first, so the coroutine is
    spawned and its events are forwarded into a stream returned right away.
    """
    event_stream = AssistantMessageEventStream()

    async def run() -> None:
        models = _get_compat_models()
        try:
            inner = await getattr(models, method)(model, context, options, **kwargs)
        except BaseException as error:
            output = AssistantMessage(
                api=model.api,
                provider=model.provider,
                model=model.id,
                stop_reason="error",
                error_message=str(error),
            )
            event_stream.push(ErrorEvent(reason="error", error=output))
            event_stream.end()
            return
        async for event in inner:
            event_stream.push(event)
        event_stream.end()

    spawn(run())
    return event_stream


async def complete_simple(
    model: Model, context: Context, options: SimpleStreamOptions | None = None, **kwargs: Any
) -> AssistantMessage:
    return await stream_simple(model, context, options, **kwargs).result()


@dataclass
class FauxProviderRegistration:
    """A faux provider reachable through the global api-provider registry."""

    api: str
    models: list[Model]
    state: FauxProviderState
    get_model: Any
    set_responses: Any
    append_responses: Any
    get_pending_response_count: Any
    unregister: Any


def register_faux_provider(options: RegisterFauxProviderOptions | None = None) -> FauxProviderRegistration:
    core = create_faux_core(options)
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    source_id = f"faux-provider-{suffix}"
    register_api_provider(core.api, core.stream, core.stream_simple, source_id=source_id)

    def unregister() -> None:
        unregister_api_providers(source_id)

    return FauxProviderRegistration(
        api=core.api,
        models=core.models,
        state=core.state,
        get_model=core.get_model,
        set_responses=core.set_responses,
        append_responses=core.append_responses,
        get_pending_response_count=core.get_pending_response_count,
        unregister=unregister,
    )


__all__ = [
    "ApiProviderEntry",
    "FauxProviderRegistration",
    "clear_api_providers",
    "complete",
    "complete_simple",
    "get_api_provider",
    "get_api_providers",
    "register_api_provider",
    "register_builtin_api_providers",
    "register_faux_provider",
    "reset_api_providers",
    "stream",
    "stream_simple",
    "unregister_api_providers",
]
