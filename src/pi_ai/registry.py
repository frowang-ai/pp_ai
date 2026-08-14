"""Provider factory and the runtime provider registry.

Python port of `createProvider` and the `Models` collection in
`packages/ai/src/models.ts`. A provider owns its id/metadata, auth, model
catalog and streaming behaviour; :class:`Models` resolves auth and delegates
each request to the provider that owns the model.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .auth.resolve import resolve_provider_auth
from .auth.types import (
    AuthCheck,
    AuthInteraction,
    AuthResult,
    Credential,
    CredentialStore,
    InMemoryCredentialStore,
    ProviderAuth,
)
from .models import ModelsError
from .types import Context, DeferredHandle, Model, SimpleStreamOptions, StreamOptions
from .utils.abort import AbortError, AbortSignal
from .utils.event_stream import AssistantMessageEventStream, setup_error_stream
from .utils.headers import merge_provider_headers


class ApiModule(Protocol):
    """The uniform stream contract of a module in :mod:`pi_ai.api`."""

    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream: ...

    def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream: ...


@dataclass
class Provider:
    """A concrete runtime provider."""

    id: str
    name: str
    auth: ProviderAuth
    api: Any
    """The API module implementing :class:`ApiModule`.

    Providers that expose several wire formats (Fireworks, GitHub Copilot,
    OpenCode, xAI, Cloudflare AI Gateway) pass a ``{api name: module}`` mapping
    instead, exactly like TypeScript's ``api: { "anthropic-messages": ..., ... }``
    object form. The module is then selected per request from ``model.api``.
    """
    base_url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    models: list[Model] = field(default_factory=list)
    filter_models: Callable[[list[Model], Credential | None], list[Model]] | None = None
    """Optional provider policy for credential-specific model availability.

    ``get_models()`` remains the complete catalog; :meth:`Models.get_available`
    applies this filter after confirming that provider auth is configured
    (TypeScript's `Provider.filterModels`).
    """

    def get_models(self) -> list[Model]:
        return list(self.models)

    def get_model(self, model_id: str) -> Model | None:
        return next((model for model in self.models if model.id == model_id), None)

    def api_for(self, model: Model) -> Any:
        """The API module that handles ``model``; raises when there is none."""
        module = self._api_or_none(model)
        if module is None:
            raise ModelsError(
                "provider",
                f'Provider {self.id} has no API implementation for "{model.api}"',
            )
        return module

    def _api_or_none(self, model: Model) -> Any:
        if not isinstance(self.api, dict):
            return self.api
        return self.api.get(model.api)

    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream:
        module = self._api_or_none(model)
        if module is None:
            return setup_error_stream(model, self._missing_api_error(model))
        return module.stream(model, context, options, **kwargs)

    def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream:
        module = self._api_or_none(model)
        if module is None:
            return setup_error_stream(model, self._missing_api_error(model))
        return module.stream_simple(model, context, options, **kwargs)

    def _missing_api_error(self, model: Model) -> ModelsError:
        return ModelsError("stream", f'Provider {self.id} has no API implementation for "{model.api}"')

    @property
    def fetch_deferred(self) -> Callable[..., AssistantMessageEventStream] | None:
        """The owning API module's `fetchDeferred`, when it implements one.

        TypeScript's `createProvider` only attaches `fetchDeferred`/`cancelDeferred`
        to the provider when at least one of its API modules implements them; the
        port resolves the module per model instead, so the capability is reported
        per provider and the actual dispatch re-resolves for the requested model.
        """
        if not self._any_api_has("fetch_deferred"):
            return None

        def fetch(
            model: Model, handle: DeferredHandle, options: StreamOptions | None = None, **kwargs: Any
        ) -> AssistantMessageEventStream:
            # A stream-returning entry point reports setup failures in-band, the
            # same way `lazyStream` does in TypeScript.
            implementation = getattr(self._api_or_none(model), "fetch_deferred", None)
            if implementation is None:
                return setup_error_stream(
                    model,
                    ModelsError(
                        "provider", f'Provider {self.id} does not support deferred responses for "{model.api}"'
                    ),
                )
            return implementation(model, handle, options, **kwargs)

        return fetch

    @property
    def cancel_deferred(self) -> Callable[..., Awaitable[None]] | None:
        if not self._any_api_has("cancel_deferred"):
            return None

        async def cancel(
            model: Model, handle: DeferredHandle, options: StreamOptions | None = None, **kwargs: Any
        ) -> None:
            implementation = getattr(self.api_for(model), "cancel_deferred", None)
            if implementation is None:
                raise ModelsError("provider", f'Provider {self.id} cannot cancel deferred responses for "{model.api}"')
            await implementation(model, handle, options, **kwargs)

        return cancel

    def _any_api_has(self, attribute: str) -> bool:
        modules = self.api.values() if isinstance(self.api, dict) else [self.api]
        return any(getattr(module, attribute, None) is not None for module in modules)


def create_provider(
    id: str,
    name: str,
    auth: ProviderAuth,
    api: Any,
    models: list[Model],
    base_url: str = "",
    headers: dict[str, str] | None = None,
    filter_models: Callable[[list[Model], Credential | None], list[Model]] | None = None,
) -> Provider:
    """Build a provider, stamping ``provider``/``base_url`` onto its models.

    The TypeScript catalogs carry these fields already; stamping here keeps a
    hand-written catalog from silently disagreeing with its provider.
    """
    stamped: list[Model] = []
    for model in models:
        stamped.append(
            replace(
                model,
                provider=model.provider or id,
                base_url=model.base_url or base_url,
            )
        )
    return Provider(
        id=id,
        name=name,
        auth=auth,
        api=api,
        base_url=base_url,
        headers=dict(headers or {}),
        models=stamped,
        filter_models=filter_models,
    )


class Models:
    """Runtime collection of providers plus auth resolution."""

    def __init__(
        self,
        providers: list[Provider] | None = None,
        credential_store: CredentialStore | None = None,
        env: Any = None,
    ) -> None:
        self._providers: dict[str, Provider] = {}
        for provider in providers or []:
            self._providers[provider.id] = provider
        self.credentials = credential_store or InMemoryCredentialStore()
        self._env = env

    def add(self, provider: Provider) -> None:
        self._providers[provider.id] = provider

    def get_providers(self) -> list[Provider]:
        return list(self._providers.values())

    def get_provider(self, provider_id: str) -> Provider | None:
        return self._providers.get(provider_id)

    def delete_provider(self, provider_id: str) -> None:
        self._providers.pop(provider_id, None)

    def clear_providers(self) -> None:
        self._providers.clear()

    def get_models(self, provider_id: str | None = None) -> list[Model]:
        """Every model, or one provider's models.

        A provider whose catalog source raises is skipped rather than failing
        the whole listing (TypeScript does the same); call
        ``get_provider(id).get_models()`` for the precise failure.
        """
        if provider_id is not None:
            provider = self._providers.get(provider_id)
            if provider is None:
                return []
            try:
                return provider.get_models()
            except Exception:
                return []
        models: list[Model] = []
        for provider in self._providers.values():
            try:
                models.extend(provider.get_models())
            except Exception:
                continue
        return models

    def get_model(self, provider_id: str, model_id: str) -> Model | None:
        provider = self._providers.get(provider_id)
        return provider.get_model(model_id) if provider else None

    def find_model(self, reference: str) -> Model | None:
        """Look up ``provider/model-id``, or a bare model id across providers."""
        if "/" in reference:
            provider_id, _, model_id = reference.partition("/")
            model = self.get_model(provider_id, model_id)
            if model is not None:
                return model
        for provider in self._providers.values():
            model = provider.get_model(reference)
            if model is not None:
                return model
        return None

    async def get_auth(
        self,
        target: str | Model,
        *,
        api_key: str | None = None,
        env: dict[str, str] | None = None,
        min_oauth_validity_ms: int | None = None,
        signal: AbortSignal | None = None,
    ) -> AuthResult | None:
        """Resolve provider auth by provider id or by model.

        ``api_key``/``env`` are per-request overrides: the key is handed to the
        provider's api-key auth as if it were a stored credential, and the env
        map is overlaid on this collection's env lookup so provider-owned
        resolvers (AWS profiles, ADC paths, account ids) see request values.

        Returns ``None`` when the provider is unknown or unconfigured. Raises
        :class:`ModelsError` with code ``"auth"`` when resolution itself fails.
        """
        provider_id = target if isinstance(target, str) else target.provider
        provider = self._providers.get(provider_id)
        if provider is None:
            return None

        lookup = self._env
        if env:
            base = lookup if lookup is not None else os.environ.get
            overlay = dict(env)

            def lookup(name: str) -> Any:
                if name in overlay:
                    return overlay[name]
                return base(name)

        try:
            result = await resolve_provider_auth(
                provider_id,
                provider.auth,
                self.credentials,
                lookup,
                api_key_override=api_key,
                env_override=env,
                min_oauth_validity_ms=min_oauth_validity_ms,
                signal=signal,
            )
        except (ModelsError, AbortError):
            # Cancellation is not an auth failure: it propagates unchanged so a
            # caller can tell "aborted" from "misconfigured".
            raise
        except Exception as error:
            raise ModelsError("auth", f"Failed to resolve auth for provider {provider_id}: {error}") from error

        if result is None:
            return None

        if not isinstance(target, str) and target.headers:
            # Model headers win over auth-derived headers, and a case-insensitive
            # match replaces the existing key rather than duplicating it.
            merged = merge_provider_headers(result.auth.headers, target.headers)
            result = AuthResult(
                auth=type(result.auth)(api_key=result.auth.api_key, headers=merged, base_url=result.auth.base_url),
                source=result.source,
                env=result.env,
            )
        return result

    async def _check_provider_auth(self, provider: Provider, credential: Credential | None) -> AuthCheck:
        """Whether ``provider`` is configured, without refreshing OAuth.

        Checking configuration must not spend a refresh token or hit the
        network: a stored OAuth credential is reported as configured on sight,
        exactly like TypeScript's `checkProviderAuth`.
        """
        if credential is not None and credential.type == "oauth":
            if provider.auth.oauth is not None:
                return AuthCheck(configured=True, source="OAuth", type="oauth")
            return AuthCheck(configured=False)
        result = await self.get_auth(provider.id)
        if result is None:
            return AuthCheck(configured=False)
        return AuthCheck(configured=True, source=result.source, type="api_key")

    async def _read_credential(self, provider_id: str) -> Credential | None:
        """Port of `models.ts`'s private `readCredential`.

        Wraps a failing credential-store read in a `ModelsError` naming the
        provider. Without it the raw store error propagates, so callers that
        surface the message -- `ModelRuntime.get_error()`'s "Availability
        refresh:" line -- report the store's wording instead of TypeScript's.
        """
        try:
            return await self.credentials.get(provider_id)
        except Exception as error:
            raise ModelsError("auth", f"Credential store read failed for {provider_id}", error) from error

    async def check_auth(self, provider_id: str) -> AuthCheck | None:
        provider = self._providers.get(provider_id)
        if provider is None:
            return None
        return await self._check_provider_auth(provider, await self._read_credential(provider_id))

    async def get_available(self, provider_id: str | None = None) -> list[Model]:
        """Models whose providers have complete auth configuration."""
        available: list[Model] = []
        provider_ids = [provider_id] if provider_id else list(self._providers)
        for pid in provider_ids:
            provider = self._providers.get(pid)
            if provider is None:
                continue
            credential = await self._read_credential(pid)
            check = await self._check_provider_auth(provider, credential)
            if check.configured:
                models = self.get_models(pid)
                if provider.filter_models is not None:
                    models = list(provider.filter_models(models, credential))
                available.extend(models)
        return available

    async def login(
        self,
        provider_id: str,
        api_key: str | None = None,
        *,
        interaction: AuthInteraction | None = None,
    ) -> Credential:
        """Store an api-key credential for ``provider_id``.

        With ``api_key`` the key is stored directly. With ``interaction`` the
        provider's own api-key login flow runs instead, which is how providers
        whose credential is not a single secret (Bedrock's AWS profile, Vertex's
        ADC project/location) get to store a credential carrying only ``env``.
        """
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {provider_id}")
        if interaction is not None:
            login = provider.auth.api_key.login if provider.auth.api_key is not None else None
            if login is None:
                raise ModelsError("provider", f"Provider {provider_id} does not support interactive api-key login")
            credential = await login(interaction)
        elif api_key is not None:
            credential = Credential(type="api_key", key=api_key)
        else:
            raise ModelsError("provider", "login requires either an api_key or an interaction")
        await self.credentials.set(provider_id, credential)
        return credential

    async def login_oauth(self, provider_id: str, interaction: AuthInteraction) -> Credential:
        """Run the OAuth login flow for ``provider_id`` and store the resulting credential."""
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {provider_id}")
        if provider.auth.oauth is None:
            raise ModelsError("provider", f"Provider {provider_id} does not support OAuth login")
        try:
            credential = await provider.auth.oauth.login(interaction)
        except Exception as error:
            raise ModelsError("oauth", f"OAuth login failed for {provider_id}: {error}") from error
        await self.credentials.set(provider_id, credential)
        return credential

    async def logout(self, provider_id: str) -> None:
        await self.credentials.delete(provider_id)

    async def _resolve_request(
        self, model: Model, options: StreamOptions | SimpleStreamOptions
    ) -> tuple[Provider, Model, StreamOptions | SimpleStreamOptions]:
        """Resolve the owning provider plus auth-merged model/options."""
        provider = self._providers.get(model.provider)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {model.provider}")

        if not options.api_key:
            auth = await self.get_auth(model)
        else:
            # TypeScript resolves provider auth even when the caller supplies an
            # api key (the key is handed to the provider's resolver as a stored
            # credential); the key then wins per-field, but provider-resolved
            # env, headers and base URL still apply.
            auth = await self.get_auth(model, api_key=options.api_key, env=options.env or None)
        if auth is None:
            raise ModelsError("auth", f"Provider {model.provider} is not configured")
        options = replace(options, api_key=options.api_key or auth.auth.api_key)
        if auth.auth.headers:
            options = replace(options, headers=merge_provider_headers(auth.auth.headers, options.headers))
        if auth.env:
            options = replace(options, env={**auth.env, **options.env})
        if auth.auth.base_url:
            # OAuth methods whose endpoint depends on the resolved credential
            # (e.g. GitHub Copilot's per-token proxy host) override the model's
            # configured base URL.
            model = replace(model, base_url=auth.auth.base_url)

        return provider, model, options

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> AssistantMessageEventStream:
        """Resolve auth for ``model`` and delegate to its provider.

        Setup failures (unknown provider, unconfigured auth) are reported
        in-band as an error message, the way `lazyStream` does in TypeScript,
        so a caller only has to handle one failure channel.
        """
        try:
            provider, model, resolved = await self._resolve_request(model, options or StreamOptions())
        except ModelsError as error:
            return setup_error_stream(model, error)
        return provider.stream(model, context, resolved, **kwargs)

    async def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
        **kwargs: Any,
    ) -> AssistantMessageEventStream:
        """Resolve auth for ``model`` and delegate to its provider (see :meth:`stream`)."""
        try:
            provider, model, resolved = await self._resolve_request(model, options or SimpleStreamOptions())
        except ModelsError as error:
            return setup_error_stream(model, error)
        return provider.stream_simple(model, context, resolved, **kwargs)

    async def fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> AssistantMessageEventStream:
        """Resolve auth and fetch a deferred response through its provider."""
        try:
            provider, model, resolved = await self._resolve_request(model, options or StreamOptions())
        except ModelsError as error:
            return setup_error_stream(model, error)
        fetch = provider.fetch_deferred
        if fetch is None:
            return setup_error_stream(
                model, ModelsError("provider", f"Provider {provider.id} does not support deferred responses")
            )
        return fetch(model, handle, resolved, **kwargs)

    async def cancel_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> None:
        """Resolve auth and cancel a deferred response through its provider."""
        provider, model, resolved = await self._resolve_request(model, options or StreamOptions())
        cancel = provider.cancel_deferred
        if cancel is None:
            raise ModelsError("provider", f"Provider {provider.id} does not support deferred responses")
        await cancel(model, handle, resolved, **kwargs)
