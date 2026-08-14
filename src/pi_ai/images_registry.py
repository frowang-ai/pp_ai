"""Image-generation providers and the runtime image provider registry.

Python port of `packages/ai/src/images-models.ts`: `createImagesProvider` and
the `ImagesModels` collection, the image-side counterpart of
`createProvider`/`Models` in `packages/ai/src/models.ts`, which this port has
as :mod:`pi_ai.registry`. The two are deliberately parallel, so this module
mirrors :class:`pi_ai.registry.Provider`/:class:`pi_ai.registry.Models`:
provider metadata plus auth on one side, auth resolution and delegation on the
other.

The one contract that differs from the chat side is :meth:`ImagesModels.generate_images`,
which never raises: any failure — unknown provider, auth resolution, the
provider call itself — comes back as an :class:`~pi_ai.types.AssistantImages`
with ``stop_reason="error"`` and ``error_message`` set, because there is no
event stream to report the failure on.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace

from .auth.resolve import resolve_provider_auth
from .auth.types import (
    AuthResult,
    Credential,
    CredentialStore,
    EnvLookup,
    InMemoryCredentialStore,
    ProviderAuth,
)
from .models import ModelsError
from .types import AssistantImages, ImagesContext, ImagesModel, ImagesOptions, now_ms
from .utils.abort import AbortSignal

ImagesGenerator = Callable[[ImagesModel, ImagesContext, "ImagesOptions | None"], Awaitable[AssistantImages]]
"""Port of `ProviderImages.generateImages` / `ImagesFunction`."""

ImagesRefresh = Callable[[], Awaitable[Sequence[ImagesModel]]]
"""Port of `CreateImagesProviderOptions.refreshModels`."""


@dataclass
class ImagesProvider:
    """An image-generation provider: the image-side counterpart of :class:`~pi_ai.registry.Provider`.

    Owns id/name metadata, auth, the model list and generation behaviour.
    """

    id: str
    name: str
    auth: ProviderAuth
    api: ImagesGenerator
    """The function implementing this provider's image api (see :mod:`pi_ai.api.openrouter_images`)."""
    models: list[ImagesModel] = field(default_factory=list)
    refresh: ImagesRefresh | None = None
    """Dynamic providers only: fetch the current model list."""

    _inflight_refresh: asyncio.Task[None] | None = field(default=None, repr=False, compare=False)

    def get_models(self) -> list[ImagesModel]:
        """Current known models.

        Static providers return their catalog; dynamic providers return the
        list as of the last :meth:`refresh_models` (empty before the first).
        """
        return list(self.models)

    def get_model(self, model_id: str) -> ImagesModel | None:
        return next((model for model in self.models if model.id == model_id), None)

    async def refresh_models(self) -> None:
        """Fetch and store the current model list; a no-op for static providers.

        Concurrent calls share one in-flight fetch. On failure the stored list
        stays at its last-known state, the error propagates to the caller, and
        a later call retries.
        """
        if self.refresh is None:
            return
        if self._inflight_refresh is None or self._inflight_refresh.done():
            self._inflight_refresh = asyncio.ensure_future(self._run_refresh())
        try:
            await asyncio.shield(self._inflight_refresh)
        finally:
            if self._inflight_refresh is not None and self._inflight_refresh.done():
                self._inflight_refresh = None

    async def _run_refresh(self) -> None:
        assert self.refresh is not None
        self.models = list(await self.refresh())

    async def generate_images(
        self, model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None
    ) -> AssistantImages:
        return await self.api(model, context, options)


def create_images_provider(
    id: str,
    name: str,
    auth: ProviderAuth,
    api: ImagesGenerator,
    models: Sequence[ImagesModel] | None = None,
    refresh: ImagesRefresh | None = None,
) -> ImagesProvider:
    """Port of `createImagesProvider`: build an image provider from parts."""
    return ImagesProvider(
        id=id,
        name=name or id,
        auth=auth,
        api=api,
        models=list(models or []),
        refresh=refresh,
    )


def _error_images(model: ImagesModel, error: BaseException) -> AssistantImages:
    return AssistantImages(
        api=model.api,
        provider=model.provider,
        model=model.id,
        output=[],
        stop_reason="error",
        error_message=str(error),
        timestamp=now_ms(),
    )


class ImagesModels:
    """Runtime collection of image providers plus auth resolution.

    The image-side counterpart of :class:`pi_ai.registry.Models`. It is mutable
    in place, so it covers both `ImagesModels` and `MutableImagesModels`;
    TypeScript needs the two interfaces only to hand out a read-only view.
    """

    def __init__(
        self,
        providers: Sequence[ImagesProvider] | None = None,
        credential_store: CredentialStore | None = None,
        env: EnvLookup | None = None,
    ) -> None:
        self._providers: dict[str, ImagesProvider] = {}
        for provider in providers or []:
            self._providers[provider.id] = provider
        self.credentials = credential_store or InMemoryCredentialStore()
        self._env = env

    def add(self, provider: ImagesProvider) -> None:
        """Upsert by ``provider.id`` (port of `setProvider`)."""
        self._providers[provider.id] = provider

    def remove(self, provider_id: str) -> None:
        """Port of `deleteProvider`."""
        self._providers.pop(provider_id, None)

    def clear(self) -> None:
        """Port of `clearProviders`."""
        self._providers.clear()

    def get_providers(self) -> list[ImagesProvider]:
        return list(self._providers.values())

    def get_provider(self, provider_id: str) -> ImagesProvider | None:
        return self._providers.get(provider_id)

    def get_models(self, provider_id: str | None = None) -> list[ImagesModel]:
        """Last-known models from one provider, or from all of them.

        Best-effort: a provider whose ``get_models()`` raises yields no models.
        """
        if provider_id is not None:
            provider = self._providers.get(provider_id)
            if provider is None:
                return []
            try:
                return provider.get_models()
            except Exception:
                return []

        models: list[ImagesModel] = []
        for provider in self._providers.values():
            try:
                models.extend(provider.get_models())
            except Exception:
                # Best-effort: ill-behaved providers yield no models.
                continue
        return models

    def get_model(self, provider_id: str, model_id: str) -> ImagesModel | None:
        return next((model for model in self.get_models(provider_id) if model.id == model_id), None)

    async def refresh(self, provider_id: str | None = None) -> None:
        """Ask dynamic providers to re-fetch their model lists.

        With a provider id, a fetch failure raises :class:`ModelsError`
        (``"model_source"``); without one, every provider is refreshed
        concurrently and failures are swallowed. Static providers are no-ops.
        """
        if provider_id is not None:
            provider = self._providers.get(provider_id)
            if provider is None or provider.refresh is None:
                return
            try:
                await provider.refresh_models()
            except ModelsError:
                raise
            except Exception as error:
                raise ModelsError("model_source", f"Model refresh failed for {provider_id}") from error
            return

        await asyncio.gather(
            *(provider.refresh_models() for provider in self._providers.values()),
            return_exceptions=True,
        )

    async def get_auth(
        self,
        target: str | ImagesModel,
        *,
        api_key_override: str | None = None,
        min_oauth_validity_ms: int | None = None,
        signal: AbortSignal | None = None,
    ) -> AuthResult | None:
        """Resolve provider auth by provider id or by image model.

        Same contract as :meth:`pi_ai.registry.Models.get_auth`: ``None`` when
        the provider is unknown or unconfigured, :class:`ModelsError` when
        resolution itself fails. The keyword arguments are the port of
        TypeScript's `AuthResolutionOverrides`; an explicit
        ``api_key_override`` wins over any stored or ambient credential.
        """
        provider_id = target if isinstance(target, str) else target.provider
        provider = self._providers.get(provider_id)
        if provider is None:
            return None

        try:
            return await resolve_provider_auth(
                provider_id,
                provider.auth,
                self.credentials,
                self._env,
                api_key_override=api_key_override,
                min_oauth_validity_ms=min_oauth_validity_ms,
                signal=signal,
            )
        except ModelsError:
            raise
        except Exception as error:
            raise ModelsError("auth", f"Failed to resolve auth for provider {provider_id}: {error}") from error

    async def login(self, provider_id: str, api_key: str) -> Credential:
        """Store an api-key credential for ``provider_id``."""
        if provider_id not in self._providers:
            raise ModelsError("provider", f"Unknown provider: {provider_id}")
        credential = Credential(type="api_key", key=api_key)
        await self.credentials.set(provider_id, credential)
        return credential

    async def logout(self, provider_id: str) -> None:
        await self.credentials.delete(provider_id)

    async def generate_images(
        self, model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None
    ) -> AssistantImages:
        """Generate images through the provider that owns ``model``.

        Auth is resolved and merged into the request, with explicit options
        winning per field and headers/env merging per key. Never raises:
        failures come back as an :class:`~pi_ai.types.AssistantImages` with
        ``stop_reason="error"``.
        """
        try:
            provider = self._providers.get(model.provider)
            if provider is None:
                raise ModelsError("provider", f"Unknown provider: {model.provider}")

            options = options or ImagesOptions()
            resolution = await self.get_auth(model)
            auth = resolution.auth if resolution is not None else None
            if auth is None:
                return await provider.generate_images(model, context, options)

            request_model = replace(model, base_url=auth.base_url) if auth.base_url else model
            request_options = replace(
                options,
                api_key=options.api_key or auth.api_key,
                headers={**auth.headers, **options.headers} if auth.headers or options.headers else {},
                env={**(resolution.env or {}), **options.env} if resolution.env or options.env else {},
            )
            return await provider.generate_images(request_model, context, request_options)
        except Exception as error:
            return _error_images(model, error)


def create_images_models(
    providers: Sequence[ImagesProvider] | None = None,
    credential_store: CredentialStore | None = None,
    env: EnvLookup | None = None,
) -> ImagesModels:
    """Port of `createImagesModels`."""
    return ImagesModels(providers=providers, credential_store=credential_store, env=env)
