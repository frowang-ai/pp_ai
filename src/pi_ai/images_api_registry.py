"""The image-generation API registry.

Python port of `packages/ai/src/images-api-registry.ts`, the image-side
counterpart of the chat api registry: it maps an api name
(``"openrouter-images"``) to the function that implements it, so
:func:`pi_ai.images.generate_images` can dispatch on ``model.api`` alone.

Registration wraps the function in a guard that rejects a model belonging to a
different api, which is what makes the single-argument dispatch safe.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .types import AssistantImages, ImagesApi, ImagesContext, ImagesModel, ImagesOptions

ImagesApiFunction = Callable[[ImagesModel, ImagesContext, "ImagesOptions | None"], Awaitable[AssistantImages]]
"""Port of `ImagesApiFunction`: the erased, api-checked generation function."""


@dataclass
class ImagesApiProvider:
    """An api implementation, as handed to :func:`register_images_api_provider`."""

    api: ImagesApi
    generate_images: ImagesApiFunction


_registry: dict[str, ImagesApiProvider] = {}
_source_ids: dict[str, str] = {}


def _wrap_generate_images(api: ImagesApi, generate_images: ImagesApiFunction) -> ImagesApiFunction:
    """Port of `wrapGenerateImages`: refuse a model that belongs to another api."""

    async def guarded(
        model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None
    ) -> AssistantImages:
        if model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return await generate_images(model, context, options)

    return guarded


def register_images_api_provider(provider: ImagesApiProvider, source_id: str | None = None) -> None:
    """Register (or replace) the implementation of one image api."""
    _registry[provider.api] = ImagesApiProvider(
        api=provider.api,
        generate_images=_wrap_generate_images(provider.api, provider.generate_images),
    )
    if source_id is None:
        _source_ids.pop(provider.api, None)
    else:
        _source_ids[provider.api] = source_id


def get_images_api_provider(api: ImagesApi) -> ImagesApiProvider | None:
    """The registered implementation of ``api``, or ``None``."""
    return _registry.get(api)


def get_images_api_provider_source_id(api: ImagesApi) -> str | None:
    """The ``source_id`` an api was registered with, if any.

    TypeScript stores `sourceId` alongside the provider but never reads it back;
    this accessor exists so the stored value is not silently unreachable.
    """
    return _source_ids.get(api)
