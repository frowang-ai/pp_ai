"""The `generate_images` entry point.

Python port of `packages/ai/src/images.ts`: resolve the api implementation
registered for ``model.api`` and delegate to it.

Importing this module registers the built-in image api providers, exactly as
the TypeScript source's bare `import "./providers/images/register-builtins.ts"`
does.
"""

from __future__ import annotations

from .images_api_registry import ImagesApiProvider, get_images_api_provider
from .providers.images import register_builtins as _register_builtins  # noqa: F401 - import for its side effect
from .types import AssistantImages, ImagesApi, ImagesContext, ImagesModel, ImagesOptions


def resolve_images_api_provider(api: ImagesApi) -> ImagesApiProvider:
    """Port of `resolveImagesApiProvider`."""
    provider = get_images_api_provider(api)
    if provider is None:
        raise ValueError(f"No API provider registered for api: {api}")
    return provider


async def generate_images(
    model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None
) -> AssistantImages:
    """Generate images with the api implementation registered for ``model.api``."""
    return await resolve_images_api_provider(model.api).generate_images(model, context, options)
