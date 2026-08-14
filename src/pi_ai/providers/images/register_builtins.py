"""Registration of the built-in image api providers.

Python port of `packages/ai/src/providers/images/register-builtins.ts`.

The TypeScript source loads `api/openrouter-images.ts` through a dynamic
`import()` so bundlers keep it out of the initial chunk, and turns a failed
load into an `AssistantImages` with `stopReason: "error"` since an
`ImagesFunction` must not reject. A Python import has no bundler to defer and
no load step that can fail at call time, so the api module is imported directly
here — the same reason `api/lazy.ts` and the `*.lazy.ts` wrappers have no
counterpart in this port.
"""

from __future__ import annotations

from ...api import openrouter_images
from ...images_api_registry import ImagesApiProvider, register_images_api_provider
from ...types import AssistantImages, ImagesContext, ImagesModel, ImagesOptions

OPENROUTER_IMAGES_API = "openrouter-images"


async def generate_images_openrouter(
    model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None
) -> AssistantImages:
    """Generate images through `pi_ai.api.openrouter_images`."""
    return await openrouter_images.generate_images(model, context, options)


def register_builtin_images_api_providers() -> None:
    """Register every built-in image api implementation."""
    register_images_api_provider(
        ImagesApiProvider(api=OPENROUTER_IMAGES_API, generate_images=generate_images_openrouter)
    )


register_builtin_images_api_providers()
