"""OpenRouter image-generation provider factory.

Python port of `packages/ai/src/providers/openrouter-images.ts`. The model list
comes from the generated image catalog shard
`pi_ai/providers/data/images/openrouter.json` (see :mod:`pi_ai.image_models`),
the Python equivalent of TypeScript's `IMAGE_MODELS.openrouter`.

Auth is the same api key and OAuth flow as the chat-side
:mod:`pi_ai.providers.openrouter` provider, so a single OpenRouter credential
covers both.
"""

from __future__ import annotations

from ..auth.helpers import env_api_key_auth, lazy_oauth
from ..auth.oauth.load import load_openrouter_oauth
from ..auth.types import ProviderAuth
from ..image_models import get_image_models
from ..images_registry import ImagesProvider, create_images_provider
from ..providers.images.register_builtins import generate_images_openrouter
from ..types import ImagesModel

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

OPENROUTER_IMAGES_MODELS: list[ImagesModel] = get_image_models("openrouter")


def openrouter_images_provider() -> ImagesProvider:
    """Build the built-in OpenRouter image-generation provider."""
    return create_images_provider(
        id="openrouter",
        name="OpenRouter",
        auth=ProviderAuth(
            api_key=env_api_key_auth("OpenRouter API key", [OPENROUTER_API_KEY_ENV]),
            oauth=lazy_oauth("OpenRouter OAuth", load_openrouter_oauth, login_label="Sign in with OpenRouter"),
        ),
        api=generate_images_openrouter,
        models=OPENROUTER_IMAGES_MODELS,
    )
