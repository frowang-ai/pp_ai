"""The built-in provider set.

Python port of `packages/ai/src/providers/all.ts`.

TypeScript reads the built-in catalog from a generated `models.generated.ts`
module and its `.manifest.json`. This port reads the same information straight
from the committed JSON shards under `pi_ai/providers/data/` through
:mod:`pi_ai.model_catalog`, so there is no code-generation step at import time.
The image half (`builtinImagesProviders`/`builtinImagesModels`) reads the
generated image shards under `pi_ai/providers/data/images/` the same way, via
:mod:`pi_ai.image_models`.
"""

from __future__ import annotations

from ..auth.types import CredentialStore, EnvLookup
from ..images_registry import ImagesModels, ImagesProvider, create_images_models
from ..model_catalog import get_model_data_generated_at, get_model_data_provider_ids, load_model_catalog
from ..registry import Models, Provider
from ..types import Model
from .amazon_bedrock import amazon_bedrock_provider
from .ant_ling import ant_ling_provider
from .anthropic import anthropic_provider
from .azure_openai_responses import azure_openai_responses_provider
from .baseten import baseten_provider
from .cerebras import cerebras_provider
from .cloudflare_ai_gateway import cloudflare_ai_gateway_provider
from .cloudflare_workers_ai import cloudflare_workers_ai_provider
from .deepseek import deepseek_provider
from .fireworks import fireworks_provider
from .github_copilot import github_copilot_provider
from .google import google_provider
from .google_vertex import google_vertex_provider
from .groq import groq_provider
from .huggingface import huggingface_provider
from .kimi_coding import kimi_coding_provider
from .minimax import minimax_provider
from .minimax_cn import minimax_cn_provider
from .mistral import mistral_provider
from .moonshotai import moonshotai_provider
from .moonshotai_cn import moonshotai_cn_provider
from .nvidia import nvidia_provider
from .openai import openai_provider
from .openai_codex import openai_codex_provider
from .opencode import opencode_provider
from .opencode_go import opencode_go_provider
from .openrouter import openrouter_provider
from .openrouter_images import openrouter_images_provider
from .qwen_token_plan import qwen_token_plan_provider
from .qwen_token_plan_cn import qwen_token_plan_cn_provider
from .qwen_token_plan_individual import qwen_token_plan_individual_provider
from .radius import radius_provider
from .together import together_provider
from .vercel_ai_gateway import vercel_ai_gateway_provider
from .xai import xai_provider
from .xiaomi import xiaomi_provider
from .xiaomi_token_plan_ams import xiaomi_token_plan_ams_provider
from .xiaomi_token_plan_cn import xiaomi_token_plan_cn_provider
from .xiaomi_token_plan_sgp import xiaomi_token_plan_sgp_provider
from .zai import zai_provider
from .zai_coding_cn import zai_coding_cn_provider


def get_builtin_providers() -> list[str]:
    """Provider ids present in the generated catalog.

    Purely dynamic providers such as ``radius`` have no generated shard and are
    therefore absent here, while still being returned by
    :func:`builtin_providers`.
    """
    return get_model_data_provider_ids()


def get_builtin_models(provider: str) -> list[Model]:
    """Every generated model of one provider."""
    return list(load_model_catalog(provider).values())


def get_builtin_model(provider: str, model_id: str) -> Model | None:
    """One generated model, or ``None`` when the provider or id is unknown."""
    return load_model_catalog(provider).get(model_id)


def get_builtin_model_data_generated_at() -> int | None:
    """Generation timestamp (ms since epoch) shared by all built-in catalogs."""
    return get_model_data_generated_at()


def builtin_providers() -> list[Provider]:
    """All built-in providers, freshly constructed."""
    return [
        amazon_bedrock_provider(),
        ant_ling_provider(),
        anthropic_provider(),
        azure_openai_responses_provider(),
        baseten_provider(),
        cerebras_provider(),
        cloudflare_ai_gateway_provider(),
        cloudflare_workers_ai_provider(),
        deepseek_provider(),
        fireworks_provider(),
        github_copilot_provider(),
        google_provider(),
        google_vertex_provider(),
        groq_provider(),
        huggingface_provider(),
        kimi_coding_provider(),
        minimax_provider(),
        minimax_cn_provider(),
        mistral_provider(),
        moonshotai_provider(),
        moonshotai_cn_provider(),
        nvidia_provider(),
        openai_provider(),
        openai_codex_provider(),
        opencode_provider(),
        opencode_go_provider(),
        openrouter_provider(),
        qwen_token_plan_provider(),
        qwen_token_plan_cn_provider(),
        qwen_token_plan_individual_provider(),
        radius_provider(),
        together_provider(),
        vercel_ai_gateway_provider(),
        xai_provider(),
        xiaomi_provider(),
        xiaomi_token_plan_ams_provider(),
        xiaomi_token_plan_cn_provider(),
        xiaomi_token_plan_sgp_provider(),
        zai_provider(),
        zai_coding_cn_provider(),
    ]


def builtin_models() -> Models:
    """A :class:`~pi_ai.registry.Models` registry with every built-in provider."""
    return Models(providers=builtin_providers())


def builtin_images_providers() -> list[ImagesProvider]:
    """All built-in image-generation providers, freshly constructed."""
    return [openrouter_images_provider()]


def builtin_images_models(
    credential_store: CredentialStore | None = None,
    env: EnvLookup | None = None,
) -> ImagesModels:
    """An :class:`~pi_ai.images_registry.ImagesModels` collection with every built-in image provider."""
    models = create_images_models(credential_store=credential_store, env=env)
    for provider in builtin_images_providers():
        models.add(provider)
    return models
