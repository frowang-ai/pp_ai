"""Built-in providers.

Python port of `packages/ai/src/providers/` (one module per TypeScript
provider factory). :mod:`pi_ai.providers.all` is the port of
`packages/ai/src/providers/all.ts` and owns the built-in provider set; this
package re-exports it plus every individual factory and its generated model
list.
"""

from __future__ import annotations

from ..registry import Provider
from .all import (
    builtin_images_models,
    builtin_images_providers,
    builtin_models,
    builtin_providers,
    get_builtin_model,
    get_builtin_model_data_generated_at,
    get_builtin_models,
    get_builtin_providers,
)
from .amazon_bedrock import AMAZON_BEDROCK_MODELS, amazon_bedrock_provider
from .ant_ling import ANT_LING_MODELS, ant_ling_provider
from .anthropic import ANTHROPIC_MODELS, anthropic_provider
from .azure_openai_responses import AZURE_OPENAI_RESPONSES_MODELS, azure_openai_responses_provider
from .baseten import BASETEN_MODELS, baseten_provider
from .cerebras import CEREBRAS_MODELS, cerebras_provider
from .cloudflare_ai_gateway import CLOUDFLARE_AI_GATEWAY_MODELS, cloudflare_ai_gateway_provider
from .cloudflare_auth import cloudflare_ai_gateway_auth, cloudflare_workers_ai_auth
from .cloudflare_stream import cloudflare_streams, resolve_cloudflare_model
from .cloudflare_workers_ai import CLOUDFLARE_WORKERS_AI_MODELS, cloudflare_workers_ai_provider
from .deepseek import DEEPSEEK_MODELS, deepseek_provider
from .faux import (
    FauxCore,
    FauxModelDefinition,
    FauxProviderHandle,
    FauxProviderState,
    RegisterFauxProviderOptions,
    create_faux_core,
    faux_assistant_message,
    faux_provider,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from .fireworks import FIREWORKS_MODELS, fireworks_provider
from .github_copilot import GITHUB_COPILOT_MODELS, filter_github_copilot_models, github_copilot_provider
from .google import GOOGLE_MODELS, google_provider
from .google_vertex import GOOGLE_VERTEX_MODELS, google_vertex_provider
from .groq import GROQ_MODELS, groq_provider
from .huggingface import HUGGINGFACE_MODELS, huggingface_provider
from .kimi_coding import KIMI_CODING_MODELS, kimi_coding_provider
from .minimax import MINIMAX_MODELS, minimax_provider
from .minimax_cn import MINIMAX_CN_MODELS, minimax_cn_provider
from .mistral import MISTRAL_MODELS, mistral_provider
from .moonshotai import MOONSHOTAI_MODELS, moonshotai_provider
from .moonshotai_cn import MOONSHOTAI_CN_MODELS, moonshotai_cn_provider
from .nvidia import NVIDIA_MODELS, nvidia_provider
from .openai import OPENAI_MODELS, openai_provider
from .openai_codex import OPENAI_CODEX_MODELS, openai_codex_provider
from .openai_compatible import OPENAI_COMPLETIONS_MODELS, openai_compatible_provider
from .openai_responses import OPENAI_RESPONSES_MODELS, openai_responses_provider
from .opencode import OPENCODE_MODELS, opencode_provider
from .opencode_go import OPENCODE_GO_MODELS, opencode_go_provider
from .openrouter import OPENROUTER_MODELS, openrouter_provider
from .openrouter_images import OPENROUTER_IMAGES_MODELS, openrouter_images_provider
from .qwen_token_plan import QWEN_TOKEN_PLAN_MODELS, qwen_token_plan_provider
from .qwen_token_plan_cn import QWEN_TOKEN_PLAN_CN_MODELS, qwen_token_plan_cn_provider
from .qwen_token_plan_individual import (
    QWEN_TOKEN_PLAN_INDIVIDUAL_MODELS,
    qwen_token_plan_individual_provider,
)
from .radius import radius_provider, refresh_radius_models
from .radius_config import (
    DEFAULT_RADIUS_GATEWAY,
    RadiusGatewayConfig,
    RadiusGatewayModel,
    get_radius_models,
    get_radius_models_from_config,
    load_radius_gateway_config,
    normalize_radius_gateway_url,
)
from .together import TOGETHER_MODELS, together_provider
from .vercel_ai_gateway import VERCEL_AI_GATEWAY_MODELS, vercel_ai_gateway_provider
from .xai import XAI_MODELS, xai_provider
from .xiaomi import XIAOMI_MODELS, xiaomi_provider
from .xiaomi_token_plan_ams import XIAOMI_TOKEN_PLAN_AMS_MODELS, xiaomi_token_plan_ams_provider
from .xiaomi_token_plan_cn import XIAOMI_TOKEN_PLAN_CN_MODELS, xiaomi_token_plan_cn_provider
from .xiaomi_token_plan_sgp import XIAOMI_TOKEN_PLAN_SGP_MODELS, xiaomi_token_plan_sgp_provider
from .zai import ZAI_MODELS, zai_provider
from .zai_coding_cn import ZAI_CODING_CN_MODELS, zai_coding_cn_provider


def all_providers() -> list[Provider]:
    """Every built-in provider, freshly constructed.

    Alias of :func:`builtin_providers` (the port of `builtinProviders()`), kept
    because the coding-agent package and its tests import this name.

    Deliberately excludes `faux_provider`: it is a scripted test double, not a
    real provider, and callers that want it opt in explicitly. It also excludes
    :func:`openai_responses_provider`, a Python-only convenience factory that
    would collide with the generated `openai` catalog.
    """
    return builtin_providers()


__all__ = [
    "AMAZON_BEDROCK_MODELS",
    "ANTHROPIC_MODELS",
    "ANT_LING_MODELS",
    "AZURE_OPENAI_RESPONSES_MODELS",
    "BASETEN_MODELS",
    "CEREBRAS_MODELS",
    "CLOUDFLARE_AI_GATEWAY_MODELS",
    "CLOUDFLARE_WORKERS_AI_MODELS",
    "DEEPSEEK_MODELS",
    "DEFAULT_RADIUS_GATEWAY",
    "FIREWORKS_MODELS",
    "GITHUB_COPILOT_MODELS",
    "GOOGLE_MODELS",
    "GOOGLE_VERTEX_MODELS",
    "GROQ_MODELS",
    "HUGGINGFACE_MODELS",
    "KIMI_CODING_MODELS",
    "MINIMAX_CN_MODELS",
    "MINIMAX_MODELS",
    "MISTRAL_MODELS",
    "MOONSHOTAI_CN_MODELS",
    "MOONSHOTAI_MODELS",
    "NVIDIA_MODELS",
    "OPENAI_CODEX_MODELS",
    "OPENAI_COMPLETIONS_MODELS",
    "OPENAI_MODELS",
    "OPENAI_RESPONSES_MODELS",
    "OPENCODE_GO_MODELS",
    "OPENCODE_MODELS",
    "OPENROUTER_IMAGES_MODELS",
    "OPENROUTER_MODELS",
    "QWEN_TOKEN_PLAN_CN_MODELS",
    "QWEN_TOKEN_PLAN_INDIVIDUAL_MODELS",
    "QWEN_TOKEN_PLAN_MODELS",
    "TOGETHER_MODELS",
    "VERCEL_AI_GATEWAY_MODELS",
    "XAI_MODELS",
    "XIAOMI_MODELS",
    "XIAOMI_TOKEN_PLAN_AMS_MODELS",
    "XIAOMI_TOKEN_PLAN_CN_MODELS",
    "XIAOMI_TOKEN_PLAN_SGP_MODELS",
    "ZAI_CODING_CN_MODELS",
    "ZAI_MODELS",
    "FauxCore",
    "FauxModelDefinition",
    "FauxProviderHandle",
    "FauxProviderState",
    "RadiusGatewayConfig",
    "RadiusGatewayModel",
    "RegisterFauxProviderOptions",
    "all_providers",
    "amazon_bedrock_provider",
    "ant_ling_provider",
    "anthropic_provider",
    "azure_openai_responses_provider",
    "baseten_provider",
    "builtin_images_models",
    "builtin_images_providers",
    "builtin_models",
    "builtin_providers",
    "cerebras_provider",
    "cloudflare_ai_gateway_auth",
    "cloudflare_ai_gateway_provider",
    "cloudflare_streams",
    "cloudflare_workers_ai_auth",
    "cloudflare_workers_ai_provider",
    "create_faux_core",
    "deepseek_provider",
    "faux_assistant_message",
    "faux_provider",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
    "filter_github_copilot_models",
    "fireworks_provider",
    "get_builtin_model",
    "get_builtin_model_data_generated_at",
    "get_builtin_models",
    "get_builtin_providers",
    "get_radius_models",
    "get_radius_models_from_config",
    "github_copilot_provider",
    "google_provider",
    "google_vertex_provider",
    "groq_provider",
    "huggingface_provider",
    "kimi_coding_provider",
    "load_radius_gateway_config",
    "minimax_cn_provider",
    "minimax_provider",
    "mistral_provider",
    "moonshotai_cn_provider",
    "moonshotai_provider",
    "normalize_radius_gateway_url",
    "nvidia_provider",
    "openai_codex_provider",
    "openai_compatible_provider",
    "openai_provider",
    "openai_responses_provider",
    "opencode_go_provider",
    "opencode_provider",
    "openrouter_images_provider",
    "openrouter_provider",
    "qwen_token_plan_cn_provider",
    "qwen_token_plan_individual_provider",
    "qwen_token_plan_provider",
    "radius_provider",
    "refresh_radius_models",
    "resolve_cloudflare_model",
    "together_provider",
    "vercel_ai_gateway_provider",
    "xai_provider",
    "xiaomi_provider",
    "xiaomi_token_plan_ams_provider",
    "xiaomi_token_plan_cn_provider",
    "xiaomi_token_plan_sgp_provider",
    "zai_coding_cn_provider",
    "zai_provider",
]
