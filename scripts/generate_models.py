#!/usr/bin/env python3
"""Generate the committed per-provider model catalog.

Python port of `packages/ai/scripts/generate-models.ts`, together with its
helpers `packages/ai/scripts/model-data.ts` (re-exported from
``scripts/model_data.py``) and
`packages/ai/scripts/models-dev-reasoning-options.ts` (re-exported from
``scripts/models_dev_reasoning_options.py``).

The script fetches https://models.dev/api.json plus the live NVIDIA NIM,
OpenRouter and Vercel AI Gateway model lists, applies the per-provider
corrections that make up the bulk of the TypeScript file, then writes one JSON
shard per provider into ``pi_ai/providers/data/`` next to a ``.manifest.json``.

Two intentional differences from TypeScript:

* TypeScript also emits ``src/providers/<provider>.models.ts`` modules and
  ``src/models.generated.ts``. Python needs no code generation: the shards are
  read at runtime by :mod:`pi_ai.model_catalog`, so only the JSON is written.
* The generated JSON is committed here (TypeScript gitignores it and hydrates
  at build time) so the package works offline.

Usage::

    uv run python packages/pi-ai/scripts/generate_models.py [--strict] [--pretty]
        [--data-only] [--json-only] [--json-output DIR] [--models-dev-file FILE]

``--models-dev-file`` is a port addition: it reads a cached copy of
``models.dev/api.json`` instead of fetching, for offline regeneration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_data import (
    MODEL_DATA_MANIFEST_FILE,
    ModelDataStructure,
    assert_exact_model_ids,
    create_model_data_manifest,
    read_model_data_provider_ids,
    validate_generated_model_data,
    validate_model_data_directory,
)
from models_dev_reasoning_options import get_effort_thinking_level_map
from pi_ai.api.cloudflare import (
    CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL,
    CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL,
    CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL,
    CLOUDFLARE_WORKERS_AI_BASE_URL,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROVIDERS_DIR = PACKAGE_ROOT / "src" / "pi_ai" / "providers"
DATA_DIR = PROVIDERS_DIR / "data"

Json = dict[str, Any]


@dataclass
class GeneratorOptions:
    """Port of the object returned by `readGeneratorOptions`."""

    strict: bool = False
    data_only: bool = False
    json_only: bool = False
    json_output_dir: Path | None = None
    pretty: bool = False
    models_dev_file: Path | None = None


generator_options = GeneratorOptions()


def read_generator_options(args: list[str]) -> GeneratorOptions:
    parser = argparse.ArgumentParser(add_help=True, description=__doc__)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--json-output", metavar="DIR")
    parser.add_argument("--models-dev-file", metavar="FILE")
    parsed = parser.parse_args(args)

    json_output_dir = Path(parsed.json_output).resolve() if parsed.json_output else None
    if parsed.json_only and json_output_dir is None:
        parser.error("--json-only requires --json-output")
    if parsed.data_only and (parsed.json_only or json_output_dir is not None):
        parser.error("--data-only cannot be combined with JSON catalog output")
    return GeneratorOptions(
        strict=parsed.strict,
        data_only=parsed.data_only,
        json_only=parsed.json_only,
        json_output_dir=json_output_dir,
        pretty=parsed.pretty,
        models_dev_file=Path(parsed.models_dev_file).resolve() if parsed.models_dev_file else None,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COPILOT_STATIC_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}

KIMI_STATIC_HEADERS = {"User-Agent": "KimiCLI/1.5"}

TOGETHER_BASE_URL = "https://api.together.ai/v1"
TOGETHER_BASE_COMPAT: Json = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
    "supportsStrictMode": False,
    "supportsLongCacheRetention": False,
}
TOGETHER_TOGGLE_REASONING_COMPAT: Json = {**TOGETHER_BASE_COMPAT, "thinkingFormat": "together"}
TOGETHER_REASONING_EFFORT_COMPAT: Json = {
    **TOGETHER_BASE_COMPAT,
    "supportsReasoningEffort": True,
    "thinkingFormat": "openai",
}
TOGETHER_TOGGLE_REASONING_EFFORT_COMPAT: Json = {
    **TOGETHER_TOGGLE_REASONING_COMPAT,
    "supportsReasoningEffort": True,
}
TOGETHER_REASONING_ONLY_MODELS = {"deepseek-ai/DeepSeek-R1", "MiniMaxAI/MiniMax-M2.7"}
TOGETHER_REASONING_EFFORT_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
TOGETHER_TOGGLE_REASONING_EFFORT_MODELS = {"deepseek-ai/DeepSeek-V4-Pro"}
TOGETHER_FIXED_REASONING_LEVEL_MAP: Json = {"off": None, "minimal": None, "low": None, "medium": None}
TOGETHER_REASONING_EFFORT_LEVEL_MAP: Json = {"off": None, "minimal": None}
TOGETHER_DEEPSEEK_V4_THINKING_LEVEL_MAP: Json = {
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
}
TOGETHER_TOGGLE_REASONING_LEVEL_MAP: Json = {"minimal": None, "low": None, "medium": None}

AI_GATEWAY_MODELS_URL = "https://ai-gateway.vercel.sh/v1"
AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh"
VERTEX_BASE_URL = "https://{location}-aiplatform.googleapis.com"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_HEADERS = {"NVCF-POLL-SECONDS": "3600"}
NVIDIA_OPENAI_COMPAT: Json = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
    "supportsStrictMode": False,
    "supportsLongCacheRetention": False,
}
NVIDIA_NIM_UNSUPPORTED_MODELS = {
    "abacusai/dracarys-llama-3.1-70b-instruct",
    "bytedance/seed-oss-36b-instruct",
    "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-pro",
    "google/gemma-2-2b-it",
    "google/gemma-3n-e2b-it",
    "google/gemma-3n-e4b-it",
    "google/gemma-4-31b-it",
    "meta/llama-3.2-1b-instruct",
    "meta/llama-4-maverick-17b-128e-instruct",
    "microsoft/phi-4-mini-instruct",
    "minimaxai/minimax-m2.7",
    "mistralai/mistral-nemotron",
    "nvidia/nemotron-mini-4b-instruct",
    "qwen/qwen3-next-80b-a3b-instruct",
    "qwen/qwen3.5-397b-a17b",
    "sarvamai/sarvam-m",
    "upstage/solar-10.7b-instruct",
}
ZAI_TOOL_STREAM_UNSUPPORTED_MODELS = {"glm-4.5", "glm-4.5-air", "glm-4.5-flash", "glm-4.5v"}
ZAI_GLM52_THINKING_LEVEL_MAP: Json = {
    "minimal": None,
    "low": "high",
    "medium": "high",
    "high": "high",
    "max": "max",
}
OPENCODE_GO_GLM52_THINKING_LEVEL_MAP: Json = {
    "off": None,
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "max": "max",
}
EAGER_TOOL_INPUT_STREAMING_UNSUPPORTED_ANTHROPIC_MODELS = {
    "github-copilot:claude-haiku-4.5",
    "github-copilot:claude-sonnet-4",
    "github-copilot:claude-sonnet-4.5",
}

DEEPSEEK_V4_THINKING_LEVEL_MAP: Json = {
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "max": "max",
}
QWEN_TOKEN_PLAN_HIGH_MAX_THINKING_LEVEL_MAP: Json = {
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": "max",
}
QWEN_TOKEN_PLAN_QWEN38_THINKING_LEVEL_MAP: Json = {
    "minimal": None,
    "low": "low",
    "medium": "medium",
    "high": None,
    "xhigh": "xhigh",
    "max": None,
}
QWEN_TOKEN_PLAN_REASONING_EFFORT_UNSUPPORTED_MODEL_IDS = {
    "MiniMax-M2.5",
    "deepseek-v3.2",
    "kimi-k2.5",
    "kimi-k2.6",
    "kimi-k2.7-code",
    "qwen3.6-flash",
    "qwen3.6-plus",
    "qwen3.7-max",
    "qwen3.7-plus",
}
# Retired preview id -- models.dev may still list it after GA ships.
QWEN_TOKEN_PLAN_EXCLUDED_MODEL_IDS = {"qwen3.8-max-preview"}
QWEN_TOKEN_PLAN_PROVIDER_IDS = {
    "qwen-token-plan",
    "qwen-token-plan-cn",
    "qwen-token-plan-individual",
}
# QwenCloud Token Plan Individual text-model allowlist, verified 2026-08-05.
# Retired models remain excluded above even if the public catalog lags.
# https://docs.qwencloud.com/token-plan/personal/token-plan-personal-overview
QWEN_TOKEN_PLAN_INDIVIDUAL_MODEL_IDS = {
    "deepseek-v4-flash-0731",
    "deepseek-v4-pro",
    "glm-5.2",
    "qwen3.6-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.8-max",
}

KIMI_K3_MAX_TOKENS = 131072
KIMI_K3_COST: Json = {"input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 0}
# Kimi Coding is subscription-backed, so models.dev reports zero cost. Use the
# equivalent Moonshot API rates to estimate the value of subscription usage.
KIMI_CODING_IMPLIED_COSTS: dict[str, Json] = {
    "k3": KIMI_K3_COST,
    "kimi-for-coding": {"input": 0.95, "output": 4, "cacheRead": 0.19, "cacheWrite": 0},
    "kimi-for-coding-highspeed": {"input": 1.9, "output": 8, "cacheRead": 0.38, "cacheWrite": 0},
    "kimi-k2-thinking": {"input": 0.6, "output": 2.5, "cacheRead": 0.15, "cacheWrite": 0},
}
OPENROUTER_KIMI_K3_MODEL_IDS = {"moonshotai/kimi-k3", "~moonshotai/kimi-latest"}

ANT_LING_RING_THINKING_LEVEL_MAP: Json = {
    "off": None,
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": "xhigh",
}

BEDROCK_INFERENCE_PROFILE_ONLY_MODEL_IDS = {"anthropic.claude-opus-5"}
MODELS_DEV_OPENAI_UNSUPPORTED_MODEL_IDS = {"gpt-5.6"}
OPENAI_TOOL_SEARCH_MODEL_IDS = {
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-pro",
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
}
# Public OpenAI documents additional_tools for applications that load tools
# outside the normal tool-search flow. Codex currently uses the input item for
# its Responses Lite GPT-5.6 models.
OPENAI_ADDITIONAL_TOOLS_MODEL_IDS = OPENAI_TOOL_SEARCH_MODEL_IDS
OPENAI_CODEX_ADDITIONAL_TOOLS_MODEL_IDS = {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}
OPENAI_LONG_CONTEXT_INPUT_THRESHOLD = 272000
OPENAI_SHORT_CONTEXT_CAPPED_MODEL_IDS = {
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
}
OPENAI_LONG_CONTEXT_PRICING_MODEL_IDS = {
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
}

# OpenAI reduced GPT-5.6 Terra and Luna prices on 2026-07-30. Keep these
# authoritative values until models.dev and passthrough catalogs catch up.
OPENAI_GPT_56_STANDARD_COSTS: dict[str, Json] = {
    "gpt-5.6-luna": {"input": 0.2, "output": 1.2, "cacheRead": 0.02, "cacheWrite": 0.25},
    "gpt-5.6-terra": {"input": 2, "output": 12, "cacheRead": 0.2, "cacheWrite": 2.5},
}

OPENAI_RESPONSES_NONE_REASONING_MODELS = {
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
}
XAI_RESPONSES_MODEL_ID = "grok-4.5"
XAI_BUILTIN_EXCLUDED_MODEL_IDS = {
    "grok-3",
    "grok-3-fast",
    "grok-4.20-0309-non-reasoning",
    "grok-4.20-0309-reasoning",
    "grok-code-fast-1",
}
XAI_RESPONSES_EFFORT_LEVEL_MAP: Json = {"off": None, "minimal": None}
XAI_RESPONSES_COMPAT: Json = {"supportsLongCacheRetention": False}

OPENCODE_OPENAI_COMPLETIONS_LONG_CACHE_RETENTION_UNSUPPORTED_MODELS = {
    "opencode:deepseek-v4-flash",
    "opencode:deepseek-v4-pro",
    "opencode:kimi-k2.5",
    "opencode:kimi-k2.6",
    "opencode:minimax-m2.7",
    "opencode-go:kimi-k2.6",
}

# GitHub's "Models with extended capabilities" table lists these Copilot models
# as supporting the extended 1 million token context window.
GITHUB_COPILOT_EXTENDED_CONTEXT_MODELS = {
    "claude-fable-5",
    "claude-opus-4.6",
    "claude-opus-4.7",
    "claude-opus-4.8",
    "claude-opus-5",
    "claude-sonnet-4.6",
    "claude-sonnet-5",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.5",
}

# Checked manually against the authenticated GitHub Copilot /models endpoint on
# 2026-06-15. Keep this to narrow corrections over models.dev metadata instead
# of snapshotting Copilot's catalog.
GITHUB_COPILOT_THINKING_LEVEL_OVERRIDES: dict[str, Json] = {
    "claude-opus-4.7": {"minimal": "low"},
    "claude-opus-4.8": {"minimal": "low"},
    "claude-opus-5": {"minimal": "low"},
    "claude-sonnet-4.6": {"minimal": "low", "max": "max"},
}

OPENAI_COMPLETIONS_DEFAULT_COMPAT: Json = {
    "supportsStore": True,
    "supportsDeveloperRole": True,
    "supportsReasoningEffort": True,
    "supportsUsageInStreaming": True,
    "supportsFinishReason": True,
    "maxTokensField": "max_completion_tokens",
    "requiresToolResultName": False,
    "requiresAssistantAfterToolResult": False,
    "requiresThinkingAsText": False,
    "requiresReasoningContentOnAssistantMessages": False,
    "thinkingFormat": "openai",
    "openRouterRouting": {},
    "vercelGatewayRouting": {},
    "chatTemplateKwargs": {},
    "chatTemplateArgs": {},
    "zaiToolStream": False,
    "supportsStrictMode": True,
    "supportsOpenAIGrammarTools": False,
    "sendSessionAffinityHeaders": False,
    "supportsLongCacheRetention": True,
}

# Responses endpoints verified (OpenAI, ChatGPT Codex backend, GitHub Copilot,
# opencode zen) or documented (Azure OpenAI, Cloudflare AI Gateway) to pass
# OpenAI custom grammar tools through. OpenAI rejects `type: "custom"` tools
# for pre-GPT-5 models (gpt-4.x, gpt-4o, o-series).
OPENAI_GRAMMAR_TOOL_PROVIDERS = {
    "openai",
    "openai-codex",
    "azure-openai-responses",
    "github-copilot",
    "opencode",
    "cloudflare-ai-gateway",
}
OPENAI_GRAMMAR_TOOL_APIS = {"openai-responses", "azure-openai-responses", "openai-codex-responses"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def round_cost(value: float) -> float:
    """Port of `roundCost`: `Number(value.toFixed(6))`."""
    return float(Decimal(value).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def with_openai_long_context_pricing(cost: Json) -> Json:
    return {
        **cost,
        "tiers": [
            {
                "inputTokensAbove": OPENAI_LONG_CONTEXT_INPUT_THRESHOLD,
                "input": round_cost(cost["input"] * 2),
                "output": round_cost(cost["output"] * 1.5),
                "cacheRead": round_cost(cost["cacheRead"] * 2),
                "cacheWrite": round_cost(cost["cacheWrite"] * 2),
            }
        ],
    }


def make_model(
    *,
    id: str,
    name: str,
    api: str,
    provider: str,
    base_url: str,
    reasoning: bool,
    input: list[str],
    cost: Json,
    context_window: int,
    max_tokens: int,
    thinking_level_map: Json | None = None,
    compat: Json | None = None,
    headers: dict[str, str] | None = None,
) -> Json:
    """Build one generated model entry, using the TypeScript JSON key spelling."""
    model: Json = {
        "id": id,
        "name": name or id,
        "api": api,
        "provider": provider,
        "baseUrl": base_url,
        "reasoning": reasoning,
    }
    if thinking_level_map is not None:
        model["thinkingLevelMap"] = dict(thinking_level_map)
    model["input"] = list(input)
    model["cost"] = deepcopy(cost)
    if compat is not None:
        model["compat"] = deepcopy(compat)
    model["contextWindow"] = context_window
    model["maxTokens"] = max_tokens
    if headers is not None:
        model["headers"] = dict(headers)
    return model


def merge_thinking_level_map(model: Json, level_map: Json) -> None:
    """Port of `mergeThinkingLevelMap`."""
    model["thinkingLevelMap"] = {**model.get("thinkingLevelMap", {}), **level_map}


def merge_compat(model: Json, compat: Json) -> None:
    model["compat"] = {**model.get("compat", {}), **compat}


def supports_image(source: Json) -> bool:
    modalities = source.get("modalities") or {}
    return "image" in (modalities.get("input") or [])


def input_modalities(source: Json) -> list[str]:
    return ["text", "image"] if supports_image(source) else ["text"]


def models_dev_simple_cost(source: Json) -> Json:
    cost = source.get("cost") or {}
    return {
        "input": cost.get("input") or 0,
        "output": cost.get("output") or 0,
        "cacheRead": cost.get("cache_read") or 0,
        "cacheWrite": cost.get("cache_write") or 0,
    }


def get_models_dev_cost(cost: Json | None) -> Json:
    """Port of `getModelsDevCost`, including context-size cost tiers."""
    cost = cost or {}
    tiers: list[Json] = []
    for tier in cost.get("tiers") or []:
        context = tier.get("tier") or {}
        if context.get("type") != "context" or context.get("size") is None:
            continue
        tiers.append(
            {
                "inputTokensAbove": context["size"],
                "input": tier.get("input") or 0,
                "output": tier.get("output") or 0,
                "cacheRead": tier.get("cache_read") or 0,
                "cacheWrite": tier.get("cache_write") or 0,
            }
        )

    result: Json = {
        "input": cost.get("input") or 0,
        "output": cost.get("output") or 0,
        "cacheRead": cost.get("cache_read") or 0,
        "cacheWrite": cost.get("cache_write") or 0,
    }
    if tiers:
        result["tiers"] = tiers
    return result


def get_bedrock_base_url(model_id: str) -> str:
    if model_id.startswith("eu."):
        return "https://bedrock-runtime.eu-central-1.amazonaws.com"
    return "https://bedrock-runtime.us-east-1.amazonaws.com"


def normalize_nvidia_model_id(model_id: str) -> str:
    return model_id.lower().replace("_", ".")


def get_anthropic_messages_compat(provider: str, model_id: str) -> Json | None:
    """Port of `getAnthropicMessagesCompat`."""
    compat: Json = {}
    if f"{provider}:{model_id}" in EAGER_TOOL_INPUT_STREAMING_UNSUPPORTED_ANTHROPIC_MODELS:
        compat["supportsEagerToolInputStreaming"] = False
    if provider == "xiaomi" or provider.startswith("xiaomi-token-plan-"):
        compat["allowEmptySignature"] = True
    return compat or None


# ---------------------------------------------------------------------------
# models.dev reasoning options
# ---------------------------------------------------------------------------

models_dev_reasoning_options: dict[str, list[Json]] = {}


def get_model_key(model: Json) -> str:
    return f"{model['provider']}:{model['id']}"


def record_models_dev_reasoning_options(provider: str, model_id: str, source: Json) -> None:
    if "reasoning_options" in source and source["reasoning_options"] is not None:
        models_dev_reasoning_options[f"{provider}:{model_id}"] = source["reasoning_options"]


# ---------------------------------------------------------------------------
# openai-completions compat detection
# ---------------------------------------------------------------------------


def detect_openai_completions_compat(model: Json) -> Json:
    """Port of `detectOpenAICompletionsCompat`."""
    provider = model["provider"]
    base_url = model["baseUrl"]

    is_zai = (
        provider == "zai" or provider == "zai-coding-cn" or "api.z.ai" in base_url or "open.bigmodel.cn" in base_url
    )
    is_together = provider == "together" or "api.together.ai" in base_url or "api.together.xyz" in base_url
    is_moonshot = provider in ("moonshotai", "moonshotai-cn") or "api.moonshot." in base_url
    is_openrouter = provider == "openrouter" or "openrouter.ai" in base_url
    is_cloudflare_workers_ai = provider == "cloudflare-workers-ai" or "api.cloudflare.com" in base_url
    is_cloudflare_ai_gateway = provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in base_url
    is_nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in base_url
    is_ant_ling = provider == "ant-ling" or "api.ant-ling.com" in base_url
    is_together_reasoning_only = is_together and model["id"] in TOGETHER_REASONING_ONLY_MODELS

    is_non_standard = (
        is_nvidia
        or provider == "cerebras"
        or "cerebras.ai" in base_url
        or provider == "xai"
        or "api.x.ai" in base_url
        or is_together
        or "chutes.ai" in base_url
        or "deepseek.com" in base_url
        or is_zai
        or is_moonshot
        or provider == "opencode"
        or "opencode.ai" in base_url
        or is_cloudflare_workers_ai
        or is_cloudflare_ai_gateway
        or is_ant_ling
    )

    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url
    use_max_tokens = (
        "chutes.ai" in base_url
        or is_deepseek
        or is_moonshot
        or is_cloudflare_ai_gateway
        or is_together
        or is_nvidia
        or is_ant_ling
        or is_zai
    )

    is_grok = provider == "xai" or "api.x.ai" in base_url
    is_openrouter_developer_role_model = is_openrouter and (
        model["id"].startswith("anthropic/") or model["id"].startswith("openai/")
    )
    cache_control_format = "anthropic" if provider == "openrouter" and re.match(r"^~?anthropic/", model["id"]) else None

    if is_deepseek:
        thinking_format = "deepseek"
    elif is_zai:
        thinking_format = "zai"
    elif is_together and not is_together_reasoning_only:
        thinking_format = "together"
    elif is_ant_ling:
        thinking_format = "ant-ling"
    elif is_openrouter:
        thinking_format = "openrouter"
    else:
        thinking_format = "openai"

    compat: Json = {
        "supportsStore": not is_non_standard,
        "supportsDeveloperRole": is_openrouter_developer_role_model or (not is_non_standard and not is_openrouter),
        "supportsReasoningEffort": not (
            is_grok or is_zai or is_moonshot or is_together or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling
        ),
        "supportsUsageInStreaming": True,
        "supportsFinishReason": True,
        "maxTokensField": "max_tokens" if use_max_tokens else "max_completion_tokens",
        "requiresToolResultName": False,
        "requiresAssistantAfterToolResult": False,
        "requiresThinkingAsText": False,
        "requiresReasoningContentOnAssistantMessages": is_deepseek,
        "thinkingFormat": thinking_format,
        "openRouterRouting": {},
        "vercelGatewayRouting": {},
        "chatTemplateKwargs": {},
        "chatTemplateArgs": {},
        "zaiToolStream": False,
        "supportsStrictMode": not (is_moonshot or is_together or is_cloudflare_ai_gateway or is_nvidia),
        "supportsOpenAIGrammarTools": False,
    }
    if cache_control_format:
        compat["cacheControlFormat"] = cache_control_format
    compat["sendSessionAffinityHeaders"] = False
    compat["supportsLongCacheRetention"] = not (
        is_together or is_cloudflare_workers_ai or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling
    )
    return compat


_MISSING = object()


def _is_plain_empty_object(value: object) -> bool:
    return isinstance(value, dict) and len(value) == 0


def openai_completions_compat_delta(compat: Json) -> Json:
    """Port of `openAICompletionsCompatDelta`: emit only non-default keys."""
    delta: Json = {}
    for key, value in compat.items():
        default = OPENAI_COMPLETIONS_DEFAULT_COMPAT.get(key, _MISSING)
        if _is_plain_empty_object(value) and _is_plain_empty_object(default):
            continue
        if default is _MISSING or value != default:
            delta[key] = value
    return delta


def supports_direct_reasoning_effort(model: Json) -> bool:
    """Port of `supportsDirectReasoningEffort`."""
    api = model["api"]
    if api == "anthropic-messages":
        return (model.get("compat") or {}).get("forceAdaptiveThinking") is True
    if api in ("openai-responses", "azure-openai-responses", "openai-codex-responses"):
        return True
    if api != "openai-completions":
        return False
    compat = {**detect_openai_completions_compat(model), **(model.get("compat") or {})}
    return compat.get("thinkingFormat") == "openai" and bool(compat.get("supportsReasoningEffort"))


# ---------------------------------------------------------------------------
# Together / OpenAI / Anthropic / Google predicates
# ---------------------------------------------------------------------------


def get_together_compat(model_id: str, reasoning: bool) -> Json:
    if not reasoning:
        return TOGETHER_BASE_COMPAT
    if model_id in TOGETHER_REASONING_EFFORT_MODELS:
        return TOGETHER_REASONING_EFFORT_COMPAT
    if model_id in TOGETHER_TOGGLE_REASONING_EFFORT_MODELS:
        return TOGETHER_TOGGLE_REASONING_EFFORT_COMPAT
    if model_id in TOGETHER_REASONING_ONLY_MODELS:
        return TOGETHER_BASE_COMPAT
    return TOGETHER_TOGGLE_REASONING_COMPAT


def get_together_thinking_level_map(model_id: str, reasoning: bool) -> Json | None:
    if not reasoning:
        return None
    if model_id in TOGETHER_REASONING_EFFORT_MODELS:
        return dict(TOGETHER_REASONING_EFFORT_LEVEL_MAP)
    if model_id in TOGETHER_TOGGLE_REASONING_EFFORT_MODELS:
        return dict(TOGETHER_DEEPSEEK_V4_THINKING_LEVEL_MAP)
    if model_id in TOGETHER_REASONING_ONLY_MODELS:
        return dict(TOGETHER_FIXED_REASONING_LEVEL_MAP)
    return dict(TOGETHER_TOGGLE_REASONING_LEVEL_MAP)


def supports_openai_xhigh(model_id: str) -> bool:
    return any(marker in model_id for marker in ("gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5", "gpt-5.6"))


def supports_openai_max(model: Json) -> bool:
    return "gpt-5.6" in model["id"] and model["api"] in (
        "openai-responses",
        "azure-openai-responses",
        "openai-codex-responses",
        "openai-completions",
    )


def is_google_thinking_api(model: Json) -> bool:
    return model["api"] in ("google-generative-ai", "google-vertex")


def is_anthropic_adaptive_thinking_model(model_id: str) -> bool:
    markers = (
        "opus-4-6",
        "opus-4.6",
        "opus-4-7",
        "opus-4.7",
        "opus-4-8",
        "opus-4.8",
        "opus-5",
        "opus.5",
        "sonnet-4-6",
        "sonnet-4.6",
        "sonnet-5",
        "sonnet.5",
        "fable-5",
    )
    return any(marker in model_id for marker in markers)


def is_anthropic_temperature_unsupported_model(model_id: str) -> bool:
    model_id = model_id.lower()
    markers = ("opus-4-7", "opus-4.7", "opus-4-8", "opus-4.8", "opus-5", "opus.5")
    return any(marker in model_id for marker in markers)


def is_gemini_3_pro_model(model_id: str) -> bool:
    return re.search(r"gemini-3(?:\.\d+)?-pro", model_id.lower()) is not None


def is_gemini_3_flash_model(model_id: str) -> bool:
    model_id = model_id.lower()
    return (
        re.search(r"gemini-3(?:\.\d+)?-flash", model_id) is not None
        or model_id == "gemini-flash-latest"
        or model_id == "gemini-flash-lite-latest"
    )


def is_gemma_4_model(model_id: str) -> bool:
    return re.search(r"gemma-?4", model_id.lower()) is not None


# ---------------------------------------------------------------------------
# Metadata pipeline
# ---------------------------------------------------------------------------


def apply_openai_completions_compat_metadata(model: Json) -> None:
    if model["api"] != "openai-completions":
        return
    detected = openai_completions_compat_delta(detect_openai_completions_compat(model))
    model["compat"] = {**detected, **(model.get("compat") or {})}
    if not model["compat"]:
        del model["compat"]


def apply_models_dev_reasoning_option_metadata(model: Json) -> None:
    reasoning_options = models_dev_reasoning_options.get(get_model_key(model))
    if not reasoning_options or not supports_direct_reasoning_effort(model):
        return
    thinking_level_map = get_effort_thinking_level_map(reasoning_options)
    if thinking_level_map:
        merge_thinking_level_map(model, thinking_level_map)


def apply_thinking_level_metadata(model: Json) -> None:
    """Port of `applyThinkingLevelMetadata`."""
    model_id = model["id"]
    provider = model["provider"]
    api = model["api"]

    if api in ("openai-responses", "azure-openai-responses") and model_id.startswith("gpt-5"):
        merge_thinking_level_map(model, {"off": None})
    if provider == "github-copilot" and model_id.startswith("gpt-5"):
        merge_thinking_level_map(model, {"minimal": "low"})
    if api == "openai-responses" and provider == "openai" and model_id in OPENAI_RESPONSES_NONE_REASONING_MODELS:
        merge_thinking_level_map(model, {"off": "none"})
    if provider == "xai" and api == "openai-responses" and model_id == XAI_RESPONSES_MODEL_ID:
        merge_thinking_level_map(model, XAI_RESPONSES_EFFORT_LEVEL_MAP)
    if supports_openai_xhigh(model_id):
        merge_thinking_level_map(model, {"xhigh": "xhigh"})
    if supports_openai_max(model):
        merge_thinking_level_map(model, {"max": "max"})
    if provider == "openai" and model_id == "gpt-5.5":
        merge_thinking_level_map(model, {"minimal": None})
    if model_id.endswith("gpt-5.5-pro"):
        merge_thinking_level_map(model, {"off": None, "minimal": None, "low": None})

    # Anthropic adaptive-thinking effort support (per Anthropic adaptive thinking docs):
    # - "max" is available on all adaptive-thinking Claude models.
    # - "xhigh" is only available on Opus 4.7/4.8/5, Sonnet 5, and Fable 5.
    if any(marker in model_id for marker in ("opus-4-6", "opus-4.6", "sonnet-4-6", "sonnet-4.6")):
        merge_thinking_level_map(model, {"max": "max"})
    if any(
        marker in model_id
        for marker in ("opus-4-7", "opus-4.7", "opus-4-8", "opus-4.8", "opus-5", "opus.5", "sonnet-5", "sonnet.5")
    ):
        merge_thinking_level_map(model, {"xhigh": "xhigh", "max": "max"})
    if "fable-5" in model_id:
        merge_thinking_level_map(model, {"off": None, "xhigh": "xhigh", "max": "max"})
    if api == "anthropic-messages" and is_anthropic_adaptive_thinking_model(model_id):
        merge_compat(model, {"forceAdaptiveThinking": True})
    if api == "anthropic-messages" and is_anthropic_temperature_unsupported_model(model_id):
        merge_compat(model, {"supportsTemperature": False})
    if api == "openai-completions" and "deepseek-v4" in model_id:
        merge_thinking_level_map(
            model,
            {**DEEPSEEK_V4_THINKING_LEVEL_MAP, "xhigh": "xhigh", "max": None}
            if provider == "openrouter"
            else DEEPSEEK_V4_THINKING_LEVEL_MAP,
        )
    if is_google_thinking_api(model) and is_gemini_3_pro_model(model_id):
        merge_thinking_level_map(model, {"off": None, "minimal": None, "low": "LOW", "medium": None, "high": "HIGH"})
    if is_google_thinking_api(model) and is_gemini_3_flash_model(model_id):
        merge_thinking_level_map(model, {"off": None})
    if is_google_thinking_api(model) and is_gemma_4_model(model_id):
        merge_thinking_level_map(
            model, {"off": None, "minimal": "MINIMAL", "low": None, "medium": None, "high": "HIGH"}
        )
    if provider == "groq" and model_id == "qwen/qwen3.6-27b":
        merge_thinking_level_map(model, {"minimal": None, "low": None, "medium": None, "high": "default"})
    if provider == "openai-codex" and supports_openai_xhigh(model_id):
        merge_thinking_level_map(model, {"minimal": "low"})
    if provider in ("moonshotai", "moonshotai-cn") and model_id in ("kimi-k2.7-code", "kimi-k2.7-code-highspeed"):
        # Kimi K2.7 Code is always-thinking. Official docs say
        # `thinking: { type: "disabled" }` is rejected, and callers can omit the
        # thinking parameter to use the enabled default.
        merge_thinking_level_map(model, {"off": None})
    if provider == "openrouter" and model_id.startswith("inception/mercury-2"):
        # Mercury 2 in instant mode (reasoning_effort: "none") disables tool calling.
        merge_thinking_level_map(model, {"off": None})
    if provider == "openrouter" and model_id == "z-ai/glm-5.2":
        merge_thinking_level_map(model, {"xhigh": "xhigh"})
    if provider == "fireworks" and "glm-5p2" in model_id:
        merge_thinking_level_map(model, {"off": "none", "minimal": None, "low": "high", "medium": "high", "max": "max"})
    if provider == "opencode-go" and model_id == "glm-5.2":
        merge_thinking_level_map(model, OPENCODE_GO_GLM52_THINKING_LEVEL_MAP)
    if provider == "opencode-go" and model_id == "kimi-k2.6":
        # OpenCode Go exposes Kimi K2.6 thinking as on/off, not distinct effort tiers.
        merge_thinking_level_map(model, {"minimal": None, "low": None, "medium": None})
    if provider == "opencode" and model_id == "grok-build-0.1":
        # OpenCode Zen Grok Build reasons by default but rejects explicit reasoningEffort.
        merge_thinking_level_map(model, {"off": None, "minimal": None, "low": None, "medium": None})
    if provider == "ant-ling" and model["reasoning"]:
        # Ring reasons by default. Only high/xhigh have documented explicit effort controls.
        merge_thinking_level_map(model, ANT_LING_RING_THINKING_LEVEL_MAP)
    if provider == "github-copilot":
        override = GITHUB_COPILOT_THINKING_LEVEL_OVERRIDES.get(model_id)
        if override:
            merge_thinking_level_map(model, override)


def apply_strict_tool_compat_metadata(model: Json) -> None:
    if model["provider"] == "openai" and model["api"] == "openai-responses":
        merge_compat(model, {"supportsStrictMode": True})
    elif model["provider"] == "anthropic" and model["api"] == "anthropic-messages":
        merge_compat(model, {"supportsStrictTools": True})


def apply_openai_grammar_tool_compat_metadata(model: Json) -> None:
    if model["api"] not in OPENAI_GRAMMAR_TOOL_APIS or model["provider"] not in OPENAI_GRAMMAR_TOOL_PROVIDERS:
        return
    match = re.match(r"^gpt-(\d+)", model["id"])
    if not match or int(match.group(1)) < 5:
        return
    merge_compat(model, {"supportsOpenAIGrammarTools": True})


def apply_openai_tool_search_metadata(model: Json) -> None:
    is_openai_responses = model["provider"] == "openai" and model["api"] == "openai-responses"
    is_openai_codex = model["provider"] == "openai-codex" and model["api"] == "openai-codex-responses"
    if not (is_openai_responses or is_openai_codex) or model["id"] not in OPENAI_TOOL_SEARCH_MODEL_IDS:
        return
    supports_additional_tools = (is_openai_responses and model["id"] in OPENAI_ADDITIONAL_TOOLS_MODEL_IDS) or (
        is_openai_codex and model["id"] in OPENAI_CODEX_ADDITIONAL_TOOLS_MODEL_IDS
    )
    if supports_additional_tools:
        merge_compat(model, {"supportsAdditionalTools": True})
    merge_compat(model, {"supportsToolSearch": True})


def apply_openai_explicit_prompt_cache_metadata(model: Json) -> None:
    """OpenAI charges prompt-cache writes starting with the GPT-5.6 family."""
    if model["provider"] != "openai" or model["api"] != "openai-responses":
        return
    if not model["cost"]["cacheWrite"] > 0:
        return
    merge_compat(model, {"supportsExplicitPromptCacheMode": True})


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "pi-ai-generate-models/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned {response.status}")
        return json.loads(response.read().decode("utf-8"))


def fetch_nvidia_nim_model_ids() -> dict[str, str]:
    """Port of `fetchNvidiaNimModelIds`."""
    try:
        print("Fetching models from NVIDIA NIM API...")
        data = fetch_json(f"{NVIDIA_BASE_URL}/models")
        model_ids: dict[str, str] = {}
        items = data.get("data") or []
        for model in items:
            model_ids[model["id"]] = model["id"]
            model_ids[normalize_nvidia_model_id(model["id"])] = model["id"]
        print(f"Fetched {len(items)} model IDs from NVIDIA NIM")
        return model_ids
    except Exception as error:
        print(f"Failed to fetch NVIDIA NIM models: {error}", file=sys.stderr)
        if generator_options.strict:
            raise
        return {}


def fetch_openrouter_models() -> list[Json]:
    """Port of `fetchOpenRouterModels`."""
    try:
        print("Fetching models from OpenRouter API...")
        data = fetch_json("https://openrouter.ai/api/v1/models")
        models: list[Json] = []

        for model in data.get("data") or []:
            supported = model.get("supported_parameters") or []
            # Only include models that support tools
            if "tools" not in supported:
                continue

            architecture = model.get("architecture") or {}
            modality = architecture.get("modality") or ""
            model_input = ["text"]
            if "image" in modality:
                model_input.append("image")

            pricing = model.get("pricing") or {}
            # Convert pricing from $/token to $/million tokens
            input_cost = round_cost(float(pricing.get("prompt") or "0") * 1_000_000)
            output_cost = round_cost(float(pricing.get("completion") or "0") * 1_000_000)
            cache_read_cost = round_cost(float(pricing.get("input_cache_read") or "0") * 1_000_000)
            cache_write_cost = round_cost(float(pricing.get("input_cache_write") or "0") * 1_000_000)

            top_provider = model.get("top_provider") or {}
            context_window = top_provider.get("context_length") or model.get("context_length") or 4096

            models.append(
                make_model(
                    id=model["id"],
                    name=model.get("name") or model["id"],
                    api="openai-completions",
                    base_url="https://openrouter.ai/api/v1",
                    provider="openrouter",
                    reasoning="reasoning" in supported,
                    input=model_input,
                    cost={
                        "input": input_cost,
                        "output": output_cost,
                        "cacheRead": cache_read_cost,
                        "cacheWrite": cache_write_cost,
                    },
                    context_window=context_window,
                    max_tokens=top_provider.get("max_completion_tokens") or 4096,
                )
            )

        print(f"Fetched {len(models)} tool-capable models from OpenRouter")
        return models
    except Exception as error:
        print(f"Failed to fetch OpenRouter models: {error}", file=sys.stderr)
        if generator_options.strict:
            raise
        return []


def _to_number(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    try:
        return float(str(value if value is not None else "0"))
    except ValueError:
        return 0.0


def fetch_ai_gateway_models() -> list[Json]:
    """Port of `fetchAiGatewayModels`."""
    try:
        print("Fetching models from Vercel AI Gateway API...")
        data = fetch_json(f"{AI_GATEWAY_MODELS_URL}/models")
        models: list[Json] = []

        items = data.get("data") if isinstance(data.get("data"), list) else []
        for model in items:
            tags = model.get("tags") if isinstance(model.get("tags"), list) else []
            # Only include models that support tools
            if "tool-use" not in tags:
                continue

            model_input = ["text"]
            if "vision" in tags:
                model_input.append("image")

            pricing = model.get("pricing") or {}
            models.append(
                make_model(
                    id=model["id"],
                    name=model.get("name") or model["id"],
                    api="anthropic-messages",
                    base_url=AI_GATEWAY_BASE_URL,
                    provider="vercel-ai-gateway",
                    reasoning="reasoning" in tags,
                    input=model_input,
                    cost={
                        "input": round_cost(_to_number(pricing.get("input")) * 1_000_000),
                        "output": round_cost(_to_number(pricing.get("output")) * 1_000_000),
                        "cacheRead": round_cost(_to_number(pricing.get("input_cache_read")) * 1_000_000),
                        "cacheWrite": round_cost(_to_number(pricing.get("input_cache_write")) * 1_000_000),
                    },
                    context_window=model.get("context_window") or 4096,
                    max_tokens=model.get("max_tokens") or 4096,
                )
            )

        print(f"Fetched {len(models)} tool-capable models from Vercel AI Gateway")
        return models
    except Exception as error:
        print(f"Failed to fetch Vercel AI Gateway models: {error}", file=sys.stderr)
        if generator_options.strict:
            raise
        return []


# ---------------------------------------------------------------------------
# models.dev provider processing
# ---------------------------------------------------------------------------


def process_baseten_models(provider: Json | None) -> list[Json]:
    """Port of `processBasetenModels`."""
    if not provider or not provider.get("models"):
        return []

    base_url = "https://inference.baseten.co/v1"
    base_compat: Json = {
        "supportsStore": False,
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": False,
        "supportsUsageInStreaming": True,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": True,
        "supportsLongCacheRetention": False,
    }
    reasoning_effort_compat: Json = {
        **base_compat,
        "supportsReasoningEffort": True,
        "thinkingFormat": "openai",
    }
    toggle_reasoning_compat: Json = {
        **base_compat,
        "thinkingFormat": "baseten",
        "chatTemplateArgs": {"enable_thinking": {"$var": "thinking.enabled"}},
    }
    toggle_reasoning_effort_compat: Json = {
        **reasoning_effort_compat,
        "thinkingFormat": "baseten",
        "chatTemplateArgs": {"enable_thinking": {"$var": "thinking.enabled"}},
    }
    toggle_thinking_level_map: Json = {
        "off": "off",
        "minimal": None,
        "low": None,
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": None,
    }
    glm52_thinking_level_map: Json = {
        "off": "none",
        "minimal": None,
        "low": None,
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": "max",
    }
    models: list[Json] = []

    for model_id, model in provider["models"].items():
        if model.get("status") == "deprecated":
            continue

        reasoning = model.get("reasoning") is True
        reasoning_options = model.get("reasoning_options") or []
        is_glm52 = model_id in ("zai-org/GLM-5.2", "zai-org/GLM-5.2-Fast")
        supports_toggle = any(option.get("type") == "toggle" for option in reasoning_options) or is_glm52
        supports_effort = any(option.get("type") == "effort" for option in reasoning_options) or is_glm52
        if supports_toggle and supports_effort:
            compat = toggle_reasoning_effort_compat
        elif supports_toggle:
            compat = toggle_reasoning_compat
        elif supports_effort:
            compat = reasoning_effort_compat
        else:
            compat = base_compat
        if is_glm52:
            thinking_level_map: Json | None = glm52_thinking_level_map
        elif supports_toggle:
            thinking_level_map = toggle_thinking_level_map
        else:
            thinking_level_map = get_effort_thinking_level_map(reasoning_options)

        limit = model.get("limit") or {}
        models.append(
            make_model(
                id=model_id,
                name=model.get("name") or model_id,
                api="openai-completions",
                provider="baseten",
                base_url=base_url,
                reasoning=reasoning,
                thinking_level_map=thinking_level_map,
                input=input_modalities(model),
                cost=models_dev_simple_cost(model),
                compat=compat,
                context_window=limit.get("context") or 4096,
                max_tokens=limit.get("output") or 4096,
            )
        )

    return models


def process_fireworks_models(provider: Json | None) -> list[Json]:
    """Port of `processFireworksModels`."""
    if not provider or not provider.get("models"):
        return []

    anthropic_compat: Json = {
        "sendSessionAffinityHeaders": True,
        "supportsEagerToolInputStreaming": False,
        "supportsCacheControlOnTools": False,
        "supportsLongCacheRetention": False,
    }
    openai_compat: Json = {
        "supportsStore": False,
        "supportsDeveloperRole": False,
        "sendSessionAffinityHeaders": True,
        "supportsLongCacheRetention": False,
    }
    kimi_k3_compat: Json = {
        **openai_compat,
        "requiresReasoningContentOnAssistantMessages": True,
        "thinkingFormat": "openai",
        "deferredToolsMode": "kimi",
    }
    models: list[Json] = []

    for model_id, model in provider["models"].items():
        if model.get("tool_call") is not True:
            continue

        limit = model.get("limit") or {}
        common = {
            "id": model_id,
            "name": model.get("name") or model_id,
            "provider": "fireworks",
            "reasoning": model.get("reasoning") is True,
            "input": input_modalities(model),
            "cost": models_dev_simple_cost(model),
            "context_window": limit.get("context") or 4096,
            "max_tokens": limit.get("output") or 4096,
        }

        if "glm-5p2" in model_id:
            models.append(
                make_model(
                    **common,
                    api="openai-completions",
                    base_url="https://api.fireworks.ai/inference/v1",
                    compat=openai_compat,
                )
            )
        elif "kimi-k3" in model_id:
            models.append(
                make_model(
                    **common,
                    api="openai-completions",
                    base_url="https://api.fireworks.ai/inference/v1",
                    compat=kimi_k3_compat,
                )
            )
        else:
            # Fireworks Anthropic-compatible API - SDK appends /v1/messages.
            # Prompt caching uses automatic prefix matching + session affinity;
            # cache_control on tools and eager_input_streaming are unsupported.
            # https://docs.fireworks.ai/tools-sdks/anthropic-compatibility
            models.append(
                make_model(
                    **common,
                    api="anthropic-messages",
                    base_url="https://api.fireworks.ai/inference",
                    compat=anthropic_compat,
                )
            )
        record_models_dev_reasoning_options("fireworks", model_id, model)

    return models


def _load_models_dev_catalog() -> Json:
    if generator_options.models_dev_file is not None:
        print(f"Reading models.dev catalog from {generator_options.models_dev_file}...")
        with open(generator_options.models_dev_file, encoding="utf-8") as handle:
            return json.load(handle)
    print("Fetching models from models.dev API...")
    return fetch_json("https://models.dev/api.json")


def load_models_dev_data() -> list[Json]:
    """Port of `loadModelsDevData`."""
    try:
        data = _load_models_dev_catalog()

        models: list[Json] = []
        nvidia_nim_model_ids = fetch_nvidia_nim_model_ids() if (data.get("nvidia") or {}).get("models") else {}

        def provider_models(key: str) -> Json:
            return (data.get(key) or {}).get("models") or {}

        # Process Amazon Bedrock models
        for model_id, m in provider_models("amazon-bedrock").items():
            if m.get("tool_call") is not True:
                continue
            if model_id in BEDROCK_INFERENCE_PROFILE_ONLY_MODEL_IDS:
                continue
            # These models don't support tool use in streaming mode
            if model_id.startswith("ai21.jamba"):
                continue
            # These models don't support system messages
            if model_id.startswith("mistral.mistral-7b-instruct-v0"):
                continue

            limit = m.get("limit") or {}
            models.append(
                make_model(
                    id=model_id,
                    name=m.get("name") or model_id,
                    api="bedrock-converse-stream",
                    provider="amazon-bedrock",
                    base_url=get_bedrock_base_url(model_id),
                    reasoning=m.get("reasoning") is True,
                    input=input_modalities(m),
                    cost=models_dev_simple_cost(m),
                    context_window=limit.get("context") or 4096,
                    max_tokens=limit.get("output") or 4096,
                    compat={"supportsStrictMode": True} if m.get("structured_output") is True else None,
                )
            )
            record_models_dev_reasoning_options("amazon-bedrock", model_id, m)

        # Process Anthropic models
        for model_id, m in provider_models("anthropic").items():
            if m.get("tool_call") is not True:
                continue
            limit = m.get("limit") or {}
            models.append(
                make_model(
                    id=model_id,
                    name=m.get("name") or model_id,
                    api="anthropic-messages",
                    provider="anthropic",
                    base_url="https://api.anthropic.com",
                    reasoning=m.get("reasoning") is True,
                    input=input_modalities(m),
                    cost=models_dev_simple_cost(m),
                    context_window=limit.get("context") or 4096,
                    max_tokens=limit.get("output") or 4096,
                )
            )
            record_models_dev_reasoning_options("anthropic", model_id, m)

        # Process Google models
        google_models = provider_models("google")
        for model_id, m in google_models.items():
            if m.get("tool_call") is not True:
                continue
            source = m
            if model_id == "gemini-flash-latest":
                source = google_models.get("gemini-3.5-flash") or m
            if model_id == "gemini-flash-lite-latest":
                source = google_models.get("gemini-3.1-flash-lite") or m

            limit = source.get("limit") or {}
            models.append(
                make_model(
                    id=model_id,
                    name=m.get("name") or model_id,
                    api="google-generative-ai",
                    provider="google",
                    base_url="https://generativelanguage.googleapis.com/v1beta",
                    reasoning=source.get("reasoning") is True,
                    input=input_modalities(source),
                    cost=models_dev_simple_cost(source),
                    context_window=limit.get("context") or 4096,
                    max_tokens=limit.get("output") or 4096,
                )
            )
            record_models_dev_reasoning_options("google", model_id, source)

        # Process Google Vertex Gemini models. The google-vertex models.dev catalog also
        # includes Claude, OpenAI, and other MaaS models that do not use the Gemini
        # streaming path implemented by the google-vertex provider.
        vertex_models = provider_models("google-vertex")
        for model_id, m in vertex_models.items():
            if m.get("tool_call") is not True:
                continue
            if not model_id.startswith("gemini-"):
                continue
            if model_id == "gemini-3.1-flash-lite-preview":
                continue
            source = m
            if model_id == "gemini-flash-latest":
                source = vertex_models.get("gemini-3.5-flash") or m
            if model_id == "gemini-flash-lite-latest":
                source = vertex_models.get("gemini-3.1-flash-lite") or m

            # models.dev reports Vertex cache_read/cache_write values for Gemini 2.5
            # Flash that do not match the official Gemini API standard pricing table.
            # pi only accounts cachedContentTokenCount as cacheRead.
            source_cost = source.get("cost") or {}
            cache_read = 0.03 if model_id == "gemini-2.5-flash" else (source_cost.get("cache_read") or 0)
            limit = source.get("limit") or {}
            models.append(
                make_model(
                    id=model_id,
                    name=m.get("name") or model_id,
                    api="google-vertex",
                    provider="google-vertex",
                    base_url=VERTEX_BASE_URL,
                    reasoning=source.get("reasoning") is True,
                    input=input_modalities(source),
                    cost={
                        "input": source_cost.get("input") or 0,
                        "output": source_cost.get("output") or 0,
                        "cacheRead": cache_read,
                        "cacheWrite": 0,
                    },
                    context_window=limit.get("context") or 4096,
                    max_tokens=limit.get("output") or 4096,
                )
            )
            record_models_dev_reasoning_options("google-vertex", model_id, source)

        # Process OpenAI models
        for model_id, m in provider_models("openai").items():
            if m.get("tool_call") is not True:
                continue
            # models.dev lists this alias, but it is not accepted by OpenAI APIs.
            if model_id in MODELS_DEV_OPENAI_UNSUPPORTED_MODEL_IDS:
                continue
            limit = m.get("limit") or {}
            models.append(
                make_model(
                    id=model_id,
                    name=m.get("name") or model_id,
                    api="openai-responses",
                    provider="openai",
                    base_url="https://api.openai.com/v1",
                    reasoning=m.get("reasoning") is True,
                    input=input_modalities(m),
                    cost=models_dev_simple_cost(m),
                    context_window=limit.get("context") or 4096,
                    max_tokens=limit.get("output") or 4096,
                )
            )
            record_models_dev_reasoning_options("openai", model_id, m)

        # Process Groq / Cerebras models
        for key, provider_id, base_url in (
            ("groq", "groq", "https://api.groq.com/openai/v1"),
            ("cerebras", "cerebras", "https://api.cerebras.ai/v1"),
        ):
            for model_id, m in provider_models(key).items():
                if m.get("tool_call") is not True:
                    continue
                limit = m.get("limit") or {}
                models.append(
                    make_model(
                        id=model_id,
                        name=m.get("name") or model_id,
                        api="openai-completions",
                        provider=provider_id,
                        base_url=base_url,
                        reasoning=m.get("reasoning") is True,
                        input=input_modalities(m),
                        cost=models_dev_simple_cost(m),
                        context_window=limit.get("context") or 4096,
                        max_tokens=limit.get("output") or 4096,
                    )
                )
                record_models_dev_reasoning_options(provider_id, model_id, m)

        # Process Cloudflare Workers AI models
        for model_id, m in provider_models("cloudflare-workers-ai").items():
            if m.get("tool_call") is not True:
                continue
            limit = m.get("limit") or {}
            models.append(
                make_model(
                    id=model_id,
                    name=m.get("name") or model_id,
                    api="openai-completions",
                    provider="cloudflare-workers-ai",
                    base_url=CLOUDFLARE_WORKERS_AI_BASE_URL,
                    reasoning=m.get("reasoning") is True,
                    input=input_modalities(m),
                    cost=models_dev_simple_cost(m),
                    context_window=limit.get("context") or 4096,
                    max_tokens=limit.get("output") or 4096,
                    compat={"sendSessionAffinityHeaders": True},
                )
            )
            record_models_dev_reasoning_options("cloudflare-workers-ai", model_id, m)

        # Process Cloudflare AI Gateway models
        for prefixed_id, m in provider_models("cloudflare-ai-gateway").items():
            if m.get("tool_call") is not True:
                continue
            slash_index = prefixed_id.find("/")
            if slash_index == -1:
                continue
            upstream = prefixed_id[:slash_index]
            native_id = prefixed_id[slash_index + 1 :]

            if upstream == "openai":
                api = "openai-responses"
                base_url = CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL
                model_id = native_id
            elif upstream == "anthropic":
                api = "anthropic-messages"
                base_url = CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL
                model_id = native_id
            elif upstream == "workers-ai":
                api = "openai-completions"
                base_url = CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL
                model_id = prefixed_id
            else:
                continue

            # Gateway passthroughs forward session affinity headers to upstreams
            # that use them for cache/routing affinity.
            compat = {"sendSessionAffinityHeaders": True} if upstream in ("anthropic", "workers-ai") else None
            limit = m.get("limit") or {}
            models.append(
                make_model(
                    id=model_id,
                    name=m.get("name") or model_id,
                    api=api,
                    provider="cloudflare-ai-gateway",
                    base_url=base_url,
                    reasoning=m.get("reasoning") is True,
                    input=input_modalities(m),
                    cost=models_dev_simple_cost(m),
                    context_window=limit.get("context") or 4096,
                    max_tokens=limit.get("output") or 4096,
                    compat=compat,
                )
            )
            record_models_dev_reasoning_options("cloudflare-ai-gateway", model_id, m)

        # Process xAI models
        for model_id, m in provider_models("xai").items():
            if m.get("tool_call") is not True:
                continue
            use_responses_api = model_id == XAI_RESPONSES_MODEL_ID
            limit = m.get("limit") or {}
            models.append(
                make_model(
                    id=model_id,
                    name=m.get("name") or model_id,
                    api="openai-responses" if use_responses_api else "openai-completions",
                    provider="xai",
                    base_url="https://api.x.ai/v1",
                    compat=dict(XAI_RESPONSES_COMPAT) if use_responses_api else None,
                    reasoning=m.get("reasoning") is True,
                    input=input_modalities(m),
                    cost=models_dev_simple_cost(m),
                    context_window=limit.get("context") or 4096,
                    max_tokens=limit.get("output") or 4096,
                )
            )
            record_models_dev_reasoning_options("xai", model_id, m)

        # Process zAI coding-plan models
        zai_coding_plan_variants = (
            ("zai", "https://api.z.ai/api/coding/paas/v4"),
            ("zai-coding-cn", "https://open.bigmodel.cn/api/coding/paas/v4"),
        )
        for provider_id, base_url in zai_coding_plan_variants:
            for model_id, m in provider_models("zai-coding-plan").items():
                if m.get("tool_call") is not True:
                    continue
                is_glm52 = model_id == "glm-5.2"
                compat: Json = {"supportsDeveloperRole": False, "thinkingFormat": "zai"}
                if is_glm52:
                    compat["supportsReasoningEffort"] = True
                if model_id not in ZAI_TOOL_STREAM_UNSUPPORTED_MODELS:
                    compat["zaiToolStream"] = True
                limit = m.get("limit") or {}
                models.append(
                    make_model(
                        id=model_id,
                        name=m.get("name") or model_id,
                        api="openai-completions",
                        provider=provider_id,
                        base_url=base_url,
                        reasoning=m.get("reasoning") is True,
                        thinking_level_map=ZAI_GLM52_THINKING_LEVEL_MAP if is_glm52 else None,
                        input=input_modalities(m),
                        cost=models_dev_simple_cost(m),
                        compat=compat,
                        context_window=limit.get("context") or 4096,
                        max_tokens=limit.get("output") or 4096,
                    )
                )
                record_models_dev_reasoning_options(provider_id, model_id, m)

        # Process Mistral models
        for model_id, m in provider_models("mistral").items():
            if m.get("tool_call") is not True:
                continue
            cost = m.get("cost") or {}
            cache_read = cost.get("cache_read")
            if cache_read is None:
                cache_read = round_cost(cost["input"] * 0.1) if cost.get("input") else 0
            limit = m.get("limit") or {}
            models.append(
                make_model(
                    id=model_id,
                    name=m.get("name") or model_id,
                    api="mistral-conversations",
                    provider="mistral",
                    base_url="https://api.mistral.ai",
                    reasoning=m.get("reasoning") is True,
                    input=input_modalities(m),
                    cost={
                        "input": cost.get("input") or 0,
                        "output": cost.get("output") or 0,
                        "cacheRead": cache_read,
                        "cacheWrite": cost.get("cache_write") or 0,
                    },
                    context_window=limit.get("context") or 4096,
                    max_tokens=limit.get("output") or 4096,
                )
            )
            record_models_dev_reasoning_options("mistral", model_id, m)

        # Process Hugging Face models
        for model_id, m in provider_models("huggingface").items():
            if m.get("tool_call") is not True:
                continue
            limit = m.get("limit") or {}
            models.append(
                make_model(
                    id=model_id,
                    name=m.get("name") or model_id,
                    api="openai-completions",
                    provider="huggingface",
                    base_url="https://router.huggingface.co/v1",
                    reasoning=m.get("reasoning") is True,
                    input=input_modalities(m),
                    cost=models_dev_simple_cost(m),
                    compat={"supportsDeveloperRole": False},
                    context_window=limit.get("context") or 4096,
                    max_tokens=limit.get("output") or 4096,
                )
            )
            record_models_dev_reasoning_options("huggingface", model_id, m)

        models.extend(process_fireworks_models(data.get("fireworks-ai")))

        # Process NVIDIA NIM models
        for model_id, m in provider_models("nvidia").items():
            if m.get("tool_call") is not True:
                continue
            modalities = m.get("modalities") or {}
            if "text" not in (modalities.get("input") or []):
                continue
            if "text" not in (modalities.get("output") or []):
                continue

            live_model_id = nvidia_nim_model_ids.get(model_id) or nvidia_nim_model_ids.get(
                normalize_nvidia_model_id(model_id)
            )
            if not live_model_id:
                continue
            if live_model_id in NVIDIA_NIM_UNSUPPORTED_MODELS:
                continue

            limit = m.get("limit") or {}
            models.append(
                make_model(
                    id=live_model_id,
                    name=m.get("name") or live_model_id,
                    api="openai-completions",
                    provider="nvidia",
                    base_url=NVIDIA_BASE_URL,
                    headers=dict(NVIDIA_HEADERS),
                    reasoning=m.get("reasoning") is True,
                    input=input_modalities(m),
                    cost=models_dev_simple_cost(m),
                    compat=NVIDIA_OPENAI_COMPAT,
                    context_window=limit.get("context") or 4096,
                    max_tokens=limit.get("output") or 4096,
                )
            )
            record_models_dev_reasoning_options("nvidia", live_model_id, m)

        # Process Together AI models
        together_provider = data.get("together") or data.get("togetherai") or data.get("together-ai")
        for model_id, m in ((together_provider or {}).get("models") or {}).items():
            if m.get("tool_call") is not True:
                continue
            if m.get("status") == "deprecated":
                continue

            reasoning = m.get("reasoning") is True
            limit = m.get("limit") or {}
            models.append(
                make_model(
                    id=model_id,
                    name=m.get("name") or model_id,
                    api="openai-completions",
                    provider="together",
                    base_url=TOGETHER_BASE_URL,
                    reasoning=reasoning,
                    thinking_level_map=get_together_thinking_level_map(model_id, reasoning),
                    input=input_modalities(m),
                    cost=models_dev_simple_cost(m),
                    compat=get_together_compat(model_id, reasoning),
                    context_window=limit.get("context") or 4096,
                    max_tokens=limit.get("output") or 4096,
                )
            )
            record_models_dev_reasoning_options("together", model_id, m)

        models.extend(process_baseten_models(data.get("baseten")))

        # Process OpenCode models (Zen and Go). API mapping is based on the
        # models.dev provider.npm field:
        # - @ai-sdk/openai -> openai-responses
        # - @ai-sdk/anthropic -> anthropic-messages
        # - @ai-sdk/google -> google-generative-ai
        # - null/undefined/@ai-sdk/openai-compatible -> openai-completions
        opencode_variants = (
            ("opencode", "opencode", "https://opencode.ai/zen"),
            ("opencode-go", "opencode-go", "https://opencode.ai/zen/go"),
        )
        for key, provider_id, base_path in opencode_variants:
            for model_id, m in provider_models(key).items():
                if m.get("tool_call") is not True:
                    continue
                if m.get("status") == "deprecated":
                    continue

                npm = (m.get("provider") or {}).get("npm")
                compat = None
                if npm == "@ai-sdk/openai":
                    api = "openai-responses"
                    base_url = f"{base_path}/v1"
                    compat = {"sessionAffinityFormat": "openai-nosession"}
                elif npm == "@ai-sdk/anthropic":
                    api = "anthropic-messages"
                    # Anthropic SDK appends /v1/messages to baseURL
                    base_url = base_path
                elif npm == "@ai-sdk/google":
                    api = "google-generative-ai"
                    base_url = f"{base_path}/v1"
                elif npm == "@ai-sdk/alibaba":
                    api = "openai-completions"
                    base_url = f"{base_path}/v1"
                    compat = {"cacheControlFormat": "anthropic"}
                else:
                    api = "openai-completions"
                    base_url = f"{base_path}/v1"

                if provider_id == "opencode" and model_id == "grok-build-0.1":
                    compat = {**(compat or {}), "supportsReasoningEffort": False}

                if model_id == "kimi-k2.6":
                    # OpenCode Kimi K2.6 accepts Anthropic-style thinking objects
                    # and rejects string thinking values or combined reasoning_effort.
                    compat = {**(compat or {}), "thinkingFormat": "deepseek", "supportsReasoningEffort": False}

                # Fix known mismatches between models.dev npm data and actual OpenCode
                # Go endpoint behaviour: the Go endpoints either don't accept Anthropic
                # SDK auth (MiniMax M2.7) or are served through the OpenAI-compatible
                # /v1/chat/completions path (Qwen 3.5/3.6).
                if provider_id == "opencode-go":
                    if model_id == "minimax-m2.7":
                        api = "openai-completions"
                        base_url = f"{base_path}/v1"
                    if model_id in ("qwen3.5-plus", "qwen3.6-plus"):
                        api = "openai-completions"
                        base_url = f"{base_path}/v1"
                        # Qwen/DashScope uses enable_thinking at the top level.
                        compat = {**(compat or {}), "thinkingFormat": "qwen"}

                if api == "openai-completions":
                    compat = {**(compat or {}), "maxTokensField": "max_tokens"}
                    if (
                        f"{provider_id}:{model_id}"
                        in OPENCODE_OPENAI_COMPLETIONS_LONG_CACHE_RETENTION_UNSUPPORTED_MODELS
                    ):
                        compat = {**compat, "supportsLongCacheRetention": False}

                limit = m.get("limit") or {}
                models.append(
                    make_model(
                        id=model_id,
                        name=m.get("name") or model_id,
                        api=api,
                        provider=provider_id,
                        base_url=base_url,
                        reasoning=m.get("reasoning") is True,
                        input=input_modalities(m),
                        cost=models_dev_simple_cost(m),
                        compat=compat,
                        context_window=limit.get("context") or 4096,
                        max_tokens=limit.get("output") or 4096,
                    )
                )
                record_models_dev_reasoning_options(provider_id, model_id, m)

        # Process GitHub Copilot models
        for model_id, m in provider_models("github-copilot").items():
            if m.get("tool_call") is not True:
                continue
            if m.get("status") == "deprecated":
                continue

            # Claude 4.x and 5.x models route to the Anthropic Messages API.
            is_copilot_claude = re.match(r"^claude-(haiku|sonnet|opus)-[45]([.\-]|$)", model_id) is not None
            # Grok 4.5, gpt-5, oswe, and MAI-Code models are only served through
            # the Copilot /responses endpoint.
            needs_responses_api = (
                model_id == "grok-4.5"
                or model_id.startswith("gpt-5")
                or model_id.startswith("oswe")
                or model_id.startswith("mai-")
            )
            if is_copilot_claude:
                api = "anthropic-messages"
            elif needs_responses_api:
                api = "openai-responses"
            else:
                api = "openai-completions"

            compat = get_anthropic_messages_compat("github-copilot", model_id) if api == "anthropic-messages" else None
            # compat only applies to openai-completions
            if api == "openai-completions":
                compat = {
                    "supportsStore": False,
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                }

            limit = m.get("limit") or {}
            models.append(
                make_model(
                    id=model_id,
                    name=m.get("name") or model_id,
                    api=api,
                    provider="github-copilot",
                    base_url="https://api.individual.githubcopilot.com",
                    reasoning=m.get("reasoning") is True,
                    input=input_modalities(m),
                    cost=get_models_dev_cost(m.get("cost")),
                    context_window=limit.get("context") or 128000,
                    max_tokens=limit.get("output") or 8192,
                    headers=dict(COPILOT_STATIC_HEADERS),
                    compat=compat,
                )
            )
            record_models_dev_reasoning_options("github-copilot", model_id, m)

        # Process MiniMax models
        minimax_variants = (
            ("minimax", "minimax", "https://api.minimax.io/anthropic"),
            ("minimax-cn", "minimax-cn", "https://api.minimaxi.com/anthropic"),
        )
        for key, provider_id, base_url in minimax_variants:
            for model_id, m in provider_models(key).items():
                if m.get("tool_call") is not True:
                    continue
                limit = m.get("limit") or {}
                models.append(
                    make_model(
                        id=model_id,
                        name=m.get("name") or model_id,
                        api="anthropic-messages",
                        provider=provider_id,
                        # MiniMax's Anthropic-compatible API - SDK appends /v1/messages
                        base_url=base_url,
                        reasoning=m.get("reasoning") is True,
                        input=input_modalities(m),
                        cost=models_dev_simple_cost(m),
                        context_window=limit.get("context") or 4096,
                        max_tokens=limit.get("output") or 4096,
                    )
                )
                record_models_dev_reasoning_options(provider_id, model_id, m)

        # Process Kimi For Coding models
        kimi_models = provider_models("kimi-for-coding")
        has_canonical_model = "kimi-for-coding" in kimi_models
        kimi_aliases = {"k2p5", "k2p6", "k2p7"}
        for model_id, m in kimi_models.items():
            if m.get("tool_call") is not True:
                continue
            # models.dev may expose versioned aliases (e.g. k2p5/k2p6/k2p7).
            # Normalize aliases to the canonical id and drop duplicates.
            if model_id in kimi_aliases and has_canonical_model:
                continue

            normalized_id = "kimi-for-coding" if model_id in kimi_aliases else model_id
            normalized_name = "Kimi For Coding" if model_id in kimi_aliases else (m.get("name") or normalized_id)
            is_kimi_k3 = normalized_id == "k3"
            allow_empty_signature = is_kimi_k3 or normalized_id == "kimi-for-coding"
            implied_cost = KIMI_CODING_IMPLIED_COSTS.get(normalized_id) or {}

            compat = {}
            if allow_empty_signature:
                compat["allowEmptySignature"] = True
            compat["forceAdaptiveThinking"] = True

            cost = m.get("cost") or {}
            limit = m.get("limit") or {}
            models.append(
                make_model(
                    id=normalized_id,
                    name=normalized_name,
                    api="anthropic-messages",
                    provider="kimi-coding",
                    # Kimi For Coding's Anthropic-compatible API - SDK appends /v1/messages
                    base_url="https://api.kimi.com/coding",
                    headers=dict(KIMI_STATIC_HEADERS),
                    compat=compat,
                    reasoning=is_kimi_k3 or m.get("reasoning") is True,
                    input=input_modalities(m),
                    cost={
                        "input": cost.get("input") or implied_cost.get("input") or 0,
                        "output": cost.get("output") or implied_cost.get("output") or 0,
                        "cacheRead": cost.get("cache_read") or implied_cost.get("cacheRead") or 0,
                        "cacheWrite": cost.get("cache_write") or implied_cost.get("cacheWrite") or 0,
                    },
                    context_window=limit.get("context") or 4096,
                    max_tokens=limit.get("output") or 4096,
                )
            )
            record_models_dev_reasoning_options("kimi-coding", normalized_id, m)

        # Process Moonshot AI models
        moonshot_variants = (
            ("moonshotai", "moonshotai", "https://api.moonshot.ai/v1"),
            ("moonshotai-cn", "moonshotai-cn", "https://api.moonshot.cn/v1"),
        )
        moonshot_compat: Json = {
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "maxTokensField": "max_tokens",
            "supportsStrictMode": False,
            "thinkingFormat": "deepseek",
        }
        for key, provider_id, base_url in moonshot_variants:
            for model_id, m in provider_models(key).items():
                if m.get("tool_call") is not True:
                    continue

                is_kimi_k3 = model_id == "kimi-k3"
                compat = dict(moonshot_compat)
                if is_kimi_k3:
                    compat["requiresReasoningContentOnAssistantMessages"] = True
                    compat["deferredToolsMode"] = "kimi"
                    compat["thinkingFormat"] = "openai"
                    compat["supportsReasoningEffort"] = True

                cost = m.get("cost") or {}
                limit = m.get("limit") or {}
                models.append(
                    make_model(
                        id=model_id,
                        name=m.get("name") or model_id,
                        api="openai-completions",
                        provider=provider_id,
                        base_url=base_url,
                        reasoning=is_kimi_k3 or m.get("reasoning") is True,
                        input=input_modalities(m),
                        cost={
                            "input": cost.get("input") or (KIMI_K3_COST["input"] if is_kimi_k3 else 0),
                            "output": cost.get("output") or (KIMI_K3_COST["output"] if is_kimi_k3 else 0),
                            "cacheRead": cost.get("cache_read") or (KIMI_K3_COST["cacheRead"] if is_kimi_k3 else 0),
                            "cacheWrite": cost.get("cache_write") or (KIMI_K3_COST["cacheWrite"] if is_kimi_k3 else 0),
                        },
                        context_window=limit.get("context") or 4096,
                        max_tokens=limit.get("output") or 4096,
                        compat=compat,
                    )
                )
                record_models_dev_reasoning_options(provider_id, model_id, m)

        # Process Xiaomi MiMo models. Built-in `xiaomi` targets the API billing
        # endpoint (single stable URL, keys from platform.xiaomimimo.com). The three
        # `xiaomi-token-plan-*` providers cover prepaid Token Plan endpoints.
        xiaomi_compat: Json = {
            "requiresReasoningContentOnAssistantMessages": True,
            "thinkingFormat": "deepseek",
        }
        xiaomi_variants = (
            ("xiaomi", "xiaomi", "https://api.xiaomimimo.com/v1"),
            ("xiaomi-token-plan-cn", "xiaomi-token-plan-cn", "https://token-plan-cn.xiaomimimo.com/v1"),
            ("xiaomi-token-plan-ams", "xiaomi-token-plan-ams", "https://token-plan-ams.xiaomimimo.com/v1"),
            ("xiaomi-token-plan-sgp", "xiaomi-token-plan-sgp", "https://token-plan-sgp.xiaomimimo.com/v1"),
        )
        for source_key, provider_id, base_url in xiaomi_variants:
            for model_id, m in provider_models(source_key).items():
                if m.get("tool_call") is not True:
                    continue
                limit = m.get("limit") or {}
                models.append(
                    make_model(
                        id=model_id,
                        name=m.get("name") or model_id,
                        api="openai-completions",
                        provider=provider_id,
                        base_url=base_url,
                        compat=xiaomi_compat,
                        reasoning=m.get("reasoning") is True,
                        input=input_modalities(m),
                        cost=models_dev_simple_cost(m),
                        context_window=limit.get("context") or 4096,
                        max_tokens=limit.get("output") or 4096,
                    )
                )
                record_models_dev_reasoning_options(provider_id, model_id, m)

        # Process Alibaba Cloud Model Studio Token Plan models. International and
        # China use separate endpoints and API keys (sk-sp- prefix). The Individual
        # provider reuses the international source and endpoint with a narrower
        # catalog. models.dev keys are "alibaba-token-plan[-cn]"; pi exposes them as
        # "qwen-token-plan[-cn]" plus the Individual catalog view.
        qwen_token_plan_compat: Json = {
            "thinkingFormat": "qwen",
            "supportsDeveloperRole": False,
            "supportsStore": False,
            "supportsReasoningEffort": True,
        }
        qwen_token_plan_variants = (
            (
                "alibaba-token-plan",
                "qwen-token-plan",
                "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
                None,
            ),
            (
                "alibaba-token-plan",
                "qwen-token-plan-individual",
                "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
                QWEN_TOKEN_PLAN_INDIVIDUAL_MODEL_IDS,
            ),
            (
                "alibaba-token-plan-cn",
                "qwen-token-plan-cn",
                "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                None,
            ),
        )
        for source_key, provider_id, base_url, model_ids in qwen_token_plan_variants:
            emitted_model_ids: set[str] | None = set() if model_ids else None

            for model_id, m in provider_models(source_key).items():
                if m.get("tool_call") is not True:
                    continue
                if model_id in QWEN_TOKEN_PLAN_EXCLUDED_MODEL_IDS:
                    continue
                if model_ids and model_id not in model_ids:
                    continue
                supports_reasoning_effort = model_id not in QWEN_TOKEN_PLAN_REASONING_EFFORT_UNSUPPORTED_MODEL_IDS

                thinking_level_map = None
                if supports_reasoning_effort:
                    thinking_level_map = (
                        QWEN_TOKEN_PLAN_QWEN38_THINKING_LEVEL_MAP
                        if model_id == "qwen3.8-max"
                        else QWEN_TOKEN_PLAN_HIGH_MAX_THINKING_LEVEL_MAP
                    )

                limit = m.get("limit") or {}
                models.append(
                    make_model(
                        id=model_id,
                        name=m.get("name") or model_id,
                        api="openai-completions",
                        provider=provider_id,
                        base_url=base_url,
                        compat=qwen_token_plan_compat
                        if supports_reasoning_effort
                        else {**qwen_token_plan_compat, "supportsReasoningEffort": False},
                        thinking_level_map=thinking_level_map,
                        reasoning=m.get("reasoning") is True,
                        input=input_modalities(m),
                        cost=models_dev_simple_cost(m),
                        context_window=limit.get("context") or 4096,
                        max_tokens=limit.get("output") or 4096,
                    )
                )
                if emitted_model_ids is not None:
                    emitted_model_ids.add(model_id)
                record_models_dev_reasoning_options(provider_id, model_id, m)

            if model_ids and emitted_model_ids is not None and generator_options.strict:
                assert_exact_model_ids(provider_id, model_ids, emitted_model_ids)

        print(f"Loaded {len(models)} tool-capable models from models.dev")
        return models
    except Exception as error:
        print(f"Failed to load models.dev data: {error}", file=sys.stderr)
        if generator_options.strict:
            raise
        return []


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _normalize_numbers(value: Any) -> Any:
    """Serialize integral floats as integers, matching `JSON.stringify` output."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _normalize_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_numbers(item) for item in value]
    return value


def serialize_json(value: Any) -> str:
    normalized = _normalize_numbers(value)
    if generator_options.pretty:
        return json.dumps(normalized, ensure_ascii=False, indent=2) + "\n"
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.write_text(serialize_json(value), encoding="utf-8")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _apply_temporary_overrides(all_models: list[Json]) -> None:
    """Temporary overrides until upstream model metadata is corrected."""
    for candidate in all_models:
        provider = candidate["provider"]
        model_id = candidate["id"]

        if provider == "github-copilot" and model_id in GITHUB_COPILOT_EXTENDED_CONTEXT_MODELS:
            candidate["contextWindow"] = 1000000

        if provider in ("anthropic", "opencode", "opencode-go") and model_id in (
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-4.6",
            "claude-sonnet-4.6",
        ):
            candidate["contextWindow"] = 1000000

        # OpenCode variants list Claude Sonnet 4/4.5 with 1M context, actual limit is 200K
        if provider in ("opencode", "opencode-go") and model_id in ("claude-sonnet-4-5", "claude-sonnet-4"):
            candidate["contextWindow"] = 200000
        if provider in ("opencode", "opencode-go") and model_id == "gpt-5.4":
            candidate["contextWindow"] = 272000
            candidate["maxTokens"] = 128000

        # Keep direct OpenAI requests in the short-context pricing tier by default.
        # Users can opt into the larger context through model overrides, so retain
        # long-context cost metadata on the capped models.
        if provider == "openai" and model_id in OPENAI_SHORT_CONTEXT_CAPPED_MODEL_IDS:
            candidate["contextWindow"] = OPENAI_LONG_CONTEXT_INPUT_THRESHOLD
            candidate["maxTokens"] = 128000
        if provider == "openai" and model_id in OPENAI_LONG_CONTEXT_PRICING_MODEL_IDS:
            standard_cost = OPENAI_GPT_56_STANDARD_COSTS.get(model_id)
            candidate["cost"] = with_openai_long_context_pricing(standard_cost or candidate["cost"])
        # Cloudflare AI Gateway passes OpenAI usage through at OpenAI list prices.
        if provider == "cloudflare-ai-gateway":
            standard_cost = OPENAI_GPT_56_STANDARD_COSTS.get(model_id)
            if standard_cost:
                candidate["cost"] = with_openai_long_context_pricing(standard_cost)
        # models.dev reports gpt-5-pro output as 272000 (a duplicate of the input
        # sub-limit); the actual max output is 128000.
        if provider == "openai" and model_id == "gpt-5-pro":
            candidate["maxTokens"] = 128000
        # Keep Kimi K3's canonical output limit when gateway metadata is missing.
        if (provider == "openrouter" and model_id in OPENROUTER_KIMI_K3_MODEL_IDS) or (
            provider == "vercel-ai-gateway" and model_id == "moonshotai/kimi-k3"
        ):
            candidate["maxTokens"] = KIMI_K3_MAX_TOKENS
        # Keep selected OpenRouter model metadata stable until upstream settles.
        if provider == "openrouter" and model_id == "moonshotai/kimi-k2.5":
            candidate["cost"]["input"] = 0.41
            candidate["cost"]["output"] = 2.06
            candidate["cost"]["cacheRead"] = 0.07
            candidate["maxTokens"] = 4096
        if provider == "openrouter" and model_id.startswith("moonshotai/kimi-k2.6"):
            merge_compat(
                candidate,
                {"supportsDeveloperRole": False, "requiresReasoningContentOnAssistantMessages": True},
            )
        if provider == "openrouter" and model_id == "z-ai/glm-5":
            candidate["cost"]["input"] = 0.6
            candidate["cost"]["output"] = 1.9
            candidate["cost"]["cacheRead"] = 0.119


def _missing_openai_models() -> list[Json]:
    return [
        make_model(
            id="gpt-5.6-sol",
            name="GPT-5.6 Sol",
            api="openai-responses",
            base_url="https://api.openai.com/v1",
            provider="openai",
            reasoning=True,
            input=["text", "image"],
            cost=with_openai_long_context_pricing({"input": 5, "output": 30, "cacheRead": 0.5, "cacheWrite": 6.25}),
            context_window=OPENAI_LONG_CONTEXT_INPUT_THRESHOLD,
            max_tokens=128000,
        ),
        make_model(
            id="gpt-5.6-terra",
            name="GPT-5.6 Terra",
            api="openai-responses",
            base_url="https://api.openai.com/v1",
            provider="openai",
            reasoning=True,
            input=["text", "image"],
            cost=with_openai_long_context_pricing(OPENAI_GPT_56_STANDARD_COSTS["gpt-5.6-terra"]),
            context_window=OPENAI_LONG_CONTEXT_INPUT_THRESHOLD,
            max_tokens=128000,
        ),
        make_model(
            id="gpt-5.6-luna",
            name="GPT-5.6 Luna",
            api="openai-responses",
            base_url="https://api.openai.com/v1",
            provider="openai",
            reasoning=True,
            input=["text", "image"],
            cost=with_openai_long_context_pricing(OPENAI_GPT_56_STANDARD_COSTS["gpt-5.6-luna"]),
            context_window=OPENAI_LONG_CONTEXT_INPUT_THRESHOLD,
            max_tokens=128000,
        ),
        make_model(
            id="gpt-5-chat-latest",
            name="GPT-5 Chat Latest",
            api="openai-responses",
            base_url="https://api.openai.com/v1",
            provider="openai",
            reasoning=False,
            input=["text", "image"],
            cost={"input": 1.25, "output": 10, "cacheRead": 0.125, "cacheWrite": 0},
            context_window=128000,
            max_tokens=16384,
        ),
    ]


DEEPSEEK_COMPAT: Json = {
    "requiresReasoningContentOnAssistantMessages": True,
    "thinkingFormat": "deepseek",
}


def _deepseek_v4_models() -> list[Json]:
    return [
        make_model(
            id="deepseek-v4-flash",
            name="DeepSeek V4 Flash",
            api="openai-completions",
            base_url="https://api.deepseek.com",
            provider="deepseek",
            reasoning=True,
            input=["text"],
            cost={"input": 0.14, "output": 0.28, "cacheRead": 0.0028, "cacheWrite": 0},
            context_window=1000000,
            max_tokens=384000,
            compat=DEEPSEEK_COMPAT,
        ),
        make_model(
            id="deepseek-v4-pro",
            name="DeepSeek V4 Pro",
            api="openai-completions",
            base_url="https://api.deepseek.com",
            provider="deepseek",
            reasoning=True,
            input=["text"],
            cost={"input": 0.435, "output": 0.87, "cacheRead": 0.003625, "cacheWrite": 0},
            context_window=1000000,
            max_tokens=384000,
            compat=DEEPSEEK_COMPAT,
        ),
    ]


def _ant_ling_models() -> list[Json]:
    ant_ling_compat: Json = {
        "supportsStore": False,
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": False,
        "maxTokensField": "max_tokens",
        "supportsLongCacheRetention": False,
    }
    return [
        make_model(
            id="Ling-2.6-flash",
            name="Ling 2.6 Flash",
            api="openai-completions",
            base_url="https://api.ant-ling.com/v1",
            provider="ant-ling",
            reasoning=False,
            input=["text"],
            cost={"input": 0.01, "output": 0.02, "cacheRead": 0, "cacheWrite": 0},
            context_window=262144,
            max_tokens=65536,
            compat=ant_ling_compat,
        ),
        make_model(
            id="Ling-2.6-1T",
            name="Ling 2.6 1T",
            api="openai-completions",
            base_url="https://api.ant-ling.com/v1",
            provider="ant-ling",
            reasoning=False,
            input=["text"],
            cost={"input": 0.06, "output": 0.25, "cacheRead": 0, "cacheWrite": 0},
            context_window=262144,
            max_tokens=65536,
            compat=ant_ling_compat,
        ),
        make_model(
            id="Ring-2.6-1T",
            name="Ring 2.6 1T",
            api="openai-completions",
            base_url="https://api.ant-ling.com/v1",
            provider="ant-ling",
            reasoning=True,
            input=["text"],
            cost={"input": 0.06, "output": 0.25, "cacheRead": 0, "cacheWrite": 0},
            context_window=262144,
            max_tokens=65536,
            compat={**ant_ling_compat, "thinkingFormat": "ant-ling"},
        ),
    ]


# OpenAI Codex (ChatGPT OAuth) models. These are not fetched from models.dev; a
# small explicit list avoids aliases. Older model limits are based on observed
# server behavior; GPT-5.6 follows Codex's 272k catalog limit (formerly 372k).
CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_CONTEXT = 272000
CODEX_GPT_56_CONTEXT = 272000
CODEX_SPARK_CONTEXT = 128000
CODEX_MAX_TOKENS = 128000


def _codex_models() -> list[Json]:
    def codex(model_id: str, name: str, cost: Json, context: int, model_input: list[str]) -> Json:
        return make_model(
            id=model_id,
            name=name,
            api="openai-codex-responses",
            provider="openai-codex",
            base_url=CODEX_BASE_URL,
            reasoning=True,
            input=model_input,
            cost=cost,
            context_window=context,
            max_tokens=CODEX_MAX_TOKENS,
        )

    return [
        codex(
            "gpt-5.3-codex-spark",
            "GPT-5.3 Codex Spark",
            {"input": 1.75, "output": 14, "cacheRead": 0.175, "cacheWrite": 0},
            CODEX_SPARK_CONTEXT,
            ["text"],
        ),
        codex(
            "gpt-5.4",
            "GPT-5.4",
            with_openai_long_context_pricing({"input": 2.5, "output": 15, "cacheRead": 0.25, "cacheWrite": 0}),
            CODEX_CONTEXT,
            ["text", "image"],
        ),
        codex(
            "gpt-5.4-mini",
            "GPT-5.4 mini",
            {"input": 0.75, "output": 4.5, "cacheRead": 0.075, "cacheWrite": 0},
            CODEX_CONTEXT,
            ["text", "image"],
        ),
        codex(
            "gpt-5.5",
            "GPT-5.5",
            with_openai_long_context_pricing({"input": 5, "output": 30, "cacheRead": 0.5, "cacheWrite": 0}),
            CODEX_CONTEXT,
            ["text", "image"],
        ),
        codex(
            "gpt-5.6-luna",
            "GPT-5.6 Luna",
            with_openai_long_context_pricing(OPENAI_GPT_56_STANDARD_COSTS["gpt-5.6-luna"]),
            CODEX_GPT_56_CONTEXT,
            ["text", "image"],
        ),
        codex(
            "gpt-5.6-sol",
            "GPT-5.6 Sol",
            with_openai_long_context_pricing({"input": 5, "output": 30, "cacheRead": 0.5, "cacheWrite": 6.25}),
            CODEX_GPT_56_CONTEXT,
            ["text", "image"],
        ),
        codex(
            "gpt-5.6-terra",
            "GPT-5.6 Terra",
            with_openai_long_context_pricing(OPENAI_GPT_56_STANDARD_COSTS["gpt-5.6-terra"]),
            CODEX_GPT_56_CONTEXT,
            ["text", "image"],
        ),
    ]


# Azure Foundry deploys these with larger context windows than OpenAI's own
# short-tier defaults. See models-sold-directly-by-azure docs.
AZURE_CONTEXT_WINDOW_OVERRIDES = {
    "gpt-5.4": 1050000,
    "gpt-5.5": 1050000,
    "gpt-5.6-luna": 1050000,
    "gpt-5.6-sol": 1050000,
    "gpt-5.6-terra": 1050000,
}


def _collect_models() -> list[Json]:
    """Port of the model collection half of `generateModels`."""
    models_dev_models = load_models_dev_data()
    openrouter_models = fetch_openrouter_models()
    ai_gateway_models = fetch_ai_gateway_models()

    # Combine models (models.dev has priority)
    all_models = [
        model
        for model in [*models_dev_models, *openrouter_models, *ai_gateway_models]
        if not (model["provider"] == "xai" and model["id"] in XAI_BUILTIN_EXCLUDED_MODEL_IDS)
        and not (model["provider"] in ("opencode", "opencode-go") and model["id"] == "gpt-5.3-codex-spark")
    ]

    _apply_temporary_overrides(all_models)

    # Add missing gpt models
    for model in _missing_openai_models():
        if not any(m["provider"] == model["provider"] and m["id"] == model["id"] for m in all_models):
            all_models.append(model)

    all_models.extend(_deepseek_v4_models())
    all_models.extend(_ant_ling_models())

    for candidate in all_models:
        if (
            candidate["api"] == "openai-completions"
            and "deepseek-v4" in candidate["id"]
            and candidate["provider"] not in QWEN_TOKEN_PLAN_PROVIDER_IDS
        ):
            preserves_native_reasoning_effort = candidate["provider"] in ("openrouter", "opencode")
            merge_compat(
                candidate,
                {
                    "requiresReasoningContentOnAssistantMessages": DEEPSEEK_COMPAT[
                        "requiresReasoningContentOnAssistantMessages"
                    ]
                }
                if preserves_native_reasoning_effort
                else DEEPSEEK_COMPAT,
            )

    minimax_direct_supported_ids = {"MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M3"}
    all_models = [
        model
        for model in all_models
        if not (model["provider"] in ("minimax", "minimax-cn") and model["id"] not in minimax_direct_supported_ids)
    ]

    all_models.extend(_codex_models())

    # Add missing Mistral Medium 3.5 model until models.dev includes it
    if not any(m["provider"] == "mistral" and m["id"] == "mistral-medium-3.5" for m in all_models):
        all_models.append(
            make_model(
                id="mistral-medium-3.5",
                name="Mistral Medium 3.5",
                api="mistral-conversations",
                provider="mistral",
                base_url="https://api.mistral.ai",
                reasoning=True,
                input=["text", "image"],
                cost={"input": 1.5, "output": 7.5, "cacheRead": 0, "cacheWrite": 0},
                context_window=262144,
                max_tokens=262144,
            )
        )

    # Add "auto" alias for openrouter/auto. Costs are unknown because OpenRouter
    # routes to different models and charges for the underlying model.
    if not any(m["provider"] == "openrouter" and m["id"] == "auto" for m in all_models):
        all_models.append(
            make_model(
                id="auto",
                name="Auto",
                api="openai-completions",
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                reasoning=True,
                input=["text", "image"],
                cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                context_window=2000000,
                max_tokens=30000,
            )
        )

    # Add "fusion" alias for openrouter/fusion. OpenRouter exposes Fusion as a
    # router alias entry point; its metadata does not advertise tools, but the
    # alias resolves to a concrete model that can invoke caller tools.
    if not any(m["provider"] == "openrouter" and m["id"] == "openrouter/fusion" for m in all_models):
        all_models.append(
            make_model(
                id="openrouter/fusion",
                name="OpenRouter: Fusion",
                api="openai-completions",
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                reasoning=True,
                input=["text"],
                cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                context_window=1000000,
                max_tokens=30000,
            )
        )

    azure_openai_models: list[Json] = []
    for model in all_models:
        if model["provider"] != "openai" or model["api"] != "openai-responses":
            continue
        clone = deepcopy(model)
        clone["api"] = "azure-openai-responses"
        clone["provider"] = "azure-openai-responses"
        clone["baseUrl"] = ""
        clone["cost"] = {
            "input": model["cost"]["input"],
            "output": model["cost"]["output"],
            "cacheRead": model["cost"]["cacheRead"],
            "cacheWrite": model["cost"]["cacheWrite"],
        }
        clone["contextWindow"] = AZURE_CONTEXT_WINDOW_OVERRIDES.get(model["id"], model["contextWindow"])
        azure_openai_models.append(clone)
    all_models.extend(azure_openai_models)

    for model in all_models:
        apply_openai_completions_compat_metadata(model)
        apply_models_dev_reasoning_option_metadata(model)
        apply_thinking_level_metadata(model)
        apply_strict_tool_compat_metadata(model)
        apply_openai_grammar_tool_compat_metadata(model)
        apply_openai_tool_search_metadata(model)
        apply_openai_explicit_prompt_cache_metadata(model)

    return all_models


def generate_models() -> None:
    """Port of `generateModels`."""
    all_models = _collect_models()

    # Group by provider and deduplicate by model ID (models.dev takes priority)
    providers: dict[str, dict[str, Json]] = {}
    for model in all_models:
        providers.setdefault(model["provider"], {})
        if model["id"] not in providers[model["provider"]]:
            providers[model["provider"]][model["id"]] = model

    sorted_provider_ids = sorted(providers)
    json_providers: dict[str, dict[str, Json]] = {
        provider_id: {model_id: providers[provider_id][model_id] for model_id in sorted(providers[provider_id])}
        for provider_id in sorted_provider_ids
    }

    generated_data_provider_ids = (
        read_model_data_provider_ids(DATA_DIR) if generator_options.data_only else sorted_provider_ids
    )
    missing_provider_ids = [
        provider_id for provider_id in generated_data_provider_ids if provider_id not in json_providers
    ]
    if missing_provider_ids:
        raise ValueError(f"Cannot hydrate missing providers: {', '.join(missing_provider_ids)}")

    # Only the committed shard data is grouped by API. Public JSON catalog output stays flat.
    generated_data_providers: dict[str, dict[str, dict[str, Json]]] = {}
    model_data_structure: ModelDataStructure = {}
    for provider_id in generated_data_provider_ids:
        models = json_providers[provider_id]
        generated_data_providers[provider_id] = {}
        model_data_structure[provider_id] = {}
        for api in sorted({model["api"] for model in models.values()}):
            generated_data_providers[provider_id][api] = {}
            for model_id, model in models.items():
                if model["api"] != api:
                    continue
                generated_data_providers[provider_id][api][model_id] = model
                model_data_structure[provider_id][model_id] = api

    generated_at = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    if not generator_options.json_only:
        _write_generated_data(generated_data_provider_ids, generated_data_providers, model_data_structure, generated_at)

    if generator_options.json_output_dir is not None:
        output_dir = generator_options.json_output_dir
        provider_output_dir = output_dir / "providers"
        shutil.rmtree(output_dir, ignore_errors=True)
        provider_output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "models.json", json_providers)
        write_json(output_dir / "providers.json", sorted_provider_ids)
        for provider_id in sorted_provider_ids:
            write_json(provider_output_dir / f"{provider_id}.json", json_providers[provider_id])
        print(f"Generated JSON model catalog under {output_dir}")

    total_models = len(all_models)
    reasoning_models = sum(1 for model in all_models if model["reasoning"])
    print("\nModel Statistics:")
    print(f"  Total tool-capable models: {total_models}")
    print(f"  Reasoning-capable models: {reasoning_models}")
    for provider_id, models in providers.items():
        print(f"  {provider_id}: {len(models)} models")


def _write_generated_data(
    provider_ids: list[str],
    generated_data_providers: dict[str, dict[str, dict[str, Json]]],
    model_data_structure: ModelDataStructure,
    generated_at: str,
) -> None:
    """Stage, validate, then atomically replace `pi_ai/providers/data/`."""
    PROVIDERS_DIR.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".model-generation-", dir=PROVIDERS_DIR))
    staged_data_dir = staging_root / "data"
    previous_data_dir = staging_root / "previous-data"
    try:
        staged_data_dir.mkdir(parents=True, exist_ok=True)
        file_contents: dict[str, str] = {}
        for provider_id in provider_ids:
            filename = f"{provider_id}.json"
            content = serialize_json(generated_data_providers[provider_id])
            file_contents[filename] = content
            (staged_data_dir / filename).write_text(content, encoding="utf-8")
        write_json(
            staged_data_dir / MODEL_DATA_MANIFEST_FILE,
            create_model_data_manifest(model_data_structure, file_contents, generated_at),
        )
        validate_model_data_directory(model_data_structure, staged_data_dir)

        had_previous_data = DATA_DIR.exists()
        if had_previous_data:
            os.rename(DATA_DIR, previous_data_dir)
        try:
            os.rename(staged_data_dir, DATA_DIR)
            validate_generated_model_data(DATA_DIR)
        except Exception:
            shutil.rmtree(DATA_DIR, ignore_errors=True)
            if had_previous_data and previous_data_dir.exists():
                os.rename(previous_data_dir, DATA_DIR)
            raise
        print(
            "Hydrated JSON model values under pi_ai/providers/data/"
            if generator_options.data_only
            else "Generated JSON model values under pi_ai/providers/data/"
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    global generator_options
    generator_options = read_generator_options(list(sys.argv[1:] if argv is None else argv))
    try:
        generate_models()
    except Exception as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
