"""OpenAI Chat Completions provider.

Python port of `packages/ai/src/api/openai-completions.ts`. The TypeScript
version drives the official ``openai`` SDK; this port speaks the same HTTP API
directly through :mod:`pi_ai.utils.http`, which keeps request construction and
SSE handling visible and testable.

Also ports `packages/ai/src/api/openai-prompt-cache.ts` (eight lines:
``OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH`` and ``clamp_openai_prompt_cache_key``),
folded in here because this module and `openai_responses` are its only
callers.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..models import calculate_cost, clamp_thinking_level
from ..types import (
    AssistantMessage,
    CacheRetention,
    Context,
    DoneEvent,
    ErrorEvent,
    Message,
    Model,
    ProviderResponse,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
    now_ms,
)
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.hash import short_hash
from ..utils.headers import apply_header_overrides
from ..utils.http import HttpRequest, stream_sse_with_retry
from ..utils.json_parse import parse_streaming_json
from ..utils.json_stringify import json_stringify
from ..utils.provider_env import get_provider_env_value
from ..utils.provider_retry import ProviderRetryOptions
from ..utils.sanitize_unicode import sanitize_surrogates
from ..utils.tasks import spawn
from .github_copilot_headers import build_copilot_dynamic_headers, has_copilot_vision_input
from .simple_options import as_provider_options, build_base_options
from .transform_messages import transform_messages

OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH = 64

REASONING_DELTA_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


def clamp_openai_prompt_cache_key(key: str | None) -> str | None:
    if key is None:
        return None
    if len(key) <= OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH:
        return key
    return key[:OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH]


@dataclass
class OpenAICompletionsOptions(StreamOptions):
    tool_choice: Any = None
    reasoning_effort: str | None = None
    thinking_budgets: Any = None


@dataclass
class ResolvedCompat:
    """Compatibility settings resolved from provider/base URL plus model overrides."""

    supports_store: bool = True
    supports_developer_role: bool = True
    supports_reasoning_effort: bool = True
    supports_usage_in_streaming: bool = True
    supports_finish_reason: bool = True
    max_tokens_field: str = "max_completion_tokens"
    requires_tool_result_name: bool = False
    requires_assistant_after_tool_result: bool = False
    requires_thinking_as_text: bool = False
    requires_reasoning_content_on_assistant_messages: bool = False
    thinking_format: str = "openai"
    open_router_routing: dict[str, Any] = field(default_factory=dict)
    vercel_gateway_routing: dict[str, Any] = field(default_factory=dict)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    chat_template_args: dict[str, Any] = field(default_factory=dict)
    zai_tool_stream: bool = False
    supports_thinking_token_budget: bool = False
    supports_strict_mode: bool = True
    supports_openai_grammar_tools: bool = False
    cache_control_format: str | None = None
    send_session_affinity_headers: bool = False
    deferred_tools_mode: str | None = None
    session_affinity_format: str = "openai"
    supports_long_cache_retention: bool = True


def detect_compat(model: Model) -> ResolvedCompat:
    """Auto-detect compatibility settings from the provider name and base URL."""
    provider = model.provider
    base_url = model.base_url

    is_zai = provider in ("zai", "zai-coding-cn") or "api.z.ai" in base_url or "open.bigmodel.cn" in base_url
    is_together = provider == "together" or "api.together.ai" in base_url or "api.together.xyz" in base_url
    is_moonshot = provider in ("moonshotai", "moonshotai-cn") or "api.moonshot." in base_url
    is_openrouter = provider == "openrouter" or "openrouter.ai" in base_url
    is_cloudflare_workers_ai = provider == "cloudflare-workers-ai" or "api.cloudflare.com" in base_url
    is_cloudflare_ai_gateway = provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in base_url
    is_nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in base_url
    is_ant_ling = provider == "ant-ling" or "api.ant-ling.com" in base_url

    # `baseUrl.toLowerCase()` upstream: a provider configured with a
    # mixed-case host would otherwise miss every DeepSeek-specific branch.
    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url.lower()
    is_non_standard = (
        is_nvidia
        or provider == "cerebras"
        or "cerebras.ai" in base_url
        or provider == "xai"
        or "api.x.ai" in base_url
        or is_together
        or "chutes.ai" in base_url
        or is_deepseek
        or is_zai
        or is_moonshot
        or provider == "opencode"
        or "opencode.ai" in base_url
        or is_cloudflare_workers_ai
        or is_cloudflare_ai_gateway
        or is_ant_ling
    )

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
        model.id.startswith("anthropic/") or model.id.startswith("openai/")
    )
    cache_control_format = "anthropic" if provider == "openrouter" and model.id.startswith("anthropic/") else None

    if is_deepseek:
        thinking_format = "deepseek"
    elif is_zai:
        thinking_format = "zai"
    elif is_together:
        thinking_format = "together"
    elif is_ant_ling:
        thinking_format = "ant-ling"
    elif is_openrouter:
        thinking_format = "openrouter"
    else:
        thinking_format = "openai"

    return ResolvedCompat(
        supports_store=not is_non_standard,
        supports_developer_role=is_openrouter_developer_role_model or (not is_non_standard and not is_openrouter),
        supports_reasoning_effort=not (
            is_grok or is_zai or is_moonshot or is_together or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling
        ),
        supports_usage_in_streaming=True,
        supports_finish_reason=True,
        max_tokens_field="max_tokens" if use_max_tokens else "max_completion_tokens",
        requires_tool_result_name=False,
        requires_assistant_after_tool_result=False,
        requires_thinking_as_text=False,
        requires_reasoning_content_on_assistant_messages=is_deepseek,
        thinking_format=thinking_format,
        open_router_routing={},
        vercel_gateway_routing={},
        chat_template_kwargs={},
        chat_template_args={},
        zai_tool_stream=False,
        supports_thinking_token_budget=False,
        supports_strict_mode=not (is_moonshot or is_together or is_cloudflare_ai_gateway or is_nvidia),
        supports_openai_grammar_tools=False,
        cache_control_format=cache_control_format,
        send_session_affinity_headers=False,
        deferred_tools_mode=None,
        session_affinity_format="openrouter" if is_openrouter else "openai",
        supports_long_cache_retention=not (
            is_together or is_cloudflare_workers_ai or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling
        ),
    )


_COMPAT_FIELDS = {
    "supportsStore": "supports_store",
    "supportsDeveloperRole": "supports_developer_role",
    "supportsReasoningEffort": "supports_reasoning_effort",
    "supportsUsageInStreaming": "supports_usage_in_streaming",
    "supportsFinishReason": "supports_finish_reason",
    "maxTokensField": "max_tokens_field",
    "requiresToolResultName": "requires_tool_result_name",
    "requiresAssistantAfterToolResult": "requires_assistant_after_tool_result",
    "requiresThinkingAsText": "requires_thinking_as_text",
    "requiresReasoningContentOnAssistantMessages": "requires_reasoning_content_on_assistant_messages",
    "thinkingFormat": "thinking_format",
    "openRouterRouting": "open_router_routing",
    "vercelGatewayRouting": "vercel_gateway_routing",
    "chatTemplateKwargs": "chat_template_kwargs",
    "chatTemplateArgs": "chat_template_args",
    "zaiToolStream": "zai_tool_stream",
    "supportsThinkingTokenBudget": "supports_thinking_token_budget",
    "supportsStrictMode": "supports_strict_mode",
    "supportsOpenAIGrammarTools": "supports_openai_grammar_tools",
    "cacheControlFormat": "cache_control_format",
    "sendSessionAffinityHeaders": "send_session_affinity_headers",
    "deferredToolsMode": "deferred_tools_mode",
    "sessionAffinityFormat": "session_affinity_format",
    "supportsLongCacheRetention": "supports_long_cache_retention",
}


def get_compat(model: Model) -> ResolvedCompat:
    """Detected settings overridden by explicit ``model.compat`` entries.

    Both the TypeScript camelCase keys and the Python snake_case names are
    accepted in ``model.compat`` so catalogs copied from the TypeScript project
    work unchanged.
    """
    resolved = detect_compat(model)
    if not model.compat:
        return resolved

    for key, value in model.compat.items():
        attribute = _COMPAT_FIELDS.get(key, key)
        if value is None:
            continue
        if hasattr(resolved, attribute):
            setattr(resolved, attribute, value)
    # openRouterRouting has no detected default: an unset value means "no routing".
    if "openRouterRouting" not in model.compat and "open_router_routing" not in model.compat:
        resolved.open_router_routing = {}
    return resolved


def resolve_cache_retention(cache_retention: CacheRetention | None, env: dict[str, str] | None) -> CacheRetention:
    """Resolve the cache retention preference.

    Defaults to ``"short"``; ``PI_CACHE_RETENTION=long`` opts into extended
    prompt caching.
    """
    if cache_retention is not None:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


def _has_header(headers: dict[str, str | None] | None, name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    return any(key.lower() == expected and value is not None and value.strip() for key, value in headers.items())


def get_client_api_key(provider: str, api_key: str | None, headers: dict[str, str | None] | None) -> str:
    if api_key:
        return api_key
    if _has_header(headers, "authorization") or _has_header(headers, "cf-aig-authorization"):
        return "unused"
    raise ValueError(f"No API key for provider: {provider}")


def has_tool_history(messages: list[Message]) -> bool:
    for msg in messages:
        if msg.role == "toolResult":
            return True
        if msg.role == "assistant" and any(block.type == "toolCall" for block in msg.content):
            return True
    return False


def get_deferred_tool_names(messages: list[Message]) -> dict[str, None]:
    """Deferred tool names in first-seen order.

    TypeScript accumulates these in a `Set`, which iterates in insertion order.
    A Python `set` does not, and the order reaches the wire (the Kimi
    tool-carrying system message lists the schemas in this order), so an
    insertion-ordered dict stands in for the `Set`.
    """
    names: dict[str, None] = {}
    for message in messages:
        if message.role == "toolResult":
            for name in message.added_tool_names or []:
                names[name] = None
    return names


def get_tools_by_name(tools: list[Tool] | None, names: Iterable[str]) -> list[Tool]:
    if not tools:
        return []
    by_name = {tool.name: tool for tool in tools}
    return [by_name[name] for name in names if name in by_name]


def normalize_tool_call_id(model: Model, tool_call_id: str) -> str:
    """Shorten and sanitize ids that came from the OpenAI Responses API.

    Responses ids look like ``{call_id}|{item_id}`` and can exceed 400 chars,
    while Chat Completions requires distinct ids of at most 40 chars.
    """
    if "|" in tool_call_id:
        separator_index = tool_call_id.index("|")
        call_id = _sanitize_id(tool_call_id[:separator_index])
        item_id = _sanitize_id(tool_call_id[separator_index + 1 :])
        combined = f"{call_id}_{item_id}" if item_id else call_id
        if len(combined) <= 40:
            return combined
        digest = short_hash(tool_call_id)[:8]
        prefix = call_id[: max(1, 40 - len(digest) - 1)]
        return f"{prefix}_{digest}"

    if model.provider == "openai":
        return tool_call_id[:40] if len(tool_call_id) > 40 else tool_call_id
    return tool_call_id


def _sanitize_id(value: str) -> str:
    return "".join(char if char.isascii() and (char.isalnum() or char in "_-") else "_" for char in value)


def convert_messages(model: Model, context: Context, compat: ResolvedCompat) -> list[dict[str, Any]]:
    """Convert pi messages into Chat Completions message params."""
    params: list[dict[str, Any]] = []

    transformed = transform_messages(
        context.messages, model, lambda tool_call_id, m, _source: normalize_tool_call_id(m, tool_call_id)
    )

    if context.system_prompt:
        use_developer_role = model.reasoning and compat.supports_developer_role
        role = "developer" if use_developer_role else "system"
        params.append({"role": role, "content": sanitize_surrogates(context.system_prompt)})

    last_role: str | None = None
    index = 0
    while index < len(transformed):
        msg = transformed[index]

        # Some providers reject a user message straight after tool results.
        if compat.requires_assistant_after_tool_result and last_role == "toolResult" and msg.role == "user":
            params.append({"role": "assistant", "content": "I have processed the tool results."})

        if msg.role == "user":
            if isinstance(msg.content, str):
                params.append({"role": "user", "content": sanitize_surrogates(msg.content)})
            else:
                content: list[dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        content.append({"type": "text", "text": sanitize_surrogates(item.text)})
                    else:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"},
                            }
                        )
                if not content:
                    index += 1
                    continue
                params.append({"role": "user", "content": content})
            last_role = msg.role
            index += 1
            continue

        if msg.role == "assistant":
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                # Some providers reject null content.
                "content": "" if compat.requires_assistant_after_tool_result else None,
            }

            text_parts = [
                sanitize_surrogates(block.text) for block in msg.content if block.type == "text" and block.text.strip()
            ]
            assistant_text = "".join(text_parts)

            thinking_blocks = [block for block in msg.content if block.type == "thinking" and block.thinking.strip()]
            if thinking_blocks:
                if compat.requires_thinking_as_text:
                    # Plain text, without tags, so the model does not mimic them.
                    thinking_text = "\n\n".join(sanitize_surrogates(block.thinking) for block in thinking_blocks)
                    assistant_msg["content"] = [{"type": "text", "text": thinking_text}] + [
                        {"type": "text", "text": part} for part in text_parts
                    ]
                else:
                    # Assistant content must be a plain string: an array of text
                    # blocks is non-standard here and makes some models mirror the
                    # block structure literally in their output.
                    if assistant_text:
                        assistant_msg["content"] = assistant_text

                    signature = thinking_blocks[0].thinking_signature
                    if model.provider == "opencode-go" and signature == "reasoning":
                        signature = "reasoning_content"
                    if signature:
                        assistant_msg[signature] = "\n".join(block.thinking for block in thinking_blocks)
            elif assistant_text:
                assistant_msg["content"] = assistant_text

            tool_calls = [block for block in msg.content if block.type == "toolCall"]
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json_stringify(tool_call.arguments),
                        },
                    }
                    for tool_call in tool_calls
                ]
                reasoning_details = []
                for tool_call in tool_calls:
                    if not tool_call.thought_signature:
                        continue
                    try:
                        reasoning_details.append(json.loads(tool_call.thought_signature))
                    except ValueError:
                        continue
                if reasoning_details:
                    assistant_msg["reasoning_details"] = reasoning_details

            if (
                compat.requires_reasoning_content_on_assistant_messages
                and model.reasoning
                and "reasoning_content" not in assistant_msg
            ):
                assistant_msg["reasoning_content"] = ""

            # Providers require either content or tool_calls; skip empty turns.
            # last_role is deliberately left untouched so that an aborted turn
            # between tool results and a user message still triggers the
            # synthetic bridging assistant message below.
            content_value = assistant_msg.get("content")
            has_content = bool(content_value)
            if not has_content and "tool_calls" not in assistant_msg:
                index += 1
                continue

            params.append(assistant_msg)
            last_role = msg.role
            index += 1
            continue

        # Tool results: consume the whole consecutive run.
        image_blocks: list[dict[str, Any]] = []
        deferred_tool_names: dict[str, None] = {}
        cursor = index
        while cursor < len(transformed) and transformed[cursor].role == "toolResult":
            tool_msg = transformed[cursor]

            text_result = "\n".join(block.text for block in tool_msg.content if block.type == "text")
            has_images = any(block.type == "image" for block in tool_msg.content)
            if text_result:
                tool_result_text = text_result
            elif has_images:
                tool_result_text = "(see attached image)"
            else:
                tool_result_text = "(no tool output)"

            tool_result_msg: dict[str, Any] = {
                "role": "tool",
                "content": sanitize_surrogates(tool_result_text),
                "tool_call_id": tool_msg.tool_call_id,
            }
            if compat.requires_tool_result_name and tool_msg.tool_name:
                tool_result_msg["name"] = tool_msg.tool_name
            params.append(tool_result_msg)

            if compat.deferred_tools_mode == "kimi":
                for name in tool_msg.added_tool_names or []:
                    deferred_tool_names[name] = None

            if has_images and "image" in model.input:
                for block in tool_msg.content:
                    if block.type == "image":
                        image_blocks.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                            }
                        )
            cursor += 1

        index = cursor

        if image_blocks:
            if compat.requires_assistant_after_tool_result:
                params.append({"role": "assistant", "content": "I have processed the tool results."})
            params.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Attached image(s) from tool result:"}, *image_blocks],
                }
            )
            last_role = "user"
        else:
            last_role = "toolResult"

        if deferred_tool_names:
            deferred_tools = get_tools_by_name(context.tools, deferred_tool_names)
            if deferred_tools:
                # Kimi accepts a system message carrying tools and no content field.
                params.append({"role": "system", "tools": convert_tools(deferred_tools, compat)})

    return params


def convert_tools(tools: list[Tool], compat: ResolvedCompat) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        function: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if compat.supports_strict_mode:
            strict = None
            config = tool.constrained_sampling
            if config and getattr(config, "type", None) == "json_schema":
                strict = config.strict == "require" or config.strict == "prefer"
            function["strict"] = bool(strict)
        converted.append({"type": "function", "function": function})
    return converted


def parse_chunk_usage(raw_usage: dict[str, Any], model: Model) -> Usage:
    prompt_tokens = raw_usage.get("prompt_tokens") or 0
    details = raw_usage.get("prompt_tokens_details") or {}
    cache_read_tokens = details.get("cached_tokens")
    if cache_read_tokens is None:
        cache_read_tokens = raw_usage.get("prompt_cache_hit_tokens") or 0
    cache_write_tokens = details.get("cache_write_tokens") or 0

    # cached_tokens counts cache reads; writes are reported separately by
    # OpenRouter-compatible providers. Subtracting writes from reads would
    # under-report spec-compliant providers.
    input_tokens = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
    output_tokens = raw_usage.get("completion_tokens") or 0
    completion_details = raw_usage.get("completion_tokens_details") or {}

    usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read_tokens,
        cache_write=cache_write_tokens,
        reasoning=completion_details.get("reasoning_tokens") or 0,
        total_tokens=input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
    )
    calculate_cost(model, usage)
    return usage


def map_stop_reason(reason: str | None) -> tuple[StopReason, str | None]:
    if reason is None:
        return "stop", None
    if reason in ("stop", "end"):
        return "stop", None
    if reason == "length":
        return "length", None
    if reason in ("function_call", "tool_calls"):
        return "toolUse", None
    if reason == "content_filter":
        return "error", "Provider finish_reason: content_filter"
    if reason == "network_error":
        return "error", "Provider finish_reason: network_error"
    return "error", f"Provider finish_reason: {reason}"


def build_headers(
    model: Model,
    api_key: str,
    options: OpenAICompletionsOptions,
    compat: ResolvedCompat,
    context: Context | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = dict(model.headers)
    headers.setdefault("content-type", "application/json")
    headers["authorization"] = f"Bearer {api_key}"

    if model.provider == "github-copilot" and context is not None:
        has_images = has_copilot_vision_input(context.messages)
        headers.update(build_copilot_dynamic_headers(context.messages, has_images))

    # TypeScript's caller passes `cacheRetention === "none" ? undefined :
    # options.sessionId` into `createClient`, so disabling caching also drops
    # the session-affinity headers.
    session_id = None if resolve_cache_retention(options.cache_retention, options.env) == "none" else options.session_id
    if session_id and compat.send_session_affinity_headers:
        if compat.session_affinity_format == "openrouter":
            headers["x-session-id"] = session_id
        else:
            if compat.session_affinity_format == "openai":
                headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id
            headers["x-session-affinity"] = session_id

    return apply_header_overrides(headers, options.headers)


def get_compat_cache_control(compat: ResolvedCompat, cache_retention: CacheRetention) -> dict[str, Any] | None:
    """The Anthropic-style ``cache_control`` marker to stamp, if any."""
    if compat.cache_control_format != "anthropic" or cache_retention == "none":
        return None
    marker: dict[str, Any] = {"type": "ephemeral"}
    if cache_retention == "long" and compat.supports_long_cache_retention:
        marker["ttl"] = "1h"
    return marker


def apply_anthropic_cache_control(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    cache_control: dict[str, Any],
) -> None:
    """Place Anthropic cache breakpoints on the system prompt, last tool and last message."""
    _add_cache_control_to_system_prompt(messages, cache_control)
    _add_cache_control_to_last_tool(tools, cache_control)
    _add_cache_control_to_last_conversation_message(messages, cache_control)


def _add_cache_control_to_system_prompt(messages: list[dict[str, Any]], cache_control: dict[str, Any]) -> None:
    for message in messages:
        if message.get("role") in ("system", "developer"):
            _add_cache_control_to_text_content(message, cache_control)
            return


def _add_cache_control_to_last_conversation_message(
    messages: list[dict[str, Any]], cache_control: dict[str, Any]
) -> None:
    for message in reversed(messages):
        if message.get("role") in ("user", "assistant", "tool") and _add_cache_control_to_text_content(
            message, cache_control
        ):
            return


def _add_cache_control_to_last_tool(tools: list[dict[str, Any]] | None, cache_control: dict[str, Any]) -> None:
    if not tools:
        return
    tools[-1]["cache_control"] = cache_control


def _add_cache_control_to_text_content(message: dict[str, Any], cache_control: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return False
        message["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]
        return True

    if not isinstance(content, list):
        return False

    for part in reversed(content):
        if isinstance(part, dict) and part.get("type") == "text":
            part["cache_control"] = cache_control
            return True

    return False


def build_params(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | None = None,
    compat: ResolvedCompat | None = None,
    cache_retention: CacheRetention | None = None,
) -> dict[str, Any]:
    options = as_provider_options(options, OpenAICompletionsOptions)
    compat = compat or get_compat(model)
    if cache_retention is None:
        cache_retention = resolve_cache_retention(options.cache_retention, options.env)

    messages = convert_messages(model, context, compat)

    params: dict[str, Any] = {"model": model.id, "messages": messages, "stream": True}

    wants_cache_key = ("api.openai.com" in model.base_url and cache_retention != "none") or (
        cache_retention == "long" and compat.supports_long_cache_retention
    )
    if wants_cache_key:
        cache_key = clamp_openai_prompt_cache_key(options.session_id)
        if cache_key is not None:
            params["prompt_cache_key"] = cache_key
    if cache_retention == "long" and compat.supports_long_cache_retention:
        params["prompt_cache_retention"] = "24h"

    if compat.supports_usage_in_streaming:
        params["stream_options"] = {"include_usage": True}

    if compat.supports_store:
        params["store"] = False

    if options.max_tokens:
        params[compat.max_tokens_field] = options.max_tokens

    if options.temperature is not None:
        params["temperature"] = options.temperature

    deferred_tool_names = get_deferred_tool_names(context.messages) if compat.deferred_tools_mode == "kimi" else {}
    active_tools = [tool for tool in (context.tools or []) if tool.name not in deferred_tool_names]
    if active_tools:
        params["tools"] = convert_tools(active_tools, compat)
        if compat.zai_tool_stream:
            params["tool_stream"] = True
    elif has_tool_history(context.messages):
        # Anthropic behind a proxy requires the tools field once the history
        # contains tool calls or tool results.
        params["tools"] = []

    if options.tool_choice:
        params["tool_choice"] = options.tool_choice

    cache_control = get_compat_cache_control(compat, cache_retention)
    if cache_control:
        apply_anthropic_cache_control(messages, params.get("tools"), cache_control)

    _apply_thinking_params(params, model, options, compat)

    if model.compat.get("openRouterRouting") or compat.open_router_routing:
        routing = model.compat.get("openRouterRouting") or compat.open_router_routing
        if routing:
            params["provider"] = routing

    gateway_routing = model.compat.get("vercelGatewayRouting") or compat.vercel_gateway_routing
    if gateway_routing and (gateway_routing.get("only") or gateway_routing.get("order")):
        gateway_options: dict[str, Any] = {}
        if gateway_routing.get("only"):
            gateway_options["only"] = gateway_routing["only"]
        if gateway_routing.get("order"):
            gateway_options["order"] = gateway_routing["order"]
        params["providerOptions"] = {"gateway": gateway_options}

    # Last, so custom keys override the named request fields.
    if options.sampling_params:
        params.update(options.sampling_params)

    return params


def _mapped_effort(model: Model, effort: str) -> str | None:
    """Strict lookup: only fall back to the raw effort when the level is unmapped.

    Mirrors TypeScript's ``mappedEffort === undefined ? effort : mappedEffort``. A
    level explicitly mapped to ``None`` stays ``None`` here (the caller then
    decides whether to omit the field).
    """
    if effort in model.thinking_level_map:
        return model.thinking_level_map[effort]
    return effort


def _mapped_effort_or_raw(model: Model, effort: str) -> str:
    """Nullish-coalescing lookup: an explicit ``None`` mapping also falls back.

    Mirrors TypeScript's ``model.thinkingLevelMap?.[effort] ?? effort``, which
    falls back to the raw effort string whether the level is unmapped or mapped
    to ``null``. Several thinking formats (qwen, deepseek, openrouter, together,
    string-thinking, plain openai) use this looser fallback, unlike zai/baseten
    which use the strict variant above.
    """
    mapped = model.thinking_level_map.get(effort)
    return effort if mapped is None else mapped


def _apply_thinking_params(
    params: dict[str, Any], model: Model, options: OpenAICompletionsOptions, compat: ResolvedCompat
) -> None:
    effort = options.reasoning_effort
    thinking_format = compat.thinking_format
    off_value = model.thinking_level_map.get("off")
    off_is_disabled = "off" in model.thinking_level_map and off_value is None

    if thinking_format == "zai" and model.reasoning:
        params["thinking"] = {"type": "enabled", "clear_thinking": False} if effort else {"type": "disabled"}
        if effort and compat.supports_reasoning_effort:
            mapped = _mapped_effort(model, effort)
            if isinstance(mapped, str):
                params["reasoning_effort"] = mapped
    elif thinking_format == "qwen" and model.reasoning:
        params["enable_thinking"] = bool(effort)
        if effort and compat.supports_reasoning_effort:
            mapped = _mapped_effort_or_raw(model, effort)
            if isinstance(mapped, str):
                params["reasoning_effort"] = mapped
    elif thinking_format == "qwen-chat-template" and model.reasoning:
        params["chat_template_kwargs"] = {"enable_thinking": bool(effort), "preserve_thinking": True}
    elif thinking_format == "chat-template" and model.reasoning:
        values = _build_chat_template_values(model, options, compat.chat_template_kwargs)
        if values:
            params["chat_template_kwargs"] = values
    elif thinking_format == "baseten" and model.reasoning:
        values = _build_chat_template_values(model, options, compat.chat_template_args)
        if values:
            params["chat_template_args"] = values
        if compat.supports_reasoning_effort:
            mapped = _mapped_effort(model, effort) if effort else model.thinking_level_map.get("off")
            if isinstance(mapped, str):
                params["reasoning_effort"] = mapped
    elif thinking_format == "deepseek" and model.reasoning:
        if effort:
            params["thinking"] = {"type": "enabled"}
        elif not off_is_disabled:
            params["thinking"] = {"type": "disabled"}
        if effort and compat.supports_reasoning_effort:
            params["reasoning_effort"] = _mapped_effort_or_raw(model, effort)
    elif thinking_format == "openrouter" and model.reasoning:
        # OpenRouter normalizes reasoning across providers with a nested object.
        if effort:
            params["reasoning"] = {"effort": _mapped_effort_or_raw(model, effort)}
        elif not off_is_disabled:
            params["reasoning"] = {"effort": off_value if off_value is not None else "none"}
    elif thinking_format == "ant-ling" and model.reasoning and effort:
        mapped = model.thinking_level_map.get(effort)
        if isinstance(mapped, str):
            params["reasoning"] = {"effort": mapped}
    elif thinking_format == "together" and model.reasoning:
        params["reasoning"] = {"enabled": bool(effort)}
        if effort and compat.supports_reasoning_effort:
            params["reasoning_effort"] = _mapped_effort_or_raw(model, effort)
    elif thinking_format == "string-thinking" and model.reasoning:
        if effort:
            params["thinking"] = _mapped_effort_or_raw(model, effort)
        elif not off_is_disabled:
            params["thinking"] = off_value if off_value is not None else "none"
    elif effort and model.reasoning and compat.supports_reasoning_effort:
        params["reasoning_effort"] = _mapped_effort_or_raw(model, effort)
    elif not effort and model.reasoning and compat.supports_reasoning_effort:
        if isinstance(off_value, str):
            params["reasoning_effort"] = off_value

    # vLLM caps reasoning with a top-level thinking_token_budget, independent of
    # thinkingFormat. Reasoning and the answer share max_tokens here, so an
    # uncapped reasoning phase can consume the whole response.
    if compat.supports_thinking_token_budget and effort and model.reasoning:
        from .simple_options import MIN_ANSWER_TOKENS, clamp_reasoning

        level = clamp_reasoning(effort)
        budgets = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384}
        if options.thinking_budgets:
            for key in budgets:
                override = getattr(options.thinking_budgets, key, None)
                if override is not None:
                    budgets[key] = override
        ceiling = params.get("max_tokens") or params.get("max_completion_tokens") or model.max_tokens
        budget = min(budgets[level], max(0, ceiling - MIN_ANSWER_TOKENS))
        if budget > 0:
            params["thinking_token_budget"] = budget


def _build_chat_template_values(
    model: Model, options: OpenAICompletionsOptions, values: dict[str, Any]
) -> dict[str, Any] | None:
    resolved: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict) and "$var" in value:
            variable = value["$var"]
            omit_when_off = value.get("omitWhenOff", value.get("omit_when_off", False))
            effort = options.reasoning_effort
            if variable == "thinking.enabled":
                if not effort and omit_when_off:
                    continue
                resolved[key] = bool(effort)
            elif variable == "thinking.effort":
                if not effort:
                    if omit_when_off:
                        continue
                    mapped = model.thinking_level_map.get("off")
                    if mapped is None:
                        continue
                    resolved[key] = mapped
                else:
                    # Strict lookup: an explicit `None` mapping omits the key
                    # entirely rather than falling back to the raw effort.
                    mapped = _mapped_effort(model, effort)
                    if isinstance(mapped, str):
                        resolved[key] = mapped
            continue
        resolved[key] = value
    return resolved or None


@dataclass
class _StreamingToolCall:
    block: ToolCall
    partial_args: str = ""


def stream(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Stream a chat completion. Failures are reported through the stream."""
    event_stream = AssistantMessageEventStream()
    spawn(_run_stream(event_stream, model, context, options, client))
    return event_stream


async def _run_stream(
    event_stream: AssistantMessageEventStream,
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | None,
    client: httpx.AsyncClient | None,
) -> None:
    options = as_provider_options(options, OpenAICompletionsOptions)
    output = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason="pending",
        timestamp=now_ms(),
    )

    try:
        api_key = get_client_api_key(model.provider, options.api_key, options.headers)
        compat = get_compat(model)
        cache_retention = resolve_cache_retention(options.cache_retention, options.env)
        params = build_params(model, context, options, compat, cache_retention)

        if options.on_payload is not None:
            replacement = options.on_payload(params, model)
            if hasattr(replacement, "__await__"):
                replacement = await replacement
            if replacement is not None:
                params = replacement

        request = HttpRequest(
            url=f"{model.base_url.rstrip('/')}/chat/completions",
            headers=build_headers(model, api_key, options, compat, context),
            json_body=params,
            timeout_ms=options.timeout_ms,
        )

        on_response = None
        if options.on_response is not None:
            captured = options.on_response

            async def on_response(provider_response: ProviderResponse) -> None:
                result = captured(provider_response, model)
                if hasattr(result, "__await__"):
                    await result

        state = _StreamState(event_stream, output, model)
        started = False
        sse_stream = stream_sse_with_retry(
            request,
            client=client,
            on_response=on_response,
            retry=ProviderRetryOptions(
                max_retries=options.max_retries or 0,
                max_retry_delay_ms=options.max_retry_delay_ms,
                signal=options.signal,
            ),
        )
        async for sse_event in sse_stream:
            if not started:
                event_stream.push(StartEvent(partial=output))
                started = True
            if sse_event.data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(sse_event.data)
            except ValueError:
                continue
            if not isinstance(chunk, dict):
                continue
            state.handle_chunk(chunk)

        if not started:
            event_stream.push(StartEvent(partial=output))

        state.finish_all_blocks()

        if options.signal is not None and options.signal.aborted:
            raise RuntimeError("Request was aborted")
        if output.stop_reason == "aborted":
            raise RuntimeError("Request was aborted")
        if not state.has_finish_reason and not compat.supports_finish_reason:
            output.stop_reason = "toolUse" if any(b.type == "toolCall" for b in output.content) else "stop"
        if output.stop_reason == "error":
            raise RuntimeError(output.error_message or "Provider returned an error stop reason")
        if (compat.supports_finish_reason and not state.has_finish_reason) or output.stop_reason == "pending":
            raise RuntimeError("Stream ended without finish_reason")

        event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
        event_stream.end()
    except asyncio.CancelledError:
        output.stop_reason = "aborted"
        output.error_message = "Request was aborted"
        event_stream.push(ErrorEvent(reason="aborted", error=output))
        event_stream.end()
        raise
    except BaseException as error:
        from ..utils.error_body import format_provider_error, normalize_provider_error

        aborted = options.signal is not None and options.signal.aborted
        output.stop_reason = "aborted" if aborted else "error"
        output.error_message = format_provider_error(normalize_provider_error(error))
        event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
        event_stream.end()


class _StreamState:
    """Accumulates streamed deltas into ``output.content`` and emits events."""

    def __init__(self, event_stream: AssistantMessageEventStream, output: AssistantMessage, model: Model) -> None:
        self.event_stream = event_stream
        self.output = output
        self.model = model
        self.text_block: TextContent | None = None
        self.thinking_block: ThinkingContent | None = None
        self.has_finish_reason = False
        self.tool_calls_by_index: dict[int, _StreamingToolCall] = {}
        self.tool_calls_by_id: dict[str, _StreamingToolCall] = {}
        self.pending_reasoning_details: dict[str, str] = {}

    def content_index(self, block: Any) -> int:
        for index, candidate in enumerate(self.output.content):
            if candidate is block:
                return index
        return -1

    def handle_chunk(self, chunk: dict[str, Any]) -> None:
        if not self.output.response_id and chunk.get("id"):
            self.output.response_id = chunk["id"]
        chunk_model = chunk.get("model")
        if (
            isinstance(chunk_model, str)
            and chunk_model
            and chunk_model != self.model.id
            and not self.output.response_model
        ):
            self.output.response_model = chunk_model
        if chunk.get("usage"):
            self.output.usage = parse_chunk_usage(chunk["usage"], self.model)

        choices = chunk.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if not choice:
            return

        # Some providers (Moonshot) report usage on the choice instead.
        if not chunk.get("usage") and choice.get("usage"):
            self.output.usage = parse_chunk_usage(choice["usage"], self.model)

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            self.output.raw_stop_reason = finish_reason
            stop_reason, error_message = map_stop_reason(finish_reason)
            self.output.stop_reason = stop_reason
            if error_message:
                self.output.error_message = error_message
            self.has_finish_reason = True

        delta = choice.get("delta")
        if not delta:
            return

        content = delta.get("content")
        if content:
            block = self.ensure_text_block()
            block.text += content
            self.event_stream.push(
                TextDeltaEvent(content_index=self.content_index(block), delta=content, partial=self.output)
            )

        # Reasoning arrives under different field names depending on the server;
        # use the first non-empty one to avoid duplicating identical content.
        for field_name in REASONING_DELTA_FIELDS:
            value = delta.get(field_name)
            if isinstance(value, str) and value:
                signature = (
                    "reasoning_content"
                    if self.model.provider == "opencode-go" and field_name == "reasoning"
                    else field_name
                )
                block = self.ensure_thinking_block(signature)
                block.thinking += value
                self.event_stream.push(
                    ThinkingDeltaEvent(content_index=self.content_index(block), delta=value, partial=self.output)
                )
                break

        for tool_call_delta in delta.get("tool_calls") or []:
            self.handle_tool_call_delta(tool_call_delta)

        reasoning_details = delta.get("reasoning_details")
        if isinstance(reasoning_details, list):
            for detail in reasoning_details:
                if not _is_encrypted_reasoning_detail(detail):
                    continue
                serialized = json_stringify(detail)
                matching = self.tool_calls_by_id.get(detail["id"])
                if matching:
                    matching.block.thought_signature = serialized
                else:
                    self.pending_reasoning_details[detail["id"]] = serialized

    def ensure_text_block(self) -> TextContent:
        if self.text_block is None:
            self.text_block = TextContent(text="")
            self.output.content.append(self.text_block)
            self.event_stream.push(
                TextStartEvent(content_index=self.content_index(self.text_block), partial=self.output)
            )
        return self.text_block

    def ensure_thinking_block(self, signature: str) -> ThinkingContent:
        if self.thinking_block is None:
            self.thinking_block = ThinkingContent(thinking="", thinking_signature=signature)
            self.output.content.append(self.thinking_block)
            self.event_stream.push(
                ThinkingStartEvent(content_index=self.content_index(self.thinking_block), partial=self.output)
            )
        return self.thinking_block

    def handle_tool_call_delta(self, tool_call_delta: dict[str, Any]) -> None:
        stream_index = tool_call_delta.get("index")
        function = tool_call_delta.get("function") or {}
        name = function.get("name") or ""
        tool_call_id = tool_call_delta.get("id") or ""

        entry = self.tool_calls_by_index.get(stream_index) if stream_index is not None else None
        if entry is None and tool_call_id:
            entry = self.tool_calls_by_id.get(tool_call_id)

        if entry is None:
            entry = _StreamingToolCall(block=ToolCall(id=tool_call_id, name=name, arguments={}))
            self.output.content.append(entry.block)
            if stream_index is not None:
                self.tool_calls_by_index[stream_index] = entry
            if tool_call_id:
                self.tool_calls_by_id[tool_call_id] = entry
            self.event_stream.push(
                ToolCallStartEvent(content_index=self.content_index(entry.block), partial=self.output)
            )
        else:
            if stream_index is not None and stream_index not in self.tool_calls_by_index:
                self.tool_calls_by_index[stream_index] = entry
            if tool_call_id:
                self.tool_calls_by_id[tool_call_id] = entry

        if not entry.block.name and name:
            entry.block.name = name

        # Apply any pending reasoning detail using the id the block had *before*
        # this delta: mirrors TypeScript, where the id assigned by a chunk that
        # first reveals it is only visible to `applyPendingReasoningDetail` on a
        # later delta for the same tool call, not the one that introduces it.
        self.apply_pending_reasoning_detail(entry)

        if not entry.block.id and tool_call_id:
            entry.block.id = tool_call_id

        arguments = function.get("arguments")
        delta = ""
        if arguments:
            delta = arguments
            entry.partial_args += arguments
            entry.block.arguments = parse_streaming_json(entry.partial_args)

        self.event_stream.push(
            ToolCallDeltaEvent(content_index=self.content_index(entry.block), delta=delta, partial=self.output)
        )

    def apply_pending_reasoning_detail(self, entry: _StreamingToolCall) -> None:
        if not entry.block.id:
            return
        pending = self.pending_reasoning_details.pop(entry.block.id, None)
        if pending:
            entry.block.thought_signature = pending

    def finish_all_blocks(self) -> None:
        entries_by_block = {id(entry.block): entry for entry in self.tool_calls_by_index.values()}
        for entry in self.tool_calls_by_id.values():
            entries_by_block.setdefault(id(entry.block), entry)

        for index, block in enumerate(self.output.content):
            if block.type == "text":
                self.event_stream.push(TextEndEvent(content_index=index, content=block.text, partial=self.output))
            elif block.type == "thinking":
                self.event_stream.push(
                    ThinkingEndEvent(content_index=index, content=block.thinking, partial=self.output)
                )
            elif block.type == "toolCall":
                entry = entries_by_block.get(id(block))
                if entry is not None:
                    block.arguments = parse_streaming_json(entry.partial_args)
                self.event_stream.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=self.output))


def _is_encrypted_reasoning_detail(detail: Any) -> bool:
    return (
        isinstance(detail, dict)
        and detail.get("type") == "reasoning.encrypted"
        and isinstance(detail.get("id"), str)
        and bool(detail["id"])
        and isinstance(detail.get("data"), str)
        and bool(detail["data"])
    )


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Stream with unified options, mapping ``reasoning`` to provider fields."""
    options = options or SimpleStreamOptions()
    get_client_api_key(model.provider, options.api_key, options.headers)

    base = build_base_options(model, context, options, options.api_key)
    clamped = clamp_thinking_level(model, options.reasoning) if options.reasoning else None
    reasoning_effort = None if clamped in (None, "off") else clamped

    completions_options = OpenAICompletionsOptions(
        **{key: getattr(base, key) for key in base.__dataclass_fields__},
    )
    completions_options.reasoning_effort = reasoning_effort
    completions_options.thinking_budgets = options.thinking_budgets
    completions_options.tool_choice = getattr(options, "tool_choice", None)

    return stream(model, context, completions_options, client=client)
