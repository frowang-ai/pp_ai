"""Anthropic Messages provider.

Python port of `packages/ai/src/api/anthropic-messages.ts`. The TypeScript
version drives the official `@anthropic-ai/sdk` package and hand-rolls its own
SSE line decoder (`flushSseEvent`/`decodeSseLine`/`iterateSseMessages`) because
the SDK's stream reader is unsuitable for surfacing mid-stream `error` events
as part of the returned stream. This port speaks the Anthropic Messages HTTP
API directly through :mod:`pi_ai.utils.http`, whose shared `stream_sse`
decoder replaces the hand-rolled one.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import httpx

from ..models import calculate_cost
from ..types import (
    AssistantMessage,
    CacheRetention,
    Context,
    DoneEvent,
    ErrorEvent,
    ImageContent,
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
    ThinkingLevel,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    now_ms,
)
from ..utils.deferred_tools import split_deferred_tools
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.http import HttpRequest, stream_sse
from ..utils.json_parse import parse_json_with_repair, parse_streaming_json
from ..utils.pi_user_agent import get_pi_user_agent
from ..utils.provider_env import get_provider_env_value
from ..utils.sanitize_unicode import sanitize_surrogates
from ..utils.tasks import spawn
from .constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from .github_copilot_headers import build_copilot_dynamic_headers, has_copilot_vision_input
from .simple_options import (
    adjust_max_tokens_for_thinking,
    as_provider_options,
    build_base_options,
    clamp_max_tokens_to_context,
)
from .transform_messages import transform_messages

FINE_GRAINED_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"

AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]
AnthropicThinkingDisplay = Literal["summarized", "omitted"]


# --------------------------------------------------------------------------
# Cache retention
# --------------------------------------------------------------------------


def _anthropic_messages_base(base_url: str) -> str:
    """The `/messages` parent path for ``base_url``.

    The generated catalog stores the SDK-style base URL (no `/v1` suffix), so
    the version segment is added here. Base URLs that already end in `/v1` are
    left alone, which keeps hand-written catalogs working.
    """
    trimmed = base_url.rstrip("/")
    return trimmed if trimmed.endswith("/v1") else f"{trimmed}/v1"


def resolve_cache_retention(
    cache_retention: CacheRetention | None, env: dict[str, str] | None = None
) -> CacheRetention:
    """Resolve cache retention preference.

    Defaults to "short" and uses PI_CACHE_RETENTION for backward compatibility.
    """
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


@dataclass
class CacheControlResult:
    retention: CacheRetention
    cache_control: dict[str, str] | None = None


def get_cache_control(
    model: Model, cache_retention: CacheRetention | None = None, env: dict[str, str] | None = None
) -> CacheControlResult:
    retention = resolve_cache_retention(cache_retention, env)
    if retention == "none":
        return CacheControlResult(retention=retention)
    ttl = "1h" if retention == "long" and get_anthropic_compat(model).supports_long_cache_retention else None
    cache_control: dict[str, str] = {"type": "ephemeral"}
    if ttl:
        cache_control["ttl"] = ttl
    return CacheControlResult(retention=retention, cache_control=cache_control)


# --------------------------------------------------------------------------
# Claude Code stealth-mode tool naming
# --------------------------------------------------------------------------

CLAUDE_CODE_VERSION = "2.1.75"

# Claude Code 2.x tool names (canonical casing).
# Source: https://cchistory.mariozechner.at/data/prompts-2.1.11.md
# To update: https://github.com/badlogic/cchistory
CLAUDE_CODE_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
]

_CC_TOOL_LOOKUP = {name.lower(): name for name in CLAUDE_CODE_TOOLS}


def to_claude_code_name(name: str) -> str:
    """Convert a tool name to CC canonical casing if it matches (case-insensitive)."""
    return _CC_TOOL_LOOKUP.get(name.lower(), name)


def from_claude_code_name(name: str, tools: list[Tool] | None = None) -> str:
    if tools:
        lower_name = name.lower()
        for tool in tools:
            if tool.name.lower() == lower_name:
                return tool.name
    return name


# --------------------------------------------------------------------------
# Content block conversion
# --------------------------------------------------------------------------


def convert_content_blocks(content: list[TextContent | ImageContent]) -> str | list[dict[str, Any]]:
    """Convert content blocks to Anthropic API format."""
    has_images = any(block.type == "image" for block in content)
    if not has_images:
        return sanitize_surrogates("\n".join(block.text for block in content))  # type: ignore[union-attr]

    blocks: list[dict[str, Any]] = []
    for block in content:
        if block.type == "text":
            blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
        else:
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": block.mime_type, "data": block.data},
                }
            )

    has_text = any(b["type"] == "text" for b in blocks)
    if not has_text:
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})
    return blocks


# --------------------------------------------------------------------------
# Options and compat resolution
# --------------------------------------------------------------------------


@dataclass
class AnthropicOptions(StreamOptions):
    """Anthropic-specific request options."""

    thinking_enabled: bool | None = None
    thinking_budget_tokens: int | None = None
    effort: AnthropicEffort | None = None
    thinking_display: AnthropicThinkingDisplay | None = None
    interleaved_thinking: bool | None = None
    tool_choice: Any = None


@dataclass
class ResolvedAnthropicCompat:
    """Compatibility settings resolved from `model.compat`.

    Mirrors `Required<Omit<AnthropicMessagesCompat, "forceAdaptiveThinking">>`;
    `forceAdaptiveThinking` has no detected default and is read directly from
    `model.compat` wherever it is used (see `_force_adaptive_thinking`).
    """

    supports_eager_tool_input_streaming: bool = True
    supports_long_cache_retention: bool = True
    send_session_affinity_headers: bool = False
    supports_cache_control_on_tools: bool = True
    supports_temperature: bool = True
    allow_empty_signature: bool = False
    supports_strict_tools: bool = False
    supports_tool_references: bool = False


_ANTHROPIC_COMPAT_FIELDS = {
    "supportsEagerToolInputStreaming": "supports_eager_tool_input_streaming",
    "supportsLongCacheRetention": "supports_long_cache_retention",
    "sendSessionAffinityHeaders": "send_session_affinity_headers",
    "supportsCacheControlOnTools": "supports_cache_control_on_tools",
    "supportsTemperature": "supports_temperature",
    "allowEmptySignature": "allow_empty_signature",
    "supportsStrictTools": "supports_strict_tools",
    "supportsToolReferences": "supports_tool_references",
}

_TOOL_REFERENCE_VERSION_RE = re.compile(r"^claude-(?:opus|sonnet|fable)-(\d+)(?:-(\d+))?(?:-|$)")


def default_supports_tool_references(model: Model) -> bool:
    """Default for `supportsToolReferences`.

    First-party Anthropic models except Haiku (rejects client-side
    tool_reference blocks) and models that predate tool search (Claude 3.x,
    Opus/Sonnet 4.0, Opus 4.1).
    """
    if model.provider != "anthropic" or "haiku" in model.id:
        return False
    match = _TOOL_REFERENCE_VERSION_RE.match(model.id)
    if not match:
        return False
    major = int(match.group(1))
    minor_group = match.group(2)
    minor = int(minor_group) if minor_group and len(minor_group) < 8 else 0
    return major > 4 or (major == 4 and minor >= 5)


def _force_adaptive_thinking(model: Model) -> bool:
    compat = model.compat
    return compat.get("forceAdaptiveThinking", compat.get("force_adaptive_thinking")) is True


def detect_anthropic_compat(model: Model) -> ResolvedAnthropicCompat:
    return ResolvedAnthropicCompat(supports_tool_references=default_supports_tool_references(model))


def get_anthropic_compat(model: Model) -> ResolvedAnthropicCompat:
    """Detected settings overridden by explicit `model.compat` entries.

    Both the TypeScript camelCase keys and the Python snake_case names are
    accepted in `model.compat` so catalogs copied from the TypeScript project
    work unchanged. `forceAdaptiveThinking`/`force_adaptive_thinking` is not a
    field of this struct; read it via `_force_adaptive_thinking`.
    """
    resolved = detect_anthropic_compat(model)
    if not model.compat:
        return resolved
    for key, value in model.compat.items():
        if key in ("forceAdaptiveThinking", "force_adaptive_thinking"):
            continue
        attribute = _ANTHROPIC_COMPAT_FIELDS.get(key, key)
        if value is None:
            continue
        if hasattr(resolved, attribute):
            setattr(resolved, attribute, value)
    return resolved


# --------------------------------------------------------------------------
# Auth / header helpers
# --------------------------------------------------------------------------


def merge_headers(*header_sources: dict[str, str] | None) -> dict[str, str]:
    merged: dict[str, str] = {}
    for headers in header_sources:
        if headers:
            merged.update(headers)
    return merged


def merge_client_headers(model: Model[Any], *header_sources: dict[str, str] | None) -> dict[str, str]:
    """`merge_headers`, but Kimi Coding is identified as pi.

    The generated catalog gives Kimi models a `User-Agent: KimiCLI/1.5`, which
    claims to be a different client. Their API is served on that identity, so
    the header is replaced rather than merged -- case-insensitively, because
    the incoming spelling is whatever the catalog and provider supplied.
    """
    merged = merge_headers(*header_sources)
    if model.provider == "kimi-coding":
        for name in [key for key in merged if key.lower() == "user-agent"]:
            del merged[name]
        merged["User-Agent"] = get_pi_user_agent()
    return merged


def has_header(headers: dict[str, str | None] | None, name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    return any(key.lower() == expected and value is not None and value.strip() for key, value in headers.items())


def assert_request_auth(provider: str, api_key: str | None, headers: dict[str, str | None] | None) -> None:
    if api_key:
        return
    if (
        has_header(headers, "authorization")
        or has_header(headers, "x-api-key")
        or has_header(headers, "cf-aig-authorization")
    ):
        return
    raise ValueError(f"No API key for provider: {provider}")


def is_oauth_token(api_key: str) -> bool:
    return "sk-ant-oat" in api_key


def should_use_fine_grained_tool_streaming_beta(model: Model, context: Context) -> bool:
    return bool(context.tools) and not get_anthropic_compat(model).supports_eager_tool_input_streaming


def _apply_header_overrides(headers: dict[str, str], options_headers: dict[str, str | None] | None) -> dict[str, str]:
    if not options_headers:
        return headers
    result = dict(headers)
    for key, value in options_headers.items():
        if value is None:
            result.pop(key, None)
            result.pop(key.lower(), None)
        else:
            result[key] = value
    return result


def build_headers(
    model: Model,
    api_key: str | None,
    interleaved_thinking: bool,
    use_fine_grained_tool_streaming_beta: bool,
    options_headers: dict[str, str | None] | None = None,
    session_id: str | None = None,
    dynamic_headers: dict[str, str] | None = None,
) -> tuple[dict[str, str], bool]:
    """Build request headers and report whether `api_key` is an OAuth token.

    Mirrors `createClient`'s header construction in the TypeScript source
    (minus the client object itself, since this port speaks HTTP directly).
    """
    compat = get_anthropic_compat(model)
    needs_interleaved_beta = interleaved_thinking and not _force_adaptive_thinking(model)
    beta_features: list[str] = []
    if use_fine_grained_tool_streaming_beta:
        beta_features.append(FINE_GRAINED_TOOL_STREAMING_BETA)
    if needs_interleaved_beta:
        beta_features.append(INTERLEAVED_THINKING_BETA)

    if model.provider == "github-copilot":
        # Copilot: dynamic per-request headers, no x-api-key.
        headers = merge_client_headers(
            model,
            {
                "accept": "application/json",
                "anthropic-dangerous-direct-browser-access": "true",
                **({"anthropic-beta": ",".join(beta_features)} if beta_features else {}),
                **({"authorization": f"Bearer {api_key}"} if api_key else {}),
            },
            model.headers,
            dynamic_headers,
        )
        return _apply_header_overrides(headers, options_headers), False

    if api_key and is_oauth_token(api_key):
        # OAuth: send Claude Code identity headers.
        oauth_betas = ["claude-code-20250219", "oauth-2025-04-20", *beta_features]
        headers = merge_client_headers(
            model,
            {
                "accept": "application/json",
                "anthropic-dangerous-direct-browser-access": "true",
                "anthropic-beta": ",".join(oauth_betas),
                "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION}",
                "x-app": "cli",
                "authorization": f"Bearer {api_key}",
            },
            model.headers,
        )
        return _apply_header_overrides(headers, options_headers), True

    # API key or header-owned auth.
    session_affinity_headers: dict[str, str] = {}
    if session_id and compat.send_session_affinity_headers:
        session_affinity_headers = {"x-session-affinity": session_id}

    headers = merge_client_headers(
        model,
        {
            "accept": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
            **({"anthropic-beta": ",".join(beta_features)} if beta_features else {}),
        },
        session_affinity_headers,
        model.headers,
    )
    if api_key:
        headers["x-api-key"] = api_key
    return _apply_header_overrides(headers, options_headers), False


# --------------------------------------------------------------------------
# Tool call id normalization
# --------------------------------------------------------------------------

_TOOL_CALL_ID_INVALID_CHARS_RE = re.compile(r"[^a-zA-Z0-9_-]")


def normalize_tool_call_id(tool_call_id: str) -> str:
    """Normalize tool call ids to match Anthropic's required pattern and length."""
    return _TOOL_CALL_ID_INVALID_CHARS_RE.sub("_", tool_call_id)[:64]


# --------------------------------------------------------------------------
# Message conversion
# --------------------------------------------------------------------------


@dataclass
class ConvertedToolResult:
    tool_result: dict[str, Any]
    sibling_content: list[dict[str, Any]] = field(default_factory=list)


def convert_tool_result(
    msg: ToolResultMessage,
    is_oauth_token: bool,
    deferred_tool_names: set[str],
    loaded_tool_names: set[str],
    normalize_tool_name: Any = None,
) -> ConvertedToolResult:
    normalize_tool_name = normalize_tool_name or (lambda name: name)
    references: list[dict[str, str]] = []
    for name in msg.added_tool_names or []:
        normalized_name = normalize_tool_name(name)
        if normalized_name not in deferred_tool_names or normalized_name in loaded_tool_names:
            continue
        loaded_tool_names.add(normalized_name)
        references.append(
            {"type": "tool_reference", "tool_name": to_claude_code_name(name) if is_oauth_token else name}
        )

    converted_content = convert_content_blocks(msg.content)
    # Anthropic rejects tool references mixed with ordinary tool-result content.
    tool_result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": msg.tool_call_id,
        "content": references if references else converted_content,
        "is_error": msg.is_error,
    }
    if not references:
        sibling_content: list[dict[str, Any]] = []
    elif isinstance(converted_content, str):
        sibling_content = [{"type": "text", "text": converted_content}]
    else:
        sibling_content = converted_content
    return ConvertedToolResult(tool_result=tool_result, sibling_content=sibling_content)


def convert_messages(
    transformed_messages: list[Message],
    is_oauth_token: bool,
    cache_control: dict[str, str] | None = None,
    allow_empty_signature: bool = False,
    deferred_tool_names: set[str] | None = None,
    normalize_tool_name: Any = None,
) -> list[dict[str, Any]]:
    deferred_tool_names = deferred_tool_names or set()
    normalize_tool_name = normalize_tool_name or (lambda name: name)
    params: list[dict[str, Any]] = []
    loaded_tool_names: set[str] = set()

    index = 0
    count = len(transformed_messages)
    while index < count:
        msg = transformed_messages[index]

        if msg.role == "user":
            if isinstance(msg.content, str):
                if msg.content.strip():
                    params.append({"role": "user", "content": sanitize_surrogates(msg.content)})
            else:
                blocks: list[dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        blocks.append({"type": "text", "text": sanitize_surrogates(item.text)})
                    else:
                        blocks.append(
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": item.mime_type, "data": item.data},
                            }
                        )
                filtered_blocks = [b for b in blocks if b["type"] != "text" or b["text"].strip()]
                if filtered_blocks:
                    params.append({"role": "user", "content": filtered_blocks})
            index += 1
            continue

        if msg.role == "assistant":
            blocks = []
            for block in msg.content:
                if block.type == "text":
                    if not block.text.strip():
                        continue
                    blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
                elif block.type == "thinking":
                    if block.redacted:
                        blocks.append({"type": "redacted_thinking", "data": block.thinking_signature})
                        continue
                    signature = block.thinking_signature
                    has_signature = bool(signature and signature.strip())
                    if not block.thinking.strip() and not has_signature:
                        continue
                    if not has_signature:
                        # If thinking signature is missing/empty (e.g., from aborted
                        # stream), convert to plain text for Anthropic. Some compatible
                        # providers emit and accept empty signatures, so let marked
                        # models preserve the block.
                        if allow_empty_signature:
                            blocks.append(
                                {"type": "thinking", "thinking": sanitize_surrogates(block.thinking), "signature": ""}
                            )
                        else:
                            blocks.append({"type": "text", "text": sanitize_surrogates(block.thinking)})
                    else:
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": sanitize_surrogates(block.thinking),
                                "signature": signature,
                            }
                        )
                elif block.type == "toolCall":
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": to_claude_code_name(block.name) if is_oauth_token else block.name,
                            "input": block.arguments or {},
                        }
                    )
            if blocks:
                params.append({"role": "assistant", "content": blocks})
            index += 1
            continue

        if msg.role == "toolResult":
            # Collect all consecutive toolResult messages, needed for z.ai Anthropic endpoint.
            tool_results: list[dict[str, Any]] = []
            sibling_content: list[dict[str, Any]] = []
            cursor = index
            while cursor < count and transformed_messages[cursor].role == "toolResult":
                converted = convert_tool_result(
                    transformed_messages[cursor],  # type: ignore[arg-type]
                    is_oauth_token,
                    deferred_tool_names,
                    loaded_tool_names,
                    normalize_tool_name,
                )
                tool_results.append(converted.tool_result)
                sibling_content.extend(converted.sibling_content)
                cursor += 1
            index = cursor
            # Displaced reference-bearing results must follow every tool_result block.
            params.append({"role": "user", "content": [*tool_results, *sibling_content]})
            continue

        index += 1

    # Add cache_control to the last user message to cache conversation history.
    if cache_control and params:
        last_message = params[-1]
        if last_message["role"] == "user":
            content = last_message["content"]
            if isinstance(content, list) and content:
                last_block = content[-1]
                if last_block.get("type") in ("text", "image", "tool_result"):
                    last_block["cache_control"] = cache_control
            elif isinstance(content, str):
                last_message["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]

    return params


def convert_tools(
    tools: list[Tool],
    is_oauth_token: bool,
    supports_eager_tool_input_streaming: bool,
    supports_strict_tools: bool,
    cache_control: dict[str, str] | None = None,
    defer_loading: bool = False,
) -> list[dict[str, Any]]:
    if not tools:
        return []

    converted: list[dict[str, Any]] = []
    last_index = len(tools) - 1
    for index, tool in enumerate(tools):
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_tools)
        # `getJsonSchemaToolParameters` (`constrained-sampling.ts:129`): under
        # strict sampling the schema must first be narrowed to the strict
        # subset. Using the raw `tool.parameters` here sent providers a schema
        # they reject when strict mode is on.
        parameters = get_json_schema_tool_parameters(tool, strict)
        legacy_input_schema = {
            "type": "object",
            "properties": parameters.get("properties", {}),
            "required": parameters.get("required", []),
        }
        input_schema = {**parameters, **legacy_input_schema} if strict is True else legacy_input_schema

        entry: dict[str, Any] = {
            "name": to_claude_code_name(tool.name) if is_oauth_token else tool.name,
            "description": tool.description,
        }
        if supports_eager_tool_input_streaming:
            entry["eager_input_streaming"] = True
        if strict is True:
            entry["strict"] = True
        entry["input_schema"] = input_schema
        if defer_loading:
            entry["defer_loading"] = True
        if cache_control and index == last_index:
            entry["cache_control"] = cache_control
        converted.append(entry)
    return converted


# --------------------------------------------------------------------------
# Stop reason mapping
# --------------------------------------------------------------------------


def map_stop_reason(reason: str, stop_details: dict[str, Any] | None = None) -> tuple[StopReason, str | None]:
    if reason == "end_turn":
        return "stop", None
    if reason == "max_tokens":
        return "length", None
    if reason == "tool_use":
        return "toolUse", None
    if reason == "refusal":
        explanation = stop_details.get("explanation") if stop_details else None
        return "error", explanation or "The model refused to complete the request"
    if reason == "pause_turn":
        # Stop is good enough -> resubmit.
        return "stop", None
    if reason == "stop_sequence":
        # We don't supply stop sequences, so this should never happen.
        return "stop", None
    if reason == "sensitive":
        # Content flagged by safety filters (not yet in SDK types).
        return "error", "Provider stopped with: sensitive"
    # Handle unknown stop reasons gracefully (API may add new values).
    raise ValueError(f"Unhandled stop reason: {reason}")


# --------------------------------------------------------------------------
# build_params
# --------------------------------------------------------------------------


def build_params(
    model: Model,
    context: Context,
    is_oauth_token: bool,
    options: AnthropicOptions | None = None,
) -> dict[str, Any]:
    options = as_provider_options(options, AnthropicOptions)
    cache_result = get_cache_control(model, options.cache_retention, options.env)
    cache_control = cache_result.cache_control
    compat = get_anthropic_compat(model)
    transformed_messages = transform_messages(
        context.messages, model, lambda tool_call_id, m, msg: normalize_tool_call_id(tool_call_id)
    )
    normalize_tool_name = to_claude_code_name if is_oauth_token else (lambda name: name)

    tool_context = replace(context, messages=transformed_messages)
    tool_placement = split_deferred_tools(tool_context, compat.supports_tool_references, normalize_tool_name)
    immediate_tools = tool_placement.immediate
    deferred_tools = list(tool_placement.deferred.values())
    if not immediate_tools and deferred_tools:
        immediate_tools, deferred_tools = deferred_tools, []
    deferred_tool_names = {normalize_tool_name(tool.name) for tool in deferred_tools}

    params: dict[str, Any] = {
        "model": model.id,
        "messages": convert_messages(
            transformed_messages,
            is_oauth_token,
            cache_control,
            compat.allow_empty_signature,
            deferred_tool_names,
            normalize_tool_name,
        ),
        "max_tokens": options.max_tokens if options.max_tokens is not None else model.max_tokens,
        "stream": True,
    }

    # For OAuth tokens, we MUST include Claude Code identity.
    if is_oauth_token:
        system_entry: dict[str, Any] = {
            "type": "text",
            "text": "You are Claude Code, Anthropic's official CLI for Claude.",
        }
        if cache_control:
            system_entry["cache_control"] = cache_control
        params["system"] = [system_entry]
        if context.system_prompt:
            prompt_entry: dict[str, Any] = {"type": "text", "text": sanitize_surrogates(context.system_prompt)}
            if cache_control:
                prompt_entry["cache_control"] = cache_control
            params["system"].append(prompt_entry)
    elif context.system_prompt:
        # Add cache control to system prompt for non-OAuth tokens.
        prompt_entry = {"type": "text", "text": sanitize_surrogates(context.system_prompt)}
        if cache_control:
            prompt_entry["cache_control"] = cache_control
        params["system"] = [prompt_entry]

    # Temperature is incompatible with extended thinking and unsupported on Claude Opus 4.7+.
    if options.temperature is not None and not options.thinking_enabled and compat.supports_temperature:
        params["temperature"] = options.temperature

    if immediate_tools or deferred_tools:
        params["tools"] = [
            *convert_tools(
                immediate_tools,
                is_oauth_token,
                compat.supports_eager_tool_input_streaming,
                compat.supports_strict_tools,
                cache_control if compat.supports_cache_control_on_tools else None,
            ),
            *convert_tools(
                deferred_tools,
                is_oauth_token,
                compat.supports_eager_tool_input_streaming,
                compat.supports_strict_tools,
                None,
                True,
            ),
        ]

    # Configure thinking mode: adaptive, budget-based, or explicitly disabled.
    if model.reasoning:
        if options.thinking_enabled:
            # Default to "summarized" so Opus 4.7 and Mythos Preview behave like
            # older Claude 4 models (whose API default is also "summarized").
            display = options.thinking_display or "summarized"
            if _force_adaptive_thinking(model):
                # Adaptive thinking: Claude decides when and how much to think.
                params["thinking"] = {"type": "adaptive", "display": display}
                if options.effort:
                    params["output_config"] = {"effort": options.effort}
            else:
                # Budget-based thinking for older models.
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": options.thinking_budget_tokens or 1024,
                    "display": display,
                }
        elif options.thinking_enabled is False:
            has_off_key = "off" in model.thinking_level_map
            off_is_disabled = has_off_key and model.thinking_level_map["off"] is None
            if not off_is_disabled:
                params["thinking"] = {"type": "disabled"}

    if options.metadata:
        user_id = options.metadata.get("user_id")
        if isinstance(user_id, str):
            params["metadata"] = {"user_id": user_id}

    if options.tool_choice:
        if isinstance(options.tool_choice, str):
            params["tool_choice"] = {"type": options.tool_choice}
        else:
            params["tool_choice"] = options.tool_choice

    return params


# --------------------------------------------------------------------------
# streamSimple thinking-level mapping
# --------------------------------------------------------------------------


def map_thinking_level_to_effort(model: Model, level: ThinkingLevel | None) -> AnthropicEffort:
    """Map ThinkingLevel to Anthropic effort levels for adaptive thinking.

    Note: effort "max" is available on all adaptive-thinking Claude models,
    while native "xhigh" is only available on Opus 4.7/4.8, Sonnet 5, and
    Fable 5.
    """
    mapped = model.thinking_level_map.get(level) if level else None
    if isinstance(mapped, str):
        return mapped  # type: ignore[return-value]

    if level in ("minimal", "low"):
        return "low"
    if level == "medium":
        return "medium"
    if level == "high":
        return "high"
    return "high"


# --------------------------------------------------------------------------
# SSE decoding
# --------------------------------------------------------------------------

_ANTHROPIC_MESSAGE_EVENTS = frozenset(
    {
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    }
)


async def iterate_anthropic_events(
    request: HttpRequest, client: httpx.AsyncClient | None = None, on_response: Any = None
):
    """Decode a `POST /messages` SSE response into parsed Anthropic events.

    Errors (`ProviderHttpError` from a non-2xx response, or a mid-stream SSE
    `error` event) are raised out of this generator; callers report them
    through the returned :class:`AssistantMessageEventStream` instead of
    letting them propagate.
    """
    saw_message_start = False
    saw_message_stop = False
    async for sse_event in stream_sse(request, client=client, on_response=on_response):
        if sse_event.event == "error":
            raise RuntimeError(sse_event.data)

        if sse_event.event not in _ANTHROPIC_MESSAGE_EVENTS:
            continue

        try:
            event = parse_json_with_repair(sse_event.data)
        except ValueError as error:
            raise RuntimeError(
                f"Could not parse Anthropic SSE event {sse_event.event}: {error}; data={sse_event.data}"
            ) from error

        event_type = event.get("type") if isinstance(event, dict) else None
        if event_type == "message_start":
            saw_message_start = True
        elif event_type == "message_stop":
            saw_message_stop = True
        yield event

    if saw_message_start and not saw_message_stop:
        raise RuntimeError("Anthropic stream ended before message_stop")


# --------------------------------------------------------------------------
# Streaming state machine
# --------------------------------------------------------------------------


class _AnthropicStreamState:
    """Accumulates streamed Anthropic content blocks into `output.content`."""

    def __init__(
        self,
        event_stream: AssistantMessageEventStream,
        output: AssistantMessage,
        model: Model,
        is_oauth_token: bool,
        tools: list[Tool] | None,
    ) -> None:
        self.event_stream = event_stream
        self.output = output
        self.model = model
        self.is_oauth_token = is_oauth_token
        self.tools = tools
        self._blocks_by_index: dict[int, Any] = {}
        self._partial_json_by_index: dict[int, str] = {}

    def content_index(self, block: Any) -> int:
        for i, candidate in enumerate(self.output.content):
            if candidate is block:
                return i
        return -1

    def handle_message_start(self, event: dict[str, Any]) -> None:
        message = event.get("message") or {}
        self.output.response_id = message.get("id")
        usage = message.get("usage") or {}
        # Capture initial token usage from message_start event. This ensures we
        # have input token counts even if the stream is aborted early.
        self.output.usage.input = usage.get("input_tokens") or 0
        self.output.usage.output = usage.get("output_tokens") or 0
        self.output.usage.cache_read = usage.get("cache_read_input_tokens") or 0
        self.output.usage.cache_write = usage.get("cache_creation_input_tokens") or 0
        cache_creation = usage.get("cache_creation") or {}
        self.output.usage.cache_write_1h = cache_creation.get("ephemeral_1h_input_tokens") or 0
        self._recompute_total_tokens()

    def handle_content_block_start(self, event: dict[str, Any]) -> None:
        index = event["index"]
        content_block = event["content_block"]
        block_type = content_block.get("type")

        if block_type == "text":
            block: Any = TextContent(text=content_block.get("text") or "")
            self.output.content.append(block)
            self._blocks_by_index[index] = block
            self.event_stream.push(TextStartEvent(content_index=self.content_index(block), partial=self.output))
        elif block_type == "thinking":
            block = ThinkingContent(
                thinking=content_block.get("thinking") or "", thinking_signature=content_block.get("signature") or ""
            )
            self.output.content.append(block)
            self._blocks_by_index[index] = block
            self.event_stream.push(ThinkingStartEvent(content_index=self.content_index(block), partial=self.output))
        elif block_type == "redacted_thinking":
            block = ThinkingContent(
                thinking="[Reasoning redacted]", thinking_signature=content_block.get("data"), redacted=True
            )
            self.output.content.append(block)
            self._blocks_by_index[index] = block
            self.event_stream.push(ThinkingStartEvent(content_index=self.content_index(block), partial=self.output))
        elif block_type == "tool_use":
            name = content_block["name"]
            if self.is_oauth_token:
                name = from_claude_code_name(name, self.tools)
            block = ToolCall(id=content_block["id"], name=name, arguments=content_block.get("input") or {})
            self.output.content.append(block)
            self._blocks_by_index[index] = block
            self._partial_json_by_index[index] = ""
            self.event_stream.push(ToolCallStartEvent(content_index=self.content_index(block), partial=self.output))

    def handle_content_block_delta(self, event: dict[str, Any]) -> None:
        index = event["index"]
        block = self._blocks_by_index.get(index)
        if block is None:
            return
        delta = event["delta"]
        delta_type = delta.get("type")

        if delta_type == "text_delta" and block.type == "text":
            block.text += delta["text"]
            self.event_stream.push(
                TextDeltaEvent(content_index=self.content_index(block), delta=delta["text"], partial=self.output)
            )
        elif delta_type == "thinking_delta" and block.type == "thinking":
            block.thinking += delta["thinking"]
            self.event_stream.push(
                ThinkingDeltaEvent(
                    content_index=self.content_index(block), delta=delta["thinking"], partial=self.output
                )
            )
        elif delta_type == "input_json_delta" and block.type == "toolCall":
            self._partial_json_by_index[index] = self._partial_json_by_index.get(index, "") + delta["partial_json"]
            block.arguments = parse_streaming_json(self._partial_json_by_index[index])
            self.event_stream.push(
                ToolCallDeltaEvent(
                    content_index=self.content_index(block), delta=delta["partial_json"], partial=self.output
                )
            )
        elif delta_type == "signature_delta" and block.type == "thinking":
            block.thinking_signature = (block.thinking_signature or "") + delta["signature"]

    def handle_content_block_stop(self, event: dict[str, Any]) -> None:
        index = event["index"]
        block = self._blocks_by_index.pop(index, None)
        if block is None:
            return

        if block.type == "text":
            self.event_stream.push(
                TextEndEvent(content_index=self.content_index(block), content=block.text, partial=self.output)
            )
        elif block.type == "thinking":
            self.event_stream.push(
                ThinkingEndEvent(content_index=self.content_index(block), content=block.thinking, partial=self.output)
            )
        elif block.type == "toolCall":
            partial_json = self._partial_json_by_index.pop(index, "")
            block.arguments = parse_streaming_json(partial_json)
            self.event_stream.push(
                ToolCallEndEvent(content_index=self.content_index(block), tool_call=block, partial=self.output)
            )

    def handle_message_delta(self, event: dict[str, Any]) -> None:
        delta = event.get("delta") or {}
        stop_reason = delta.get("stop_reason")
        if stop_reason:
            self.output.raw_stop_reason = stop_reason
            mapped_reason, error_message = map_stop_reason(stop_reason, delta.get("stop_details"))
            self.output.stop_reason = mapped_reason
            if error_message:
                self.output.error_message = error_message

        # Only update usage fields if present (not null). Preserves input_tokens
        # from message_start when proxies omit it in message_delta.
        usage = event.get("usage")
        if usage:
            if usage.get("input_tokens") is not None:
                self.output.usage.input = usage["input_tokens"]
            if usage.get("output_tokens") is not None:
                self.output.usage.output = usage["output_tokens"]
            if usage.get("cache_read_input_tokens") is not None:
                self.output.usage.cache_read = usage["cache_read_input_tokens"]
            if usage.get("cache_creation_input_tokens") is not None:
                self.output.usage.cache_write = usage["cache_creation_input_tokens"]
            # Anthropic reports reasoning tokens in output_tokens_details.thinking_tokens
            # on the final message_delta usage (a subset of output_tokens).
            thinking_tokens = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
            if thinking_tokens is not None:
                self.output.usage.reasoning = thinking_tokens

        self._recompute_total_tokens()

    def _recompute_total_tokens(self) -> None:
        # Anthropic doesn't provide total_tokens, compute from components.
        usage = self.output.usage
        usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write
        calculate_cost(self.model, usage)


# --------------------------------------------------------------------------
# stream
# --------------------------------------------------------------------------


def stream(
    model: Model,
    context: Context,
    options: AnthropicOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Stream a Messages API completion. Failures are reported through the stream."""
    event_stream = AssistantMessageEventStream()
    spawn(_run_stream(event_stream, model, context, options, client))
    return event_stream


async def _run_stream(
    event_stream: AssistantMessageEventStream,
    model: Model,
    context: Context,
    options: AnthropicOptions | None,
    client: httpx.AsyncClient | None,
) -> None:
    options = as_provider_options(options, AnthropicOptions)
    output = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason="pending",
        timestamp=now_ms(),
    )

    try:
        assert_request_auth(model.provider, options.api_key, options.headers)

        cache_retention = resolve_cache_retention(options.cache_retention, options.env)
        cache_session_id = options.session_id if cache_retention != "none" else None

        copilot_dynamic_headers: dict[str, str] | None = None
        if model.provider == "github-copilot":
            has_images = has_copilot_vision_input(context.messages)
            copilot_dynamic_headers = build_copilot_dynamic_headers(context.messages, has_images)

        headers, is_oauth = build_headers(
            model,
            options.api_key,
            options.interleaved_thinking if options.interleaved_thinking is not None else True,
            should_use_fine_grained_tool_streaming_beta(model, context),
            options.headers,
            cache_session_id,
            copilot_dynamic_headers,
        )

        params = build_params(model, context, is_oauth, options)

        if options.on_payload is not None:
            replacement = options.on_payload(params, model)
            if hasattr(replacement, "__await__"):
                replacement = await replacement
            if replacement is not None:
                params = replacement

        request = HttpRequest(
            # TypeScript passes `model.baseUrl` to the Anthropic SDK as `baseURL`
            # and the SDK appends `/v1/messages`, so the generated catalog's base
            # URLs stop at the host prefix (`https://api.anthropic.com`,
            # `https://api.minimax.io/anthropic`, ...). A base URL that already
            # ends in `/v1` keeps working.
            url=f"{_anthropic_messages_base(model.base_url)}/messages",
            headers=headers,
            json_body=params,
            timeout_ms=options.timeout_ms,
        )

        on_response = None
        if options.on_response is not None:
            captured_on_response = options.on_response

            async def on_response(provider_response: ProviderResponse) -> None:
                result = captured_on_response(provider_response, model)
                if hasattr(result, "__await__"):
                    await result

        state = _AnthropicStreamState(event_stream, output, model, is_oauth, context.tools)
        started = False

        async for event in iterate_anthropic_events(request, client=client, on_response=on_response):
            if not started:
                event_stream.push(StartEvent(partial=output))
                started = True

            event_type = event.get("type")
            if event_type == "message_start":
                state.handle_message_start(event)
            elif event_type == "content_block_start":
                state.handle_content_block_start(event)
            elif event_type == "content_block_delta":
                state.handle_content_block_delta(event)
            elif event_type == "content_block_stop":
                state.handle_content_block_stop(event)
            elif event_type == "message_delta":
                state.handle_message_delta(event)
            # message_stop / ping carry no additional data.

        if not started:
            event_stream.push(StartEvent(partial=output))

        if options.signal is not None and options.signal.aborted:
            raise RuntimeError("Request was aborted")

        if output.stop_reason == "pending":
            raise RuntimeError("Anthropic stream ended without a stop reason")
        if output.stop_reason in ("aborted", "error"):
            raise RuntimeError(output.error_message or "An unknown error occurred")

        event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
        event_stream.end()
    except asyncio.CancelledError:
        output.stop_reason = "aborted"
        output.error_message = "Request was aborted"
        event_stream.push(ErrorEvent(reason="aborted", error=output))
        event_stream.end()
        raise
    except BaseException as error:
        aborted = bool(options is not None and options.signal is not None and options.signal.aborted)
        output.stop_reason = "aborted" if aborted else "error"
        output.error_message = format_provider_error(normalize_provider_error(error))
        event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
        event_stream.end()


# --------------------------------------------------------------------------
# stream_simple
# --------------------------------------------------------------------------


def _base_to_anthropic_options(base: StreamOptions) -> AnthropicOptions:
    return AnthropicOptions(**{key: getattr(base, key) for key in base.__dataclass_fields__})


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Stream with unified options, mapping `reasoning` to thinking mode."""
    options = options or SimpleStreamOptions()
    assert_request_auth(model.provider, options.api_key, options.headers)

    base = build_base_options(model, context, options, options.api_key)
    if not options.reasoning:
        anthropic_options = _base_to_anthropic_options(base)
        anthropic_options.thinking_enabled = False
        return stream(model, context, anthropic_options, client=client)

    # For models with adaptive thinking: use an effort level.
    # For older models: use budget-based thinking.
    if _force_adaptive_thinking(model):
        effort = map_thinking_level_to_effort(model, options.reasoning)
        anthropic_options = _base_to_anthropic_options(base)
        anthropic_options.thinking_enabled = True
        anthropic_options.effort = effort
        return stream(model, context, anthropic_options, client=client)

    # Undefined means the caller did not request an output cap; let the helper
    # use the model cap. Do not coerce to 0 here, or the thinking budget would
    # become the entire max_tokens value.
    adjusted = adjust_max_tokens_for_thinking(
        base.max_tokens, model.max_tokens, options.reasoning, options.thinking_budgets
    )
    max_tokens = clamp_max_tokens_to_context(model, context, adjusted.max_tokens)

    anthropic_options = _base_to_anthropic_options(base)
    anthropic_options.max_tokens = max_tokens
    anthropic_options.thinking_enabled = True
    anthropic_options.thinking_budget_tokens = min(adjusted.thinking_budget, max(0, max_tokens - 1024))
    return stream(model, context, anthropic_options, client=client)
