"""OpenAI Responses API provider.

Python port of `packages/ai/src/api/openai-responses.ts`. Speaks the Responses
HTTP API directly through :mod:`pi_ai.utils.http` rather than the official
`openai` SDK, mirroring the sibling `openai_completions.py` port.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx

from ..models import clamp_thinking_level
from ..types import (
    AssistantMessage,
    CacheRetention,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    ProviderResponse,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    Usage,
    now_ms,
)
from ..utils.deferred_tools import split_deferred_tools
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.headers import apply_header_overrides
from ..utils.http import HttpRequest, stream_sse
from ..utils.tasks import spawn
from .constrained_sampling import create_grammar_tool_input_properties
from .openai_completions import clamp_openai_prompt_cache_key, get_client_api_key, resolve_cache_retention
from .openai_responses_shared import (
    ConvertResponsesMessagesOptions,
    ConvertResponsesToolsOptions,
    OpenAIResponsesStreamOptions,
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from .simple_options import as_provider_options, build_base_options

OPENAI_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode", "openai-responses"}
# OpenAI Responses rejects max_output_tokens below 16: https://github.com/earendil-works/pi/issues/6265
OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16

_COMPAT_FIELDS = {
    "supportsDeveloperRole": "supports_developer_role",
    "sessionAffinityFormat": "session_affinity_format",
    "supportsLongCacheRetention": "supports_long_cache_retention",
    "supportsStrictMode": "supports_strict_mode",
    "supportsOpenAIGrammarTools": "supports_openai_grammar_tools",
    "supportsAdditionalTools": "supports_additional_tools",
    "supportsToolSearch": "supports_tool_search",
    "supportsExplicitPromptCacheMode": "supports_explicit_prompt_cache_mode",
}


@dataclass
class ResolvedResponsesCompat:
    """Compatibility settings resolved from provider/base URL plus model overrides."""

    supports_developer_role: bool = True
    session_affinity_format: str = "openai"
    supports_long_cache_retention: bool = True
    supports_strict_mode: bool = False
    supports_openai_grammar_tools: bool = False
    supports_additional_tools: bool = False
    supports_tool_search: bool = False
    supports_explicit_prompt_cache_mode: bool = False


def _detect_session_affinity_format(model: Model) -> str:
    return "openrouter" if model.provider == "openrouter" or "openrouter.ai" in model.base_url else "openai"


def detect_compat(model: Model) -> ResolvedResponsesCompat:
    return ResolvedResponsesCompat(session_affinity_format=_detect_session_affinity_format(model))


def get_compat(model: Model) -> ResolvedResponsesCompat:
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
    return resolved


def get_prompt_cache_retention(compat: ResolvedResponsesCompat, cache_retention: CacheRetention) -> str | None:
    return "24h" if cache_retention == "long" and compat.supports_long_cache_retention else None


def _format_openai_responses_error(error: BaseException) -> str:
    return format_provider_error(normalize_provider_error(error), "OpenAI API error")


@dataclass
class OpenAIResponsesOptions(StreamOptions):
    reasoning_effort: str | None = None
    """minimal | low | medium | high | xhigh | max"""
    reasoning_summary: str | None = None
    """auto | detailed | concise | None"""
    service_tier: str | None = None
    tool_choice: Any = None


def build_headers(
    model: Model,
    api_key: str,
    options: OpenAIResponsesOptions,
    compat: ResolvedResponsesCompat,
    session_id: str | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = dict(model.headers)
    headers.setdefault("content-type", "application/json")
    headers["authorization"] = f"Bearer {api_key}"

    if session_id:
        if compat.session_affinity_format == "openrouter":
            headers["x-session-id"] = session_id
        else:
            if compat.session_affinity_format == "openai":
                headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id

    return apply_header_overrides(headers, options.headers)


def build_params(
    model: Model,
    context: Context,
    options: OpenAIResponsesOptions | None = None,
    compat: ResolvedResponsesCompat | None = None,
    grammar_tool_input_properties: dict[str, str] | None = None,
) -> dict[str, Any]:
    options = as_provider_options(options, OpenAIResponsesOptions)
    compat = compat or get_compat(model)
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            context.tools, compat.supports_openai_grammar_tools
        )

    deferred_tools_mode = (
        "additional-tools"
        if compat.supports_additional_tools
        else "tool-search"
        if compat.supports_tool_search
        else None
    )
    tool_placement = split_deferred_tools(context, deferred_tools_mode is not None)
    messages = convert_responses_messages(
        model,
        context,
        OPENAI_TOOL_CALL_PROVIDERS,
        ConvertResponsesMessagesOptions(
            grammar_tool_input_properties=grammar_tool_input_properties,
            deferred_tools=tool_placement.deferred,
            deferred_tools_mode=deferred_tools_mode,
            tool_options=ConvertResponsesToolsOptions(
                supports_strict_mode=compat.supports_strict_mode,
                supports_openai_grammar_tools=compat.supports_openai_grammar_tools,
            ),
        ),
    )

    cache_retention = resolve_cache_retention(options.cache_retention, options.env)
    disable_implicit_prompt_cache = cache_retention == "none" and compat.supports_explicit_prompt_cache_mode

    params: dict[str, Any] = {
        "model": model.id,
        "input": messages,
        "stream": True,
        "store": False,
    }

    if cache_retention != "none":
        cache_key = clamp_openai_prompt_cache_key(options.session_id)
        if cache_key is not None:
            params["prompt_cache_key"] = cache_key
    retention = get_prompt_cache_retention(compat, cache_retention)
    if retention is not None:
        params["prompt_cache_retention"] = retention
    if disable_implicit_prompt_cache:
        params["prompt_cache_options"] = {"mode": "explicit"}

    if options.max_tokens:
        params["max_output_tokens"] = max(options.max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)

    if options.temperature is not None:
        params["temperature"] = options.temperature

    if options.service_tier is not None:
        params["service_tier"] = options.service_tier

    if tool_placement.immediate:
        params["tools"] = convert_responses_tools(
            tool_placement.immediate,
            ConvertResponsesToolsOptions(
                supports_strict_mode=compat.supports_strict_mode,
                supports_openai_grammar_tools=compat.supports_openai_grammar_tools,
            ),
        )

    if options.tool_choice is not None:
        params["tool_choice"] = options.tool_choice

    if model.reasoning:
        if options.reasoning_effort or options.reasoning_summary:
            if options.reasoning_effort:
                effort = model.thinking_level_map.get(options.reasoning_effort)
                if effort is None:
                    effort = options.reasoning_effort
            else:
                effort = "medium"
            params["reasoning"] = {"effort": effort, "summary": options.reasoning_summary or "auto"}
            params["include"] = ["reasoning.encrypted_content"]
        elif model.provider != "github-copilot" and not (
            "off" in model.thinking_level_map and model.thinking_level_map["off"] is None
        ):
            off_value = model.thinking_level_map.get("off")
            params["reasoning"] = {"effort": off_value if off_value is not None else "none"}
        if model.provider == "xai":
            params["include"] = ["reasoning.encrypted_content"]

    # Last so custom keys override the named request fields.
    if options.sampling_params:
        params.update(options.sampling_params)

    return params


def get_service_tier_cost_multiplier(model_id: str, service_tier: str | None) -> float:
    if service_tier == "flex":
        return 0.5
    if service_tier == "priority":
        return 2.5 if model_id == "gpt-5.5" else 2.0
    return 1.0


def apply_service_tier_pricing(usage: Usage, service_tier: str | None, model_id: str) -> None:
    multiplier = get_service_tier_cost_multiplier(model_id, service_tier)
    if multiplier == 1.0:
        return
    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cache_read *= multiplier
    usage.cost.cache_write *= multiplier
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write


def stream(
    model: Model,
    context: Context,
    options: OpenAIResponsesOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Stream a Responses API completion. Failures are reported through the stream."""
    event_stream = AssistantMessageEventStream()
    spawn(_run_stream(event_stream, model, context, options, client))
    return event_stream


async def _run_stream(
    event_stream: AssistantMessageEventStream,
    model: Model,
    context: Context,
    options: OpenAIResponsesOptions | None,
    client: httpx.AsyncClient | None,
) -> None:
    options = as_provider_options(options, OpenAIResponsesOptions)
    output = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason="pending",
        timestamp=now_ms(),
    )

    try:
        api_key = get_client_api_key(model.provider, options.api_key, options.headers)
        cache_retention = resolve_cache_retention(options.cache_retention, options.env)
        cache_session_id = options.session_id if cache_retention != "none" else None
        compat = get_compat(model)
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            context.tools, compat.supports_openai_grammar_tools
        )
        headers = build_headers(model, api_key, options, compat, cache_session_id)
        params = build_params(model, context, options, compat, grammar_tool_input_properties)

        if options.on_payload is not None:
            replacement = options.on_payload(params, model)
            if hasattr(replacement, "__await__"):
                replacement = await replacement
            if replacement is not None:
                params = replacement

        request = HttpRequest(
            url=f"{model.base_url.rstrip('/')}/responses",
            headers=headers,
            json_body=params,
            timeout_ms=options.timeout_ms,
        )

        started = False

        async def on_response(provider_response: ProviderResponse) -> None:
            nonlocal started
            if options.on_response is not None:
                result = options.on_response(provider_response, model)
                if hasattr(result, "__await__"):
                    await result
            if not started:
                event_stream.push(StartEvent(partial=output))
                started = True

        async def event_iterator():
            async for sse_event in stream_sse(request, client=client, on_response=on_response):
                try:
                    event = json.loads(sse_event.data)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    yield event

        await process_responses_stream(
            event_iterator(),
            output,
            event_stream,
            model,
            OpenAIResponsesStreamOptions(
                service_tier=options.service_tier,
                grammar_tool_input_properties=grammar_tool_input_properties,
                apply_service_tier_pricing=lambda usage, service_tier: apply_service_tier_pricing(
                    usage, service_tier, model.id
                ),
            ),
        )

        if not started:
            event_stream.push(StartEvent(partial=output))

        if options.signal is not None and options.signal.aborted:
            raise RuntimeError("Request was aborted")

        if output.stop_reason == "pending":
            raise RuntimeError("OpenAI Responses stream ended without a stop reason")
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
        aborted = options.signal is not None and options.signal.aborted
        output.stop_reason = "aborted" if aborted else "error"
        output.error_message = _format_openai_responses_error(error)
        event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
        event_stream.end()


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Stream with unified options, mapping ``reasoning`` to ``reasoning_effort``."""
    options = options or SimpleStreamOptions()
    get_client_api_key(model.provider, options.api_key, options.headers)

    base = build_base_options(model, context, options, options.api_key)
    clamped = clamp_thinking_level(model, options.reasoning) if options.reasoning else None
    reasoning_effort = None if clamped in (None, "off") else clamped

    responses_options = OpenAIResponsesOptions(**{key: getattr(base, key) for key in base.__dataclass_fields__})
    responses_options.reasoning_effort = reasoning_effort

    return stream(model, context, responses_options, client=client)
