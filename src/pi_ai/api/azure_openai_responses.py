"""Azure OpenAI Responses API provider.

Python port of `packages/ai/src/api/azure-openai-responses.ts`. Differs from
`openai_responses.py` mainly in URL construction (Azure resource/deployment
name plus an `api-version` query parameter) and auth (`api-key` header instead
of `Authorization: Bearer`).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from ..models import clamp_thinking_level
from ..types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    ProviderResponse,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    now_ms,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.headers import apply_header_overrides
from ..utils.http import HttpRequest, stream_sse
from ..utils.provider_env import get_provider_env_value
from ..utils.tasks import spawn
from .constrained_sampling import create_grammar_tool_input_properties
from .openai_completions import clamp_openai_prompt_cache_key
from .openai_responses_shared import (
    ConvertResponsesMessagesOptions,
    ConvertResponsesToolsOptions,
    OpenAIResponsesStreamOptions,
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from .simple_options import build_base_options

DEFAULT_AZURE_API_VERSION = "v1"
AZURE_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode", "azure-openai-responses"}
# OpenAI Responses rejects max_output_tokens below 16: https://github.com/earendil-works/pi/issues/6265
OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16

_AZURE_COMPAT_FIELDS = {
    "supportsStrictMode": "supports_strict_mode",
    "supportsOpenAIGrammarTools": "supports_openai_grammar_tools",
}


@dataclass
class ResolvedAzureCompat:
    supports_strict_mode: bool = True
    supports_openai_grammar_tools: bool = False


def get_compat(model: Model) -> ResolvedAzureCompat:
    """Detected settings overridden by explicit ``model.compat`` entries.

    Both the TypeScript camelCase keys and the Python snake_case names are
    accepted in ``model.compat``.
    """
    resolved = ResolvedAzureCompat()
    if not model.compat:
        return resolved
    for key, value in model.compat.items():
        attribute = _AZURE_COMPAT_FIELDS.get(key, key)
        if value is None:
            continue
        if hasattr(resolved, attribute):
            setattr(resolved, attribute, value)
    return resolved


def _parse_deployment_name_map(value: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not value:
        return mapping
    for entry in value.split(","):
        trimmed = entry.strip()
        if not trimmed:
            continue
        # JavaScript's `trimmed.split("=", 2)` keeps only the first two
        # segments and drops the rest, rather than joining the remainder back.
        parts = trimmed.split("=")
        if len(parts) < 2:
            continue
        model_id, deployment_name = parts[0].strip(), parts[1].strip()
        if not model_id or not deployment_name:
            continue
        mapping[model_id] = deployment_name
    return mapping


def _format_azure_openai_error(error: BaseException) -> str:
    return format_provider_error(normalize_provider_error(error), "Azure OpenAI API error")


@dataclass
class AzureOpenAIResponsesOptions(StreamOptions):
    reasoning_effort: str | None = None
    """minimal | low | medium | high | xhigh | max"""
    reasoning_summary: str | None = None
    """auto | detailed | concise | None"""
    azure_api_version: str | None = None
    azure_resource_name: str | None = None
    azure_base_url: str | None = None
    azure_deployment_name: str | None = None


def resolve_deployment_name(model: Model, options: AzureOpenAIResponsesOptions | None = None) -> str:
    if options is not None and options.azure_deployment_name:
        return options.azure_deployment_name
    mapped = _parse_deployment_name_map(
        get_provider_env_value("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", options.env if options else None)
    ).get(model.id)
    return mapped or model.id


def _normalize_azure_base_url(base_url: str) -> str:
    trimmed = base_url.strip().rstrip("/")
    parts = urlsplit(trimmed)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"Invalid Azure OpenAI base URL: {base_url}")

    hostname = parts.hostname or ""
    is_azure_host = (
        hostname.endswith(".openai.azure.com")
        or hostname.endswith(".cognitiveservices.azure.com")
        or hostname.endswith(".ai.azure.com")
    )
    normalized_path = parts.path.rstrip("/")

    # Ensure Azure hosts have /openai/v1 as base path so requests can append
    # /deployments/<model>/... and ?api-version=v1 correctly.
    if is_azure_host and normalized_path in ("", "/", "/openai", "/openai/v1/responses"):
        parts = parts._replace(path="/openai/v1", query="")

    return urlunsplit(parts).rstrip("/")


def _build_default_base_url(resource_name: str) -> str:
    return f"https://{resource_name}.openai.azure.com/openai/v1"


def _resolve_azure_config(model: Model, options: AzureOpenAIResponsesOptions | None = None) -> tuple[str, str]:
    env = options.env if options else None
    api_version = (
        (options.azure_api_version if options else None)
        or get_provider_env_value("AZURE_OPENAI_API_VERSION", env)
        or DEFAULT_AZURE_API_VERSION
    )

    azure_base_url_opt = options.azure_base_url.strip() if options and options.azure_base_url else None
    env_base_url = get_provider_env_value("AZURE_OPENAI_BASE_URL", env)
    env_base_url_trimmed = env_base_url.strip() if env_base_url else None
    base_url = azure_base_url_opt or env_base_url_trimmed or None

    resource_name = (options.azure_resource_name if options else None) or get_provider_env_value(
        "AZURE_OPENAI_RESOURCE_NAME", env
    )

    resolved_base_url = base_url
    if not resolved_base_url and resource_name:
        resolved_base_url = _build_default_base_url(resource_name)
    if not resolved_base_url and model.base_url:
        resolved_base_url = model.base_url
    if not resolved_base_url:
        raise ValueError(
            "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, "
            "or pass azure_base_url, azure_resource_name, or model.base_url."
        )

    return _normalize_azure_base_url(resolved_base_url), api_version


def build_headers(model: Model, api_key: str, options: AzureOpenAIResponsesOptions | None = None) -> dict[str, str]:
    headers: dict[str, str] = dict(model.headers)
    headers.setdefault("content-type", "application/json")
    headers["api-key"] = api_key

    return apply_header_overrides(headers, options.headers if options is not None else None)


def build_params(
    model: Model,
    context: Context,
    options: AzureOpenAIResponsesOptions | None = None,
    deployment_name: str | None = None,
    grammar_tool_input_properties: dict[str, str] | None = None,
) -> dict[str, Any]:
    compat = get_compat(model)
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            context.tools, compat.supports_openai_grammar_tools
        )

    messages = convert_responses_messages(
        model,
        context,
        AZURE_TOOL_CALL_PROVIDERS,
        ConvertResponsesMessagesOptions(grammar_tool_input_properties=grammar_tool_input_properties),
    )

    params: dict[str, Any] = {
        "model": deployment_name or model.id,
        "input": messages,
        "stream": True,
        "store": False,
    }

    cache_key = clamp_openai_prompt_cache_key(options.session_id if options else None)
    if cache_key is not None:
        params["prompt_cache_key"] = cache_key

    if options is not None and options.max_tokens:
        params["max_output_tokens"] = max(options.max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)

    if options is not None and options.temperature is not None:
        params["temperature"] = options.temperature

    if context.tools:
        params["tools"] = convert_responses_tools(
            context.tools,
            ConvertResponsesToolsOptions(
                supports_strict_mode=compat.supports_strict_mode,
                supports_openai_grammar_tools=compat.supports_openai_grammar_tools,
            ),
        )

    if model.reasoning:
        reasoning_effort = options.reasoning_effort if options else None
        reasoning_summary = options.reasoning_summary if options else None
        if reasoning_effort or reasoning_summary:
            if reasoning_effort:
                effort = model.thinking_level_map.get(reasoning_effort)
                if effort is None:
                    effort = reasoning_effort
            else:
                effort = "medium"
            params["reasoning"] = {"effort": effort, "summary": reasoning_summary or "auto"}
            params["include"] = ["reasoning.encrypted_content"]
        elif not ("off" in model.thinking_level_map and model.thinking_level_map["off"] is None):
            off_value = model.thinking_level_map.get("off")
            params["reasoning"] = {"effort": off_value if off_value is not None else "none"}

    # Last so custom keys override the named request fields.
    if options is not None and options.sampling_params:
        params.update(options.sampling_params)

    return params


def stream(
    model: Model,
    context: Context,
    options: AzureOpenAIResponsesOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Stream an Azure Responses API completion. Failures are reported through the stream."""
    event_stream = AssistantMessageEventStream()
    spawn(_run_stream(event_stream, model, context, options, client))
    return event_stream


async def _run_stream(
    event_stream: AssistantMessageEventStream,
    model: Model,
    context: Context,
    options: AzureOpenAIResponsesOptions | None,
    client: httpx.AsyncClient | None,
) -> None:
    deployment_name = resolve_deployment_name(model, options)
    output = AssistantMessage(
        api="azure-openai-responses",
        provider=model.provider,
        model=model.id,
        stop_reason="pending",
        timestamp=now_ms(),
    )

    try:
        api_key = options.api_key if options else None
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")

        base_url, api_version = _resolve_azure_config(model, options)
        headers = build_headers(model, api_key, options)
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            context.tools, get_compat(model).supports_openai_grammar_tools
        )
        params = build_params(model, context, options, deployment_name, grammar_tool_input_properties)

        if options is not None and options.on_payload is not None:
            replacement = options.on_payload(params, model)
            if hasattr(replacement, "__await__"):
                replacement = await replacement
            if replacement is not None:
                params = replacement

        url = f"{base_url}/deployments/{quote(deployment_name, safe='')}/responses?api-version={quote(api_version, safe='')}"
        request = HttpRequest(
            url=url,
            headers=headers,
            json_body=params,
            timeout_ms=options.timeout_ms if options else None,
        )

        started = False

        async def on_response(provider_response: ProviderResponse) -> None:
            nonlocal started
            if options is not None and options.on_response is not None:
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
            OpenAIResponsesStreamOptions(grammar_tool_input_properties=grammar_tool_input_properties),
        )

        if not started:
            event_stream.push(StartEvent(partial=output))

        if options is not None and options.signal is not None and options.signal.aborted:
            raise RuntimeError("Request was aborted")

        if output.stop_reason == "pending":
            raise RuntimeError("Azure OpenAI Responses stream ended without a stop reason")
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
        output.error_message = _format_azure_openai_error(error)
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
    api_key = options.api_key
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, context, options, api_key)
    clamped = clamp_thinking_level(model, options.reasoning) if options.reasoning else None
    reasoning_effort = None if clamped in (None, "off") else clamped

    azure_options = AzureOpenAIResponsesOptions(**{key: getattr(base, key) for key in base.__dataclass_fields__})
    azure_options.reasoning_effort = reasoning_effort

    return stream(model, context, azure_options, client=client)
