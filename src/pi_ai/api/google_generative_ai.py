"""Google Generative AI (Gemini) provider.

Python port of `packages/ai/src/api/google-generative-ai.ts`. The TypeScript
version drives the `@google/genai` SDK; this port speaks the Gemini REST API
directly through :mod:`pi_ai.utils.http`:

    POST {base_url}/models/{model_id}:streamGenerateContent?alt=sse
    x-goog-api-key: <api key>

`base_url` defaults to `https://generativelanguage.googleapis.com/v1beta`
(the same default the `google` provider factory stamps onto its models).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, fields
from typing import Any, Literal

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
    ThinkingBudgets,
    now_ms,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.headers import provider_headers_to_record
from ..utils.http import HttpRequest
from ..utils.sanitize_unicode import sanitize_surrogates
from ..utils.tasks import spawn
from .google_shared import (
    GoogleStreamState,
    GoogleThinkingLevel,
    convert_messages,
    convert_tools,
    is_gemini3_flash_model,
    is_gemini3_pro_model,
    iterate_google_chunks,
    resolve_google_function_calling_mode,
    supports_google_strict_tool_sampling,
)
from .simple_options import as_provider_options, build_base_options

GoogleToolChoice = Literal["auto", "none", "any"]


@dataclass
class GoogleThinkingOptions:
    enabled: bool
    budget_tokens: int | None = None
    """-1 for dynamic, 0 to disable."""
    level: GoogleThinkingLevel | None = None


@dataclass
class GoogleOptions(StreamOptions):
    """Google Generative AI request options."""

    tool_choice: GoogleToolChoice | None = None
    thinking: GoogleThinkingOptions | None = None


_GEMMA_RE = re.compile(r"gemma-?4")


def _is_gemma4_model(model_id: str) -> bool:
    return _GEMMA_RE.search(model_id.lower()) is not None


# --------------------------------------------------------------------------
# Request construction
# --------------------------------------------------------------------------


def build_headers(model: Model, api_key: str, options_headers: dict[str, str | None] | None = None) -> dict[str, str]:
    headers = dict(model.headers)
    override = provider_headers_to_record(options_headers)
    if override:
        headers.update(override)
    headers["x-goog-api-key"] = api_key
    return headers


def build_url(model: Model) -> str:
    base_url = (model.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    return f"{base_url}/models/{model.id}:streamGenerateContent?alt=sse"


def build_params(model: Model, context: Context, options: GoogleOptions | None = None) -> dict[str, Any]:
    options = as_provider_options(options, GoogleOptions)
    contents = convert_messages(model, context)

    generation_config: dict[str, Any] = {}
    if options.temperature is not None:
        generation_config["temperature"] = options.temperature
    if options.max_tokens is not None:
        generation_config["maxOutputTokens"] = options.max_tokens

    body: dict[str, Any] = {"contents": contents}
    if context.system_prompt:
        body["systemInstruction"] = {"parts": [{"text": sanitize_surrogates(context.system_prompt)}]}
    if context.tools:
        tools = convert_tools(context.tools, False, supports_google_strict_tool_sampling(model.id))
        if tools:
            body["tools"] = tools
        function_calling_mode = resolve_google_function_calling_mode(
            context.tools, options.tool_choice, supports_google_strict_tool_sampling(model.id)
        )
        if function_calling_mode is not None:
            body["toolConfig"] = {"functionCallingConfig": {"mode": function_calling_mode}}

    if options.thinking is not None and options.thinking.enabled and model.reasoning:
        thinking_config: dict[str, Any] = {"includeThoughts": True}
        if options.thinking.level is not None:
            thinking_config["thinkingLevel"] = options.thinking.level
        elif options.thinking.budget_tokens is not None:
            thinking_config["thinkingBudget"] = options.thinking.budget_tokens
        generation_config["thinkingConfig"] = thinking_config
    elif model.reasoning and options.thinking is not None and not options.thinking.enabled:
        generation_config["thinkingConfig"] = _get_disabled_thinking_config(model)

    if generation_config:
        body["generationConfig"] = generation_config

    if options.signal is not None and options.signal.aborted:
        raise RuntimeError("Request aborted")

    return body


ClampedThinkingLevel = Literal["minimal", "low", "medium", "high"]


def _get_disabled_thinking_config(model: Model) -> dict[str, Any]:
    # Google docs: Gemini 3.1 Pro cannot disable thinking, and Gemini 3 Flash /
    # Flash-Lite do not support full thinking-off either. For Gemini 3 models,
    # use the lowest supported thinkingLevel without includeThoughts so hidden
    # thinking remains invisible to pi.
    if is_gemini3_pro_model(model.id):
        return {"thinkingLevel": "LOW"}
    if is_gemini3_flash_model(model.id):
        return {"thinkingLevel": "MINIMAL"}
    if _is_gemma4_model(model.id):
        return {"thinkingLevel": "MINIMAL"}
    # Gemini 2.x supports disabling via thinkingBudget = 0.
    return {"thinkingBudget": 0}


def _get_thinking_level(effort: ClampedThinkingLevel, model: Model) -> GoogleThinkingLevel:
    if is_gemini3_pro_model(model.id):
        return "LOW" if effort in ("minimal", "low") else "HIGH"
    if _is_gemma4_model(model.id):
        return "MINIMAL" if effort in ("minimal", "low") else "HIGH"
    return {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}[effort]


def _get_google_budget(model: Model, effort: ClampedThinkingLevel, custom_budgets: ThinkingBudgets | None) -> int:
    if custom_budgets is not None:
        override = getattr(custom_budgets, effort, None)
        if override is not None:
            return override

    if "2.5-pro" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}[effort]
    if "2.5-flash-lite" in model.id:
        return {"minimal": 512, "low": 2048, "medium": 8192, "high": 24576}[effort]
    if "2.5-flash" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}[effort]
    return -1


# --------------------------------------------------------------------------
# stream
# --------------------------------------------------------------------------


def stream(
    model: Model,
    context: Context,
    options: GoogleOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Stream a `streamGenerateContent` completion. Failures are reported through the stream."""
    event_stream = AssistantMessageEventStream()
    spawn(_run_stream(event_stream, model, context, options, client))
    return event_stream


async def _run_stream(
    event_stream: AssistantMessageEventStream,
    model: Model,
    context: Context,
    options: GoogleOptions | None,
    client: httpx.AsyncClient | None,
) -> None:
    options = as_provider_options(options, GoogleOptions)
    output = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason="pending",
        timestamp=now_ms(),
    )

    try:
        api_key = options.api_key
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")

        params = build_params(model, context, options)

        if options.on_payload is not None:
            replacement = options.on_payload(params, model)
            if hasattr(replacement, "__await__"):
                replacement = await replacement
            if replacement is not None:
                params = replacement

        request = HttpRequest(
            url=build_url(model),
            headers=build_headers(model, api_key, options.headers),
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

        state = GoogleStreamState(event_stream, output, model)
        started = False

        async for chunk in iterate_google_chunks(request, client=client, on_response=on_response, options=options):
            if not started:
                event_stream.push(StartEvent(partial=output))
                started = True
            state.handle_chunk(chunk)

        if not started:
            event_stream.push(StartEvent(partial=output))

        state.finalize()

        if options.signal is not None and options.signal.aborted:
            raise RuntimeError("Request was aborted")

        if output.stop_reason == "pending":
            raise RuntimeError("Google stream ended without a finish reason")
        if output.stop_reason in ("aborted", "error"):
            error_message = (
                f"Provider stopped with: {output.raw_stop_reason}"
                if output.raw_stop_reason
                else "An unknown error occurred"
            )
            raise RuntimeError(error_message)

        event_stream.push(DoneEvent(reason=output.stop_reason, message=output))  # type: ignore[arg-type]
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
        event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))  # type: ignore[arg-type]
        event_stream.end()


# --------------------------------------------------------------------------
# stream_simple
# --------------------------------------------------------------------------


def _base_to_google_options(base: StreamOptions, **overrides: Any) -> GoogleOptions:
    values = {f.name: getattr(base, f.name) for f in fields(base)}
    values.update(overrides)
    return GoogleOptions(**values)


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Stream with unified options, mapping `reasoning` to a Gemini thinking config."""
    options = options or SimpleStreamOptions()
    api_key = options.api_key
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, context, options, api_key)
    if not options.reasoning:
        return stream(
            model, context, _base_to_google_options(base, thinking=GoogleThinkingOptions(enabled=False)), client=client
        )

    clamped_reasoning = clamp_thinking_level(model, options.reasoning)
    effort: ClampedThinkingLevel = "high" if clamped_reasoning == "off" else clamped_reasoning  # type: ignore[assignment]

    if is_gemini3_pro_model(model.id) or is_gemini3_flash_model(model.id) or _is_gemma4_model(model.id):
        return stream(
            model,
            context,
            _base_to_google_options(
                base, thinking=GoogleThinkingOptions(enabled=True, level=_get_thinking_level(effort, model))
            ),
            client=client,
        )

    return stream(
        model,
        context,
        _base_to_google_options(
            base,
            thinking=GoogleThinkingOptions(
                enabled=True, budget_tokens=_get_google_budget(model, effort, options.thinking_budgets)
            ),
        ),
        client=client,
    )
