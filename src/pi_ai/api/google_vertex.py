"""Google Vertex AI provider.

Python port of `packages/ai/src/api/google-vertex.ts`. The TypeScript version
drives the `@google/genai` SDK, which authenticates with Application Default
Credentials (ADC) - a Google-auth-library flow that can read a user's `gcloud
auth application-default login` refresh token or a service account JSON key
and mint short-lived OAuth access tokens, refreshing them automatically for
the whole SDK lifetime.

This port speaks the Vertex REST API directly through
:mod:`pi_ai.utils.http` and does **not** implement ADC. Minting an OAuth
access token from a service-account private key requires RS256 JWT signing
(the `iat`/`exp`/`scope` JWT-bearer grant, RFC 7523), which needs an
asymmetric-crypto library this workspace does not depend on (the standard
library has no RSA "sign arbitrary bytes" primitive); shelling out to an
external `openssl` process to sign the assertion would put credential
material through an untrusted subprocess boundary, which is worse than not
supporting the flow. So this port implements only:

  * An explicit Google Cloud **API key** (`GoogleVertexOptions.api_key`,
    resolved the same way as the TypeScript `resolveApiKey`/env
    `GOOGLE_CLOUD_API_KEY`) - sent as `x-goog-api-key` against the
    project/location-free "Express Mode" endpoint
    (`https://aiplatform.googleapis.com/v1/publishers/google/models/...`).
  * An explicit, already-minted OAuth **access token**
    (`GoogleVertexOptions.access_token`, or env `GOOGLE_VERTEX_ACCESS_TOKEN` -
    a pp-only addition, since the TypeScript version has no equivalent
    option and expects ADC to produce this token internally) - sent as
    `Authorization: Bearer <token>` against the regional endpoint
    `https://{location}-aiplatform.googleapis.com/v1/projects/{project}/
    locations/{location}/publishers/google/models/...`.

If neither is configured, `stream`/`stream_simple` raise a `ValueError`
telling the caller to supply one of these two, explaining that full ADC
(including reading `GOOGLE_APPLICATION_CREDENTIALS`) is out of scope for this
port.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, fields
from typing import Any, Literal
from urllib.parse import urlparse

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
from ..utils.provider_env import get_provider_env_value
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

API_VERSION = "v1"
GCP_VERTEX_CREDENTIALS_MARKER = "gcp-vertex-credentials"

GoogleToolChoice = Literal["auto", "none", "any"]

ClampedThinkingLevel = Literal["minimal", "low", "medium", "high"]


@dataclass
class GoogleThinkingOptions:
    enabled: bool
    budget_tokens: int | None = None
    """-1 for dynamic, 0 to disable."""
    level: GoogleThinkingLevel | None = None


@dataclass
class GoogleVertexOptions(StreamOptions):
    """Google Vertex request options."""

    tool_choice: GoogleToolChoice | None = None
    thinking: GoogleThinkingOptions | None = None
    project: str | None = None
    location: str | None = None
    access_token: str | None = None
    """Pre-obtained OAuth access token. See module docstring: this port does
    not implement Application Default Credentials, so this (or
    `GOOGLE_VERTEX_ACCESS_TOKEN`) is required when not using an API key."""


# --------------------------------------------------------------------------
# Auth / URL resolution
# --------------------------------------------------------------------------

_PLACEHOLDER_API_KEY_RE = re.compile(r"^<[^>]+>$")


def resolve_api_key(options: GoogleVertexOptions | None) -> str | None:
    api_key = (options.api_key if options else None) or None
    if api_key:
        api_key = api_key.strip()
    if not api_key or api_key == GCP_VERTEX_CREDENTIALS_MARKER or _PLACEHOLDER_API_KEY_RE.match(api_key):
        return None
    return api_key


def resolve_project(options: GoogleVertexOptions | None) -> str:
    project = (
        (options.project if options else None)
        or get_provider_env_value("GOOGLE_CLOUD_PROJECT", options.env if options else None)
        or get_provider_env_value("GCLOUD_PROJECT", options.env if options else None)
    )
    if not project:
        raise ValueError(
            "Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
        )
    return project


def resolve_location(options: GoogleVertexOptions | None) -> str:
    location = (options.location if options else None) or get_provider_env_value(
        "GOOGLE_CLOUD_LOCATION", options.env if options else None
    )
    if not location:
        raise ValueError("Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.")
    return location


def resolve_access_token(options: GoogleVertexOptions | None) -> str:
    access_token = (options.access_token if options else None) or get_provider_env_value(
        "GOOGLE_VERTEX_ACCESS_TOKEN", options.env if options else None
    )
    if access_token:
        return access_token

    adc_path = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", options.env if options else None)
    if adc_path:
        raise ValueError(
            "Vertex AI requires an OAuth access token, and this port does not implement Application "
            f"Default Credentials (found GOOGLE_APPLICATION_CREDENTIALS={adc_path!r}, but minting a token "
            "from it is out of scope - see the module docstring). Provide a pre-obtained access token via "
            "GoogleVertexOptions.access_token or the GOOGLE_VERTEX_ACCESS_TOKEN env var, or use a Google "
            "Cloud API key instead (GOOGLE_CLOUD_API_KEY)."
        )
    raise ValueError(
        "Vertex AI requires an OAuth access token. Set GOOGLE_VERTEX_ACCESS_TOKEN, pass access_token in "
        "options, or provide a Google Cloud API key (GOOGLE_CLOUD_API_KEY) instead."
    )


def _resolve_custom_base_url(base_url: str) -> str | None:
    trimmed = (base_url or "").strip()
    if not trimmed or "{location}" in trimmed:
        return None
    return trimmed


def _base_url_includes_api_version(base_url: str) -> bool:
    try:
        parsed = urlparse(base_url)
        path = parsed.path
    except ValueError:
        path = base_url
    if not path or (path == base_url and "://" not in base_url):
        return bool(re.search(r"(?:^|/)v\d+(?:beta\d*)?(?:/|$)", base_url))
    return any(re.match(r"^v\d+(?:beta\d*)?$", part) for part in path.split("/"))


def build_url_and_headers(model: Model, options: GoogleVertexOptions | None) -> tuple[str, dict[str, str]]:
    options = as_provider_options(options, GoogleVertexOptions)
    headers = dict(model.headers)
    override = provider_headers_to_record(options.headers)
    if override:
        headers.update(override)

    api_key = resolve_api_key(options)
    custom_base_url = _resolve_custom_base_url(model.base_url)
    # A custom base URL is treated as already scoped to the resource collection
    # (`ResourceScope.COLLECTION` in the TypeScript adapter's httpOptions), so the
    # `projects/<project>/locations/<location>` segment is not prepended to it. A
    # custom base URL that already carries an API version also suppresses the
    # version segment (TypeScript sets `httpOptions.apiVersion = ""`).
    version_segment = "" if custom_base_url and _base_url_includes_api_version(custom_base_url) else f"/{API_VERSION}"

    if api_key:
        base_url = custom_base_url or "https://aiplatform.googleapis.com"
        url = (
            f"{base_url.rstrip('/')}{version_segment}/publishers/google/models/{model.id}:streamGenerateContent?alt=sse"
        )
        headers["x-goog-api-key"] = api_key
        return url, headers

    project = resolve_project(options)
    location = resolve_location(options)
    access_token = resolve_access_token(options)

    if custom_base_url:
        base_url = custom_base_url
        resource_path = ""
    else:
        base_url = (
            "https://aiplatform.googleapis.com"
            if location == "global"
            else f"https://{location}-aiplatform.googleapis.com"
        )
        resource_path = f"/projects/{project}/locations/{location}"
    url = (
        f"{base_url.rstrip('/')}{version_segment}{resource_path}"
        f"/publishers/google/models/{model.id}:streamGenerateContent?alt=sse"
    )
    headers["authorization"] = f"Bearer {access_token}"
    return url, headers


# --------------------------------------------------------------------------
# Request body construction
# --------------------------------------------------------------------------


def build_params(model: Model, context: Context, options: GoogleVertexOptions | None = None) -> dict[str, Any]:
    options = as_provider_options(options, GoogleVertexOptions)
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


def _get_disabled_thinking_config(model: Model) -> dict[str, Any]:
    # Google docs: Gemini 3.1 Pro cannot disable thinking, and Gemini 3 Flash /
    # Flash-Lite do not support full thinking-off either. For Gemini 3 models,
    # use the lowest supported thinkingLevel without includeThoughts so hidden
    # thinking remains invisible to pi.
    if is_gemini3_pro_model(model.id):
        return {"thinkingLevel": "LOW"}
    if is_gemini3_flash_model(model.id):
        return {"thinkingLevel": "MINIMAL"}
    # Gemini 2.x supports disabling via thinkingBudget = 0.
    return {"thinkingBudget": 0}


def _get_gemini3_thinking_level(effort: ClampedThinkingLevel, model: Model) -> GoogleThinkingLevel:
    if is_gemini3_pro_model(model.id):
        return "LOW" if effort in ("minimal", "low") else "HIGH"
    return {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}[effort]


def _get_google_budget(model: Model, effort: ClampedThinkingLevel, custom_budgets: ThinkingBudgets | None) -> int:
    if custom_budgets is not None:
        override = getattr(custom_budgets, effort, None)
        if override is not None:
            return override

    if "2.5-pro" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}[effort]
    if "2.5-flash" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}[effort]
    return -1


# --------------------------------------------------------------------------
# stream
# --------------------------------------------------------------------------


def stream(
    model: Model,
    context: Context,
    options: GoogleVertexOptions | None = None,
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
    options: GoogleVertexOptions | None,
    client: httpx.AsyncClient | None,
) -> None:
    options = as_provider_options(options, GoogleVertexOptions)
    output = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason="pending",
        timestamp=now_ms(),
    )

    try:
        url, headers = build_url_and_headers(model, options)
        params = build_params(model, context, options)

        if options.on_payload is not None:
            replacement = options.on_payload(params, model)
            if hasattr(replacement, "__await__"):
                replacement = await replacement
            if replacement is not None:
                params = replacement

        request = HttpRequest(url=url, headers=headers, json_body=params, timeout_ms=options.timeout_ms)

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
            raise RuntimeError("Google Vertex stream ended without a finish reason")
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


def _base_to_google_options(base: StreamOptions, **overrides: Any) -> GoogleVertexOptions:
    values = {f.name: getattr(base, f.name) for f in fields(base)}
    values.update(overrides)
    return GoogleVertexOptions(**values)


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Stream with unified options, mapping `reasoning` to a Gemini thinking config."""
    base = build_base_options(model, context, options, None)
    if options is None or not options.reasoning:
        return stream(
            model, context, _base_to_google_options(base, thinking=GoogleThinkingOptions(enabled=False)), client=client
        )

    clamped_reasoning = clamp_thinking_level(model, options.reasoning)
    effort: ClampedThinkingLevel = "high" if clamped_reasoning == "off" else clamped_reasoning  # type: ignore[assignment]

    if is_gemini3_pro_model(model.id) or is_gemini3_flash_model(model.id):
        return stream(
            model,
            context,
            _base_to_google_options(
                base, thinking=GoogleThinkingOptions(enabled=True, level=_get_gemini3_thinking_level(effort, model))
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
