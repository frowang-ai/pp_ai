"""pi-messages API implementation.

Python port of `packages/ai/src/api/pi-messages.ts`.

Streams pi's own message protocol directly to a backend: the request is a
single POST of ``{model, context, options}`` to ``<baseUrl>/messages``, and
the response is an SSE stream of serialized assistant-message events plus a
terminal ``done``/``error`` event. This is the wire protocol spoken by the
Radius gateway, but any backend implementing it can be used, e.g. via a
models.json custom provider with ``"api": "pi-messages"``.

Unlike the other ported providers, this API needs no vendor wire-format
translation: the backend understands pi's own `Context`/`Message`/`Tool`
shapes directly. What this module does instead is serialize those Python
dataclasses to the exact camelCase JSON shape the TypeScript types describe
(`_context_to_wire`/`_message_to_wire`/...), and parse incoming SSE frames
that are already pi's own event protocol (`_PiMessagesEventConverter`).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, fields
from typing import Any
from urllib.parse import urlencode

import httpx

from ..types import (
    AssistantContent,
    AssistantMessage,
    AssistantMessageDiagnostic,
    Context,
    Cost,
    DeferredHandle,
    DoneEvent,
    ErrorEvent,
    GrammarConstrainedSampling,
    ImageContent,
    JsonSchemaConstrainedSampling,
    Message,
    Model,
    SimpleStreamOptions,
    StartEvent,
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
    ToolResultMessage,
    Usage,
    UserMessage,
    now_ms,
)
from ..utils.diagnostics import append_assistant_message_diagnostic, create_assistant_message_diagnostic
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.headers import provider_headers_to_record
from ..utils.http import HttpRequest, ProviderHttpError, stream_sse
from ..utils.json_parse import parse_streaming_json
from ..utils.provider_env import get_provider_env_value
from ..utils.tasks import spawn
from .simple_options import as_provider_options

MAX_DIAGNOSTIC_STRING_CHARS = 8192


@dataclass
class PiMessagesOptions(StreamOptions):
    """Provider-specific options for the pi-messages API."""

    reasoning: str | None = None
    """A `ThinkingLevel` (`"minimal" | "low" | "medium" | "high" | "xhigh" | "max"`)."""
    tool_choice: Any = None
    """`"auto" | "none" | "required" | {"type": "function", "function": {"name": str}}`."""
    debug: bool = False
    """Ask the backend for debug metadata (e.g. routing response headers)."""


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class PiMessagesResponseError(Exception):
    """Raised for a non-2xx pi-messages response, matching `PiMessagesResponseError`."""

    def __init__(self, message: str, code: str | None, diagnostic_details: dict[str, Any]) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostic_details = diagnostic_details


def _parse_pi_messages_error_body(body: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    error = parsed.get("error")
    if isinstance(error, dict):
        return parsed
    return None


def _truncate_diagnostic_string(value: str) -> str:
    if len(value) > MAX_DIAGNOSTIC_STRING_CHARS:
        return f"{value[:MAX_DIAGNOSTIC_STRING_CHARS]}\u2026"
    return value


def _format_pi_messages_response_error(
    status: int, reason_phrase: str, body: str, error_body: dict[str, Any] | None
) -> str:
    error = error_body.get("error") if error_body else None
    message = error.get("message") if isinstance(error, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    suffix = message if isinstance(message, str) else body
    code_suffix = f" ({code})" if isinstance(code, str) else ""
    return f"{status} {reason_phrase}: {suffix}{code_suffix}"


def _create_pi_messages_response_error(
    model: Model, url: str, status: int, reason_phrase: str, body: str
) -> PiMessagesResponseError:
    error_body = _parse_pi_messages_error_body(body)
    error = error_body.get("error") if error_body else None
    code = error.get("code") if isinstance(error, dict) else None
    return PiMessagesResponseError(
        _format_pi_messages_response_error(status, reason_phrase, body, error_body),
        code if isinstance(code, str) else None,
        {
            "version": 1,
            "provider": model.provider,
            "model": model.id,
            "url": url,
            "status": status,
            "statusText": reason_phrase,
            "error": error,
            "body": None if error_body else _truncate_diagnostic_string(body),
            "timestampMs": now_ms(),
        },
    )


def _create_error_event(model: Model, error: BaseException, aborted: bool) -> ErrorEvent:
    reason = "aborted" if aborted else "error"
    assistant_message = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        content=[],
        usage=Usage(),
        stop_reason=reason,
        error_message=str(error) if str(error) else type(error).__name__,
        timestamp=now_ms(),
    )
    if not aborted and isinstance(error, PiMessagesResponseError):
        append_assistant_message_diagnostic(
            assistant_message,
            create_assistant_message_diagnostic("pi_messages_response_failure", error, error.diagnostic_details),
        )
    return ErrorEvent(reason=reason, error=assistant_message)


# --------------------------------------------------------------------------
# Wire (de)serialization
# --------------------------------------------------------------------------


def _omit_none(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _content_block_to_wire(block: TextContent | ThinkingContent | ImageContent | ToolCall) -> dict[str, Any]:
    if isinstance(block, TextContent):
        return _omit_none({"type": "text", "text": block.text, "textSignature": block.text_signature})
    if isinstance(block, ThinkingContent):
        return _omit_none(
            {
                "type": "thinking",
                "thinking": block.thinking,
                "thinkingSignature": block.thinking_signature,
                "redacted": block.redacted,
            }
        )
    if isinstance(block, ImageContent):
        return {"type": "image", "data": block.data, "mimeType": block.mime_type}
    if isinstance(block, ToolCall):
        return _omit_none(
            {
                "type": "toolCall",
                "id": block.id,
                "name": block.name,
                "arguments": block.arguments,
                "thoughtSignature": block.thought_signature,
                "namespace": block.namespace,
            }
        )
    raise TypeError(f"Unknown content block type: {type(block)!r}")


def _usage_to_wire(usage: Usage) -> dict[str, Any]:
    return _omit_none(
        {
            "input": usage.input,
            "output": usage.output,
            "cacheRead": usage.cache_read,
            "cacheWrite": usage.cache_write,
            "cacheWrite1h": usage.cache_write_1h,
            "reasoning": usage.reasoning,
            "totalTokens": usage.total_tokens,
            "cost": {
                "input": usage.cost.input,
                "output": usage.cost.output,
                "cacheRead": usage.cost.cache_read,
                "cacheWrite": usage.cost.cache_write,
                "total": usage.cost.total,
            },
        }
    )


def _usage_from_wire(data: dict[str, Any] | None) -> Usage:
    if not data:
        return Usage()
    cost_data = data.get("cost") or {}
    return Usage(
        input=data.get("input", 0),
        output=data.get("output", 0),
        cache_read=data.get("cacheRead", 0),
        cache_write=data.get("cacheWrite", 0),
        cache_write_1h=data.get("cacheWrite1h"),
        reasoning=data.get("reasoning"),
        total_tokens=data.get("totalTokens", 0),
        cost=Cost(
            input=cost_data.get("input", 0.0),
            output=cost_data.get("output", 0.0),
            cache_read=cost_data.get("cacheRead", 0.0),
            cache_write=cost_data.get("cacheWrite", 0.0),
            total=cost_data.get("total", 0.0),
        ),
    )


def _deferred_to_wire(handle: DeferredHandle) -> dict[str, Any]:
    return _omit_none(
        {
            "provider": handle.provider,
            "modelId": handle.model_id,
            "api": handle.api,
            "id": handle.id,
            "expiresAt": handle.expires_at,
            "pollAfterMs": handle.poll_after_ms,
            "data": handle.data,
        }
    )


def _diagnostic_to_wire(diagnostic: AssistantMessageDiagnostic) -> dict[str, Any]:
    # The Python `AssistantMessageDiagnostic` is a flatter shape than the
    # TypeScript one (see `utils/diagnostics.py`); there is no lossless
    # reconstruction of the original TS `{type, timestamp, error, details}`
    # shape, so this forwards the Python fields as-is.
    return _omit_none(
        {
            "kind": diagnostic.kind,
            "message": diagnostic.message,
            "detail": diagnostic.detail,
            "timestamp": diagnostic.timestamp,
        }
    )


def _message_to_wire(message: Message) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        content = (
            message.content
            if isinstance(message.content, str)
            else [_content_block_to_wire(block) for block in message.content]
        )
        return {"role": "user", "content": content, "timestamp": message.timestamp}
    if isinstance(message, AssistantMessage):
        return _omit_none(
            {
                "role": "assistant",
                "content": [_content_block_to_wire(block) for block in message.content],
                "api": message.api,
                "provider": message.provider,
                "model": message.model,
                "responseModel": message.response_model,
                "responseId": message.response_id,
                "diagnostics": [_diagnostic_to_wire(d) for d in message.diagnostics] or None,
                "usage": _usage_to_wire(message.usage),
                "stopReason": message.stop_reason,
                "deferred": _deferred_to_wire(message.deferred) if message.deferred else None,
                "errorMessage": message.error_message,
                "rawStopReason": message.raw_stop_reason,
                "endTurn": message.end_turn,
                "timestamp": message.timestamp,
            }
        )
    if isinstance(message, ToolResultMessage):
        return _omit_none(
            {
                "role": "toolResult",
                "toolCallId": message.tool_call_id,
                "toolName": message.tool_name,
                "content": [_content_block_to_wire(block) for block in message.content],
                "details": message.details,
                "usage": _usage_to_wire(message.usage) if message.usage is not None else None,
                "addedToolNames": message.added_tool_names,
                "isError": message.is_error,
                "timestamp": message.timestamp,
            }
        )
    raise TypeError(f"Unknown message type: {type(message)!r}")


def _constrained_sampling_to_wire(config: Any) -> Any:
    if config is False:
        return False
    if isinstance(config, JsonSchemaConstrainedSampling):
        return {"type": "json_schema", "strict": config.strict}
    if isinstance(config, GrammarConstrainedSampling):
        return {"type": "grammar", "variants": config.variants}
    return config


def _tool_to_wire(tool: Tool) -> dict[str, Any]:
    result: dict[str, Any] = {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
    if tool.constrained_sampling is not None:
        result["constrainedSampling"] = _constrained_sampling_to_wire(tool.constrained_sampling)
    return result


def _context_to_wire(context: Context) -> dict[str, Any]:
    return _omit_none(
        {
            "systemPrompt": context.system_prompt,
            "messages": [_message_to_wire(message) for message in context.messages],
            "tools": [_tool_to_wire(tool) for tool in context.tools] if context.tools else None,
        }
    )


def _tool_call_from_wire(data: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=data["id"],
        name=data["name"],
        arguments=data.get("arguments") or {},
        thought_signature=data.get("thoughtSignature"),
        namespace=data.get("namespace"),
    )


# --------------------------------------------------------------------------
# Event conversion
# --------------------------------------------------------------------------


def _start_content_block(partial: AssistantMessage, index: int, content: AssistantContent) -> None:
    if index == len(partial.content):
        partial.content.append(content)
    else:
        partial.content[index] = content


class _PiMessagesEventConverter:
    """Stateful conversion of wire pi-messages events into `AssistantMessageEvent`s.

    Mirrors the closure returned by TypeScript's `createEventConverter`: builds
    up a single `partial` `AssistantMessage` across the whole stream and mutates
    it in place as events arrive.
    """

    def __init__(self, model: Model) -> None:
        self.partial = AssistantMessage(
            api=model.api,
            provider=model.provider,
            model=model.id,
            content=[],
            usage=Usage(),
            stop_reason="pending",
            timestamp=now_ms(),
        )
        self.tool_json: dict[int, str] = {}

    def convert(self, event: dict[str, Any]) -> Any:
        event_type = event.get("type")
        partial = self.partial

        if event_type == "done":
            partial.stop_reason = event["reason"]
            partial.usage = _usage_from_wire(event.get("usage"))
            partial.response_id = event.get("responseId")
            self._append_rewrite_diagnostic(event.get("rewrite"))
            return DoneEvent(reason=event["reason"], message=partial)
        if event_type == "error":
            partial.stop_reason = event["reason"]
            partial.usage = _usage_from_wire(event.get("usage"))
            partial.error_message = event.get("errorMessage")
            partial.response_id = event.get("responseId")
            self._append_rewrite_diagnostic(event.get("rewrite"))
            return ErrorEvent(reason=event["reason"], error=partial)
        if event_type == "start":
            return StartEvent(partial=partial)
        if event_type == "text_start":
            index = event["contentIndex"]
            _start_content_block(partial, index, TextContent(text=""))
            return TextStartEvent(content_index=index, partial=partial)
        if event_type == "text_delta":
            index = event["contentIndex"]
            block = partial.content[index]
            assert isinstance(block, TextContent)
            block.text += event["delta"]
            return TextDeltaEvent(content_index=index, delta=event["delta"], partial=partial)
        if event_type == "text_end":
            index = event["contentIndex"]
            block = partial.content[index]
            assert isinstance(block, TextContent)
            block.text = event["content"]
            block.text_signature = event.get("contentSignature")
            return TextEndEvent(content_index=index, content=event["content"], partial=partial)
        if event_type == "thinking_start":
            index = event["contentIndex"]
            _start_content_block(partial, index, ThinkingContent(thinking=""))
            return ThinkingStartEvent(content_index=index, partial=partial)
        if event_type == "thinking_delta":
            index = event["contentIndex"]
            block = partial.content[index]
            assert isinstance(block, ThinkingContent)
            block.thinking += event["delta"]
            return ThinkingDeltaEvent(content_index=index, delta=event["delta"], partial=partial)
        if event_type == "thinking_end":
            index = event["contentIndex"]
            block = partial.content[index]
            assert isinstance(block, ThinkingContent)
            block.thinking = event["content"]
            block.thinking_signature = event.get("contentSignature")
            block.redacted = event.get("redacted")
            return ThinkingEndEvent(content_index=index, content=event["content"], partial=partial)
        if event_type == "toolcall_start":
            index = event["contentIndex"]
            _start_content_block(partial, index, ToolCall(id=event["id"], name=event["toolName"], arguments={}))
            self.tool_json[index] = ""
            return ToolCallStartEvent(content_index=index, partial=partial)
        if event_type == "toolcall_delta":
            index = event["contentIndex"]
            json_text = f"{self.tool_json.get(index, '')}{event['delta']}"
            self.tool_json[index] = json_text
            block = partial.content[index]
            assert isinstance(block, ToolCall)
            block.arguments = parse_streaming_json(json_text)
            return ToolCallDeltaEvent(content_index=index, delta=event["delta"], partial=partial)
        if event_type == "toolcall_end":
            index = event["contentIndex"]
            tool_call = _tool_call_from_wire(event["toolCall"])
            partial.content[index] = tool_call
            self.tool_json.pop(index, None)
            return ToolCallEndEvent(content_index=index, tool_call=tool_call, partial=partial)

        raise ValueError(f"Unknown pi-messages event type: {event_type!r}")

    def _append_rewrite_diagnostic(self, rewrite: dict[str, Any] | None) -> None:
        if not rewrite:
            return
        append_assistant_message_diagnostic(
            self.partial,
            AssistantMessageDiagnostic(
                kind="pi_messages_rewrite", message="", detail=dict(rewrite), timestamp=now_ms()
            ),
        )


def _resolve_cache_retention(cache_retention: str | None, env: dict[str, str] | None) -> str | None:
    if cache_retention:
        return cache_retention
    # Backend defaults apply when unset; only the legacy env opt-in is mapped.
    return "long" if get_provider_env_value("PI_CACHE_RETENTION", env) == "long" else None


def _status_reason_phrase(status: int) -> str:
    """Best-effort HTTP reason phrase for `status`, matching `Response.statusText`.

    `ProviderHttpError` (shared HTTP transport) does not carry the server's
    reason phrase, so this falls back to the standard phrase for the status
    code, or an empty string for non-standard codes.
    """
    try:
        return httpx.codes(status).phrase
    except ValueError:
        return ""


def _pi_messages_url(model: Model, debug: bool) -> str:
    url = f"{model.base_url.rstrip('/')}/messages"
    if debug:
        url = f"{url}?{urlencode({'debug': '1'})}"
    return url


# --------------------------------------------------------------------------
# stream
# --------------------------------------------------------------------------


def stream(
    model: Model,
    context: Context,
    options: PiMessagesOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Stream responses from a pi-messages backend.

    Failures are reported through the returned stream, not raised.
    """
    event_stream = AssistantMessageEventStream()
    spawn(_run_stream(event_stream, model, context, options, client))
    return event_stream


async def _run_stream(
    event_stream: AssistantMessageEventStream,
    model: Model,
    context: Context,
    options: PiMessagesOptions | None,
    client: httpx.AsyncClient | None,
) -> None:
    options = as_provider_options(options, PiMessagesOptions)
    converter = _PiMessagesEventConverter(model)
    url = _pi_messages_url(model, options.debug)

    try:
        api_key = options.api_key
        if not api_key:
            raise ValueError(f'No API key provided for provider "{model.provider}"')

        payload: Any = {
            "model": model.id,
            "context": _context_to_wire(context),
            "options": _omit_none(
                {
                    "temperature": options.temperature,
                    "maxTokens": options.max_tokens,
                    "reasoning": options.reasoning,
                    "cacheRetention": _resolve_cache_retention(options.cache_retention, options.env),
                    "sessionId": options.session_id,
                    "toolChoice": options.tool_choice,
                }
            ),
        }
        if options.on_payload is not None:
            replacement = options.on_payload(payload, model)
            if hasattr(replacement, "__await__"):
                replacement = await replacement
            if replacement is not None:
                payload = replacement

        headers: dict[str, str] = {
            "authorization": f"Bearer {api_key}",
            "accept": "text/event-stream",
            "content-type": "application/json",
        }
        headers.update(provider_headers_to_record(options.headers) or {})

        request = HttpRequest(url=url, headers=headers, json_body=payload, timeout_ms=options.timeout_ms)

        on_response = None
        if options.on_response is not None:
            captured_on_response = options.on_response

            async def on_response(provider_response: Any) -> None:
                result = captured_on_response(provider_response, model)
                if hasattr(result, "__await__"):
                    await result

        try:
            async for sse_event in stream_sse(request, client, on_response):
                if not sse_event.data or sse_event.data == "[DONE]":
                    continue
                raw_event = json.loads(sse_event.data)
                event = converter.convert(raw_event)
                event_stream.push(event)
                if event.type in ("done", "error"):
                    event_stream.end()
                    return
        except ProviderHttpError as http_error:
            raise _create_pi_messages_response_error(
                model, url, http_error.status_code, _status_reason_phrase(http_error.status_code), http_error.body
            ) from http_error

        raise RuntimeError(f"{model.provider} stream ended without a terminal event")
    except asyncio.CancelledError:
        event_stream.push(_create_error_event(model, RuntimeError("Request was aborted"), aborted=True))
        event_stream.end()
        raise
    except BaseException as error:
        aborted = options.signal is not None and options.signal.aborted
        event_stream.push(_create_error_event(model, error, aborted))
        event_stream.end()


# --------------------------------------------------------------------------
# stream_simple
# --------------------------------------------------------------------------


def _base_to_pi_messages_options(base: StreamOptions, **overrides: Any) -> PiMessagesOptions:
    # `base` may be a `SimpleStreamOptions` (or another `StreamOptions`
    # subclass) carrying fields `PiMessagesOptions` doesn't declare (e.g.
    # `deferred`, `thinking_budgets`), so only the shared `StreamOptions`
    # fields are copied across; `**overrides` supplies the pi-messages-specific
    # ones.
    values = {f.name: getattr(base, f.name) for f in fields(StreamOptions)}
    values.update(overrides)
    return PiMessagesOptions(**values)


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Map provider-agnostic `SimpleStreamOptions` to pi-messages options.

    Unlike other providers, this forwards options straight through: the
    backend does its own sampling/context-window handling, so there is no
    local `build_base_options`/max-tokens clamping step (matching the
    TypeScript source, which also skips it for this API).
    """
    options = options or SimpleStreamOptions()
    return stream(
        model,
        context,
        _base_to_pi_messages_options(
            options,
            reasoning=options.reasoning,
            tool_choice=getattr(options, "tool_choice", None),
            debug=getattr(options, "debug", False) or False,
        ),
        client=client,
    )
