"""Native Mistral Chat Completions streaming.

Python port of `packages/ai/src/api/mistral-conversations.ts`.

Talks directly to `POST {baseUrl}/v1/chat/completions` with ``stream: true``
via :func:`pi_ai.utils.http.stream_sse` (no vendor SDK). Mistral's wire
protocol uses snake_case keys (``max_tokens``, ``tool_calls``, ...); the
internal payload/message dicts built here use the same camelCase-ish keys as
the TypeScript source and are remapped to snake_case only at serialization
time by :func:`to_mistral_wire_payload`, mirroring ``toMistralWirePayload``.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, fields
from typing import Any

import httpx

from ..models import calculate_cost, clamp_thinking_level
from ..types import (
    AssistantMessage,
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
    now_ms,
)
from ..utils.error_body import safe_json_stringify, truncate_error_text
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.hash import short_hash
from ..utils.http import HttpRequest, ProviderHttpError, stream_sse
from ..utils.json_parse import parse_streaming_json
from ..utils.json_stringify import json_stringify
from ..utils.sanitize_unicode import sanitize_surrogates
from ..utils.tasks import spawn
from .constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from .simple_options import as_provider_options, build_base_options
from .transform_messages import transform_messages

MISTRAL_TOOL_CALL_ID_LENGTH = 9
MAX_MISTRAL_ERROR_BODY_CHARS = 4000

_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]")


@dataclass
class MistralOptions(StreamOptions):
    """Provider-specific options for the Mistral API."""

    tool_choice: Any = None
    prompt_mode: str | None = None
    """``"reasoning"`` for models routed through `promptMode` instead of `reasoningEffort`."""
    reasoning_effort: str | None = None
    """``"none" | "high"``, for models routed through `reasoningEffort`."""


# --------------------------------------------------------------------------
# stream
# --------------------------------------------------------------------------


def stream(
    model: Model,
    context: Context,
    options: MistralOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Stream responses from the native Mistral Chat Completions endpoint.

    Failures are reported through the returned stream, not raised.
    """
    event_stream = AssistantMessageEventStream()
    spawn(_run_stream(event_stream, model, context, options, client))
    return event_stream


async def _run_stream(
    event_stream: AssistantMessageEventStream,
    model: Model,
    context: Context,
    options: MistralOptions | None,
    client: httpx.AsyncClient | None,
) -> None:
    options = as_provider_options(options, MistralOptions)
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

        normalize_mistral_tool_call_id = create_mistral_tool_call_id_normalizer()
        transformed_messages = transform_messages(
            context.messages,
            model,
            lambda tool_call_id, _model, _assistant: normalize_mistral_tool_call_id(tool_call_id),
        )

        payload = build_chat_payload(model, context, transformed_messages, options)
        if options.on_payload is not None:
            replacement = options.on_payload(payload, model)
            if hasattr(replacement, "__await__"):
                replacement = await replacement
            if replacement is not None:
                payload = replacement

        request = HttpRequest(
            url=_mistral_url(model),
            headers=build_mistral_headers(model, api_key, options),
            json_body=to_mistral_wire_payload(payload),
            timeout_ms=options.timeout_ms,
            signal=options.signal,
        )

        on_response = None
        if options.on_response is not None:
            captured_on_response = options.on_response

            async def on_response(provider_response: ProviderResponse) -> None:
                result = captured_on_response(provider_response, model)
                if hasattr(result, "__await__"):
                    await result

        state = _MistralStreamState(event_stream, output, model)
        started = False
        async for chunk in _iterate_mistral_chunks(request, client, on_response):
            if not started:
                event_stream.push(StartEvent(partial=output))
                started = True
            state.handle_chunk(chunk)

        if not started:
            event_stream.push(StartEvent(partial=output))

        state.finish_all_blocks()

        if options.signal is not None and options.signal.aborted:
            raise RuntimeError("Request was aborted")

        if output.stop_reason == "pending":
            raise RuntimeError("Mistral stream ended without a finish reason")
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
        # Unlike the TypeScript source, tool-call argument JSON accumulates in a
        # separate scratch dict (`_MistralStreamState.tool_partial_args`), never
        # on the `ToolCall` block itself, so there is no `partialArgs` field to
        # strip from `output.content` before surfacing the error.
        aborted = options.signal is not None and options.signal.aborted
        output.stop_reason = "aborted" if aborted else "error"
        output.error_message = format_mistral_error(error)
        event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
        event_stream.end()


# --------------------------------------------------------------------------
# stream_simple
# --------------------------------------------------------------------------


def _base_to_mistral_options(base: StreamOptions, **overrides: Any) -> MistralOptions:
    values = {f.name: getattr(base, f.name) for f in fields(base)}
    values.update(overrides)
    return MistralOptions(**values)


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantMessageEventStream:
    """Map provider-agnostic `SimpleStreamOptions` to Mistral options."""
    options = options or SimpleStreamOptions()
    api_key = options.api_key
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, context, options, api_key)
    clamped_reasoning = clamp_thinking_level(model, options.reasoning) if options.reasoning else None
    reasoning = None if clamped_reasoning == "off" else clamped_reasoning
    should_use_reasoning = model.reasoning and reasoning is not None

    return stream(
        model,
        context,
        _base_to_mistral_options(
            base,
            prompt_mode="reasoning" if should_use_reasoning and _uses_prompt_mode_reasoning(model) else None,
            reasoning_effort=(
                _map_reasoning_effort(model, reasoning)  # type: ignore[arg-type]
                if should_use_reasoning and _uses_reasoning_effort(model)
                else None
            ),
        ),
        client=client,
    )


# --------------------------------------------------------------------------
# Tool call id normalization
# --------------------------------------------------------------------------


def create_mistral_tool_call_id_normalizer() -> Any:
    id_map: dict[str, str] = {}
    reverse_map: dict[str, str] = {}

    def normalize(tool_call_id: str) -> str:
        existing = id_map.get(tool_call_id)
        if existing:
            return existing

        attempt = 0
        while True:
            candidate = derive_mistral_tool_call_id(tool_call_id, attempt)
            owner = reverse_map.get(candidate)
            if not owner or owner == tool_call_id:
                id_map[tool_call_id] = candidate
                reverse_map[candidate] = tool_call_id
                return candidate
            attempt += 1

    return normalize


def derive_mistral_tool_call_id(tool_call_id: str, attempt: int) -> str:
    normalized = _ALNUM_RE.sub("", tool_call_id)
    if attempt == 0 and len(normalized) == MISTRAL_TOOL_CALL_ID_LENGTH:
        return normalized
    seed_base = normalized or tool_call_id
    seed = seed_base if attempt == 0 else f"{seed_base}:{attempt}"
    return _ALNUM_RE.sub("", short_hash(seed))[:MISTRAL_TOOL_CALL_ID_LENGTH]


# --------------------------------------------------------------------------
# Error formatting
# --------------------------------------------------------------------------


def format_mistral_error(error: object) -> str:
    if isinstance(error, ProviderHttpError):
        body_text = error.body.strip() if error.body else ""
        if body_text:
            return f"Mistral API error ({error.status_code}): {truncate_error_text(body_text, MAX_MISTRAL_ERROR_BODY_CHARS)}"
        return f"Mistral API error ({error.status_code}): {error}"
    if isinstance(error, BaseException):
        return str(error)
    return safe_json_stringify(error)


# --------------------------------------------------------------------------
# HTTP transport
# --------------------------------------------------------------------------


def _mistral_url(model: Model) -> str:
    return f"{model.base_url.rstrip('/')}/v1/chat/completions"


def build_mistral_headers(model: Model, api_key: str, options: MistralOptions | None = None) -> dict[str, str]:
    headers: dict[str, str] = {
        "accept": "text/event-stream",
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    _apply_mistral_header_overrides(headers, model.headers)
    _apply_mistral_header_overrides(headers, options.headers if options is not None else None)

    has_explicit_affinity = _has_mistral_header_override(model.headers, "x-affinity") or _has_mistral_header_override(
        options.headers if options is not None else None, "x-affinity"
    )
    if _should_use_prompt_caching(options) and not has_explicit_affinity:
        headers["x-affinity"] = options.session_id  # type: ignore[union-attr,assignment]

    return headers


def _apply_mistral_header_overrides(headers: dict[str, str], overrides: dict[str, str | None] | None) -> None:
    # TypeScript builds a `Headers` object, whose names are case-insensitive, so
    # `{ Authorization: "..." }` replaces the default `authorization` entry and
    # `{ authorization: null }` deletes it whatever case it was set with. A
    # plain dict is case-sensitive, so fold names to lower case here.
    if not overrides:
        return
    for name, value in overrides.items():
        key = name.lower()
        if value is None:
            headers.pop(key, None)
        else:
            headers[key] = value


def _has_mistral_header_override(overrides: dict[str, str | None] | None, target: str) -> bool:
    return bool(overrides) and any(name.lower() == target for name in overrides)


async def _iterate_mistral_chunks(
    request: HttpRequest, client: httpx.AsyncClient | None, on_response: Any
) -> AsyncIterator[dict[str, Any]]:
    async for sse_event in stream_sse(request, client=client, on_response=on_response):
        data = sse_event.data.strip()
        if not data or data == "[DONE]":
            continue
        parsed = json.loads(data)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("choices"), list):
            raise ValueError("Invalid Mistral streaming event")
        yield parsed


# --------------------------------------------------------------------------
# Wire payload construction
# --------------------------------------------------------------------------


def build_chat_payload(
    model: Model,
    context: Context,
    messages: list[Message],
    options: MistralOptions | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.id,
        "stream": True,
        "messages": to_chat_messages(messages, "image" in model.input),
    }

    if context.tools:
        payload["tools"] = to_function_tools(context.tools)
    if options is not None:
        if options.temperature is not None:
            payload["temperature"] = options.temperature
        if options.max_tokens is not None:
            payload["maxTokens"] = options.max_tokens
        # TypeScript's `ProviderStreamOptions` is `StreamOptions & Record<string,
        # unknown>`, so a caller passing the plain base options simply leaves
        # these provider-specific keys `undefined`. `getattr` reproduces that:
        # `compat.stream()` hands the api module a base `StreamOptions`.
        tool_choice = getattr(options, "tool_choice", None)
        if tool_choice:
            mapped_choice = map_tool_choice(tool_choice)
            if mapped_choice is not None:
                payload["toolChoice"] = mapped_choice
        prompt_mode = getattr(options, "prompt_mode", None)
        if prompt_mode:
            payload["promptMode"] = prompt_mode
        reasoning_effort = getattr(options, "reasoning_effort", None)
        if reasoning_effort:
            payload["reasoningEffort"] = reasoning_effort
        if _should_use_prompt_caching(options):
            payload["promptCacheKey"] = options.session_id

    if context.system_prompt:
        payload["messages"] = [
            {"role": "system", "content": sanitize_surrogates(context.system_prompt)},
            *payload["messages"],
        ]

    return payload


def _should_use_prompt_caching(options: MistralOptions | None) -> bool:
    return options is not None and options.cache_retention != "none" and bool(options.session_id)


def _get_mistral_cached_prompt_tokens(usage: dict[str, Any], prompt_tokens: int) -> int:
    def _nested(key1: str, key2: str) -> Any:
        sub = usage.get(key1)
        return sub.get(key2) if isinstance(sub, dict) else None

    raw_cached_tokens = _nested("promptTokensDetails", "cachedTokens")
    if raw_cached_tokens is None:
        raw_cached_tokens = _nested("prompt_tokens_details", "cached_tokens")
    if raw_cached_tokens is None:
        raw_cached_tokens = _nested("promptTokenDetails", "cachedTokens")
    if raw_cached_tokens is None:
        raw_cached_tokens = _nested("prompt_token_details", "cached_tokens")
    if raw_cached_tokens is None:
        raw_cached_tokens = usage.get("numCachedTokens")
    if raw_cached_tokens is None:
        raw_cached_tokens = usage.get("num_cached_tokens")
    if raw_cached_tokens is None:
        raw_cached_tokens = 0

    cached_tokens = (
        raw_cached_tokens
        if isinstance(raw_cached_tokens, (int, float)) and not isinstance(raw_cached_tokens, bool)
        else 0
    )
    return min(prompt_tokens, max(0, int(cached_tokens)))


_TOP_LEVEL_REMAP = (
    ("topP", "top_p"),
    ("maxTokens", "max_tokens"),
    ("randomSeed", "random_seed"),
    ("responseFormat", "response_format"),
    ("toolChoice", "tool_choice"),
    ("presencePenalty", "presence_penalty"),
    ("frequencyPenalty", "frequency_penalty"),
    ("parallelToolCalls", "parallel_tool_calls"),
    ("reasoningEffort", "reasoning_effort"),
    ("promptMode", "prompt_mode"),
    ("promptCacheKey", "prompt_cache_key"),
    ("safePrompt", "safe_prompt"),
)

_CONTENT_CHUNK_REMAP = (
    ("imageUrl", "image_url"),
    ("documentUrl", "document_url"),
    ("documentName", "document_name"),
    ("fileId", "file_id"),
    ("referenceIds", "reference_ids"),
    ("inputAudio", "input_audio"),
)


def _remap_mistral_property(record: dict[str, Any], source: str, target: str) -> None:
    if source not in record:
        return
    record[target] = record.pop(source)


def to_mistral_wire_payload(payload: dict[str, Any]) -> dict[str, Any]:
    wire_payload: dict[str, Any] = dict(payload)
    for source, target in _TOP_LEVEL_REMAP:
        _remap_mistral_property(wire_payload, source, target)
    wire_payload["messages"] = [to_mistral_wire_message(message) for message in payload["messages"]]

    response_format = wire_payload.get("response_format")
    if isinstance(response_format, dict):
        wire_response_format = dict(response_format)
        _remap_mistral_property(wire_response_format, "jsonSchema", "json_schema")
        json_schema = wire_response_format.get("json_schema")
        if isinstance(json_schema, dict):
            wire_json_schema = dict(json_schema)
            _remap_mistral_property(wire_json_schema, "schemaDefinition", "schema")
            wire_response_format["json_schema"] = wire_json_schema
        wire_payload["response_format"] = wire_response_format

    return wire_payload


def to_mistral_wire_message(message: dict[str, Any]) -> dict[str, Any]:
    wire_message: dict[str, Any] = dict(message)
    _remap_mistral_property(wire_message, "toolCalls", "tool_calls")
    _remap_mistral_property(wire_message, "toolCallId", "tool_call_id")
    content = message.get("content")
    if isinstance(content, list):
        wire_message["content"] = [to_mistral_wire_content_chunk(chunk) for chunk in content]
    return wire_message


def to_mistral_wire_content_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    wire_chunk: dict[str, Any] = dict(chunk)
    for source, target in _CONTENT_CHUNK_REMAP:
        _remap_mistral_property(wire_chunk, source, target)
    return wire_chunk


# --------------------------------------------------------------------------
# Chat stream consumption
# --------------------------------------------------------------------------


class _MistralStreamState:
    """Accumulates streamed deltas into ``output.content`` and emits events."""

    def __init__(self, event_stream: AssistantMessageEventStream, output: AssistantMessage, model: Model) -> None:
        self.event_stream = event_stream
        self.output = output
        self.model = model
        self.current_block: TextContent | ThinkingContent | None = None
        self.tool_blocks_by_key: dict[str, int] = {}
        # Streaming scratch buffers for tool-call argument JSON, keyed by
        # content index. Never persisted onto the `ToolCall` block itself.
        self.tool_partial_args: dict[int, str] = {}

    def _block_index(self) -> int:
        return len(self.output.content) - 1

    def _finish_current_block(self) -> None:
        block = self.current_block
        if block is None:
            return
        if isinstance(block, TextContent):
            self.event_stream.push(
                TextEndEvent(content_index=self._block_index(), content=block.text, partial=self.output)
            )
            return
        if isinstance(block, ThinkingContent):
            self.event_stream.push(
                ThinkingEndEvent(content_index=self._block_index(), content=block.thinking, partial=self.output)
            )

    def handle_chunk(self, chunk: dict[str, Any]) -> None:
        output = self.output
        # Mistral's streamed CompletionChunk carries an id field. Keep the first
        # non-empty one, mirroring how OpenAI-style streaming exposes a stable
        # response identifier per stream.
        if not output.response_id:
            output.response_id = chunk.get("id")

        usage = chunk.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens") or 0
            cached_prompt_tokens = _get_mistral_cached_prompt_tokens(usage, prompt_tokens)

            output.usage.input = max(0, prompt_tokens - cached_prompt_tokens)
            output.usage.output = usage.get("completion_tokens") or 0
            output.usage.cache_read = cached_prompt_tokens
            output.usage.cache_write = 0
            output.usage.total_tokens = usage.get("total_tokens") or (
                output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write
            )
            calculate_cost(self.model, output.usage)

        choices = chunk.get("choices") or []
        if not choices:
            return
        choice = choices[0]

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            output.raw_stop_reason = finish_reason
            stop_reason, error_message = map_chat_stop_reason(finish_reason)
            output.stop_reason = stop_reason
            if error_message:
                output.error_message = error_message

        delta = choice.get("delta") or {}
        self._handle_content_delta(delta.get("content"))
        self._handle_tool_call_deltas(delta.get("tool_calls") or [])

    def _handle_content_delta(self, content: Any) -> None:
        if content is None:
            return
        content_items = [content] if isinstance(content, str) else content
        for item in content_items:
            if isinstance(item, str):
                self._append_text_delta(sanitize_surrogates(item))
                continue

            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type == "thinking":
                thinking_parts = item.get("thinking") or []
                delta_text = "".join(
                    part.get("text") or "" for part in thinking_parts if isinstance(part, dict) and part.get("text")
                )
                thinking_delta = sanitize_surrogates(delta_text)
                if thinking_delta:
                    self._append_thinking_delta(thinking_delta)
                continue

            if item_type == "text":
                self._append_text_delta(sanitize_surrogates(item.get("text") or ""))

    def _append_text_delta(self, text_delta: str) -> None:
        output = self.output
        if self.current_block is None or not isinstance(self.current_block, TextContent):
            self._finish_current_block()
            self.current_block = TextContent(text="")
            output.content.append(self.current_block)
            self.event_stream.push(TextStartEvent(content_index=self._block_index(), partial=output))
        self.current_block.text += text_delta
        self.event_stream.push(TextDeltaEvent(content_index=self._block_index(), delta=text_delta, partial=output))

    def _append_thinking_delta(self, thinking_delta: str) -> None:
        output = self.output
        if self.current_block is None or not isinstance(self.current_block, ThinkingContent):
            self._finish_current_block()
            self.current_block = ThinkingContent(thinking="")
            output.content.append(self.current_block)
            self.event_stream.push(ThinkingStartEvent(content_index=self._block_index(), partial=output))
        self.current_block.thinking += thinking_delta
        self.event_stream.push(
            ThinkingDeltaEvent(content_index=self._block_index(), delta=thinking_delta, partial=output)
        )

    def _handle_tool_call_deltas(self, tool_calls: list[dict[str, Any]]) -> None:
        output = self.output
        for tool_call in tool_calls:
            if self.current_block is not None:
                self._finish_current_block()
                self.current_block = None

            raw_id = tool_call.get("id")
            index_hint = tool_call.get("index") or 0
            call_id = (
                raw_id if raw_id and raw_id != "null" else derive_mistral_tool_call_id(f"toolcall:{index_hint}", 0)
            )
            key = f"{call_id}:{index_hint}"
            existing_index = self.tool_blocks_by_key.get(key)
            block: ToolCall | None = None
            if existing_index is not None:
                existing = output.content[existing_index]
                if isinstance(existing, ToolCall):
                    block = existing

            function = tool_call.get("function") or {}
            if block is None:
                block = ToolCall(id=call_id, name=function.get("name", ""), arguments={})
                output.content.append(block)
                new_index = len(output.content) - 1
                self.tool_blocks_by_key[key] = new_index
                self.tool_partial_args[new_index] = ""
                self.event_stream.push(ToolCallStartEvent(content_index=new_index, partial=output))

            idx = self.tool_blocks_by_key[key]
            arguments = function.get("arguments")
            args_delta = arguments if isinstance(arguments, str) else json_stringify(arguments or {})
            self.tool_partial_args[idx] = self.tool_partial_args.get(idx, "") + args_delta
            block.arguments = parse_streaming_json(self.tool_partial_args[idx])
            self.event_stream.push(ToolCallDeltaEvent(content_index=idx, delta=args_delta, partial=output))

    def finish_all_blocks(self) -> None:
        self._finish_current_block()
        for index in self.tool_blocks_by_key.values():
            block = self.output.content[index]
            if not isinstance(block, ToolCall):
                continue
            # Finalize in-place using the accumulated scratch buffer so replay
            # only carries parsed arguments.
            block.arguments = parse_streaming_json(self.tool_partial_args.get(index, ""))
            self.event_stream.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=self.output))


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


def to_function_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, True)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    # No `stripSymbolKeys` equivalent needed: Python dict keys are
                    # always strings, unlike JS objects which can carry Symbol keys.
                    "parameters": get_json_schema_tool_parameters(tool, strict),
                    "strict": bool(strict) if strict is not None else False,
                },
            }
        )
    return result


# --------------------------------------------------------------------------
# Message conversion
# --------------------------------------------------------------------------


def _user_content_chunk(item: Any) -> dict[str, Any]:
    if item.type == "text":
        return {"type": "text", "text": sanitize_surrogates(item.text)}
    return {"type": "image_url", "imageUrl": f"data:{item.mime_type};base64,{item.data}"}


def to_chat_messages(messages: list[Message], supports_images: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                result.append({"role": "user", "content": sanitize_surrogates(msg.content)})
                continue

            had_images = any(item.type == "image" for item in msg.content)
            content = [_user_content_chunk(item) for item in msg.content if item.type == "text" or supports_images]
            if content:
                result.append({"role": "user", "content": content})
                continue
            if had_images and not supports_images:
                result.append({"role": "user", "content": "(image omitted: model does not support images)"})
            continue

        if msg.role == "assistant":
            content_parts: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []

            for block in msg.content:
                if block.type == "text":
                    if block.text.strip():
                        content_parts.append({"type": "text", "text": sanitize_surrogates(block.text)})
                    continue
                if block.type == "thinking":
                    if block.thinking.strip():
                        content_parts.append(
                            {
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": sanitize_surrogates(block.thinking)}],
                            }
                        )
                    continue
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": json_stringify(block.arguments or {})},
                        "index": 0,
                    }
                )

            assistant_message: dict[str, Any] = {"role": "assistant", "prefix": False}
            if content_parts:
                assistant_message["content"] = content_parts
            if tool_calls:
                assistant_message["toolCalls"] = tool_calls
            if content_parts or tool_calls:
                result.append(assistant_message)
            continue

        # toolResult
        tool_content: list[dict[str, Any]] = []
        text_result = "\n".join(sanitize_surrogates(part.text) for part in msg.content if part.type == "text")
        has_images = any(part.type == "image" for part in msg.content)
        tool_text = build_tool_result_text(text_result, has_images, supports_images, msg.is_error)
        tool_content.append({"type": "text", "text": tool_text})
        for part in msg.content:
            if not supports_images:
                continue
            if part.type != "image":
                continue
            tool_content.append({"type": "image_url", "imageUrl": f"data:{part.mime_type};base64,{part.data}"})
        result.append(
            {
                "role": "tool",
                "toolCallId": msg.tool_call_id,
                "name": msg.tool_name,
                "content": tool_content,
            }
        )

    return result


def build_tool_result_text(text: str, has_images: bool, supports_images: bool, is_error: bool) -> str:
    trimmed = text.strip()
    error_prefix = "[tool error] " if is_error else ""

    if trimmed:
        image_suffix = (
            "\n[tool image omitted: model does not support images]" if has_images and not supports_images else ""
        )
        return f"{error_prefix}{trimmed}{image_suffix}"

    if has_images:
        if supports_images:
            return "[tool error] (see attached image)" if is_error else "(see attached image)"
        return (
            "[tool error] (image omitted: model does not support images)"
            if is_error
            else "(image omitted: model does not support images)"
        )

    return "[tool error] (no tool output)" if is_error else "(no tool output)"


# --------------------------------------------------------------------------
# Reasoning routing
# --------------------------------------------------------------------------


def _uses_reasoning_effort(model: Model) -> bool:
    return model.id in ("mistral-small-2603", "mistral-small-latest", "mistral-medium-3.5")


def _uses_prompt_mode_reasoning(model: Model) -> bool:
    return bool(model.reasoning) and not _uses_reasoning_effort(model)


def _map_reasoning_effort(model: Model, level: str) -> str:
    mapped = model.thinking_level_map.get(level)
    return mapped if mapped is not None else "high"


def map_tool_choice(choice: Any) -> Any:
    if not choice:
        return None
    if choice in ("auto", "none", "any", "required"):
        return choice
    return {"type": "function", "function": {"name": choice["function"]["name"]}}


def map_chat_stop_reason(reason: str | None) -> tuple[StopReason, str | None]:
    if reason is None:
        return "stop", None
    if reason == "stop":
        return "stop", None
    if reason in ("length", "model_length"):
        return "length", None
    if reason == "tool_calls":
        return "toolUse", None
    if reason == "error":
        return "error", "Provider stopped with: error"
    return "error", f"Provider stopped with: {reason}"
