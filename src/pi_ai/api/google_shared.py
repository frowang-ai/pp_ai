"""Shared utilities for the Google Generative AI and Google Vertex providers.

Python port of `packages/ai/src/api/google-shared.ts`. The TypeScript version
drives the official `@google/genai` SDK; this port speaks the Gemini REST API
directly (`POST .../models/{id}:streamGenerateContent?alt=sse` for Generative
Language, `POST .../publishers/google/models/{id}:streamGenerateContent?alt=sse`
for Vertex) through :mod:`pi_ai.utils.http`. Streaming chunks are whole
`GenerateContentResponse` JSON objects sent as unlabelled SSE `data:` lines
(no `event:` field), unlike Anthropic's named SSE events.

Because `google-generative-ai.ts` and `google-vertex.ts` duplicate their
streaming loop verbatim in the TypeScript source (down to comments), this port
factors that loop into :class:`GoogleStreamState` here and has both provider
modules drive it. Behaviour is unchanged; only the duplication is removed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx

from ..models import calculate_cost
from ..types import (
    AssistantMessage,
    Context,
    Message,
    Model,
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
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.http import HttpRequest, SseEvent, stream_sse
from ..utils.json_parse import parse_json_with_repair
from ..utils.json_stringify import json_stringify
from ..utils.provider_retry import ProviderRetryOptions, retry_provider_request
from ..utils.sanitize_unicode import sanitize_surrogates
from .constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from .transform_messages import transform_messages

T = TypeVar("T")

GoogleThinkingLevel = str
"""`"THINKING_LEVEL_UNSPECIFIED" | "MINIMAL" | "LOW" | "MEDIUM" | "HIGH"`."""


# --------------------------------------------------------------------------
# Thought signature handling
# --------------------------------------------------------------------------


def is_thinking_part(part: dict[str, Any]) -> bool:
    """Determine whether a streamed Gemini `Part` should be treated as "thinking".

    Protocol note (Gemini / Vertex AI thought signatures):
    - `thought: true` is the definitive marker for thinking content (thought
      summaries).
    - `thoughtSignature` is an encrypted representation of the model's internal
      thought process used to preserve reasoning context across multi-turn
      interactions.
    - `thoughtSignature` can appear on ANY part type (text, functionCall, etc.)
      - it does NOT indicate the part itself is thinking content.
    - For non-functionCall responses, the signature appears on the last part
      for context replay.
    - When persisting/replaying model outputs, signature-bearing parts must be
      preserved as-is; do not merge/move signatures across parts.

    See: https://ai.google.dev/gemini-api/docs/thought-signatures
    """
    return part.get("thought") is True


def retain_thought_signature(existing: str | None, incoming: str | None) -> str | None:
    """Retain thought signatures during streaming.

    Some backends only send `thoughtSignature` on the first delta for a given
    part/block; later deltas may omit it. This helper preserves the last
    non-empty signature for the current block.

    Note: this does NOT merge or move signatures across distinct response
    parts. It only prevents a signature from being overwritten with `None`
    within the same streamed block.
    """
    if isinstance(incoming, str) and len(incoming) > 0:
        return incoming
    return existing


# Thought signatures must be base64 for Google APIs (TYPE_BYTES).
_BASE64_SIGNATURE_PATTERN = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _is_valid_thought_signature(signature: str | None) -> bool:
    if not signature:
        return False
    if len(signature) % 4 != 0:
        return False
    return bool(_BASE64_SIGNATURE_PATTERN.match(signature))


def resolve_thought_signature(is_same_provider_and_model: bool, signature: str | None) -> str | None:
    """Only keep signatures from the same provider/model and with valid base64."""
    return signature if is_same_provider_and_model and _is_valid_thought_signature(signature) else None


# --------------------------------------------------------------------------
# Model id helpers
# --------------------------------------------------------------------------

_GEMINI_VERSION_RE = re.compile(r"^gemini(?:-live)?-(\d+)")


def requires_tool_call_id(model_id: str) -> bool:
    """Models via Google APIs that require explicit tool call IDs in function calls/responses."""
    gemini_major_version = get_gemini_major_version(model_id)
    return (
        model_id.startswith("claude-")
        or model_id.startswith("gpt-oss-")
        or (gemini_major_version is not None and gemini_major_version >= 3)
    )


def get_gemini_major_version(model_id: str) -> int | None:
    match = _GEMINI_VERSION_RE.match(model_id.lower())
    if not match:
        return None
    return int(match.group(1))


def supports_multimodal_function_response(model_id: str) -> bool:
    gemini_major_version = get_gemini_major_version(model_id)
    if gemini_major_version is not None:
        return gemini_major_version >= 3
    return True


def is_gemini3_pro_model(model_id: str) -> bool:
    return re.search(r"gemini-3(?:\.\d+)?-pro", model_id.lower()) is not None


def is_gemini3_flash_model(model_id: str) -> bool:
    lower = model_id.lower()
    return (
        re.search(r"gemini-3(?:\.\d+)?-flash", lower) is not None
        or lower == "gemini-flash-latest"
        or lower == "gemini-flash-lite-latest"
    )


# --------------------------------------------------------------------------
# Message conversion
# --------------------------------------------------------------------------


def convert_messages(model: Model, context: Context) -> list[dict[str, Any]]:
    """Convert internal messages to Gemini `Content[]` format."""
    contents: list[dict[str, Any]] = []

    def normalize_tool_call_id(tool_call_id: str, _model: Model, _msg: AssistantMessage) -> str:
        if not requires_tool_call_id(model.id):
            return tool_call_id
        return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_call_id)[:64]

    transformed_messages: list[Message] = transform_messages(context.messages, model, normalize_tool_call_id)

    for msg in transformed_messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                contents.append({"role": "user", "parts": [{"text": sanitize_surrogates(msg.content)}]})
            else:
                parts: list[dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        parts.append({"text": sanitize_surrogates(item.text)})
                    else:
                        parts.append({"inlineData": {"mimeType": item.mime_type, "data": item.data}})
                if not parts:
                    continue
                contents.append({"role": "user", "parts": parts})

        elif msg.role == "assistant":
            parts = []
            is_same_provider_and_model = msg.provider == model.provider and msg.model == model.id

            for block in msg.content:
                if block.type == "text":
                    thought_signature = resolve_thought_signature(is_same_provider_and_model, block.text_signature)
                    # Skip empty text blocks - unless they carry a thought signature. Gemini
                    # can attach the signature to a part whose visible text is empty and
                    # requires it echoed back; dropping it breaks the reasoning chain and the
                    # model intermittently ends mid-task turns with a thought-only STOP (empty
                    # completion, no tool call).
                    if (not block.text or not block.text.strip()) and not thought_signature:
                        continue
                    part: dict[str, Any] = {"text": sanitize_surrogates(block.text)}
                    if thought_signature:
                        part["thoughtSignature"] = thought_signature
                    parts.append(part)
                elif block.type == "thinking":
                    # Only keep as thinking block if same provider AND same model.
                    # Otherwise convert to plain text (no tags to avoid model mimicking them).
                    if is_same_provider_and_model:
                        thought_signature = resolve_thought_signature(
                            is_same_provider_and_model, block.thinking_signature
                        )
                        # Same rule as text blocks: an empty thinking block is dropped only
                        # when it carries no signature (mirrors the anthropic converter's
                        # handling).
                        if (not block.thinking or not block.thinking.strip()) and not thought_signature:
                            continue
                        part = {"thought": True, "text": sanitize_surrogates(block.thinking)}
                        if thought_signature:
                            part["thoughtSignature"] = thought_signature
                        parts.append(part)
                    else:
                        # Cross-provider/model: the signature is unusable, empty blocks stay
                        # dropped.
                        if not block.thinking or not block.thinking.strip():
                            continue
                        parts.append({"text": sanitize_surrogates(block.thinking)})
                elif block.type == "toolCall":
                    thought_signature = resolve_thought_signature(is_same_provider_and_model, block.thought_signature)
                    function_call: dict[str, Any] = {"name": block.name, "args": block.arguments or {}}
                    if requires_tool_call_id(model.id):
                        function_call["id"] = block.id
                    part = {"functionCall": function_call}
                    if thought_signature:
                        part["thoughtSignature"] = thought_signature
                    parts.append(part)

            if not parts:
                continue
            contents.append({"role": "model", "parts": parts})

        elif msg.role == "toolResult":
            text_content = [c for c in msg.content if c.type == "text"]
            text_result = "\n".join(c.text for c in text_content)
            image_content = [c for c in msg.content if c.type == "image"] if "image" in model.input else []

            has_text = len(text_result) > 0
            has_images = len(image_content) > 0

            # Gemini 3+ models support multimodal function responses with images
            # nested inside functionResponse.parts. Claude and other non-Gemini
            # models behind Cloud Code Assist / Gemini < 3 still need a separate
            # user image turn.
            model_supports_multimodal_function_response = supports_multimodal_function_response(model.id)

            # Use "output" key for success, "error" key for errors as per SDK
            # documentation.
            response_value = (
                sanitize_surrogates(text_result) if has_text else ("(see attached image)" if has_images else "")
            )

            image_parts = [
                {"inlineData": {"mimeType": image_block.mime_type, "data": image_block.data}}
                for image_block in image_content
            ]

            include_id = requires_tool_call_id(model.id)
            function_response: dict[str, Any] = {
                "name": msg.tool_name,
                "response": {"error": response_value} if msg.is_error else {"output": response_value},
            }
            if has_images and model_supports_multimodal_function_response:
                function_response["parts"] = image_parts
            if include_id:
                function_response["id"] = msg.tool_call_id
            function_response_part = {"functionResponse": function_response}

            # Cloud Code Assist API requires all function responses to be in a
            # single user turn. Check if the last content is already a user turn
            # with function responses and merge.
            last_content = contents[-1] if contents else None
            if (
                last_content is not None
                and last_content.get("role") == "user"
                and any("functionResponse" in p for p in last_content.get("parts") or [])
            ):
                last_content["parts"].append(function_response_part)
            else:
                contents.append({"role": "user", "parts": [function_response_part]})

            # For Gemini < 3, add images in a separate user message.
            if has_images and not model_supports_multimodal_function_response:
                contents.append({"role": "user", "parts": [{"text": "Tool result image:"}, *image_parts]})

    return contents


# --------------------------------------------------------------------------
# Tool conversion
# --------------------------------------------------------------------------

_JSON_SCHEMA_META_DECLARATIONS = frozenset(
    {
        "$schema",
        "$id",
        "$anchor",
        "$dynamicAnchor",
        "$vocabulary",
        "$comment",
        "$defs",
        "definitions",  # pre-draft-2019-09 equivalent of $defs
    }
)


def _sanitize_for_openapi(schema: Any) -> Any:
    """Strip meta-declarations from a schema object."""
    if not isinstance(schema, dict):
        return schema
    return {
        key: _sanitize_for_openapi(value) for key, value in schema.items() if key not in _JSON_SCHEMA_META_DECLARATIONS
    }


def convert_tools(
    tools: list[Tool], use_parameters: bool = False, supports_strict_mode: bool = True
) -> list[dict[str, Any]] | None:
    """Convert tools to Gemini function declarations format.

    By default uses `parametersJsonSchema`, which supports full JSON Schema
    (including anyOf, oneOf, const, etc.). Set `use_parameters` to `True` to use
    the legacy `parameters` field instead (OpenAPI 3.03 Schema). This is needed
    for Cloud Code Assist with Claude models, where the API translates
    `parameters` into Anthropic's `input_schema`.
    """
    if not tools:
        return None
    declarations = []
    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        # `getJsonSchemaToolParameters` upstream (`google-shared.ts:295`): under
        # strict sampling Gemini gets the strict subset, not the raw schema.
        parameters = get_json_schema_tool_parameters(tool, strict)
        declaration: dict[str, Any] = {"name": tool.name, "description": tool.description}
        if use_parameters:
            declaration["parameters"] = _sanitize_for_openapi(parameters)
        else:
            declaration["parametersJsonSchema"] = parameters
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}]


def supports_google_strict_tool_sampling(model_id: str) -> bool:
    """Gemini 3+ enforces required function parameters in validated tool-calling modes."""
    major_version = get_gemini_major_version(model_id)
    return major_version is not None and major_version >= 3


def map_tool_choice(choice: str) -> str:
    """Map tool choice string to Gemini's `FunctionCallingConfigMode`."""
    if choice == "none":
        return "NONE"
    if choice == "any":
        return "ANY"
    return "AUTO"


def resolve_google_function_calling_mode(
    tools: list[Tool], tool_choice: str | None, supports_strict_mode: bool
) -> str | None:
    use_strict_mode = any(resolve_json_schema_strict_sampling(tool, supports_strict_mode) is True for tool in tools)
    if tool_choice in ("none", "any"):
        return map_tool_choice(tool_choice)
    if use_strict_mode:
        return "VALIDATED"
    return map_tool_choice(tool_choice) if tool_choice else None


# --------------------------------------------------------------------------
# Stop reason mapping
# --------------------------------------------------------------------------

_ERROR_FINISH_REASONS = frozenset(
    {
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "SAFETY",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
        "RECITATION",
        "FINISH_REASON_UNSPECIFIED",
        "OTHER",
        "LANGUAGE",
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "NO_IMAGE",
    }
)


def map_stop_reason(reason: str) -> StopReason:
    """Map a Gemini `FinishReason` string to our `StopReason`."""
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    # Every other known FinishReason value (safety blocks, recitation, malformed
    # calls, ...) maps to "error"; unrecognized future values fall back to
    # "error" too, since raising here would turn a provider-side stream into an
    # unrecoverable Python exception for a value this port cannot yet name.
    return "error"


def map_stop_reason_string(reason: str) -> StopReason:
    """Map string finish reason to our StopReason (for raw API responses)."""
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    return "error"


# --------------------------------------------------------------------------
# Retries
# --------------------------------------------------------------------------


async def retry_google_request(
    request: Callable[[], Awaitable[T]],
    options: StreamOptions | None = None,
) -> T:
    """Run a Google request with the shared provider retry policy.

    Port of `retryGoogleRequest` in `packages/ai/src/api/google-shared.ts`
    (408/409/429/5xx with backoff, honoring retry-after), mirroring how the
    Anthropic and OpenAI adapters wrap their initial request in
    `retryProviderRequest`. Google's SDK-shaped `ApiError` carries a `status`
    but no `headers`, and :func:`retry_provider_request` only retries errors
    that carry both, so normalize the error by adding the missing `headers`
    before re-raising.
    """

    async def attempt() -> T:
        try:
            return await request()
        except Exception as error:
            if hasattr(error, "status") and not hasattr(error, "headers"):
                error.headers = None  # type: ignore[attr-defined]
            raise

    return await retry_provider_request(
        attempt,
        ProviderRetryOptions(
            max_retries=options.max_retries if options is not None and options.max_retries is not None else 0,
            max_retry_delay_ms=options.max_retry_delay_ms if options is not None else None,
            signal=options.signal if options is not None else None,
        ),
    )


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


def _decode_google_chunk(data: str) -> Any:
    try:
        chunk = parse_json_with_repair(data)
    except ValueError as error:
        raise RuntimeError(f"Could not parse Google SSE chunk: {error}; data={data}") from error
    if isinstance(chunk, dict) and chunk.get("error") and not chunk.get("candidates"):
        error_body = chunk["error"]
        message = error_body.get("message") if isinstance(error_body, dict) else None
        raise RuntimeError(message or f"Google API error: {json.dumps(error_body)}")
    return chunk


async def iterate_google_chunks(
    request: HttpRequest,
    client: httpx.AsyncClient | None = None,
    on_response: Any = None,
    options: StreamOptions | None = None,
):
    """Decode a `:streamGenerateContent?alt=sse` response into parsed JSON chunks.

    Gemini's SSE stream carries unlabelled `data:` lines, each a whole
    `GenerateContentResponse` JSON object; there is no separate `event:` name to
    dispatch on the way Anthropic's SSE does. A non-2xx response raises
    :class:`~pi_ai.utils.http.ProviderHttpError` from `stream_sse` before this
    generator yields anything. A chunk carrying a top-level `error` object (some
    gateways emit these instead of a non-2xx status) is also raised as an error.

    Opening the stream goes through :func:`retry_google_request`, mirroring how
    `google-generative-ai.ts` and `google-vertex.ts` wrap their
    `generateContentStream` call: only the initial request is retried, never a
    stream that has already started emitting chunks.
    """

    async def open_stream() -> tuple[Any, SseEvent | None]:
        iterator = stream_sse(request, client=client, on_response=on_response).__aiter__()
        try:
            first = await iterator.__anext__()
        except StopAsyncIteration:
            return iterator, None
        return iterator, first

    iterator, first_event = await retry_google_request(open_stream, options)

    async def events():
        if first_event is not None:
            yield first_event
            async for event in iterator:
                yield event

    async for sse_event in events():
        if not sse_event.data or sse_event.data == "[DONE]":
            continue
        yield _decode_google_chunk(sse_event.data)


class GoogleStreamState:
    """Accumulates streamed `GenerateContentResponse` chunks into `output.content`.

    Shared by the Google Generative AI and Google Vertex providers, whose
    TypeScript sources duplicate this exact loop.
    """

    def __init__(self, event_stream: AssistantMessageEventStream, output: AssistantMessage, model: Model) -> None:
        self.event_stream = event_stream
        self.output = output
        self.model = model
        self._current_block: TextContent | ThinkingContent | None = None
        self._tool_call_counter = 0

    def _content_index(self) -> int:
        return len(self.output.content) - 1

    def _close_current_block(self) -> None:
        block = self._current_block
        if block is None:
            return
        index = self._content_index()
        if isinstance(block, TextContent):
            self.event_stream.push(TextEndEvent(content_index=index, content=block.text, partial=self.output))
        else:
            self.event_stream.push(ThinkingEndEvent(content_index=index, content=block.thinking, partial=self.output))
        self._current_block = None

    def handle_chunk(self, chunk: dict[str, Any]) -> None:
        if not self.output.response_id:
            self.output.response_id = chunk.get("responseId")

        candidates = chunk.get("candidates") or []
        candidate = candidates[0] if candidates else None
        content = (candidate or {}).get("content") or {}
        for part in content.get("parts") or []:
            self._handle_part(part)

        finish_reason = (candidate or {}).get("finishReason")
        if finish_reason:
            self.output.raw_stop_reason = finish_reason
            self.output.stop_reason = map_stop_reason(finish_reason)
            if any(block.type == "toolCall" for block in self.output.content):
                self.output.stop_reason = "toolUse"

        usage_metadata = chunk.get("usageMetadata")
        if usage_metadata:
            cache_read = usage_metadata.get("cachedContentTokenCount") or 0
            self.output.usage.input = (usage_metadata.get("promptTokenCount") or 0) - cache_read
            self.output.usage.output = (usage_metadata.get("candidatesTokenCount") or 0) + (
                usage_metadata.get("thoughtsTokenCount") or 0
            )
            self.output.usage.cache_read = cache_read
            self.output.usage.cache_write = 0
            self.output.usage.reasoning = usage_metadata.get("thoughtsTokenCount") or 0
            self.output.usage.total_tokens = usage_metadata.get("totalTokenCount") or 0
            calculate_cost(self.model, self.output.usage)

    def _handle_part(self, part: dict[str, Any]) -> None:
        text = part.get("text")
        if text is not None:
            is_thinking = is_thinking_part(part)
            if (
                self._current_block is None
                or (is_thinking and not isinstance(self._current_block, ThinkingContent))
                or (not is_thinking and not isinstance(self._current_block, TextContent))
            ):
                self._close_current_block()
                if is_thinking:
                    self._current_block = ThinkingContent(thinking="", thinking_signature=None)
                    self.output.content.append(self._current_block)
                    self.event_stream.push(ThinkingStartEvent(content_index=self._content_index(), partial=self.output))
                else:
                    self._current_block = TextContent(text="")
                    self.output.content.append(self._current_block)
                    self.event_stream.push(TextStartEvent(content_index=self._content_index(), partial=self.output))

            if isinstance(self._current_block, ThinkingContent):
                self._current_block.thinking += text
                self._current_block.thinking_signature = retain_thought_signature(
                    self._current_block.thinking_signature, part.get("thoughtSignature")
                )
                self.event_stream.push(
                    ThinkingDeltaEvent(content_index=self._content_index(), delta=text, partial=self.output)
                )
            else:
                self._current_block.text += text
                self._current_block.text_signature = retain_thought_signature(
                    self._current_block.text_signature, part.get("thoughtSignature")
                )
                self.event_stream.push(
                    TextDeltaEvent(content_index=self._content_index(), delta=text, partial=self.output)
                )

        function_call = part.get("functionCall")
        if function_call:
            self._close_current_block()

            provided_id = function_call.get("id")
            needs_new_id = not provided_id or any(
                block.type == "toolCall" and block.id == provided_id for block in self.output.content
            )
            if needs_new_id:
                self._tool_call_counter += 1
                tool_call_id = f"{function_call.get('name', '')}_{now_ms()}_{self._tool_call_counter}"
            else:
                tool_call_id = provided_id

            tool_call = ToolCall(
                id=tool_call_id,
                name=function_call.get("name") or "",
                arguments=function_call.get("args") or {},
                thought_signature=part.get("thoughtSignature") or None,
            )
            self.output.content.append(tool_call)
            index = self._content_index()
            self.event_stream.push(ToolCallStartEvent(content_index=index, partial=self.output))
            self.event_stream.push(
                ToolCallDeltaEvent(content_index=index, delta=json_stringify(tool_call.arguments), partial=self.output)
            )
            self.event_stream.push(ToolCallEndEvent(content_index=index, tool_call=tool_call, partial=self.output))

    def finalize(self) -> None:
        self._close_current_block()
