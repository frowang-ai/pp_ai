"""Scripted/fake provider used for deterministic testing.

Python port of `packages/ai/src/providers/faux.ts`. Tests queue one
`AssistantMessage` (or a response factory) per expected model call, then drive
`stream`/`stream_simple` and assert on the emitted event sequence or the final
message returned by `pi_ai.models.complete`. The provider fabricates a
plausible `Usage` from the serialized request context so cost/usage-dependent
code can be exercised without a real model.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias

from ..auth.types import ApiKeyAuth, AuthResult, ProviderAuth, ResolvedAuth
from ..registry import Provider, create_provider
from ..types import (
    AssistantMessage,
    Context,
    Cost,
    DeferredHandle,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Message,
    Model,
    ModelCost,
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
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    Usage,
    UserContent,
    now_ms,
)
from ..utils.abort import AbortSignal
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.tasks import spawn

DEFAULT_API = "faux"
DEFAULT_PROVIDER = "faux"
DEFAULT_MODEL_ID = "faux-1"
DEFAULT_MODEL_NAME = "Faux Model"
DEFAULT_BASE_URL = "http://localhost:0"
DEFAULT_MIN_TOKEN_SIZE = 3
DEFAULT_MAX_TOKEN_SIZE = 5

_BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def _default_usage() -> Usage:
    return Usage()


FauxContentBlock: TypeAlias = "TextContent | ThinkingContent | ToolCall"


def faux_text(text: str) -> TextContent:
    return TextContent(text=text)


def faux_thinking(thinking: str) -> ThinkingContent:
    return ThinkingContent(thinking=thinking)


def faux_tool_call(name: str, arguments: dict[str, Any], *, id: str | None = None) -> ToolCall:
    return ToolCall(id=id or random_id("tool"), name=name, arguments=arguments)


def _normalize_faux_assistant_content(
    content: str | FauxContentBlock | list[FauxContentBlock],
) -> list[FauxContentBlock]:
    if isinstance(content, str):
        return [faux_text(content)]
    if isinstance(content, list):
        return list(content)
    return [content]


def faux_assistant_message(
    content: str | FauxContentBlock | list[FauxContentBlock],
    *,
    stop_reason: str = "stop",
    deferred: DeferredHandle | None = None,
    error_message: str | None = None,
    response_id: str | None = None,
    timestamp: int | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=_normalize_faux_assistant_content(content),
        api=DEFAULT_API,
        provider=DEFAULT_PROVIDER,
        model=DEFAULT_MODEL_ID,
        usage=_default_usage(),
        stop_reason=stop_reason,
        deferred=deferred,
        error_message=error_message,
        response_id=response_id,
        timestamp=timestamp if timestamp is not None else now_ms(),
    )


@dataclass
class FauxModelCost:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass
class FauxModelDefinition:
    id: str
    name: str | None = None
    reasoning: bool = False
    input: list[str] | None = None
    cost: FauxModelCost | None = None
    context_window: int | None = None
    max_tokens: int | None = None


@dataclass
class FauxDeferredOptions:
    """Number of fetches that return the original handle before the scripted
    response becomes ready, plus the poll-after hint attached to the handle."""

    pending_fetches: int | None = None
    poll_after_ms: int | None = None


@dataclass
class FauxTokenSizeOptions:
    min: int | None = None
    max: int | None = None


@dataclass
class RegisterFauxProviderOptions:
    api: str | None = None
    provider: str | None = None
    models: list[FauxModelDefinition] | None = None
    deferred: FauxDeferredOptions | None = None
    tokens_per_second: float | None = None
    token_size: FauxTokenSizeOptions | None = None


@dataclass
class FauxProviderState:
    call_count: int = 0
    deferred_fetch_count: int = 0
    cancelled_deferred: list[DeferredHandle] = field(default_factory=list)


FauxResponseFactory: TypeAlias = Callable[
    [Context, SimpleStreamOptions | None, FauxProviderState, Model],
    "AssistantMessage | Awaitable[AssistantMessage]",
]
FauxResponseStep: TypeAlias = "AssistantMessage | FauxResponseFactory"


def estimate_tokens(text: str) -> int:
    return -(-len(text) // 4)


def random_id(prefix: str) -> str:
    suffix = "".join(random.choice(_BASE36_ALPHABET) for _ in range(11))
    return f"{prefix}:{now_ms()}:{suffix}"


def _content_to_text(content: str | list[UserContent]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ImageContent):
            parts.append(f"[image:{block.mime_type}:{len(block.data)}]")
    return "\n".join(parts)


def _assistant_content_to_text(content: list[TextContent | ThinkingContent | ToolCall]) -> str:
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ThinkingContent):
            parts.append(block.thinking)
        else:
            parts.append(f"{block.name}:{json.dumps(block.arguments)}")
    return "\n".join(parts)


def _tool_result_to_text(message: ToolResultMessage) -> str:
    parts = [message.tool_name] + [_content_to_text([block]) for block in message.content]
    return "\n".join(parts)


def _message_to_text(message: Message) -> str:
    if message.role == "user":
        return _content_to_text(message.content)
    if message.role == "assistant":
        return _assistant_content_to_text(message.content)
    return _tool_result_to_text(message)


def _tool_to_json(tool: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
    if tool.constrained_sampling is not None:
        payload["constrainedSampling"] = tool.constrained_sampling
    return payload


def serialize_context(context: Context) -> str:
    parts: list[str] = []
    if context.system_prompt:
        parts.append(f"system:{context.system_prompt}")
    for message in context.messages:
        parts.append(f"{message.role}:{_message_to_text(message)}")
    if context.tools:
        parts.append(f"tools:{json.dumps([_tool_to_json(tool) for tool in context.tools])}")
    return "\n\n".join(parts)


def _common_prefix_length(a: str, b: str) -> int:
    length = min(len(a), len(b))
    index = 0
    while index < length and a[index] == b[index]:
        index += 1
    return index


def _with_usage_estimate(
    message: AssistantMessage,
    context: Context,
    options: StreamOptions | None,
    prompt_cache: dict[str, str],
) -> AssistantMessage:
    prompt_text = serialize_context(context)
    prompt_tokens = estimate_tokens(prompt_text)
    output_tokens = estimate_tokens(_assistant_content_to_text(message.content))
    input_tokens = prompt_tokens
    cache_read = 0
    cache_write = 0
    session_id = options.session_id if options else None

    if session_id and (options is None or options.cache_retention != "none"):
        previous_prompt = prompt_cache.get(session_id)
        if previous_prompt:
            cached_chars = _common_prefix_length(previous_prompt, prompt_text)
            cache_read = estimate_tokens(previous_prompt[:cached_chars])
            cache_write = estimate_tokens(prompt_text[cached_chars:])
            input_tokens = max(0, prompt_tokens - cache_read)
        else:
            cache_write = prompt_tokens
        prompt_cache[session_id] = prompt_text

    message.usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input_tokens + output_tokens + cache_read + cache_write,
        cost=Cost(),
    )
    return message


def _split_string_by_token_size(text: str, min_token_size: int, max_token_size: int) -> list[str]:
    chunks: list[str] = []
    index = 0
    while index < len(text):
        token_size = min_token_size + random.randint(0, max_token_size - min_token_size)
        char_size = max(1, token_size * 4)
        chunks.append(text[index : index + char_size])
        index += char_size
    return chunks if chunks else [""]


def _clone_message(message: AssistantMessage, api: str, provider: str, model_id: str) -> AssistantMessage:
    cloned = copy.deepcopy(message)
    cloned.api = api
    cloned.provider = provider
    cloned.model = model_id
    cloned.timestamp = cloned.timestamp or now_ms()
    cloned.usage = cloned.usage or _default_usage()
    return cloned


def _create_deferred_message(model: Model, handle: DeferredHandle) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=_default_usage(),
        stop_reason="deferred",
        deferred=handle,
        timestamp=now_ms(),
    )


def _create_error_message(error: BaseException | str, api: str, provider: str, model_id: str) -> AssistantMessage:
    message = str(error) or type(error).__name__ if isinstance(error, BaseException) else str(error)
    return AssistantMessage(
        role="assistant",
        content=[],
        api=api,
        provider=provider,
        model=model_id,
        usage=_default_usage(),
        stop_reason="error",
        error_message=message,
        timestamp=now_ms(),
    )


def _create_aborted_message(partial: AssistantMessage) -> AssistantMessage:
    return replace(partial, stop_reason="aborted", error_message="Request was aborted", timestamp=now_ms())


async def _schedule_chunk(chunk: str, tokens_per_second: float | None) -> None:
    if not tokens_per_second or tokens_per_second <= 0:
        await asyncio.sleep(0)
        return
    delay_seconds = estimate_tokens(chunk) / tokens_per_second
    await asyncio.sleep(delay_seconds)


def _snapshot(partial: AssistantMessage) -> AssistantMessage:
    """Shallow copy, mirroring the TypeScript `{ ...partial }` spread."""
    return replace(partial)


class _FauxStreamEndedWithoutStopReason(RuntimeError):
    pass


async def _stream_with_deltas(
    out_stream: AssistantMessageEventStream,
    message: AssistantMessage,
    min_token_size: int,
    max_token_size: int,
    tokens_per_second: float | None,
    signal: AbortSignal | None,
) -> None:
    partial = replace(message, content=[], stop_reason="pending")
    if signal is not None and signal.aborted:
        aborted = _create_aborted_message(partial)
        out_stream.push(ErrorEvent(reason="aborted", error=aborted))
        out_stream.end(aborted)
        return

    out_stream.push(StartEvent(partial=_snapshot(partial)))

    for index, block in enumerate(message.content):
        if signal is not None and signal.aborted:
            aborted = _create_aborted_message(partial)
            out_stream.push(ErrorEvent(reason="aborted", error=aborted))
            out_stream.end(aborted)
            return

        if isinstance(block, ThinkingContent):
            new_block = ThinkingContent(thinking="")
            partial.content = [*partial.content, new_block]
            out_stream.push(ThinkingStartEvent(content_index=index, partial=_snapshot(partial)))
            for chunk in _split_string_by_token_size(block.thinking, min_token_size, max_token_size):
                await _schedule_chunk(chunk, tokens_per_second)
                if signal is not None and signal.aborted:
                    aborted = _create_aborted_message(partial)
                    out_stream.push(ErrorEvent(reason="aborted", error=aborted))
                    out_stream.end(aborted)
                    return
                new_block.thinking += chunk
                out_stream.push(ThinkingDeltaEvent(content_index=index, delta=chunk, partial=_snapshot(partial)))
            out_stream.push(ThinkingEndEvent(content_index=index, content=block.thinking, partial=_snapshot(partial)))
            continue

        if isinstance(block, TextContent):
            new_text_block = TextContent(text="")
            partial.content = [*partial.content, new_text_block]
            out_stream.push(TextStartEvent(content_index=index, partial=_snapshot(partial)))
            for chunk in _split_string_by_token_size(block.text, min_token_size, max_token_size):
                await _schedule_chunk(chunk, tokens_per_second)
                if signal is not None and signal.aborted:
                    aborted = _create_aborted_message(partial)
                    out_stream.push(ErrorEvent(reason="aborted", error=aborted))
                    out_stream.end(aborted)
                    return
                new_text_block.text += chunk
                out_stream.push(TextDeltaEvent(content_index=index, delta=chunk, partial=_snapshot(partial)))
            out_stream.push(TextEndEvent(content_index=index, content=block.text, partial=_snapshot(partial)))
            continue

        new_tool_block = ToolCall(id=block.id, name=block.name, arguments={})
        partial.content = [*partial.content, new_tool_block]
        out_stream.push(ToolCallStartEvent(content_index=index, partial=_snapshot(partial)))
        for chunk in _split_string_by_token_size(json.dumps(block.arguments), min_token_size, max_token_size):
            await _schedule_chunk(chunk, tokens_per_second)
            if signal is not None and signal.aborted:
                aborted = _create_aborted_message(partial)
                out_stream.push(ErrorEvent(reason="aborted", error=aborted))
                out_stream.end(aborted)
                return
            out_stream.push(ToolCallDeltaEvent(content_index=index, delta=chunk, partial=_snapshot(partial)))
        new_tool_block.arguments = block.arguments
        out_stream.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=_snapshot(partial)))

    if message.stop_reason == "pending":
        raise _FauxStreamEndedWithoutStopReason("Faux response ended without a stop reason")

    if message.stop_reason in ("error", "aborted"):
        out_stream.push(ErrorEvent(reason=message.stop_reason, error=message))
        out_stream.end(message)
        return

    out_stream.push(DoneEvent(reason=message.stop_reason, message=message))
    out_stream.end(message)


@dataclass
class _DeferredEntry:
    handle: DeferredHandle
    step: FauxResponseStep
    context: Context
    options: SimpleStreamOptions | None
    model: Model
    pending_fetches: int
    cancelled: bool = False
    final: AssistantMessage | None = None


@dataclass
class FauxCore:
    """The scripting API shared by `faux_provider` and `compat.register_faux_provider`."""

    api: str
    provider: str
    models: list[Model]
    state: FauxProviderState
    stream: Callable[..., AssistantMessageEventStream]
    stream_simple: Callable[..., AssistantMessageEventStream]
    fetch_deferred: Callable[..., AssistantMessageEventStream]
    cancel_deferred: Callable[..., Awaitable[None]]
    get_model: Callable[[str | None], Model | None]
    set_responses: Callable[[list[FauxResponseStep]], None]
    append_responses: Callable[[list[FauxResponseStep]], None]
    get_pending_response_count: Callable[[], int]


def create_faux_core(options: RegisterFauxProviderOptions | None = None) -> FauxCore:
    options = options or RegisterFauxProviderOptions()
    api = options.api or random_id(DEFAULT_API)
    provider = options.provider or DEFAULT_PROVIDER
    token_size = options.token_size or FauxTokenSizeOptions()
    requested_min = token_size.min if token_size.min is not None else DEFAULT_MIN_TOKEN_SIZE
    requested_max = token_size.max if token_size.max is not None else DEFAULT_MAX_TOKEN_SIZE
    min_token_size = max(1, min(requested_min, requested_max))
    max_token_size = max(min_token_size, requested_max)
    tokens_per_second = options.tokens_per_second

    pending_responses: list[FauxResponseStep] = []
    state = FauxProviderState()
    prompt_cache: dict[str, str] = {}
    deferred_responses: dict[str, _DeferredEntry] = {}

    model_definitions = options.models or [
        FauxModelDefinition(
            id=DEFAULT_MODEL_ID,
            name=DEFAULT_MODEL_NAME,
            reasoning=False,
            input=["text", "image"],
            cost=FauxModelCost(),
            context_window=128_000,
            max_tokens=16_384,
        )
    ]
    models: list[Model] = []
    for definition in model_definitions:
        cost = definition.cost or FauxModelCost()
        models.append(
            Model(
                id=definition.id,
                name=definition.name or definition.id,
                api=api,
                provider=provider,
                base_url=DEFAULT_BASE_URL,
                reasoning=definition.reasoning,
                input=definition.input or ["text", "image"],
                cost=ModelCost(
                    input=cost.input, output=cost.output, cache_read=cost.cache_read, cache_write=cost.cache_write
                ),
                context_window=definition.context_window or 128_000,
                max_tokens=definition.max_tokens or 16_384,
            )
        )

    def get_model(model_id: str | None = None) -> Model | None:
        if model_id is None:
            return models[0] if models else None
        return next((candidate for candidate in models if candidate.id == model_id), None)

    async def resolve_response(
        step: FauxResponseStep,
        context: Context,
        stream_options: SimpleStreamOptions | None,
        request_model: Model,
    ) -> AssistantMessage:
        if callable(step):
            resolved = step(context, stream_options, state, request_model)
            if inspect.isawaitable(resolved):
                resolved = await resolved
        else:
            resolved = step
        return _with_usage_estimate(
            _clone_message(resolved, api, provider, request_model.id), context, stream_options, prompt_cache
        )

    def stream(
        request_model: Model,
        context: Context,
        stream_options: SimpleStreamOptions | None = None,
        client: Any = None,
    ) -> AssistantMessageEventStream:
        out_stream = AssistantMessageEventStream()
        step = pending_responses.pop(0) if pending_responses else None
        state.call_count += 1

        async def produce() -> None:
            try:
                if stream_options is not None and stream_options.on_response is not None:
                    from ..types import ProviderResponse

                    result = stream_options.on_response(ProviderResponse(status=200, headers={}), request_model)
                    if inspect.isawaitable(result):
                        await result

                if step is None:
                    message = _create_error_message(
                        RuntimeError("No more faux responses queued"), api, provider, request_model.id
                    )
                    message = _with_usage_estimate(message, context, stream_options, prompt_cache)
                    out_stream.push(ErrorEvent(reason="error", error=message))
                    out_stream.end(message)
                    return

                if stream_options is not None and stream_options.deferred:
                    handle = DeferredHandle(
                        provider=request_model.provider,
                        model_id=request_model.id,
                        api=request_model.api,
                        id=random_id("deferred"),
                        poll_after_ms=options.deferred.poll_after_ms if options.deferred else None,
                    )
                    deferred_responses[handle.id] = _DeferredEntry(
                        handle=handle,
                        step=step,
                        context=context,
                        options=stream_options,
                        model=request_model,
                        pending_fetches=max(0, int(options.deferred.pending_fetches or 0)) if options.deferred else 0,
                    )
                    await _stream_with_deltas(
                        out_stream,
                        _create_deferred_message(request_model, handle),
                        min_token_size,
                        max_token_size,
                        tokens_per_second,
                        stream_options.signal,
                    )
                    return

                message = await resolve_response(step, context, stream_options, request_model)
                await _stream_with_deltas(
                    out_stream,
                    message,
                    min_token_size,
                    max_token_size,
                    tokens_per_second,
                    stream_options.signal if stream_options is not None else None,
                )
            except Exception as error:
                message = _create_error_message(error, api, provider, request_model.id)
                out_stream.push(ErrorEvent(reason="error", error=message))
                out_stream.end(message)

        spawn(produce())
        return out_stream

    def stream_simple(
        request_model: Model,
        context: Context,
        stream_options: SimpleStreamOptions | None = None,
        client: Any = None,
    ) -> AssistantMessageEventStream:
        return stream(request_model, context, stream_options, client=client)

    def fetch_deferred(
        request_model: Model,
        handle: DeferredHandle,
        fetch_options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        out_stream = AssistantMessageEventStream()
        state.deferred_fetch_count += 1

        async def produce() -> None:
            try:
                if fetch_options is not None and fetch_options.on_response is not None:
                    from ..types import ProviderResponse

                    result = fetch_options.on_response(ProviderResponse(status=200, headers={}), request_model)
                    if inspect.isawaitable(result):
                        await result

                entry = deferred_responses.get(handle.id)
                if (
                    entry is None
                    or entry.handle.provider != handle.provider
                    or entry.handle.model_id != handle.model_id
                    or entry.handle.api != handle.api
                ):
                    raise RuntimeError(f"Unknown faux deferred response: {handle.id}")
                if entry.cancelled:
                    raise RuntimeError(f"Faux deferred response was cancelled: {handle.id}")

                signal = fetch_options.signal if fetch_options is not None else None

                if entry.pending_fetches > 0:
                    entry.pending_fetches -= 1
                    await _stream_with_deltas(
                        out_stream,
                        _create_deferred_message(request_model, entry.handle),
                        min_token_size,
                        max_token_size,
                        tokens_per_second,
                        signal,
                    )
                    return

                if entry.final is None:
                    submission_options = (
                        replace(entry.options, deferred=None, signal=None, on_response=None)
                        if entry.options is not None
                        else None
                    )
                    try:
                        entry.final = await resolve_response(entry.step, entry.context, submission_options, entry.model)
                    except Exception as error:
                        entry.final = _create_error_message(error, api, provider, entry.model.id)

                await _stream_with_deltas(
                    out_stream, entry.final, min_token_size, max_token_size, tokens_per_second, signal
                )
            except Exception as error:
                message = _create_error_message(error, api, provider, request_model.id)
                out_stream.push(ErrorEvent(reason="error", error=message))
                out_stream.end(message)

        spawn(produce())
        return out_stream

    async def cancel_deferred(
        request_model: Model,
        handle: DeferredHandle,
        cancel_options: StreamOptions | None = None,
    ) -> None:
        state.cancelled_deferred.append(copy.deepcopy(handle))
        entry = deferred_responses.get(handle.id)
        if entry is not None:
            entry.cancelled = True
        if cancel_options is not None and cancel_options.on_response is not None:
            from ..types import ProviderResponse

            result = cancel_options.on_response(ProviderResponse(status=200, headers={}), request_model)
            if inspect.isawaitable(result):
                await result

    def set_responses(responses: list[FauxResponseStep]) -> None:
        pending_responses[:] = list(responses)

    def append_responses(responses: list[FauxResponseStep]) -> None:
        pending_responses.extend(responses)

    def get_pending_response_count() -> int:
        return len(pending_responses)

    return FauxCore(
        api=api,
        provider=provider,
        models=models,
        state=state,
        stream=stream,
        stream_simple=stream_simple,
        fetch_deferred=fetch_deferred,
        cancel_deferred=cancel_deferred,
        get_model=get_model,
        set_responses=set_responses,
        append_responses=append_responses,
        get_pending_response_count=get_pending_response_count,
    )


@dataclass
class FauxProviderHandle:
    """Faux provider for tests built on explicit `Models` collections.

    ```python
    faux = faux_provider()
    models = Models()
    models.add(faux.provider)
    faux.set_responses([faux_assistant_message("hi")])
    ```
    """

    provider: Provider
    api: str
    models: list[Model]
    state: FauxProviderState
    get_model: Callable[[str | None], Model | None]
    set_responses: Callable[[list[FauxResponseStep]], None]
    append_responses: Callable[[list[FauxResponseStep]], None]
    get_pending_response_count: Callable[[], int]


class _FauxAuthResolve:
    def __call__(self, *, credential: Any = None, env: Any = None) -> AuthResult:
        return AuthResult(auth=ResolvedAuth(), source="faux")


def faux_provider(options: RegisterFauxProviderOptions | None = None) -> FauxProviderHandle:
    core = create_faux_core(options)
    faux_auth = ApiKeyAuth(name="Faux", resolve=_FauxAuthResolve())
    api_module = _FauxApiModule(core)
    registry_provider = create_provider(
        id=core.provider,
        name="Faux",
        auth=ProviderAuth(api_key=faux_auth),
        api=api_module,
        models=core.models,
    )
    return FauxProviderHandle(
        provider=registry_provider,
        api=core.api,
        models=core.models,
        state=core.state,
        get_model=core.get_model,
        set_responses=core.set_responses,
        append_responses=core.append_responses,
        get_pending_response_count=core.get_pending_response_count,
    )


@dataclass
class _FauxApiModule:
    """Adapts `FauxCore` to the `ApiModule` protocol expected by `registry.Provider`."""

    core: FauxCore

    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream:
        return self.core.stream(model, context, options, **kwargs)

    def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream:
        return self.core.stream_simple(model, context, options, **kwargs)

    def fetch_deferred(
        self, model: Model, handle: DeferredHandle, options: StreamOptions | None = None
    ) -> AssistantMessageEventStream:
        return self.core.fetch_deferred(model, handle, options)

    async def cancel_deferred(self, model: Model, handle: DeferredHandle, options: StreamOptions | None = None) -> None:
        await self.core.cancel_deferred(model, handle, options)


__all__ = [
    "DEFAULT_API",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_TOKEN_SIZE",
    "DEFAULT_MIN_TOKEN_SIZE",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_PROVIDER",
    "FauxCore",
    "FauxDeferredOptions",
    "FauxModelCost",
    "FauxModelDefinition",
    "FauxProviderHandle",
    "FauxProviderState",
    "FauxResponseFactory",
    "FauxResponseStep",
    "FauxTokenSizeOptions",
    "RegisterFauxProviderOptions",
    "create_faux_core",
    "estimate_tokens",
    "faux_assistant_message",
    "faux_provider",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
    "random_id",
    "serialize_context",
]
