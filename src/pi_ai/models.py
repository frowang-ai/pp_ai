"""Model helpers and the provider protocol.

Python port of the runtime helpers in `packages/ai/src/models.ts`. The large
auth/credential-store machinery is ported separately in :mod:`pi_ai.auth`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import (
    THINKING_LEVELS,
    AssistantMessage,
    Context,
    Cost,
    Model,
    ModelCostRates,
    ModelThinkingLevel,
    SimpleStreamOptions,
    StreamOptions,
    Usage,
)
from .utils.diagnostics import format_thrown_value
from .utils.event_stream import AssistantMessageEventStream


class ModelsError(Exception):
    """Error raised by provider/auth resolution.

    ``code`` mirrors ``ModelsErrorCode`` in TypeScript: ``"auth"``, ``"oauth"``,
    ``"provider"``, or ``"model"``.
    """

    def __init__(self, code: str, message: str, cause: BaseException | None = None) -> None:
        # Callers surface the message only, so keep the underlying reason in it
        # (TypeScript's `withCauseDetail`).
        if cause is not None:
            detail = format_thrown_value(cause).strip()
            if detail and detail not in message:
                message = f"{message}: {detail}"
        super().__init__(message)
        self.code = code
        self.__cause__ = cause


def calculate_cost(model: Model, usage: Usage) -> Cost:
    """Fill ``usage.cost`` from the model's rate card and return it."""
    input_tokens = usage.input + usage.cache_read + usage.cache_write
    rates: ModelCostRates = model.cost
    matched_threshold = -1
    for tier in model.cost.tiers:
        if input_tokens > tier.input_tokens_above and tier.input_tokens_above > matched_threshold:
            rates = tier
            matched_threshold = tier.input_tokens_above

    # Anthropic charges 2x base input for 1h cache writes.
    long_write = usage.cache_write_1h or 0
    short_write = usage.cache_write - long_write
    usage.cost.input = (rates.input / 1_000_000) * usage.input
    usage.cost.output = (rates.output / 1_000_000) * usage.output
    usage.cost.cache_read = (rates.cache_read / 1_000_000) * usage.cache_read
    usage.cost.cache_write = (rates.cache_write * short_write + rates.input * 2 * long_write) / 1_000_000
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    return usage.cost


def get_supported_thinking_levels(model: Model) -> list[ModelThinkingLevel]:
    if not model.reasoning:
        return ["off"]

    levels: list[ModelThinkingLevel] = []
    for level in THINKING_LEVELS:
        present = level in model.thinking_level_map
        mapped = model.thinking_level_map.get(level)
        if present and mapped is None:
            continue
        if level in ("xhigh", "max") and not present:
            continue
        levels.append(level)
    return levels


def clamp_thinking_level(model: Model, level: ModelThinkingLevel) -> ModelThinkingLevel:
    available = get_supported_thinking_levels(model)
    if level in available:
        return level

    fallback: ModelThinkingLevel = available[0] if available else "off"
    if level not in THINKING_LEVELS:
        return fallback

    requested_index = THINKING_LEVELS.index(level)
    for candidate in THINKING_LEVELS[requested_index:]:
        if candidate in available:
            return candidate
    for candidate in reversed(THINKING_LEVELS[:requested_index]):
        if candidate in available:
            return candidate
    return fallback


def models_are_equal(a: Model | None, b: Model | None) -> bool:
    """Compare two models by id and provider. Returns False if either is None."""
    if a is None or b is None:
        return False
    return a.id == b.id and a.provider == b.provider


def has_api(model: Model, api: str) -> bool:
    return model.api == api


@runtime_checkable
class ProviderStreams(Protocol):
    """The uniform stream contract implemented by every module in ``pi_ai.api``."""

    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessageEventStream: ...

    def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None
    ) -> AssistantMessageEventStream: ...


@runtime_checkable
class Provider(Protocol):
    """A concrete runtime provider owning metadata, model listing and streaming."""

    id: str
    name: str

    def get_models(self) -> list[Model]: ...

    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessageEventStream: ...

    def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None
    ) -> AssistantMessageEventStream: ...


async def complete(stream: AssistantMessageEventStream) -> AssistantMessage:
    """Drain a stream and return its final assistant message."""
    async for _event in stream:
        pass
    return await stream.result()
