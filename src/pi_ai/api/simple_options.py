"""Shared sampling-option helpers for `simple_stream` style provider APIs.

Python port of `packages/ai/src/api/simple-options.ts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeVar

from ..types import Context, Model, SimpleStreamOptions, StreamOptions, ThinkingBudgets, ThinkingLevel
from ..utils.estimate import estimate_context_tokens

_CONTEXT_SAFETY_TOKENS = 4096
_MIN_MAX_TOKENS = 1

MIN_ANSWER_TOKENS = 1024
"""Tokens always left for the answer when a thinking budget shares the response ceiling."""


def clamp_max_tokens_to_context(model: Model, context: Context, max_tokens: int) -> int:
    if model.context_window <= 0:
        return max(_MIN_MAX_TOKENS, max_tokens)
    available = model.context_window - estimate_context_tokens(context).tokens - _CONTEXT_SAFETY_TOKENS
    return min(max_tokens, max(_MIN_MAX_TOKENS, available))


def build_base_options(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    api_key: str | None = None,
) -> StreamOptions:
    sampling_params = (
        {**model.sampling_params, **(options.sampling_params if options is not None else {})}
        if model.sampling_params or (options is not None and options.sampling_params)
        else {}
    )
    return StreamOptions(
        temperature=options.temperature if options is not None else None,
        sampling_params=sampling_params,
        max_tokens=clamp_max_tokens_to_context(
            model, context, (options.max_tokens if options is not None else None) or model.max_tokens
        ),
        signal=options.signal if options is not None else None,
        telemetry_context=options.telemetry_context if options is not None else None,
        api_key=api_key or (options.api_key if options is not None else None),
        transport=options.transport if options is not None else None,
        cache_retention=options.cache_retention if options is not None else None,
        session_id=options.session_id if options is not None else None,
        headers=options.headers if options is not None else {},
        on_payload=options.on_payload if options is not None else None,
        on_response=options.on_response if options is not None else None,
        timeout_ms=options.timeout_ms if options is not None else None,
        websocket_connect_timeout_ms=options.websocket_connect_timeout_ms if options is not None else None,
        max_retries=options.max_retries if options is not None else None,
        max_retry_delay_ms=options.max_retry_delay_ms if options is not None else None,
        metadata=options.metadata if options is not None else {},
        env=options.env if options is not None else {},
    )


def clamp_reasoning(effort: ThinkingLevel | None) -> Literal["minimal", "low", "medium", "high"] | None:
    return "high" if effort in ("xhigh", "max") else effort


@dataclass
class AdjustedMaxTokens:
    max_tokens: int
    thinking_budget: int


def adjust_max_tokens_for_thinking(
    # None means no explicit caller cap. Use the model cap and fit thinking inside it.
    base_max_tokens: int | None,
    model_max_tokens: int,
    reasoning_level: ThinkingLevel,
    custom_budgets: ThinkingBudgets | None = None,
) -> AdjustedMaxTokens:
    budgets: dict[str, int] = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384}
    if custom_budgets is not None:
        for key in budgets:
            override = getattr(custom_budgets, key, None)
            if override is not None:
                budgets[key] = override

    level = clamp_reasoning(reasoning_level)
    assert level is not None
    thinking_budget = budgets[level]
    max_tokens = (
        model_max_tokens if base_max_tokens is None else min(base_max_tokens + thinking_budget, model_max_tokens)
    )

    if max_tokens <= thinking_budget:
        thinking_budget = max(0, max_tokens - MIN_ANSWER_TOKENS)

    return AdjustedMaxTokens(max_tokens=max_tokens, thinking_budget=thinking_budget)


_ProviderOptionsT = TypeVar("_ProviderOptionsT", bound=StreamOptions)


def as_provider_options(
    options: StreamOptions | None, provider_options_type: type[_ProviderOptionsT]
) -> _ProviderOptionsT:
    """Widen a base :class:`StreamOptions` into an adapter's own options type.

    TypeScript's `ProviderStreamOptions<TApi> = StreamOptions & Record<string,
    unknown>` is structural: reading a provider-only key such as `toolChoice`
    off a plain options object yields `undefined`. Python dataclasses raise
    `AttributeError` instead, so every adapter entry point normalizes first and
    the provider-only fields keep their declared defaults.
    """
    if options is None:
        return provider_options_type()
    if isinstance(options, provider_options_type):
        return options
    fields = provider_options_type.__dataclass_fields__
    return provider_options_type(
        **{name: getattr(options, name) for name in options.__dataclass_fields__ if name in fields}
    )
