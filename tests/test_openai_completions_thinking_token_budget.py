"""Python port of `packages/ai/test/openai-completions-thinking-token-budget.test.ts`.

The TypeScript test mocks the `openai` SDK and reads the params it was called
with. The port captures the same payload through the `on_payload` hook and
serves a canned finish chunk over an `httpx.MockTransport`, so the payload is
observed on a request that actually completes.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import httpx
from pi_ai.api import openai_completions
from pi_ai.types import (
    Context,
    Model,
    ModelCost,
    SimpleStreamOptions,
    ThinkingBudgets,
    UserMessage,
    now_ms,
)

# vLLM-served reasoning model: reasoning and the answer share max_tokens.
VLLM_MODEL = Model(
    id="zai-org/glm-5.2",
    name="GLM 5.2 (local vLLM)",
    api="openai-completions",
    provider="local-vllm",
    base_url="http://localhost:8000/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=262144,
    max_tokens=16384,
    compat={"thinkingFormat": "zai", "supportsThinkingTokenBudget": True},
)

FINISH_CHUNK = {
    "choices": [{"delta": {}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
}


def finish_client() -> httpx.AsyncClient:
    body = f"data: {json.dumps(FINISH_CHUNK)}\n\ndata: [DONE]\n\n"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def capture(
    model: Model,
    reasoning: str | None = None,
    thinking_budgets: ThinkingBudgets | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def on_payload(payload: dict[str, Any], _model: Model) -> None:
        captured["payload"] = payload

    options = SimpleStreamOptions(
        api_key="test",
        reasoning=reasoning,
        thinking_budgets=thinking_budgets,
        max_tokens=max_tokens,
        on_payload=on_payload,
    )
    await openai_completions.stream_simple(
        model,
        Context(messages=[UserMessage(content="Hi", timestamp=now_ms())]),
        options,
        client=finish_client(),
    ).result()

    assert "payload" in captured
    return captured["payload"]


async def test_sends_the_configured_budget_for_the_requested_level() -> None:
    params = await capture(VLLM_MODEL, reasoning="medium", thinking_budgets=ThinkingBudgets(medium=4096))
    assert params["thinking_token_budget"] == 4096


async def test_omits_the_budget_when_the_compat_flag_is_not_set() -> None:
    model = dataclasses.replace(VLLM_MODEL, compat={"thinkingFormat": "zai"})
    params = await capture(model, reasoning="medium", thinking_budgets=ThinkingBudgets(medium=4096))
    assert "thinking_token_budget" not in params


async def test_omits_the_budget_when_thinking_is_off() -> None:
    params = await capture(VLLM_MODEL, reasoning=None, thinking_budgets=ThinkingBudgets(high=8192))
    assert "thinking_token_budget" not in params


async def test_clamps_xhigh_and_max_to_the_high_budget() -> None:
    xhigh = await capture(VLLM_MODEL, reasoning="xhigh", thinking_budgets=ThinkingBudgets(high=8192))
    maximum = await capture(VLLM_MODEL, reasoning="max", thinking_budgets=ThinkingBudgets(high=8192))
    assert xhigh["thinking_token_budget"] == 8192
    assert maximum["thinking_token_budget"] == 8192


async def test_leaves_room_for_the_answer_when_the_budget_meets_the_response_ceiling() -> None:
    # The default high budget (16384) equals the model ceiling, which would
    # leave no room for the answer.
    params = await capture(VLLM_MODEL, reasoning="high")
    assert params["thinking_token_budget"] == 16384 - 1024


async def test_uses_the_caller_max_tokens_as_the_ceiling_when_lower_than_the_model_cap() -> None:
    params = await capture(
        VLLM_MODEL,
        reasoning="high",
        thinking_budgets=ThinkingBudgets(high=8192),
        max_tokens=4096,
    )
    assert params["thinking_token_budget"] == 4096 - 1024
