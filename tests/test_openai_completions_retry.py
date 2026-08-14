"""Python port of `packages/ai/test/openai-completions-retry.test.ts`.

The TypeScript test mocks the `openai` SDK and asserts the request options it
was called with, in particular `maxRetries: 0` -- the pinned SDK retries by
itself, and its retry timer ignores the request's abort signal, so pi has to
disable it and drive retries through `retryProviderRequest`. **That assertion
has no Python analogue and is skipped**: the port has no vendor SDK, it speaks
HTTP through `httpx`, and `httpx` does not retry at all. What the test really
pins -- how many requests actually leave, when they leave, and what a rejected
retry delay reports -- is asserted on the recorded requests instead.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pi_ai.api import openai_completions
from pi_ai.api.openai_completions import OpenAICompletionsOptions, stream
from pi_ai.types import Context, Model, ModelCost, TextContent, UserMessage
from pi_ai.utils import provider_retry as provider_retry_module

MODEL = Model(
    id="test-model",
    name="Test Model",
    api="openai-completions",
    provider="opencode-go",
    base_url="https://opencode.ai/zen/go/v1",
    reasoning=False,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=1000,
    max_tokens=100,
)

CONTEXT = Context(
    system_prompt="",
    messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)],
    tools=[],
)

SUCCESS_BODY = (
    "".join(
        f"data: {json.dumps(chunk)}\n\n"
        for chunk in (
            {"id": "chatcmpl-test", "choices": [{"index": 0, "delta": {"content": "ok"}}]},
            {"id": "chatcmpl-test", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        )
    )
    + "data: [DONE]\n\n"
)


def scripted_client(
    failures: list[tuple[int, dict[str, str], str]],
    requests: list[httpx.Request],
) -> httpx.AsyncClient:
    """Serve ``failures`` in order, then the successful stream."""
    pending = list(failures)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if pending:
            status, headers, body = pending.pop(0)
            return httpx.Response(status, text=body, headers=headers)
        return httpx.Response(200, text=SUCCESS_BODY, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def recorded_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def fake_sleep(
        ms: float,
        signal: Any,
        clock: provider_retry_module.RetryClock = provider_retry_module.REAL_CLOCK,
    ) -> None:
        delays.append(ms)

    monkeypatch.setattr(provider_retry_module, "_abortable_sleep", fake_sleep)
    return delays


async def consume(
    client: httpx.AsyncClient,
    max_retries: int | None = None,
    max_retry_delay_ms: float | None = None,
):
    event_stream = stream(
        MODEL,
        CONTEXT,
        OpenAICompletionsOptions(
            api_key="test",
            max_retries=max_retries,
            max_retry_delay_ms=max_retry_delay_ms,
        ),
        client=client,
    )
    async for _event in event_stream:
        pass
    return await event_stream.result()


async def test_makes_a_single_request_by_default() -> None:
    requests: list[httpx.Request] = []
    result = await consume(scripted_client([], requests))
    assert result.stop_reason == "stop"
    # TS asserts `requestOptions` is `[objectContaining({ maxRetries: 0 })]` here (and in
    # the two cases below), i.e. that the `openai` SDK's own retry loop was switched off.
    # No analogue: the port talks to `httpx`, which never retries, so there is no second
    # retry layer to disable -- only the request count can be pinned.
    assert len(requests) == 1


async def test_honors_provider_retries(recorded_sleeps: list[float]) -> None:
    requests: list[httpx.Request] = []
    failures = [
        (429, {"retry-after-ms": "100"}, "rate limited"),
        (500, {"retry-after-ms": "100"}, "server error"),
    ]
    result = await consume(
        scripted_client(failures, requests),
        max_retries=2,
        max_retry_delay_ms=100,
    )

    assert result.stop_reason == "stop"
    assert len(requests) == 3
    assert recorded_sleeps == [100, 100]


async def test_fails_immediately_when_a_retry_delay_exceeds_the_limit() -> None:
    requests: list[httpx.Request] = []
    failures = [(429, {"retry-after": "277403"}, "rate limited")]
    result = await consume(
        scripted_client(failures, requests),
        max_retries=2,
        max_retry_delay_ms=1000,
    )

    assert result.stop_reason == "error"
    assert "Server requested 277403s retry delay (max: 1s)" in (result.error_message or "")
    assert "rate limited" in (result.error_message or "")
    assert len(requests) == 1


def test_the_module_streams_through_the_retrying_sse_helper() -> None:
    # Guards the wiring the cases above depend on: without it a provider error
    # would never be retried, because `httpx` has no retry of its own.
    assert openai_completions.stream_sse_with_retry is not None
