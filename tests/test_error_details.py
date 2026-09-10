"""Tests for `AssistantMessage.error_details` (pi_ai-specific extension).

Provider streams swallow exceptions into a terminal ``error`` event; the HTTP
status previously survived only inside the ``error_message`` string.
`error_details` exposes the status, the server-requested retry delay, and a
normalized `kind` (auth/quota/rate_limit/server/network/aborted/unknown) so
failover logic can branch without parsing that string. Built by
`pi_ai.utils.error_details.build_error_details` and attached by every provider
stream's terminal error path.
"""

from __future__ import annotations

import json

import httpx

from pi_ai import Context, Model, ModelCost, UserMessage
from pi_ai.api.anthropic_messages import AnthropicOptions
from pi_ai.api.anthropic_messages import stream as anthropic_stream
from pi_ai.api.google_generative_ai import GoogleOptions
from pi_ai.api.google_generative_ai import stream as google_stream
from pi_ai.api.openai_completions import OpenAICompletionsOptions
from pi_ai.api.openai_completions import stream as openai_stream
from pi_ai.utils.abort import AbortSignal
from pi_ai.utils.error_details import build_error_details
from pi_ai.utils.http import ProviderHttpError
from pi_ai.utils.provider_retry import ProviderRequestAbortError


def make_model(api: str = "openai-completions", provider: str = "openai", base_url: str = "https://api.openai.com/v1") -> Model:
    return Model(
        id="gpt-test",
        name="GPT Test",
        api=api,
        provider=provider,
        base_url=base_url,
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=1.0, output=2.0),
        context_window=100_000,
        max_tokens=4096,
    )


def make_client(body: str, status: int = 200, headers: dict[str, str] | None = None) -> httpx.AsyncClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body, headers=headers or {})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def collect(event_stream):
    events = [event async for event in event_stream]
    return events, await event_stream.result()


# --------------------------------------------------------------------------
# build_error_details classification
# --------------------------------------------------------------------------


def test_classifies_auth_errors():
    assert build_error_details(ProviderHttpError(401, "bad key")).kind == "auth"
    assert build_error_details(ProviderHttpError(403, "forbidden")).kind == "auth"


def test_classifies_quota_error():
    details = build_error_details(ProviderHttpError(402, "insufficient balance"))
    assert details.kind == "quota"
    assert details.status == 402
    assert details.body == "insufficient balance"


def test_classifies_rate_limit_and_reads_retry_after_headers():
    details = build_error_details(ProviderHttpError(429, "slow down", {"retry-after-ms": "250"}))
    assert details.kind == "rate_limit"
    assert details.status == 429
    assert details.retry_after_ms == 250.0

    seconds = build_error_details(ProviderHttpError(429, "slow down", {"Retry-After": "2"}))
    assert seconds.retry_after_ms == 2000.0


def test_classifies_server_errors():
    assert build_error_details(ProviderHttpError(500, "boom")).kind == "server"
    assert build_error_details(ProviderHttpError(503, "")).kind == "server"


def test_classifies_transport_failures_as_network():
    assert build_error_details(httpx.ConnectError("connection refused")).kind == "network"
    assert build_error_details(httpx.ReadTimeout("timed out")).kind == "network"


def test_classifies_abort():
    assert build_error_details(RuntimeError("Request was aborted"), aborted=True).kind == "aborted"
    assert build_error_details(ProviderRequestAbortError()).kind == "aborted"
    # The explicit flag wins over an otherwise classifiable status.
    assert build_error_details(ProviderHttpError(500, "boom"), aborted=True).kind == "aborted"


def test_unrecognized_errors_are_unknown():
    details = build_error_details(ValueError("nope"))
    assert details.kind == "unknown"
    assert details.status is None
    assert details.retry_after_ms is None


def test_client_4xx_without_dedicated_kind_is_unknown():
    assert build_error_details(ProviderHttpError(400, "bad request")).kind == "unknown"


# --------------------------------------------------------------------------
# provider streams attach error_details to the terminal message
# --------------------------------------------------------------------------


async def test_openai_completions_401_error_event_carries_details():
    async with make_client('{"error": {"message": "bad key"}}', status=401) as client:
        events, message = await collect(
            openai_stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert message.error_details is not None
    assert message.error_details.kind == "auth"
    assert message.error_details.status == 401


async def test_openai_completions_429_error_event_carries_retry_after():
    async with make_client('{"error": {"message": "rate limited"}}', status=429, headers={"retry-after": "1.5"}) as client:
        events, message = await collect(
            openai_stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.error_details is not None
    assert message.error_details.kind == "rate_limit"
    assert message.error_details.status == 429
    assert message.error_details.retry_after_ms == 1500.0


async def test_openai_completions_network_error_event_carries_details():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events, message = await collect(
            openai_stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.error_details is not None
    assert message.error_details.kind == "network"
    assert message.error_details.status is None


async def test_openai_completions_aborted_signal_marks_details_aborted():
    signal = AbortSignal()
    signal.abort()
    body = f'data: {json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})}\n\ndata: [DONE]\n\n'
    async with make_client(body) as client:
        events, message = await collect(
            openai_stream(
                make_model(),
                Context(messages=[UserMessage(content="hi")]),
                OpenAICompletionsOptions(api_key="k", signal=signal),
                client=client,
            )
        )
    assert events[-1].type == "error"
    assert events[-1].reason == "aborted"
    assert message.error_details is not None
    assert message.error_details.kind == "aborted"


async def test_anthropic_500_error_event_carries_details():
    model = make_model(api="anthropic-messages", provider="anthropic", base_url="https://api.anthropic.com/v1")
    async with make_client('{"error": {"message": "overloaded"}}', status=500) as client:
        events, message = await collect(
            anthropic_stream(model, Context(messages=[]), AnthropicOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.error_details is not None
    assert message.error_details.kind == "server"
    assert message.error_details.status == 500


async def test_google_402_error_event_carries_details():
    model = make_model(
        api="google-generative-ai",
        provider="google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
    async with make_client('{"error": {"message": "quota exceeded"}}', status=402) as client:
        events, message = await collect(
            google_stream(model, Context(messages=[]), GoogleOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.error_details is not None
    assert message.error_details.kind == "quota"
    assert message.error_details.status == 402


async def test_successful_stream_leaves_error_details_unset():
    chunk = {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
    async with make_client(body) as client:
        _events, message = await collect(
            openai_stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    assert message.stop_reason == "stop"
    assert message.error_details is None
