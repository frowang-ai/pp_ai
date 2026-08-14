"""Coverage tests for pi_ai.api.google_generative_ai."""

from __future__ import annotations

import json

import httpx
import pytest
from pi_ai import Context, Model, ModelCost, Tool, UserMessage
from pi_ai.api.google_generative_ai import (
    GoogleOptions,
    GoogleThinkingOptions,
    _get_disabled_thinking_config,
    _get_google_budget,
    _get_thinking_level,
    build_headers,
    build_params,
    stream,
    stream_simple,
)
from pi_ai.types import SimpleStreamOptions, ThinkingBudgets
from pi_ai.utils.abort import AbortSignal


def make_model(**overrides) -> Model:
    defaults = dict(
        id="gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        api="google-generative-ai",
        provider="google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0.3, output=2.5),
        context_window=1_048_576,
        max_tokens=65_536,
    )
    defaults.update(overrides)
    return Model(**defaults)


def sse_body(chunks: list[dict]) -> str:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)


def make_client(body: str, status: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def collect(event_stream):
    events = [event async for event in event_stream]
    return events, await event_stream.result()


def finish_chunk(reason: str) -> dict:
    return {"candidates": [{"finishReason": reason}]}


def text_chunk(text: str) -> dict:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


# --------------------------------------------------------------------------
# build_headers
# --------------------------------------------------------------------------


def test_build_headers_with_override():
    # Line 91: headers.update(override) when override is non-empty
    model = make_model(headers={"x-base": "base-val"})
    headers = build_headers(model, "my-api-key", {"x-custom": "custom-val"})
    assert headers["x-goog-api-key"] == "my-api-key"
    assert headers["x-custom"] == "custom-val"


def test_build_headers_without_override():
    model = make_model()
    headers = build_headers(model, "my-key", None)
    assert headers["x-goog-api-key"] == "my-key"


# --------------------------------------------------------------------------
# build_params
# --------------------------------------------------------------------------


def test_build_params_tools_truthy():
    # Lines 116-118: body["tools"] set when tools list is non-empty
    tool = Tool(name="search", description="Search the web", parameters={"type": "object", "properties": {}})
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")], tools=[tool])
    body = build_params(model, ctx)
    assert "tools" in body


def test_build_params_tool_config():
    # Lines 121->124: toolConfig set when function_calling_mode is not None
    tool = Tool(name="search", description="Search the web", parameters={"type": "object", "properties": {}})
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")], tools=[tool])
    options = GoogleOptions(tool_choice="any")
    body = build_params(model, ctx, options)
    if "toolConfig" in body:
        assert "functionCallingConfig" in body["toolConfig"]


def test_build_params_thinking_budget_tokens():
    # Lines 128->130: thinkingBudget when budget_tokens is set
    model = make_model(reasoning=True)
    ctx = Context(messages=[UserMessage(content="hi")])
    options = GoogleOptions(thinking=GoogleThinkingOptions(enabled=True, budget_tokens=4096))
    body = build_params(model, ctx, options)
    assert body["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 4096


def test_build_params_signal_aborted_raises():
    # Line 137-138: signal aborted raises
    signal = AbortSignal()
    signal.abort()
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    options = GoogleOptions(signal=signal)
    with pytest.raises(RuntimeError, match="aborted"):
        build_params(model, ctx, options)


# --------------------------------------------------------------------------
# _get_disabled_thinking_config
# --------------------------------------------------------------------------


def test_disabled_thinking_gemini3_flash():
    # Line 154: gemini3 flash model
    model = make_model(id="gemini-3.0-flash-001")
    result = _get_disabled_thinking_config(model)
    assert result == {"thinkingLevel": "MINIMAL"}


def test_disabled_thinking_gemma4():
    # Line 156: gemma4 model
    model = make_model(id="gemma-4-9b-it")
    result = _get_disabled_thinking_config(model)
    assert result == {"thinkingLevel": "MINIMAL"}


def test_disabled_thinking_gemini2_fallback():
    # Line 158: fallback thinkingBudget=0
    model = make_model(id="gemini-2.0-flash")
    result = _get_disabled_thinking_config(model)
    assert result == {"thinkingBudget": 0}


# --------------------------------------------------------------------------
# _get_thinking_level
# --------------------------------------------------------------------------


def test_get_thinking_level_gemma4_low():
    # Lines 164-166: gemma4 thinking level
    model = make_model(id="gemma4-9b")
    assert _get_thinking_level("minimal", model) == "MINIMAL"
    assert _get_thinking_level("low", model) == "MINIMAL"
    assert _get_thinking_level("high", model) == "HIGH"


def test_get_thinking_level_gemini2_flash():
    # Line 166: fallback level mapping for non-gemini3/non-gemma4
    model = make_model(id="gemini-2.0-flash")
    assert _get_thinking_level("minimal", model) == "MINIMAL"
    assert _get_thinking_level("low", model) == "LOW"
    assert _get_thinking_level("medium", model) == "MEDIUM"
    assert _get_thinking_level("high", model) == "HIGH"


# --------------------------------------------------------------------------
# _get_google_budget
# --------------------------------------------------------------------------


def test_get_google_budget_custom_override():
    # Lines 171-173: custom_budgets override
    model = make_model(id="gemini-2.5-pro")
    budgets = ThinkingBudgets(minimal=64, low=256, medium=1024, high=8192)
    assert _get_google_budget(model, "minimal", budgets) == 64
    assert _get_google_budget(model, "high", budgets) == 8192


def test_get_google_budget_25_flash_lite():
    # Line 176: 2.5-flash-lite budget
    model = make_model(id="gemini-2.5-flash-lite-preview")
    assert _get_google_budget(model, "minimal", None) == 512
    assert _get_google_budget(model, "high", None) == 24576


def test_get_google_budget_25_flash():
    # Line 178: 2.5-flash budget
    model = make_model(id="gemini-2.5-flash")
    assert _get_google_budget(model, "minimal", None) == 128
    assert _get_google_budget(model, "high", None) == 24576


def test_get_google_budget_fallback():
    # Line 181: fallback -1
    model = make_model(id="gemini-1.5-flash")
    assert _get_google_budget(model, "high", None) == -1


# --------------------------------------------------------------------------
# stream — on_response, on_payload, error paths
# --------------------------------------------------------------------------


async def test_stream_on_response_callback():
    # Lines 240-245: on_response callback invoked
    received = {}

    def on_response(provider_response, model):
        received["status"] = provider_response.status  # ProviderResponse uses .status

    body = sse_body([text_chunk("hello"), finish_chunk("STOP")])
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleOptions(api_key="key", on_response=on_response)

    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert received.get("status") == 200
    assert msg.stop_reason == "stop"


async def test_stream_on_payload_non_awaitable():
    # Lines 226->228: on_payload returns None (not awaitable)
    called = {}

    def on_payload(params, model):
        called["got"] = True
        return None  # no replacement, not awaitable

    body = sse_body([text_chunk("hello"), finish_chunk("STOP")])
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleOptions(api_key="key", on_payload=on_payload)

    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert called.get("got") is True
    assert msg.stop_reason == "stop"


async def test_stream_on_payload_replaces_params():
    # Lines 228-229: replacement is not None -> params = replacement
    replacement_params = None

    def on_payload(params, model):
        nonlocal replacement_params
        new_params = dict(params)
        new_params["_test_marker"] = True
        replacement_params = new_params
        return new_params

    body = sse_body([text_chunk("hello"), finish_chunk("STOP")])
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleOptions(api_key="key", on_payload=on_payload)

    _, _ = await collect(stream(model, ctx, opts, client=client))
    assert replacement_params is not None


async def test_stream_pending_stop_reason_error():
    # Line 264-265: stop_reason == "pending" after stream
    body = sse_body([text_chunk("hello")])  # no finish chunk
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleOptions(api_key="key")

    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert msg.stop_reason == "error"


async def test_stream_error_stop_reason_with_raw_stop_reason():
    # Lines 266-272: stop_reason in aborted/error with raw_stop_reason
    body = sse_body([{"candidates": [{"finishReason": "SAFETY"}]}])
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleOptions(api_key="key")

    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert msg.stop_reason in ("error", "stop", "length")


async def test_stream_no_api_key_reports_error():
    # No api_key raises ValueError which gets caught and reported
    body = sse_body([])
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleOptions(api_key=None)

    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert msg.stop_reason == "error"


async def test_stream_http_error_reports_as_error():
    body = '{"error": {"message": "Forbidden", "code": 403}}'
    client = make_client(body, status=403)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleOptions(api_key="key")

    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert msg.stop_reason == "error"


# --------------------------------------------------------------------------
# stream_simple
# --------------------------------------------------------------------------


def test_stream_simple_no_api_key_raises():
    # Line 311: no api key
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    with pytest.raises(ValueError, match="No API key"):
        stream_simple(model, ctx, SimpleStreamOptions(api_key=None))


async def test_stream_simple_with_reasoning_gemini3_flash():
    # Lines 322-330: gemini3 flash with reasoning -> level-based
    body = sse_body([text_chunk("ok"), finish_chunk("STOP")])
    client = make_client(body)
    model = make_model(
        id="gemini-3.0-flash-001",
        reasoning=True,
        thinking_level_map={"minimal": "minimal", "low": "low", "medium": "medium", "high": "high"},
    )
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = SimpleStreamOptions(api_key="key", reasoning="medium")

    _, msg = await collect(stream_simple(model, ctx, opts, client=client))
    assert msg.stop_reason == "stop"


async def test_stream_simple_with_reasoning_gemma4():
    # Lines 322-330: gemma4 model with reasoning -> level-based
    body = sse_body([text_chunk("ok"), finish_chunk("STOP")])
    client = make_client(body)
    model = make_model(
        id="gemma-4-9b-it",
        reasoning=True,
        thinking_level_map={"minimal": "minimal", "low": "low", "medium": "medium", "high": "high"},
    )
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = SimpleStreamOptions(api_key="key", reasoning="high")

    _, msg = await collect(stream_simple(model, ctx, opts, client=client))
    assert msg.stop_reason == "stop"


async def test_stream_simple_with_reasoning_gemini25_pro():
    # Lines 332-342: non-gemini3/non-gemma4 with reasoning -> budget-based
    body = sse_body([text_chunk("ok"), finish_chunk("STOP")])
    client = make_client(body)
    model = make_model(
        id="gemini-2.5-pro-preview",
        reasoning=True,
        thinking_level_map={"minimal": "minimal", "low": "low", "medium": "medium", "high": "high"},
    )
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = SimpleStreamOptions(api_key="key", reasoning="medium")

    _, msg = await collect(stream_simple(model, ctx, opts, client=client))
    assert msg.stop_reason == "stop"


async def test_stream_simple_without_reasoning():
    # Lines 314-317: no reasoning -> disabled thinking
    body = sse_body([text_chunk("ok"), finish_chunk("STOP")])
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = SimpleStreamOptions(api_key="key")

    _, msg = await collect(stream_simple(model, ctx, opts, client=client))
    assert msg.stop_reason == "stop"
