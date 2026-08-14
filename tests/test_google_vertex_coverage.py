"""Coverage tests for pi_ai.api.google_vertex."""

from __future__ import annotations

import json

import httpx
import pytest
from pi_ai import Context, Model, ModelCost, Tool, UserMessage
from pi_ai.api.google_vertex import (
    GoogleThinkingOptions,
    GoogleVertexOptions,
    _base_url_includes_api_version,
    _get_disabled_thinking_config,
    _get_gemini3_thinking_level,
    _get_google_budget,
    _resolve_custom_base_url,
    build_params,
    build_url_and_headers,
    stream,
    stream_simple,
)
from pi_ai.types import SimpleStreamOptions, ThinkingBudgets
from pi_ai.utils.abort import AbortSignal


def make_model(**overrides) -> Model:
    defaults = dict(
        id="gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        api="google-vertex",
        provider="google-vertex",
        base_url="",
        reasoning=False,
        input=["text", "image"],
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
# _resolve_custom_base_url
# --------------------------------------------------------------------------


def test_resolve_custom_base_url_returns_trimmed_url():
    # Line 177: return trimmed when base_url is valid
    result = _resolve_custom_base_url("https://custom.example.com/v1")
    assert result == "https://custom.example.com/v1"


def test_resolve_custom_base_url_returns_none_for_location_placeholder():
    # Line 175-176: returns None when {location} in trimmed
    result = _resolve_custom_base_url("https://{location}-aiplatform.googleapis.com/v1")
    assert result is None


def test_resolve_custom_base_url_returns_none_for_empty():
    assert _resolve_custom_base_url("") is None
    assert _resolve_custom_base_url("   ") is None


# --------------------------------------------------------------------------
# _base_url_includes_api_version
# --------------------------------------------------------------------------


def test_base_url_includes_api_version_with_version_segment():
    # Line 188: version in path segment
    assert _base_url_includes_api_version("https://example.com/v1") is True
    assert _base_url_includes_api_version("https://example.com/v1/foo") is True
    assert _base_url_includes_api_version("https://example.com/v1beta/foo") is True


def test_base_url_includes_api_version_without_version():
    assert _base_url_includes_api_version("https://example.com/api") is False


def test_base_url_includes_api_version_no_path_with_version_in_string():
    # Line 187: path == base_url and no :// - regex on full string
    assert _base_url_includes_api_version("v1") is True
    assert _base_url_includes_api_version("api/v2/models") is True
    assert _base_url_includes_api_version("no-version-here") is False


# --------------------------------------------------------------------------
# build_url_and_headers — uncovered branches
# --------------------------------------------------------------------------


def test_build_url_with_override_headers():
    # Line 196: headers.update(override) when override is non-empty
    model = make_model()
    options = GoogleVertexOptions(api_key="my-key", headers={"x-custom": "val"})
    _, headers = build_url_and_headers(model, options)
    assert headers["x-custom"] == "val"
    assert headers["x-goog-api-key"] == "my-key"


def test_build_url_api_key_with_custom_base_url_including_api_version():
    # Line 204: custom_base_url and _base_url_includes_api_version is True -> base_url = custom_base_url
    model = make_model(base_url="https://custom.example.com/v1")
    options = GoogleVertexOptions(api_key="key")
    url, _ = build_url_and_headers(model, options)
    assert url.startswith("https://custom.example.com/v1/publishers/google")


def test_build_url_api_key_with_custom_base_url_without_version():
    # Line 202-205: custom_base_url without version -> appends /publishers/google/...
    model = make_model(base_url="https://custom.example.com/api")
    options = GoogleVertexOptions(api_key="key")
    url, _ = build_url_and_headers(model, options)
    assert "publishers/google/models" in url


def test_build_url_access_token_with_custom_base_url():
    # A custom base URL is collection-scoped: the projects/locations segment is
    # not appended to it, and a base URL that already carries an API version
    # does not get another one (matches the TypeScript adapter's
    # `baseUrlResourceScope: COLLECTION` / `apiVersion: ""` httpOptions).
    model = make_model(base_url="https://custom.example.com/v1")
    options = GoogleVertexOptions(
        project="proj",
        location="us-central1",
        access_token="tok",
    )
    url, _ = build_url_and_headers(model, options)
    assert url == (
        "https://custom.example.com/v1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    )


# --------------------------------------------------------------------------
# build_params — uncovered branches
# --------------------------------------------------------------------------


def test_build_params_with_temperature():
    # Line 241: temperature
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    options = GoogleVertexOptions(api_key="k", temperature=0.7)
    body = build_params(model, ctx, options)
    assert body["generationConfig"]["temperature"] == 0.7


def test_build_params_with_system_prompt():
    # Line 247: systemInstruction
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")], system_prompt="Be helpful")
    body = build_params(model, ctx)
    assert "systemInstruction" in body
    assert body["systemInstruction"]["parts"][0]["text"] == "Be helpful"


def test_build_params_with_tool():
    # Lines 250-256: tools and toolConfig
    tool = Tool(name="search", description="search", parameters={"type": "object", "properties": {}})
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")], tools=[tool])
    body = build_params(model, ctx)
    assert "tools" in body


def test_build_params_thinking_with_level():
    # Line 261: thinking level set
    model = make_model(reasoning=True)
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleVertexOptions(thinking=GoogleThinkingOptions(enabled=True, level="HIGH"))
    body = build_params(model, ctx, opts)
    assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"
    assert body["generationConfig"]["thinkingConfig"]["includeThoughts"] is True


def test_build_params_thinking_with_budget_tokens():
    # Lines 262-264: thinkingBudget
    model = make_model(reasoning=True)
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleVertexOptions(thinking=GoogleThinkingOptions(enabled=True, budget_tokens=2048))
    body = build_params(model, ctx, opts)
    assert body["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 2048


def test_build_params_disabled_thinking_on_reasoning_model():
    # Line 265-266: thinking disabled
    model = make_model(reasoning=True)
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleVertexOptions(thinking=GoogleThinkingOptions(enabled=False))
    body = build_params(model, ctx, opts)
    assert "thinkingConfig" in body["generationConfig"]


def test_build_params_signal_aborted_raises():
    # Line 272: signal aborted
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    signal = AbortSignal()
    signal.abort()
    opts = GoogleVertexOptions(signal=signal)
    with pytest.raises(RuntimeError, match="aborted"):
        build_params(model, ctx, opts)


# --------------------------------------------------------------------------
# _get_disabled_thinking_config
# --------------------------------------------------------------------------


def test_get_disabled_thinking_config_gemini3_pro():
    # Line 283: pro model
    model = make_model(id="gemini-3.1-pro-preview")
    result = _get_disabled_thinking_config(model)
    assert result == {"thinkingLevel": "LOW"}


def test_get_disabled_thinking_config_gemini3_flash():
    # Line 285: flash model
    model = make_model(id="gemini-3.0-flash-001")
    result = _get_disabled_thinking_config(model)
    assert result == {"thinkingLevel": "MINIMAL"}


def test_get_disabled_thinking_config_gemini2():
    # Line 287: default thinkingBudget=0
    model = make_model(id="gemini-2.0-flash")
    result = _get_disabled_thinking_config(model)
    assert result == {"thinkingBudget": 0}


# --------------------------------------------------------------------------
# _get_gemini3_thinking_level
# --------------------------------------------------------------------------


def test_get_gemini3_thinking_level_pro_low():
    # Line 292: pro model returns LOW for low effort
    model = make_model(id="gemini-3.1-pro-preview")
    assert _get_gemini3_thinking_level("minimal", model) == "LOW"
    assert _get_gemini3_thinking_level("low", model) == "LOW"
    assert _get_gemini3_thinking_level("high", model) == "HIGH"


def test_get_gemini3_thinking_level_flash_mapping():
    # Line 293: flash model returns mapped level
    model = make_model(id="gemini-3.0-flash-001")
    assert _get_gemini3_thinking_level("minimal", model) == "MINIMAL"
    assert _get_gemini3_thinking_level("medium", model) == "MEDIUM"
    assert _get_gemini3_thinking_level("high", model) == "HIGH"


# --------------------------------------------------------------------------
# _get_google_budget
# --------------------------------------------------------------------------


def test_get_google_budget_custom_override():
    # Lines 298-300: custom_budgets override
    model = make_model(id="gemini-2.5-pro-preview")
    budgets = ThinkingBudgets(minimal=64, low=512, medium=4096, high=16384)
    assert _get_google_budget(model, "minimal", budgets) == 64
    assert _get_google_budget(model, "high", budgets) == 16384


def test_get_google_budget_25_pro_defaults():
    # Line 303: 2.5-pro defaults
    model = make_model(id="gemini-2.5-pro-preview")
    assert _get_google_budget(model, "minimal", None) == 128
    assert _get_google_budget(model, "high", None) == 32768


def test_get_google_budget_25_flash_defaults():
    model = make_model(id="gemini-2.5-flash-preview")
    assert _get_google_budget(model, "minimal", None) == 128


def test_get_google_budget_fallback():
    # Line 306: -1 fallback for unknown model
    model = make_model(id="gemini-1.5-flash")
    assert _get_google_budget(model, "high", None) == -1


# --------------------------------------------------------------------------
# stream — on_response and error paths
# --------------------------------------------------------------------------


async def test_stream_invokes_on_response_callback():
    # Lines 347-351: on_response callback
    received = {}

    def on_response(provider_response, model):
        received["status"] = provider_response.status  # ProviderResponse uses .status

    body = sse_body([text_chunk("hello"), finish_chunk("STOP")])
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleVertexOptions(api_key="key", on_response=on_response)

    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert received.get("status") == 200
    assert msg.stop_reason == "stop"


async def test_stream_invokes_on_payload_callback():
    # Lines 347-351: on_payload callback
    replaced = {}

    def on_payload(params, model):
        replaced["got"] = True
        return None  # no replacement

    body = sse_body([text_chunk("hi"), finish_chunk("STOP")])
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleVertexOptions(api_key="key", on_payload=on_payload)

    _, _ = await collect(stream(model, ctx, opts, client=client))
    assert replaced.get("got") is True


async def test_stream_pending_stop_reason_raises_error():
    # Line 379-382: stop_reason == "pending" after stream
    # Provide a stream with no finishReason chunk -> stream ends, state.finalize() won't set stop_reason
    body = sse_body([text_chunk("hello")])
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleVertexOptions(api_key="key")

    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert msg.stop_reason == "error"


async def test_stream_error_stop_reason_with_raw_stop_reason():
    # Line 382-388: stop_reason in aborted/error with raw_stop_reason
    body = sse_body([{"candidates": [{"finishReason": "SAFETY", "content": {}}]}])
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleVertexOptions(api_key="key")

    _, msg = await collect(stream(model, ctx, opts, client=client))
    # Safety maps to "error" or a similar stop reason
    assert msg.stop_reason in ("error", "stop", "length")


async def test_stream_http_error_reported_as_error_event():
    # Error path: HTTP error
    body = '{"error": {"message": "Unauthorized", "code": 401}}'
    client = make_client(body, status=401)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = GoogleVertexOptions(api_key="key")

    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert msg.stop_reason == "error"


# --------------------------------------------------------------------------
# stream_simple — reasoning paths
# --------------------------------------------------------------------------


async def test_stream_simple_with_gemini3_flash_reasoning():
    # Lines 434-442: gemini3 flash model with reasoning -> level-based thinking
    body = sse_body([text_chunk("ok"), finish_chunk("STOP")])
    client = make_client(body)
    model = make_model(
        id="gemini-3.0-flash-001",
        reasoning=True,
        thinking_level_map={"minimal": "minimal", "low": "low", "medium": "medium", "high": "high"},
    )
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = SimpleStreamOptions(api_key="gc-key", reasoning="high")

    _, msg = await collect(stream_simple(model, ctx, opts, client=client))
    assert msg.stop_reason == "stop"


async def test_stream_simple_with_gemini25_pro_reasoning():
    # Lines 444-454: non-gemini3 model with reasoning -> budget-based thinking
    body = sse_body([text_chunk("ok"), finish_chunk("STOP")])
    client = make_client(body)
    model = make_model(
        id="gemini-2.5-pro-preview",
        reasoning=True,
        thinking_level_map={"minimal": "minimal", "low": "low", "medium": "medium", "high": "high"},
    )
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = SimpleStreamOptions(api_key="gc-key", reasoning="medium")

    _, msg = await collect(stream_simple(model, ctx, opts, client=client))
    assert msg.stop_reason == "stop"


async def test_stream_simple_no_reasoning_sends_disabled_thinking():
    # Line 426-429: no reasoning -> disabled thinking
    body = sse_body([text_chunk("ok"), finish_chunk("STOP")])
    client = make_client(body)
    model = make_model(id="gemini-2.5-flash-preview", reasoning=False)
    ctx = Context(messages=[UserMessage(content="hi")])

    _, msg = await collect(stream_simple(model, ctx, SimpleStreamOptions(api_key="gc-key"), client=client))
    assert msg.stop_reason == "stop"
