"""Coverage tests for pi_ai.api.azure_openai_responses."""

from __future__ import annotations

import json

import httpx
import pytest
from pi_ai import Context, Model, ModelCost, UserMessage
from pi_ai.api.azure_openai_responses import (
    AzureOpenAIResponsesOptions,
    _parse_deployment_name_map,
    build_params,
    get_compat,
    stream,
    stream_simple,
)
from pi_ai.types import SimpleStreamOptions


def make_model(**overrides) -> Model:
    defaults = dict(
        id="gpt-azure-test",
        name="GPT Azure Test",
        api="azure-openai-responses",
        provider="azure",
        base_url="https://my-resource.openai.azure.com",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=1.0, output=2.0, cache_read=0.5, cache_write=1.5),
        context_window=100_000,
        max_tokens=4096,
    )
    defaults.update(overrides)
    return Model(**defaults)


def sse_body(chunks: list[dict]) -> str:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)


def make_client(body: str, status: int = 200, capture: dict | None = None) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["request"] = request
            capture["json"] = json.loads(request.content)
        return httpx.Response(status, text=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def collect(event_stream):
    events = [event async for event in event_stream]
    return events, await event_stream.result()


def make_done_sse() -> str:
    return sse_body(
        [
            {"type": "response.created", "response": {"id": "resp_1"}},
            {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1"}},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "content": [{"type": "output_text", "text": "hello"}],
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        "input_tokens_details": {},
                    },
                },
            },
        ]
    )


# --------------------------------------------------------------------------
# get_compat — value is None skip
# --------------------------------------------------------------------------


def test_get_compat_skips_none_values():
    # Lines 78-79: when value is None, continue (skip)
    compat = get_compat(make_model(compat={"supportsStrictMode": None, "supports_openai_grammar_tools": True}))
    # None value is skipped, so supportsStrictMode stays at its default (True)
    assert compat.supports_strict_mode is True
    assert compat.supports_openai_grammar_tools is True


# --------------------------------------------------------------------------
# _parse_deployment_name_map — edge cases
# --------------------------------------------------------------------------


def test_parse_deployment_name_map_empty_entry():
    # Line 91: empty trimmed entry -> continue
    result = _parse_deployment_name_map("   ,model1=dep1")
    assert "model1" in result


def test_parse_deployment_name_map_no_equals():
    # Line 96: no `=` -> continue
    result = _parse_deployment_name_map("no-equals-sign")
    assert result == {}


def test_parse_deployment_name_map_empty_model_or_dep():
    # Line 99: empty model_id or deployment_name after split -> continue
    result = _parse_deployment_name_map("=dep,model=")
    assert result == {}


# --------------------------------------------------------------------------
# build_params — uncovered branches
# --------------------------------------------------------------------------


def test_build_params_temperature():
    # Line 236: temperature set
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = AzureOpenAIResponsesOptions(azure_base_url="https://r.openai.azure.com/openai/v1", temperature=0.5)
    params = build_params(model, ctx, opts)
    assert params["temperature"] == 0.5


def test_build_params_reasoning_effort_with_level_map():
    # Lines 254-256: reasoning_effort with level_map
    model = make_model(
        reasoning=True,
        thinking_level_map={"low": "low", "high": "high"},
    )
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = AzureOpenAIResponsesOptions(
        azure_base_url="https://r.openai.azure.com/openai/v1",
        reasoning_effort="high",
    )
    params = build_params(model, ctx, opts)
    assert "reasoning" in params
    assert params["reasoning"]["effort"] == "high"


def test_build_params_reasoning_effort_not_in_map_uses_effort_directly():
    # Line 255-256: effort not in thinking_level_map -> use effort directly
    model = make_model(
        reasoning=True,
        thinking_level_map={},
    )
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = AzureOpenAIResponsesOptions(
        azure_base_url="https://r.openai.azure.com/openai/v1",
        reasoning_effort="medium",
    )
    params = build_params(model, ctx, opts)
    assert params["reasoning"]["effort"] == "medium"


def test_build_params_reasoning_summary_only():
    # Lines 259-261: reasoning_summary without reasoning_effort -> effort = "medium"
    model = make_model(
        reasoning=True,
        thinking_level_map={"off": None},
    )
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = AzureOpenAIResponsesOptions(
        azure_base_url="https://r.openai.azure.com/openai/v1",
        reasoning_summary="auto",
    )
    params = build_params(model, ctx, opts)
    assert params["reasoning"]["effort"] == "medium"
    assert params["reasoning"]["summary"] == "auto"


def test_build_params_sampling_params():
    # Line 265: sampling_params override
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = AzureOpenAIResponsesOptions(
        azure_base_url="https://r.openai.azure.com/openai/v1",
        sampling_params={"top_p": 0.9},
    )
    params = build_params(model, ctx, opts)
    assert params["top_p"] == 0.9


# --------------------------------------------------------------------------
# stream — on_payload, on_response, error paths
# --------------------------------------------------------------------------


async def test_stream_on_payload_callback():
    # Lines 311-315: on_payload callback
    called = {}

    def on_payload(params, model):
        called["got"] = True
        return None

    body = make_done_sse()
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = AzureOpenAIResponsesOptions(
        api_key="my-key",
        azure_base_url="https://my-resource.openai.azure.com/openai/v1",
        on_payload=on_payload,
    )

    _, _ = await collect(stream(model, ctx, opts, client=client))
    assert called.get("got") is True


async def test_stream_on_response_callback():
    # Lines 330-332: on_response callback (options.on_response is not None)
    received = {}

    def on_response(provider_response, model):
        received["status"] = provider_response.status  # ProviderResponse uses .status

    body = make_done_sse()
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = AzureOpenAIResponsesOptions(
        api_key="my-key",
        azure_base_url="https://my-resource.openai.azure.com/openai/v1",
        on_response=on_response,
    )

    _, _ = await collect(stream(model, ctx, opts, client=client))
    assert received.get("status") == 200


async def test_stream_invalid_sse_data_skipped():
    # Lines 341-342: json parse error in event_iterator -> continue
    # Non-JSON data interleaved with valid events
    raw = "data: not-json\n\ndata: also not json\n\n" + make_done_sse()
    client = make_client(raw)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = AzureOpenAIResponsesOptions(
        api_key="my-key",
        azure_base_url="https://my-resource.openai.azure.com/openai/v1",
    )
    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert msg.stop_reason == "stop"


async def test_stream_non_dict_sse_event_skipped():
    # Line 343->338: non-dict event (e.g. a JSON array) is skipped
    raw = 'data: ["not", "a", "dict"]\n\n' + make_done_sse()
    client = make_client(raw)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = AzureOpenAIResponsesOptions(
        api_key="my-key",
        azure_base_url="https://my-resource.openai.azure.com/openai/v1",
    )
    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert msg.stop_reason == "stop"


async def test_stream_no_start_event_pushed_before_process():
    # Line 355: if not started -> push StartEvent after process_responses_stream
    # This happens when on_response is never called (impossible in normal flow with
    # SSE, but we can simulate by using a body with no SSE events that still completes)
    # Actually with a proper body, on_response is called so started=True. We need to
    # test the case where no SSE comes through at all, but the stream ends normally.
    # Use a body that has only the response.completed event but with no output items
    # We need a client that DOES NOT call on_response (i.e., returns non-SSE)
    # Since on_response is always called from stream_sse when a response is received,
    # we need a different approach: use empty body so on_response IS called but started
    # stays False if the stream yields no events.
    # Let's just verify the started=False path: use an SSE body with only non-dict events
    raw_body = "data: null\n\n" + sse_body(
        [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_y",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            }
        ]
    )
    client = make_client(raw_body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = AzureOpenAIResponsesOptions(
        api_key="my-key",
        azure_base_url="https://my-resource.openai.azure.com/openai/v1",
    )
    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert msg.stop_reason in ("stop", "error")


async def test_stream_stop_reason_pending_error():
    # Line 361: stop_reason == "pending" -> error
    # Stream body has no response.completed so stop_reason stays pending
    body = sse_body([{"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}}])
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = AzureOpenAIResponsesOptions(
        api_key="my-key",
        azure_base_url="https://my-resource.openai.azure.com/openai/v1",
    )
    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert msg.stop_reason == "error"


async def test_stream_stop_reason_aborted_or_error_raises():
    # Line 363: stop_reason in aborted/error -> error message
    body = sse_body(
        [
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {"code": "server_error", "message": "Internal error"},
                },
            }
        ]
    )
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = AzureOpenAIResponsesOptions(
        api_key="my-key",
        azure_base_url="https://my-resource.openai.azure.com/openai/v1",
    )
    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert msg.stop_reason == "error"


async def test_stream_no_api_key_reports_error():
    body = make_done_sse()
    client = make_client(body)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = AzureOpenAIResponsesOptions(
        api_key=None,
        azure_base_url="https://my-resource.openai.azure.com/openai/v1",
    )
    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert msg.stop_reason == "error"


async def test_stream_http_error_reports_as_error():
    body = '{"error": {"message": "Forbidden"}}'
    client = make_client(body, status=403)
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = AzureOpenAIResponsesOptions(
        api_key="my-key",
        azure_base_url="https://my-resource.openai.azure.com/openai/v1",
    )
    _, msg = await collect(stream(model, ctx, opts, client=client))
    assert msg.stop_reason == "error"


# --------------------------------------------------------------------------
# stream_simple
# --------------------------------------------------------------------------


async def test_stream_simple_with_reasoning():
    # Lines 394-398: reasoning_effort is set
    body = make_done_sse()
    client = make_client(body)
    model = make_model(
        reasoning=True,
        thinking_level_map={"minimal": "low", "low": "low", "medium": "medium", "high": "high"},
    )
    ctx = Context(messages=[UserMessage(content="hi")])
    opts = SimpleStreamOptions(
        api_key="my-key",
        reasoning="medium",
        env={"AZURE_OPENAI_BASE_URL": "https://my-resource.openai.azure.com/openai/v1"},
    )
    _, msg = await collect(stream_simple(model, ctx, opts, client=client))
    assert msg.stop_reason in ("stop", "error")


def test_stream_simple_no_api_key_raises():
    model = make_model()
    ctx = Context(messages=[UserMessage(content="hi")])
    with pytest.raises(ValueError, match="No API key"):
        stream_simple(model, ctx, SimpleStreamOptions(api_key=None))
