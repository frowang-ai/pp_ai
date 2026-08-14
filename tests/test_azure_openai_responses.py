import json

import httpx
import pytest
from pi_ai import (
    Context,
    Model,
    ModelCost,
    UserMessage,
)
from pi_ai.api.azure_openai_responses import (
    OPENAI_RESPONSES_MIN_OUTPUT_TOKENS,
    AzureOpenAIResponsesOptions,
    _normalize_azure_base_url,
    _parse_deployment_name_map,
    _resolve_azure_config,
    build_headers,
    build_params,
    get_compat,
    resolve_deployment_name,
    stream,
    stream_simple,
)


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


# --------------------------------------------------------------------------
# compat
# --------------------------------------------------------------------------


def test_get_compat_defaults():
    compat = get_compat(make_model())
    assert compat.supports_strict_mode is True
    assert compat.supports_openai_grammar_tools is False


def test_get_compat_overrides_in_both_spellings():
    camel = get_compat(make_model(compat={"supportsStrictMode": False}))
    assert camel.supports_strict_mode is False
    snake = get_compat(make_model(compat={"supports_openai_grammar_tools": True}))
    assert snake.supports_openai_grammar_tools is True


# --------------------------------------------------------------------------
# deployment name resolution
# --------------------------------------------------------------------------


def test_parse_deployment_name_map_basic():
    mapping = _parse_deployment_name_map("gpt-5.1=my-deployment, gpt-5-mini=mini-dep")
    assert mapping == {"gpt-5.1": "my-deployment", "gpt-5-mini": "mini-dep"}


def test_parse_deployment_name_map_truncates_extra_equals_segments():
    # JavaScript's `split("=", 2)` keeps only the first two segments.
    mapping = _parse_deployment_name_map("model=dep=extra")
    assert mapping == {"model": "dep"}


def test_parse_deployment_name_map_empty_value():
    assert _parse_deployment_name_map(None) == {}
    assert _parse_deployment_name_map("") == {}


def test_resolve_deployment_name_prefers_explicit_option():
    model = make_model(id="gpt-5.1")
    name = resolve_deployment_name(model, AzureOpenAIResponsesOptions(azure_deployment_name="explicit-dep"))
    assert name == "explicit-dep"


def test_resolve_deployment_name_uses_env_map(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "gpt-5.1=mapped-dep")
    model = make_model(id="gpt-5.1")
    name = resolve_deployment_name(model, AzureOpenAIResponsesOptions())
    assert name == "mapped-dep"


def test_resolve_deployment_name_falls_back_to_model_id():
    model = make_model(id="gpt-5.1")
    assert resolve_deployment_name(model, None) == "gpt-5.1"


# --------------------------------------------------------------------------
# base URL normalization
# --------------------------------------------------------------------------


def test_normalize_azure_base_url_appends_openai_v1_for_bare_azure_host():
    assert _normalize_azure_base_url("https://my-resource.openai.azure.com") == (
        "https://my-resource.openai.azure.com/openai/v1"
    )


def test_normalize_azure_base_url_appends_for_openai_path():
    assert _normalize_azure_base_url("https://my-resource.openai.azure.com/openai") == (
        "https://my-resource.openai.azure.com/openai/v1"
    )


def test_normalize_azure_base_url_appends_for_responses_path():
    assert _normalize_azure_base_url("https://my-resource.openai.azure.com/openai/v1/responses") == (
        "https://my-resource.openai.azure.com/openai/v1"
    )


def test_normalize_azure_base_url_leaves_non_azure_host_alone():
    assert _normalize_azure_base_url("https://example.com/custom/path") == "https://example.com/custom/path"


def test_normalize_azure_base_url_leaves_already_normalized_path_alone():
    assert _normalize_azure_base_url("https://my-resource.openai.azure.com/openai/v1") == (
        "https://my-resource.openai.azure.com/openai/v1"
    )


def test_normalize_azure_base_url_raises_for_invalid_url():
    with pytest.raises(ValueError, match="Invalid Azure OpenAI base URL"):
        _normalize_azure_base_url("not-a-url")


def test_resolve_azure_config_prefers_explicit_option_over_resource_name():
    model = make_model(base_url="")
    base_url, api_version = _resolve_azure_config(
        model,
        AzureOpenAIResponsesOptions(azure_base_url="https://explicit.openai.azure.com", azure_api_version="2024-10-01"),
    )
    assert base_url == "https://explicit.openai.azure.com/openai/v1"
    assert api_version == "2024-10-01"


def test_resolve_azure_config_builds_url_from_resource_name():
    model = make_model(base_url="")
    base_url, api_version = _resolve_azure_config(model, AzureOpenAIResponsesOptions(azure_resource_name="myres"))
    assert base_url == "https://myres.openai.azure.com/openai/v1"
    assert api_version == "v1"


def test_resolve_azure_config_falls_back_to_model_base_url():
    model = make_model(base_url="https://my-resource.openai.azure.com")
    base_url, _ = _resolve_azure_config(model, None)
    assert base_url == "https://my-resource.openai.azure.com/openai/v1"


def test_resolve_azure_config_raises_without_any_base_url():
    model = make_model(base_url="")
    with pytest.raises(ValueError, match="Azure OpenAI base URL is required"):
        _resolve_azure_config(model, None)


def test_resolve_azure_config_reads_env_vars(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://from-env.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")
    model = make_model(base_url="")
    base_url, api_version = _resolve_azure_config(model, AzureOpenAIResponsesOptions())
    assert base_url == "https://from-env.openai.azure.com/openai/v1"
    assert api_version == "2025-01-01"


# --------------------------------------------------------------------------
# headers
# --------------------------------------------------------------------------


def test_build_headers_uses_api_key_header():
    headers = build_headers(make_model(), "sk-azure", None)
    assert headers["api-key"] == "sk-azure"
    assert "authorization" not in headers


def test_build_headers_applies_overrides_and_null_deletes():
    headers = build_headers(
        make_model(headers={"x-default": "1"}),
        "sk-azure",
        AzureOpenAIResponsesOptions(headers={"x-default": None, "x-custom": "2"}),
    )
    assert "x-default" not in headers
    assert headers["x-custom"] == "2"


# --------------------------------------------------------------------------
# build_params
# --------------------------------------------------------------------------


def test_build_params_uses_deployment_name_as_model():
    params = build_params(make_model(), Context(messages=[]), None, deployment_name="my-deployment")
    assert params["model"] == "my-deployment"


def test_build_params_max_output_tokens_clamped_to_minimum():
    params = build_params(
        make_model(), Context(messages=[]), AzureOpenAIResponsesOptions(max_tokens=1), deployment_name="dep"
    )
    assert params["max_output_tokens"] == OPENAI_RESPONSES_MIN_OUTPUT_TOKENS


def test_build_params_prompt_cache_key():
    params = build_params(
        make_model(), Context(messages=[]), AzureOpenAIResponsesOptions(session_id="sess-1"), deployment_name="dep"
    )
    assert params["prompt_cache_key"] == "sess-1"


def test_build_params_store_always_false():
    params = build_params(make_model(), Context(messages=[]), None, deployment_name="dep")
    assert params["store"] is False


def test_build_params_reasoning_effort():
    model = make_model(reasoning=True, thinking_level_map={"high": "high-effort"})
    params = build_params(
        model,
        Context(messages=[]),
        AzureOpenAIResponsesOptions(reasoning_effort="high"),
        deployment_name="dep",
    )
    assert params["reasoning"] == {"effort": "high-effort", "summary": "auto"}


def test_build_params_tools_included_when_present():
    from pi_ai import Tool

    params = build_params(
        make_model(),
        Context(messages=[], tools=[Tool(name="read", description="reads a file")]),
        None,
        deployment_name="dep",
    )
    assert params["tools"][0]["name"] == "read"


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------


async def test_stream_emits_text_events_and_final_message():
    body = sse_body(
        [
            {"type": "response.created", "response": {"id": "resp_1"}},
            {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1"}},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "Hi"},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_1", "content": [{"type": "output_text", "text": "Hi"}]},
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
                },
            },
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(
                make_model(),
                Context(messages=[UserMessage(content="hi")]),
                AzureOpenAIResponsesOptions(api_key="k"),
                client=client,
            )
        )
    assert [event.type for event in events] == ["start", "text_start", "text_delta", "text_end", "done"]
    assert message.stop_reason == "stop"
    assert message.content[0].text == "Hi"
    assert message.api == "azure-openai-responses"


async def test_stream_function_call_with_argument_deltas():
    body = sse_body(
        [
            {"type": "response.created", "response": {"id": "resp_1"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call1", "id": "fc_1", "name": "read", "arguments": ""},
            },
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"path": "a.txt"}'},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "call1",
                    "id": "fc_1",
                    "name": "read",
                    "arguments": '{"path": "a.txt"}',
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), AzureOpenAIResponsesOptions(api_key="k"), client=client)
        )
    assert message.stop_reason == "toolUse"
    assert message.content[0].arguments == {"path": "a.txt"}
    assert [event.type for event in events] == [
        "start",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]


async def test_stream_response_failed_event_reports_error():
    body = sse_body(
        [{"type": "response.failed", "response": {"status": "failed", "error": {"code": "boom", "message": "bad"}}}]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), AzureOpenAIResponsesOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "bad" in message.error_message


async def test_stream_reports_http_error_through_stream():
    async with make_client('{"error": {"message": "bad key"}}', status=401) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), AzureOpenAIResponsesOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "bad key" in message.error_message


async def test_stream_raises_helpful_error_without_api_key():
    events, message = await collect(stream(make_model(), Context(messages=[]), AzureOpenAIResponsesOptions()))
    assert events[-1].type == "error"
    assert "No API key" in message.error_message


async def test_stream_sends_expected_request_url_and_headers():
    capture: dict = {}
    body = sse_body(
        [
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {}},
            }
        ]
    )
    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                make_model(),
                Context(messages=[]),
                AzureOpenAIResponsesOptions(api_key="sk-azure", azure_deployment_name="my-dep"),
                client=client,
            )
        )
    request = capture["request"]
    assert str(request.url) == (
        "https://my-resource.openai.azure.com/openai/v1/deployments/my-dep/responses?api-version=v1"
    )
    assert request.headers["api-key"] == "sk-azure"
    assert capture["json"]["model"] == "my-dep"


async def test_aborted_signal_terminates_the_stream_as_aborted():
    from pi_ai.utils.abort import AbortSignal

    signal = AbortSignal()
    signal.abort()
    body = sse_body(
        [
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {}},
            }
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(
                make_model(),
                Context(messages=[]),
                AzureOpenAIResponsesOptions(api_key="k", signal=signal),
                client=client,
            )
        )
    assert events[-1].type == "error"
    assert events[-1].reason == "aborted"
    assert message.stop_reason == "aborted"


async def test_stream_simple_maps_reasoning_level_to_reasoning_effort():
    from pi_ai import SimpleStreamOptions

    capture: dict = {}
    body = sse_body(
        [
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {}},
            }
        ]
    )
    model = make_model(reasoning=True, thinking_level_map={"high": "high"})
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                model, Context(messages=[]), SimpleStreamOptions(api_key="k", reasoning="high"), client=client
            )
        )
    assert capture["json"]["reasoning"]["effort"] == "high"


async def test_stream_simple_raises_without_api_key():
    from pi_ai import SimpleStreamOptions

    with pytest.raises(ValueError, match="No API key"):
        stream_simple(make_model(), Context(messages=[]), SimpleStreamOptions())
