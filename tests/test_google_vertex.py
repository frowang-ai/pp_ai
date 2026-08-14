import json

import httpx
import pytest
from pi_ai import Context, Model, ModelCost, Tool, UserMessage
from pi_ai.api.google_vertex import (
    GCP_VERTEX_CREDENTIALS_MARKER,
    GoogleThinkingOptions,
    GoogleVertexOptions,
    build_params,
    build_url_and_headers,
    resolve_access_token,
    resolve_api_key,
    resolve_location,
    resolve_project,
    stream,
    stream_simple,
)
from pi_ai.types import SimpleStreamOptions


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


def finish_chunk(reason: str) -> dict:
    return {"candidates": [{"finishReason": reason}]}


def text_chunk(text: str) -> dict:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


# --------------------------------------------------------------------------
# auth resolution
# --------------------------------------------------------------------------


def test_resolve_api_key_returns_trimmed_key():
    assert resolve_api_key(GoogleVertexOptions(api_key="  key-1  ")) == "key-1"


def test_resolve_api_key_ignores_marker_and_placeholder():
    assert resolve_api_key(GoogleVertexOptions(api_key=GCP_VERTEX_CREDENTIALS_MARKER)) is None
    assert resolve_api_key(GoogleVertexOptions(api_key="<GOOGLE_CLOUD_API_KEY>")) is None
    assert resolve_api_key(GoogleVertexOptions(api_key=None)) is None
    assert resolve_api_key(None) is None


def test_resolve_project_from_options_and_env():
    assert resolve_project(GoogleVertexOptions(project="proj-1")) == "proj-1"
    assert resolve_project(GoogleVertexOptions(env={"GOOGLE_CLOUD_PROJECT": "proj-env"})) == "proj-env"
    assert resolve_project(GoogleVertexOptions(env={"GCLOUD_PROJECT": "proj-legacy"})) == "proj-legacy"


def test_resolve_project_missing_raises():
    with pytest.raises(ValueError, match="project"):
        resolve_project(GoogleVertexOptions())


def test_resolve_location_from_options_and_env():
    assert resolve_location(GoogleVertexOptions(location="us-central1")) == "us-central1"
    assert resolve_location(GoogleVertexOptions(env={"GOOGLE_CLOUD_LOCATION": "us-east1"})) == "us-east1"


def test_resolve_location_missing_raises():
    with pytest.raises(ValueError, match="location"):
        resolve_location(GoogleVertexOptions())


def test_resolve_access_token_from_options_and_env():
    assert resolve_access_token(GoogleVertexOptions(access_token="tok-1")) == "tok-1"
    assert resolve_access_token(GoogleVertexOptions(env={"GOOGLE_VERTEX_ACCESS_TOKEN": "tok-env"})) == "tok-env"


def test_resolve_access_token_missing_raises_clear_error():
    with pytest.raises(ValueError, match="access token"):
        resolve_access_token(GoogleVertexOptions())


def test_resolve_access_token_missing_but_adc_path_present_mentions_scope_limitation():
    with pytest.raises(ValueError, match="Application"):
        resolve_access_token(GoogleVertexOptions(env={"GOOGLE_APPLICATION_CREDENTIALS": "/tmp/sa.json"}))


# --------------------------------------------------------------------------
# URL / header construction
# --------------------------------------------------------------------------


def test_build_url_with_api_key_uses_express_endpoint():
    url, headers = build_url_and_headers(make_model(), GoogleVertexOptions(api_key="gc-key"))
    assert (
        url
        == "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    )
    assert headers["x-goog-api-key"] == "gc-key"
    assert "authorization" not in headers


def test_build_url_with_access_token_uses_regional_endpoint():
    options = GoogleVertexOptions(project="proj-1", location="us-central1", access_token="tok-1")
    url, headers = build_url_and_headers(make_model(), options)
    assert url == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-central1"
        "/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    )
    assert headers["authorization"] == "Bearer tok-1"


def test_build_url_global_location_has_no_location_prefixed_host():
    options = GoogleVertexOptions(project="proj-1", location="global", access_token="tok-1")
    url, _headers = build_url_and_headers(make_model(), options)
    assert url.startswith("https://aiplatform.googleapis.com/v1/projects/proj-1/locations/global")


def test_build_url_raises_when_unconfigured():
    with pytest.raises(ValueError):
        build_url_and_headers(make_model(), GoogleVertexOptions())


# --------------------------------------------------------------------------
# build_params (shared logic sanity check via the vertex entrypoint)
# --------------------------------------------------------------------------


def test_build_params_basic_shape():
    params = build_params(make_model(), Context(messages=[UserMessage(content="hi")]))
    assert params["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_build_params_tools():
    tools = [Tool(name="read", description="d")]
    params = build_params(make_model(), Context(messages=[], tools=tools))
    assert params["tools"][0]["functionDeclarations"][0]["name"] == "read"


def test_build_params_thinking_budget():
    params = build_params(
        make_model(reasoning=True),
        Context(messages=[]),
        GoogleVertexOptions(thinking=GoogleThinkingOptions(enabled=True, budget_tokens=4096)),
    )
    assert params["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 4096


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------


async def test_stream_emits_text_events_with_api_key():
    body = sse_body([text_chunk("Hello"), finish_chunk("STOP")])
    async with make_client(body) as client:
        events, message = await collect(
            stream(
                make_model(),
                Context(messages=[UserMessage(content="hi")]),
                GoogleVertexOptions(api_key="gc-key"),
                client=client,
            )
        )
    assert [e.type for e in events] == ["start", "text_start", "text_delta", "text_end", "done"]
    assert message.stop_reason == "stop"
    assert message.content[0].text == "Hello"


async def test_stream_with_access_token_sends_bearer_header():
    body = sse_body([finish_chunk("STOP")])
    capture: dict = {}
    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                make_model(),
                Context(messages=[]),
                GoogleVertexOptions(project="p", location="us-central1", access_token="tok-1"),
                client=client,
            )
        )
    assert capture["request"].headers["authorization"] == "Bearer tok-1"


async def test_stream_safety_blocked_reports_error_through_stream():
    body = sse_body([finish_chunk("RECITATION")])
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), GoogleVertexOptions(api_key="gc-key"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "RECITATION" in message.error_message


async def test_stream_reports_http_error_through_stream():
    async with make_client('{"error": {"message": "permission denied"}}', status=403) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), GoogleVertexOptions(api_key="gc-key"), client=client)
        )
    assert events[-1].type == "error"
    assert "permission denied" in message.error_message


async def test_stream_missing_auth_reports_error_without_raising():
    _events, message = await collect(stream(make_model(), Context(messages=[]), GoogleVertexOptions()))
    assert message.stop_reason == "error"
    assert "project ID" in message.error_message


async def test_stream_missing_access_token_with_project_and_location_reports_error():
    options = GoogleVertexOptions(project="p", location="us-central1")
    _events, message = await collect(stream(make_model(), Context(messages=[]), options))
    assert message.stop_reason == "error"
    assert "access token" in message.error_message


# --------------------------------------------------------------------------
# stream_simple
# --------------------------------------------------------------------------


async def test_stream_simple_disables_thinking_without_reasoning():
    body = sse_body([finish_chunk("STOP")])
    capture: dict = {}
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                make_model(reasoning=True),
                Context(messages=[]),
                SimpleStreamOptions(api_key="gc-key"),
                client=client,
            )
        )
    assert capture["json"]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


async def test_stream_simple_budget_based_thinking_for_flash():
    body = sse_body([finish_chunk("STOP")])
    capture: dict = {}
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                make_model(reasoning=True),
                Context(messages=[]),
                SimpleStreamOptions(api_key="gc-key", reasoning="high"),
                client=client,
            )
        )
    assert capture["json"]["generationConfig"]["thinkingConfig"] == {"includeThoughts": True, "thinkingBudget": 24576}


async def test_stream_simple_level_based_thinking_for_gemini3_pro():
    body = sse_body([finish_chunk("STOP")])
    capture: dict = {}
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                make_model(id="gemini-3-pro", reasoning=True),
                Context(messages=[]),
                SimpleStreamOptions(api_key="gc-key", reasoning="high"),
                client=client,
            )
        )
    assert capture["json"]["generationConfig"]["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "HIGH"}
