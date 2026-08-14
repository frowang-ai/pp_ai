"""Python port of `packages/ai/test/openai-completions-prompt-cache.test.ts`.

TypeScript mocks the `openai` SDK and inspects the constructed client's
`defaultHeaders`. This port posts with `httpx`, so the equivalent observation
point is the recorded request's headers and JSON body.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import httpx
import pytest
from pi_ai.api.openai_completions import OpenAICompletionsOptions, stream
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Context, Model, UserMessage, now_ms

_CHUNK: dict[str, Any] = {
    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
}


@pytest.fixture(autouse=True)
def _clear_cache_retention_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI_CACHE_RETENTION", raising=False)


def create_model(**overrides: object) -> Model:
    base = get_builtin_model("openai", "gpt-4o-mini")
    assert base is not None
    fields: dict[str, object] = {"compat": {}, "api": "openai-completions"}
    fields.update(overrides)
    return dataclasses.replace(base, **fields)


async def capture_request(
    options: OpenAICompletionsOptions | None = None,
    model: Model | None = None,
) -> tuple[dict[str, Any], httpx.Headers]:
    captured: dict[str, Any] = {}
    body = f"data: {json.dumps(_CHUNK)}\n\ndata: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        captured["headers"] = request.headers
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stream_options = options or OpenAICompletionsOptions()
    stream_options.api_key = "test-key"

    await stream(
        model if model is not None else create_model(),
        Context(system_prompt="sys", messages=[UserMessage(content="hi", timestamp=now_ms())]),
        stream_options,
        client=client,
    ).result()

    return captured["payload"], captured["headers"]


async def test_sets_prompt_cache_key_for_direct_openai_requests_when_caching_is_enabled() -> None:
    payload, _headers = await capture_request(OpenAICompletionsOptions(session_id="session-123"))

    assert payload["prompt_cache_key"] == "session-123"
    assert "prompt_cache_retention" not in payload


async def test_sets_prompt_cache_retention_to_24h_when_cache_retention_is_long() -> None:
    payload, _headers = await capture_request(
        OpenAICompletionsOptions(cache_retention="long", session_id="session-456")
    )

    assert payload["prompt_cache_key"] == "session-456"
    assert payload["prompt_cache_retention"] == "24h"


async def test_clamps_prompt_cache_key_to_openais_64_character_limit() -> None:
    payload, _headers = await capture_request(OpenAICompletionsOptions(session_id="x" * 67))

    assert payload["prompt_cache_key"] == "x" * 64


async def test_omits_prompt_cache_fields_when_cache_retention_is_none() -> None:
    payload, _headers = await capture_request(
        OpenAICompletionsOptions(cache_retention="none", session_id="session-789")
    )

    assert "prompt_cache_key" not in payload
    assert "prompt_cache_retention" not in payload


async def test_omits_prompt_cache_fields_for_non_openai_base_urls_without_long_retention() -> None:
    model = create_model(
        base_url="https://proxy.example.com/v1",
        compat={"supportsLongCacheRetention": False},
    )
    payload, _headers = await capture_request(
        OpenAICompletionsOptions(cache_retention="long", session_id="session-proxy"), model
    )

    assert "prompt_cache_key" not in payload
    assert "prompt_cache_retention" not in payload


async def test_uses_pi_cache_retention_for_direct_openai_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_CACHE_RETENTION", "long")
    payload, _headers = await capture_request(OpenAICompletionsOptions(session_id="session-env"))

    assert payload["prompt_cache_key"] == "session-env"
    assert payload["prompt_cache_retention"] == "24h"


async def test_sends_known_session_affinity_headers_when_compat_enables_them() -> None:
    model = create_model(base_url="https://proxy.example.com/v1", compat={"sendSessionAffinityHeaders": True})
    _payload, headers = await capture_request(OpenAICompletionsOptions(session_id="session-affinity"), model)

    assert headers["session_id"] == "session-affinity"
    assert headers["x-client-request-id"] == "session-affinity"
    assert headers["x-session-affinity"] == "session-affinity"


@pytest.mark.parametrize(
    "model_id",
    ["accounts/fireworks/models/glm-5p2", "accounts/fireworks/routers/glm-5p2-fast"],
)
async def test_sends_fireworks_session_affinity(model_id: str) -> None:
    model = get_builtin_model("fireworks", model_id)
    assert model is not None
    _payload, headers = await capture_request(OpenAICompletionsOptions(session_id="fireworks-session"), model)

    assert headers["x-session-affinity"] == "fireworks-session"


async def test_uses_openai_no_session_format_when_configured() -> None:
    model = create_model(
        compat={"sendSessionAffinityHeaders": True, "sessionAffinityFormat": "openai-nosession"},
    )
    payload, headers = await capture_request(OpenAICompletionsOptions(session_id="session-nosession"), model)

    assert "session_id" not in payload
    assert payload["prompt_cache_key"] == "session-nosession"
    assert "session_id" not in headers
    assert headers["x-client-request-id"] == "session-nosession"
    assert headers["x-session-affinity"] == "session-nosession"
    assert "x-session-id" not in headers


async def test_uses_openrouter_session_affinity_header_when_configured() -> None:
    model = create_model(
        base_url="https://proxy.example.com/v1",
        compat={"sendSessionAffinityHeaders": True, "sessionAffinityFormat": "openrouter"},
    )
    payload, headers = await capture_request(OpenAICompletionsOptions(session_id="session-proxy"), model)

    assert "session_id" not in payload
    assert "prompt_cache_key" not in payload
    assert headers["x-session-id"] == "session-proxy"
    assert "session_id" not in headers
    assert "x-client-request-id" not in headers
    assert "x-session-affinity" not in headers


async def test_auto_detects_openrouter_session_affinity_header_for_openrouter_endpoints() -> None:
    model = create_model(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        compat={"sendSessionAffinityHeaders": True},
    )
    payload, headers = await capture_request(OpenAICompletionsOptions(session_id="session-openrouter"), model)

    assert "session_id" not in payload
    assert "prompt_cache_key" not in payload
    assert headers["x-session-id"] == "session-openrouter"
    assert "session_id" not in headers
    assert "x-client-request-id" not in headers
    assert "x-session-affinity" not in headers


async def test_omits_openrouter_session_affinity_data_when_disabled() -> None:
    model = create_model(provider="openrouter", base_url="https://openrouter.ai/api/v1")
    payload, headers = await capture_request(OpenAICompletionsOptions(session_id="session-openrouter"), model)

    assert "session_id" not in payload
    assert "prompt_cache_key" not in payload
    assert "x-session-id" not in headers


async def test_omits_session_affinity_headers_when_cache_retention_is_none() -> None:
    model = create_model(base_url="https://proxy.example.com/v1", compat={"sendSessionAffinityHeaders": True})
    _payload, headers = await capture_request(
        OpenAICompletionsOptions(cache_retention="none", session_id="session-affinity"), model
    )

    assert "session_id" not in headers
    assert "x-client-request-id" not in headers
    assert "x-session-affinity" not in headers


async def test_lets_explicit_headers_override_generated_session_affinity_headers() -> None:
    model = create_model(base_url="https://proxy.example.com/v1", compat={"sendSessionAffinityHeaders": True})
    _payload, headers = await capture_request(
        OpenAICompletionsOptions(
            session_id="session-affinity",
            headers={
                "session_id": "override-session",
                "x-client-request-id": "override-request",
                "x-session-affinity": "override-affinity",
            },
        ),
        model,
    )

    assert headers["session_id"] == "override-session"
    assert headers["x-client-request-id"] == "override-request"
    assert headers["x-session-affinity"] == "override-affinity"
