"""Python port of `packages/ai/test/anthropic-auth-token.test.ts`.

TypeScript mocks the `@anthropic-ai/sdk` constructor and asserts on
`constructorOpts.apiKey` / `constructorOpts.authToken` / `defaultHeaders`. This
port speaks HTTP directly and has no SDK client object, so the equivalent
assertions read the headers of the request an `httpx.MockTransport` receives:
`authToken` set means an `authorization: Bearer ...` header, `apiKey` set means
an `x-api-key` header, and OAuth-mode shaping shows up as the
`oauth-2025-04-20` entry in `anthropic-beta`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pi_ai.api.anthropic_messages import AnthropicOptions
from pi_ai.api.anthropic_messages import stream as stream_anthropic
from pi_ai.auth.types import AuthResult, ResolvedAuth
from pi_ai.providers.anthropic import (
    ANTHROPIC_AUTH_TOKEN_ENV,
    ANTHROPIC_OAUTH_TOKEN_ENV,
    anthropic_provider,
)
from pi_ai.registry import Models
from pi_ai.types import Context, Model, ModelCost, SimpleStreamOptions, UserMessage

SSE_BODY = "\n".join(
    [
        "event: message_start\ndata: "
        + json.dumps(
            {
                "type": "message_start",
                "message": {"id": "msg_test", "usage": {"input_tokens": 1, "output_tokens": 0}},
            }
        )
        + "\n",
        "event: message_delta\ndata: "
        + json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}})
        + "\n",
        "event: message_stop\ndata: " + json.dumps({"type": "message_stop"}) + "\n",
    ]
)

CONTEXT = Context(system_prompt="System prompt.", messages=[UserMessage(content="Hello")])

ANTHROPIC_MODEL = Model(
    id="claude-test",
    name="Claude Test",
    api="anthropic-messages",
    provider="anthropic",
    base_url="https://api.anthropic.com",
    reasoning=False,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=100000,
    max_tokens=4096,
)


def make_client(capture: dict[str, Any]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        capture["headers"] = {name.lower(): value for name, value in request.headers.items()}
        capture["json"] = json.loads(request.content)
        return httpx.Response(200, text=SSE_BODY, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def env_from(values: dict[str, str]):
    async def lookup(name: str) -> str | None:
        return values.get(name)

    return lookup


async def test_resolves_auth_token_as_a_bearer_authorization_header():
    provider = anthropic_provider()
    auth = await provider.auth.api_key.resolve(
        None,
        env_from(
            {
                "ANTHROPIC_AUTH_TOKEN": "auth-token",
                "ANTHROPIC_OAUTH_TOKEN": "oauth-token",
                "ANTHROPIC_API_KEY": "api-key",
            }
        ),
    )

    assert auth == AuthResult(
        auth=ResolvedAuth(headers={"Authorization": "Bearer auth-token"}),
        source=ANTHROPIC_AUTH_TOKEN_ENV,
    )


async def test_preserves_oauth_token_as_oauth_shaped_api_auth():
    provider = anthropic_provider()
    auth = await provider.auth.api_key.resolve(
        None,
        env_from({"ANTHROPIC_OAUTH_TOKEN": "oauth-token", "ANTHROPIC_API_KEY": "api-key"}),
    )

    assert auth == AuthResult(auth=ResolvedAuth(api_key="oauth-token"), source=ANTHROPIC_OAUTH_TOKEN_ENV)


async def test_uses_authorization_headers_without_oauth_mode_request_shaping():
    capture: dict[str, Any] = {}
    event_stream = stream_anthropic(
        ANTHROPIC_MODEL,
        CONTEXT,
        AnthropicOptions(headers={"Authorization": "Bearer header-token"}),
        client=make_client(capture),
    )
    await event_stream.result()

    headers = capture["headers"]
    assert "x-api-key" not in headers
    assert headers["authorization"] == "Bearer header-token"
    assert "oauth-2025-04-20" not in headers.get("anthropic-beta", "")
    system = capture["json"]["system"]
    assert len(system) == 1
    assert system[0]["text"] == "System prompt."


async def test_threads_auth_context_auth_token_through_request_headers():
    models = Models(env=env_from({"ANTHROPIC_AUTH_TOKEN": "ctx-token"}))
    models.add(anthropic_provider())

    capture: dict[str, Any] = {}
    event_stream = await models.stream_simple(
        ANTHROPIC_MODEL, CONTEXT, SimpleStreamOptions(), client=make_client(capture)
    )
    await event_stream.result()

    headers = capture["headers"]
    assert "x-api-key" not in headers
    assert headers["authorization"] == "Bearer ctx-token"
    assert "oauth-2025-04-20" not in headers.get("anthropic-beta", "")
    system = capture["json"]["system"]
    assert len(system) == 1
    assert system[0]["text"] == "System prompt."


async def test_preserves_oauth_request_shaping_for_oauth_token():
    models = Models(env=env_from({"ANTHROPIC_OAUTH_TOKEN": "sk-ant-oat-test"}))
    models.add(anthropic_provider())

    capture: dict[str, Any] = {}
    event_stream = await models.stream_simple(
        ANTHROPIC_MODEL, CONTEXT, SimpleStreamOptions(), client=make_client(capture)
    )
    await event_stream.result()

    headers = capture["headers"]
    assert "x-api-key" not in headers
    assert headers["authorization"] == "Bearer sk-ant-oat-test"
    assert "oauth-2025-04-20" in headers["anthropic-beta"]


async def test_lets_explicit_request_headers_override_auth_token():
    models = Models(env=env_from({"ANTHROPIC_AUTH_TOKEN": "ctx-token"}))
    models.add(anthropic_provider())

    capture: dict[str, Any] = {}
    event_stream = await models.stream_simple(
        ANTHROPIC_MODEL,
        CONTEXT,
        SimpleStreamOptions(headers={"Authorization": "Bearer explicit-token"}),
        client=make_client(capture),
    )
    await event_stream.result()

    assert capture["headers"]["authorization"] == "Bearer explicit-token"
