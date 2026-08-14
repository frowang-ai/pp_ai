"""Python port of `packages/ai/test/github-copilot-anthropic.test.ts`.

TypeScript mocks `@anthropic-ai/sdk` and inspects the constructor options it
receives (`authToken`, `defaultHeaders`) plus the `messages.create` params. This
port speaks HTTP directly, so the same two things are read off the actual
request an `httpx.MockTransport` receives: its headers and its JSON body.
"""

from __future__ import annotations

import json

import httpx
from pi_ai.api.anthropic_messages import AnthropicOptions, stream
from pi_ai.models import get_supported_thinking_levels
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Context, UserMessage

CONTEXT = Context(system_prompt="You are a helpful assistant.", messages=[UserMessage(content="Hello")])

SSE_BODY = "\n".join(
    [
        "event: message_start\ndata: "
        + json.dumps(
            {"type": "message_start", "message": {"id": "msg_test", "usage": {"input_tokens": 10, "output_tokens": 0}}}
        )
        + "\n",
        "event: message_delta\ndata: "
        + json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}})
        + "\n",
    ]
)


def make_client(capture: dict) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        capture["headers"] = {name.lower(): value for name, value in request.headers.items()}
        capture["json"] = json.loads(request.content)
        return httpx.Response(200, text=SSE_BODY, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_applies_copilot_specific_adaptive_thinking_effort_overrides():
    opus47 = get_builtin_model("github-copilot", "claude-opus-4.7")
    assert opus47.thinking_level_map["minimal"] == "low"
    assert opus47.thinking_level_map["xhigh"] == "xhigh"
    assert opus47.thinking_level_map["max"] == "max"
    assert "xhigh" in get_supported_thinking_levels(opus47)
    assert "max" in get_supported_thinking_levels(opus47)

    opus5 = get_builtin_model("github-copilot", "claude-opus-5")
    assert opus5.api == "anthropic-messages"
    assert opus5.context_window == 1000000
    assert opus5.thinking_level_map["minimal"] == "low"
    assert opus5.thinking_level_map["xhigh"] == "xhigh"
    assert opus5.thinking_level_map["max"] == "max"
    assert "xhigh" in get_supported_thinking_levels(opus5)
    assert "max" in get_supported_thinking_levels(opus5)

    sonnet46 = get_builtin_model("github-copilot", "claude-sonnet-4.6")
    assert sonnet46.thinking_level_map["minimal"] == "low"
    assert sonnet46.thinking_level_map["max"] == "max"
    assert "max" in get_supported_thinking_levels(sonnet46)
    assert "xhigh" not in get_supported_thinking_levels(sonnet46)


async def test_uses_bearer_auth_copilot_headers_and_valid_anthropic_messages_payload():
    model = get_builtin_model("github-copilot", "claude-sonnet-4.6")
    assert model.api == "anthropic-messages"

    capture: dict = {}
    event_stream = stream(
        model,
        CONTEXT,
        AnthropicOptions(api_key="tid_copilot_session_test_token"),
        client=make_client(capture),
    )
    async for event in event_stream:
        if event.type == "error":
            break

    headers = capture["headers"]

    # Auth: no x-api-key, bearer authorization for the Copilot session token.
    assert headers["authorization"] == "Bearer tid_copilot_session_test_token"
    assert "x-api-key" not in headers

    # Copilot static headers from model.headers
    assert "GitHubCopilotChat" in headers["user-agent"]
    assert headers["copilot-integration-id"] == "vscode-chat"

    # Dynamic headers
    assert headers["x-initiator"] == "user"
    assert headers["openai-intent"] == "conversation-edits"

    # No fine-grained-tool-streaming (Copilot doesn't support it)
    assert "fine-grained-tool-streaming" not in headers.get("anthropic-beta", "")

    # Payload is valid Anthropic Messages format
    params = capture["json"]
    assert params["model"] == "claude-sonnet-4.6"
    assert params["stream"] is True
    assert params["max_tokens"] == model.max_tokens
    assert isinstance(params["messages"], list)


async def test_omits_interleaved_thinking_beta_for_adaptive_thinking_models():
    model = get_builtin_model("github-copilot", "claude-sonnet-4.6")

    capture: dict = {}
    event_stream = stream(
        model,
        CONTEXT,
        AnthropicOptions(api_key="tid_copilot_session_test_token", interleaved_thinking=True),
        client=make_client(capture),
    )
    async for event in event_stream:
        if event.type == "error":
            break

    assert "interleaved-thinking-2025-05-14" not in capture["headers"].get("anthropic-beta", "")
