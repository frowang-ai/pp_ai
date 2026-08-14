"""Python port of `packages/ai/test/openai-completions-thinking-as-text.test.ts`.

TypeScript spins up a real `node:http` server on localhost to observe the request
body. This port uses `httpx.MockTransport`, which is the same observation point
without binding a socket.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pi_ai.api.openai_completions import (
    OpenAICompletionsOptions,
    convert_messages,
    get_compat,
    stream,
)
from pi_ai.types import (
    AssistantContent,
    AssistantMessage,
    Context,
    Cost,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    Usage,
    UserMessage,
)

EMPTY_USAGE = Usage(
    input=0,
    output=0,
    cache_read=0,
    cache_write=0,
    total_tokens=0,
    cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
)

# `Model.compat` mirrors TypeScript's `OpenAICompletionsCompat` and keeps its
# camelCase keys.
COMPAT: dict[str, Any] = {
    "supportsStore": True,
    "supportsDeveloperRole": True,
    "supportsReasoningEffort": True,
    "supportsUsageInStreaming": True,
    "supportsFinishReason": True,
    "maxTokensField": "max_completion_tokens",
    "requiresToolResultName": False,
    "requiresAssistantAfterToolResult": False,
    "requiresThinkingAsText": True,
    "requiresReasoningContentOnAssistantMessages": False,
    "thinkingFormat": "openai",
    "openRouterRouting": {},
    "vercelGatewayRouting": {},
    "chatTemplateKwargs": {},
    "chatTemplateArgs": {},
    "zaiToolStream": False,
    "supportsThinkingTokenBudget": False,
    "supportsStrictMode": True,
    "supportsOpenAIGrammarTools": False,
    "sendSessionAffinityHeaders": False,
    "sessionAffinityFormat": "openai",
    "supportsLongCacheRetention": True,
}


def build_model(base_url: str = "http://127.0.0.1:1") -> Model:
    return Model(
        id="repro-model",
        name="Repro Model",
        api="openai-completions",
        provider="repro-provider",
        base_url=base_url,
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
        compat=COMPAT,
    )


def build_assistant(content: list[AssistantContent]) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api="openai-completions",
        provider="repro-provider",
        model="repro-model",
        usage=EMPTY_USAGE,
        stop_reason="stop",
        timestamp=2,
    )


def build_context(assistant: AssistantMessage) -> Context:
    return Context(
        messages=[
            UserMessage(content="hello", timestamp=1),
            assistant,
            UserMessage(content="continue", timestamp=3),
        ]
    )


def test_serializes_same_model_thinking_plus_text_replay_as_assistant_text_parts() -> None:
    model = build_model()
    messages = convert_messages(
        model,
        build_context(
            build_assistant(
                [
                    ThinkingContent(thinking="internal reasoning"),
                    TextContent(text="visible answer"),
                ]
            )
        ),
        get_compat(model),
    )

    assert messages[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "internal reasoning"},
            {"type": "text", "text": "visible answer"},
        ],
    }


def test_serializes_same_model_thinking_only_replay_as_assistant_text_parts() -> None:
    model = build_model()
    messages = convert_messages(
        model,
        build_context(build_assistant([ThinkingContent(thinking="internal reasoning")])),
        get_compat(model),
    )

    assert messages[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "internal reasoning"}],
    }


async def test_reaches_the_endpoint_when_replay_contains_both_thinking_and_text() -> None:
    request_bodies: list[dict[str, Any]] = []

    chunks = [
        {
            "id": "chatcmpl-repro",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "repro-model",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-repro",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "repro-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/chat/completions"
        request_bodies.append(json.loads(request.content))
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    events = []
    async for event in stream(
        build_model("http://127.0.0.1:1"),
        build_context(
            build_assistant(
                [
                    ThinkingContent(thinking="internal reasoning"),
                    TextContent(text="visible answer"),
                ]
            )
        ),
        OpenAICompletionsOptions(api_key="test-key"),
        client=client,
    ):
        events.append(event)

    assert len(request_bodies) == 1
    assert request_bodies[0]["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "internal reasoning"},
            {"type": "text", "text": "visible answer"},
        ],
    }
    assert events[-1].type == "done"
