"""Python port of `packages/ai/test/cloudflare-stream.test.ts`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pi_ai.providers.cloudflare_stream import cloudflare_streams
from pi_ai.types import Context, Model, ModelCost, SimpleStreamOptions, StreamOptions
from pi_ai.utils.event_stream import AssistantMessageEventStream

MODEL = Model(
    id="model",
    name="model",
    api="openai-completions",
    provider="cloudflare-ai-gateway",
    base_url="https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai",
    reasoning=False,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=1000,
    max_tokens=100,
)

CONTEXT = Context(messages=[])


@dataclass
class _CapturingStreams:
    captured: list[str] = field(default_factory=list)

    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream:
        self.captured.append(model.base_url)
        return AssistantMessageEventStream()

    def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream:
        self.captured.append(model.base_url)
        return AssistantMessageEventStream()


def test_materializes_the_model_endpoint_before_dispatch():
    inner = _CapturingStreams()
    streams = cloudflare_streams(inner)
    env = {"CLOUDFLARE_ACCOUNT_ID": "account", "CLOUDFLARE_GATEWAY_ID": "gateway"}

    streams.stream(MODEL, CONTEXT, StreamOptions(env=env))
    streams.stream_simple(MODEL, CONTEXT, SimpleStreamOptions(env=env))

    assert inner.captured == [
        "https://gateway.ai.cloudflare.com/v1/account/gateway/openai",
        "https://gateway.ai.cloudflare.com/v1/account/gateway/openai",
    ]


def test_keeps_placeholders_when_the_provider_env_does_not_resolve_them():
    inner = _CapturingStreams()
    streams = cloudflare_streams(inner)

    streams.stream_simple(MODEL, CONTEXT, SimpleStreamOptions())

    assert inner.captured == [MODEL.base_url]
