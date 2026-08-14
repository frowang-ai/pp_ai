"""Python port of `packages/ai/test/provider-error-body-regression.test.ts`.

Per-tier provider regression for `issues/provider-error-body-passthrough`:
routes a 403-with-body error through the real provider catch path for one
representative per tier. TypeScript covers a body-blind text provider
(openai-completions), a status-only provider (openai-responses) and a
body-blind Bedrock provider.

The two Bedrock cases are not ported: `bedrock-converse-stream` is a documented
omission of this port (it needs SigV4 signing and the Smithy stack, see the
repository README) and `pi_ai.api.bedrock_converse_stream` raises
`NotImplementedError`, so there is no catch path to exercise.

TypeScript mocks the `openai` SDK so `create().withResponse()` throws an
`APIError` carrying the parsed body on `.error`. The port has no SDK layer, so
the 403-with-body is served by an `httpx.MockTransport` and the adapter's own
`ProviderHttpError` carries the status/body/parsed-body that
`normalize_provider_error` reads.
"""

from __future__ import annotations

import re

import httpx
import pytest
from pi_ai.api import openai_completions, openai_responses
from pi_ai.types import Context, Model, ModelCost, StreamOptions, TextContent, UserMessage
from pi_ai.utils.event_stream import AssistantMessageEventStream

CONTEXT = Context(
    system_prompt="",
    messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)],
    tools=[],
)

COMPLETIONS_MODEL = Model(
    id="test-model",
    name="Test Model",
    api="openai-completions",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    reasoning=False,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=1000,
    max_tokens=100,
)

RESPONSES_MODEL = Model(
    id="gpt-test",
    name="GPT Test",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=False,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=1000,
    max_tokens=100,
)

_REAL_ASYNC_CLIENT = httpx.AsyncClient


def forbidden_client(parsed_body: object) -> httpx.AsyncClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": parsed_body})

    return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler))


async def drain_result(stream: AssistantMessageEventStream):
    async for _event in stream:
        pass
    return await stream.result()


async def test_openai_completions_body_blind_text_surfaces_status_and_body() -> None:
    output = await drain_result(
        openai_completions.stream(
            COMPLETIONS_MODEL,
            CONTEXT,
            StreamOptions(api_key="test"),
            client=forbidden_client("blocked by gateway WAF"),
        )
    )

    assert output.stop_reason == "error"
    assert "403" in output.error_message
    assert "blocked by gateway WAF" in output.error_message
    assert output.error_message != "403 status code (no body)"


async def test_openai_completions_does_not_double_print_the_openrouter_metadata_raw_extra() -> None:
    # OpenRouter returns the extra reason under error.error.metadata.raw, which
    # is part of the parsed body normalize_provider_error already surfaces. The
    # manual append must not duplicate it.
    output = await drain_result(
        openai_completions.stream(
            COMPLETIONS_MODEL,
            CONTEXT,
            StreamOptions(api_key="test"),
            client=forbidden_client(
                {
                    "message": "Provider returned error",
                    "code": 403,
                    "metadata": {"raw": "upstream WAF blocked policy XYZ"},
                }
            ),
        )
    )

    assert "upstream WAF blocked policy XYZ" in output.error_message
    assert len(re.findall("upstream WAF blocked policy XYZ", output.error_message)) == 1


async def test_openai_responses_status_only_keeps_the_prefix_and_surfaces_the_body() -> None:
    output = await drain_result(
        openai_responses.stream(
            RESPONSES_MODEL,
            CONTEXT,
            StreamOptions(api_key="test"),
            client=forbidden_client("blocked by gateway WAF"),
        )
    )

    assert output.stop_reason == "error"
    assert "OpenAI API error (403)" in output.error_message
    assert "blocked by gateway WAF" in output.error_message


def test_bedrock_body_blind_gateway_body_case_is_not_ported() -> None:
    # TS counterpart: "bedrock (body-blind) surfaces the gateway body instead of
    # Unknown: UnknownError". It drives `streamSimple` through the Bedrock
    # adapter with a mocked SDK `send()` that throws an `UnknownError` carrying
    # `$response.body`. There is no Python catch path to reach: the adapter is a
    # documented omission and refuses the call before any request is built.
    from pi_ai.api import bedrock_converse_stream

    with pytest.raises(NotImplementedError, match="not ported to Python"):
        bedrock_converse_stream.stream_simple(COMPLETIONS_MODEL, CONTEXT, None)


def test_bedrock_streamed_body_validation_message_case_is_not_ported() -> None:
    # TS counterpart: "bedrock preserves the SDK validation message when the
    # response body is a stream". Same reason as above; the streamed-body branch
    # it pins is AWS-SDK-specific (`$response.body` being a Node readable), so it
    # has no analogue even in principle here.
    from pi_ai.api import bedrock_converse_stream

    with pytest.raises(NotImplementedError, match="not ported to Python"):
        bedrock_converse_stream.stream(COMPLETIONS_MODEL, CONTEXT, None)
