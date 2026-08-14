"""Tests ported from packages/ai/test/faux-provider.test.ts.

The agent-loop integration case that used to live here now sits in
`pp-agent-core` (`tests/test_faux_provider_agent_loop.py`): it needs
`pi_agent`, and `pp-agent-core` depends on `pp-ai`, so keeping it here made the
two packages depend on each other -- fine inside one workspace, unresolvable
once each is its own repository."""

from __future__ import annotations

import json
import math

from pi_ai import (
    AssistantMessageEvent,
    Context,
    ImageContent,
    TextContent,
    ToolResultMessage,
    UserMessage,
    now_ms,
)
from pi_ai import (
    SimpleStreamOptions as StreamOptions,
)
from pi_ai.models import complete
from pi_ai.providers.faux import (
    FauxDeferredOptions,
    FauxModelDefinition,
    FauxProviderHandle,
    FauxTokenSizeOptions,
    RegisterFauxProviderOptions,
    faux_assistant_message,
    faux_provider,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from pi_ai.types import Tool
from pi_ai.utils.abort import AbortController
from pi_ai.utils.event_stream import AssistantMessageEventStream


async def collect_events(stream: AssistantMessageEventStream) -> list[AssistantMessageEvent]:
    return [event async for event in stream]


def make_context(**overrides) -> Context:
    defaults: dict = {"messages": [UserMessage(content="hi")]}
    defaults.update(overrides)
    return Context(**defaults)


def make_handle(**options) -> FauxProviderHandle:
    if isinstance(options.get("token_size"), dict):
        options["token_size"] = FauxTokenSizeOptions(**options["token_size"])
    if isinstance(options.get("deferred"), dict):
        options["deferred"] = FauxDeferredOptions(**options["deferred"])
    return faux_provider(RegisterFauxProviderOptions(**options))


async def test_registers_a_custom_provider_and_estimates_usage() -> None:
    handle = make_handle()
    handle.set_responses([faux_assistant_message("hello world")])

    context = Context(system_prompt="Be concise.", messages=[UserMessage(content="hi there")])
    response = await complete(handle.provider.stream(handle.get_model(), context))

    assert response.content == [TextContent(text="hello world")]
    assert response.usage.input > 0
    assert response.usage.output > 0
    assert response.usage.total_tokens == response.usage.input + response.usage.output
    assert handle.state.call_count == 1


async def test_supports_helper_blocks_for_text_thinking_and_tool_calls() -> None:
    handle = make_handle()
    handle.set_responses(
        [
            faux_assistant_message(
                [faux_thinking("think"), faux_tool_call("echo", {"text": "hi"}), faux_text("done")],
                stop_reason="toolUse",
            )
        ]
    )

    response = await complete(handle.provider.stream(handle.get_model(), make_context()))

    assert response.content[0] == faux_thinking("think")
    assert response.content[1].name == "echo"
    assert response.content[1].arguments == {"text": "hi"}
    assert response.content[2] == faux_text("done")
    assert response.stop_reason == "toolUse"


async def test_supports_multiple_models_with_per_model_reasoning_and_model_aware_factories() -> None:
    handle = make_handle(
        models=[
            FauxModelDefinition(id="faux-fast", name="Faux Fast", reasoning=False),
            FauxModelDefinition(id="faux-thinker", name="Faux Thinker", reasoning=True),
        ]
    )
    handle.set_responses(
        [
            lambda _context, _options, _state, model: faux_assistant_message(f"{model.id}:{model.reasoning}"),
            lambda _context, _options, _state, model: faux_assistant_message(f"{model.id}:{model.reasoning}"),
        ]
    )

    assert [model.id for model in handle.models] == ["faux-fast", "faux-thinker"]
    assert handle.get_model() is handle.models[0]
    assert handle.get_model("faux-fast").reasoning is False
    assert handle.get_model("faux-thinker").reasoning is True

    fast = await complete(handle.provider.stream(handle.get_model("faux-fast"), make_context()))
    thinker = await complete(handle.provider.stream(handle.get_model("faux-thinker"), make_context()))

    assert fast.content == [TextContent(text="faux-fast:False")]
    assert thinker.content == [TextContent(text="faux-thinker:True")]


async def test_rewrites_api_provider_and_model_on_returned_messages() -> None:
    handle = make_handle(api="faux:test", provider="faux-provider", models=[FauxModelDefinition(id="faux-model")])
    handle.set_responses([faux_assistant_message("hello")])

    response = await complete(handle.provider.stream(handle.get_model(), make_context()))

    assert response.api == "faux:test"
    assert response.provider == "faux-provider"
    assert response.model == "faux-model"


async def test_consumes_queued_responses_in_order_and_errors_when_exhausted() -> None:
    handle = make_handle()
    handle.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])
    context = make_context()

    first = await complete(handle.provider.stream(handle.get_model(), context))
    second = await complete(handle.provider.stream(handle.get_model(), context))
    exhausted = await complete(handle.provider.stream(handle.get_model(), context))

    assert first.content == [TextContent(text="first")]
    assert second.content == [TextContent(text="second")]
    assert exhausted.stop_reason == "error"
    assert exhausted.error_message == "No more faux responses queued"
    assert handle.get_pending_response_count() == 0
    assert handle.state.call_count == 3


async def test_can_replace_and_append_queued_responses() -> None:
    handle = make_handle()
    handle.set_responses([faux_assistant_message("first")])
    context = make_context()

    assert (await complete(handle.provider.stream(handle.get_model(), context))).content == [TextContent(text="first")]
    assert handle.get_pending_response_count() == 0

    handle.set_responses([faux_assistant_message("second")])
    assert handle.get_pending_response_count() == 1
    assert (await complete(handle.provider.stream(handle.get_model(), context))).content == [TextContent(text="second")]

    handle.append_responses([faux_assistant_message("third"), faux_assistant_message("fourth")])
    assert handle.get_pending_response_count() == 2
    assert (await complete(handle.provider.stream(handle.get_model(), context))).content == [TextContent(text="third")]
    assert (await complete(handle.provider.stream(handle.get_model(), context))).content == [TextContent(text="fourth")]
    assert handle.get_pending_response_count() == 0


async def test_supports_async_response_factories() -> None:
    handle = make_handle()

    async def factory(context, _options, state, _model):
        return faux_assistant_message(f"{len(context.messages)}:{state.call_count}")

    handle.set_responses([factory])

    response = await complete(handle.provider.stream(handle.get_model(), make_context()))

    assert response.content == [TextContent(text="1:1")]


async def test_emits_an_error_when_a_response_factory_throws() -> None:
    handle = make_handle()

    def factory(*_args):
        raise RuntimeError("boom")

    handle.set_responses([factory])

    events = await collect_events(handle.provider.stream(handle.get_model(), make_context()))

    assert len(events) == 1
    assert events[0].type == "error"
    assert events[0].error.stop_reason == "error"
    assert events[0].error.error_message == "boom"


async def test_rejects_a_queued_response_without_a_terminal_stop_reason() -> None:
    handle = make_handle()
    handle.set_responses([faux_assistant_message("partial", stop_reason="pending")])

    events = await collect_events(handle.provider.stream(handle.get_model(), make_context()))

    assert not any(event.type == "done" for event in events)
    terminal = events[-1]
    assert terminal.type == "error"
    assert terminal.error.stop_reason == "error"
    assert terminal.error.error_message == "Faux response ended without a stop reason"


async def test_estimates_prompt_and_output_tokens_from_serialized_context() -> None:
    handle = make_handle()
    handle.set_responses([faux_assistant_message("done")])

    tool = Tool(
        name="echo",
        description="Echo back text",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
    )
    context = Context(
        system_prompt="sys",
        messages=[
            UserMessage(
                content=[TextContent(text="hello"), ImageContent(mime_type="image/png", data="abcd")], timestamp=1
            ),
            faux_assistant_message("prior"),
            ToolResultMessage(
                tool_call_id="tool-1",
                tool_name="echo",
                content=[TextContent(text="tool out")],
                is_error=False,
                timestamp=2,
            ),
        ],
        tools=[tool],
    )

    response = await complete(handle.provider.stream(handle.get_model(), context))
    tool_json = {"name": "echo", "description": "Echo back text", "parameters": tool.parameters}
    prompt_text = "\n\n".join(
        [
            "system:sys",
            "user:hello\n[image:image/png:4]",
            "assistant:prior",
            "toolResult:echo\ntool out",
            f"tools:{json.dumps([tool_json])}",
        ]
    )
    expected_prompt_tokens = math.ceil(len(prompt_text) / 4)
    expected_output_tokens = math.ceil(len("done") / 4)

    assert response.usage.input == expected_prompt_tokens
    assert response.usage.output == expected_output_tokens
    assert response.usage.cache_read == 0
    assert response.usage.cache_write == 0
    assert response.usage.total_tokens == expected_prompt_tokens + expected_output_tokens


async def test_does_not_share_cache_across_sessions_or_requests_without_session_id() -> None:
    handle = make_handle()
    handle.set_responses(
        [faux_assistant_message("first"), faux_assistant_message("second"), faux_assistant_message("third")]
    )
    context = make_context(messages=[UserMessage(content="hello")])

    first = await complete(
        handle.provider.stream(
            handle.get_model(), context, StreamOptions(session_id="session-1", cache_retention="short")
        )
    )
    assert first.usage.cache_write > 0
    context.messages.append(first)
    context.messages.append(UserMessage(content="follow up", timestamp=now_ms() + 1))

    second = await complete(
        handle.provider.stream(
            handle.get_model(), context, StreamOptions(session_id="session-2", cache_retention="short")
        )
    )
    assert second.usage.cache_read == 0
    assert second.usage.cache_write > 0

    third = await complete(handle.provider.stream(handle.get_model(), context))
    assert third.usage.cache_read == 0
    assert third.usage.cache_write == 0


async def test_simulates_prompt_caching_per_session_id() -> None:
    handle = make_handle()
    handle.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])
    context = Context(system_prompt="Be concise.", messages=[UserMessage(content="hello")])

    first = await complete(
        handle.provider.stream(
            handle.get_model(), context, StreamOptions(session_id="session-1", cache_retention="short")
        )
    )
    assert first.usage.cache_read == 0
    assert first.usage.cache_write > 0

    context.messages.append(first)
    context.messages.append(UserMessage(content="follow up", timestamp=now_ms() + 1))

    second = await complete(
        handle.provider.stream(
            handle.get_model(), context, StreamOptions(session_id="session-1", cache_retention="short")
        )
    )
    assert second.usage.cache_read > 0
    assert second.usage.input + second.usage.cache_read > second.usage.input


async def test_does_not_simulate_caching_when_cache_retention_is_none() -> None:
    handle = make_handle()
    handle.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])
    context = make_context(messages=[UserMessage(content="hello")])

    await complete(
        handle.provider.stream(
            handle.get_model(), context, StreamOptions(session_id="session-1", cache_retention="none")
        )
    )
    context.messages.append(faux_assistant_message("first"))
    context.messages.append(UserMessage(content="follow up", timestamp=now_ms() + 1))
    second = await complete(
        handle.provider.stream(
            handle.get_model(), context, StreamOptions(session_id="session-1", cache_retention="none")
        )
    )
    assert second.usage.cache_read == 0
    assert second.usage.cache_write == 0


async def test_streams_thinking_text_and_partial_tool_call_deltas() -> None:
    handle = make_handle()
    handle.set_responses(
        [
            faux_assistant_message(
                [
                    faux_thinking("thinking text"),
                    faux_text("answer text"),
                    faux_tool_call("echo", {"text": "hi", "count": 12}, id="tool-1"),
                ],
                stop_reason="toolUse",
            )
        ]
    )

    events: list[str] = []
    tool_call_deltas: list[str] = []
    async for event in handle.provider.stream(handle.get_model(), make_context()):
        events.append(event.type)
        if event.type == "toolcall_delta":
            tool_call_deltas.append(event.delta)

    assert "thinking_start" in events
    assert "thinking_delta" in events
    assert "text_start" in events
    assert "text_delta" in events
    assert "toolcall_start" in events
    assert "toolcall_delta" in events
    assert "toolcall_end" in events
    assert len(tool_call_deltas) > 1
    assert json.loads("".join(tool_call_deltas)) == {"text": "hi", "count": 12}


async def test_streams_an_exact_event_order_for_fixed_size_chunks() -> None:
    handle = make_handle(token_size={"min": 1, "max": 1})
    handle.set_responses(
        [
            faux_assistant_message(
                [faux_thinking("go"), faux_text("ok"), faux_tool_call("echo", {}, id="tool-1")],
                stop_reason="toolUse",
            )
        ]
    )

    events = await collect_events(handle.provider.stream(handle.get_model(), make_context()))

    assert events[0].type == "start"
    assert events[0].partial.stop_reason == "pending"
    assert [event.type for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]


async def test_streams_multiple_tool_calls_in_one_message() -> None:
    handle = make_handle()
    handle.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("echo", {"text": "one"}, id="tool-1"),
                    faux_tool_call("echo", {"text": "two"}, id="tool-2"),
                ],
                stop_reason="toolUse",
            )
        ]
    )

    events = await collect_events(handle.provider.stream(handle.get_model(), make_context()))

    assert len([e for e in events if e.type == "toolcall_start"]) == 2
    assert len([e for e in events if e.type == "toolcall_end"]) == 2


async def test_streams_an_explicit_assistant_error_message_as_a_terminal_error() -> None:
    handle = make_handle(token_size={"min": 2, "max": 2})
    message = faux_assistant_message("partial")
    message.stop_reason = "error"
    message.error_message = "upstream failed"
    handle.set_responses([message])

    events = await collect_events(handle.provider.stream(handle.get_model(), make_context()))

    assert [event.type for event in events] == ["start", "text_start", "text_delta", "text_end", "error"]
    terminal = events[-1]
    assert terminal.reason == "error"
    assert terminal.error.stop_reason == "error"
    assert terminal.error.error_message == "upstream failed"


async def test_streams_an_explicit_assistant_aborted_message_as_a_terminal_error() -> None:
    handle = make_handle(token_size={"min": 2, "max": 2})
    message = faux_assistant_message("partial")
    message.stop_reason = "aborted"
    message.error_message = "Request was aborted"
    handle.set_responses([message])

    events = await collect_events(handle.provider.stream(handle.get_model(), make_context()))

    assert [event.type for event in events] == ["start", "text_start", "text_delta", "text_end", "error"]
    terminal = events[-1]
    assert terminal.reason == "aborted"
    assert terminal.error.stop_reason == "aborted"
    assert terminal.error.error_message == "Request was aborted"


async def test_supports_aborting_before_the_first_chunk() -> None:
    handle = make_handle(tokens_per_second=50, token_size={"min": 3, "max": 3})
    handle.set_responses([faux_assistant_message("abcdefghijklmnopqrstuvwxyz")])

    controller = AbortController()
    controller.abort()
    events = await collect_events(
        handle.provider.stream(handle.get_model(), make_context(), StreamOptions(signal=controller.signal))
    )

    assert len(events) == 1
    assert events[0].type == "error"
    assert events[0].reason == "aborted"
    assert events[0].error.stop_reason == "aborted"


async def test_supports_aborting_mid_text_stream_when_paced() -> None:
    handle = make_handle(tokens_per_second=100, token_size={"min": 3, "max": 3})
    handle.set_responses([faux_assistant_message("abcdefghijklmnopqrstuvwxyz")])

    controller = AbortController()
    events: list[str] = []
    text_delta_count = 0
    async for event in handle.provider.stream(
        handle.get_model(), make_context(), StreamOptions(signal=controller.signal)
    ):
        events.append(event.type)
        if event.type == "text_delta":
            text_delta_count += 1
            controller.abort()

    assert text_delta_count == 1
    assert "text_start" in events
    assert "text_delta" in events
    assert "error" in events
    assert "text_end" not in events


async def test_supports_aborting_mid_thinking_stream_when_paced() -> None:
    handle = make_handle(tokens_per_second=100, token_size={"min": 3, "max": 3})
    message = faux_assistant_message("ignored")
    message.content = [faux_thinking("abcdefghijklmnopqrstuvwxyz")]
    handle.set_responses([message])

    controller = AbortController()
    events: list[str] = []
    thinking_delta_count = 0
    async for event in handle.provider.stream(
        handle.get_model(), make_context(), StreamOptions(signal=controller.signal)
    ):
        events.append(event.type)
        if event.type == "thinking_delta":
            thinking_delta_count += 1
            controller.abort()

    assert thinking_delta_count == 1
    assert "thinking_start" in events
    assert "thinking_delta" in events
    assert "error" in events
    assert "thinking_end" not in events


async def test_supports_aborting_mid_toolcall_stream_when_paced() -> None:
    handle = make_handle(tokens_per_second=100, token_size={"min": 3, "max": 3})
    message = faux_assistant_message("done")
    message.content = [faux_tool_call("echo", {"text": "abcdefghijklmnopqrstuvwxyz", "count": 123456789}, id="tool-1")]
    message.stop_reason = "toolUse"
    handle.set_responses([message])

    controller = AbortController()
    events: list[str] = []
    tool_call_delta_count = 0
    async for event in handle.provider.stream(
        handle.get_model(), make_context(), StreamOptions(signal=controller.signal)
    ):
        events.append(event.type)
        if event.type == "toolcall_delta":
            tool_call_delta_count += 1
            controller.abort()

    assert tool_call_delta_count == 1
    assert "toolcall_start" in events
    assert "toolcall_delta" in events
    assert "error" in events
    assert "toolcall_end" not in events
