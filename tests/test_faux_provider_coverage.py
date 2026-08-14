"""Extra coverage tests for pi_ai.providers.faux — targeting missing lines.

These tests exercise faux's own scripting/validation logic directly:
image content, constrained-sampling tools, deferred responses, cancellation,
the on_response callback, and the _FauxApiModule adapter layer.
"""

from __future__ import annotations

from pi_ai import (
    Context,
    ImageContent,
    TextContent,
    UserMessage,
)
from pi_ai import (
    SimpleStreamOptions as StreamOptions,
)
from pi_ai.models import complete
from pi_ai.providers.faux import (
    FauxDeferredOptions,
    RegisterFauxProviderOptions,
    _content_to_text,
    _normalize_faux_assistant_content,
    _tool_to_json,
    create_faux_core,
    faux_assistant_message,
    faux_provider,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from pi_ai.types import DeferredHandle, Tool
from pi_ai.utils.event_stream import AssistantMessageEventStream


async def drain(stream: AssistantMessageEventStream):
    return [event async for event in stream]


def make_context(**kw) -> Context:
    defaults: dict = {"messages": [UserMessage(content="hi")]}
    defaults.update(kw)
    return Context(**defaults)


# ---------------------------------------------------------------------------
# line 96: _normalize_faux_assistant_content with single non-str, non-list block
# ---------------------------------------------------------------------------


def test_normalize_single_content_block_returns_list() -> None:
    """Line 96: single FauxContentBlock (not str, not list) → [content]."""
    block = faux_text("hi")
    result = _normalize_faux_assistant_content(block)
    assert result == [block]


def test_normalize_single_thinking_block() -> None:
    block = faux_thinking("thought")
    result = _normalize_faux_assistant_content(block)
    assert result == [block]


def test_normalize_single_tool_call_block() -> None:
    block = faux_tool_call("do_thing", {"a": 1})
    result = _normalize_faux_assistant_content(block)
    assert result == [block]


# ---------------------------------------------------------------------------
# line 197->194: _content_to_text with ImageContent
# ---------------------------------------------------------------------------


def test_content_to_text_with_image_content() -> None:
    """Line 197: ImageContent branch in _content_to_text."""
    image = ImageContent(mime_type="image/png", data=b"\x89PNG\r\n\x1a\n")
    result = _content_to_text([image])
    assert "[image:image/png:8]" in result


def test_content_to_text_mixed_text_and_image() -> None:
    text = TextContent(text="caption")
    image = ImageContent(mime_type="image/jpeg", data=b"\xff\xd8")
    result = _content_to_text([text, image])
    assert "caption" in result
    assert "[image:image/jpeg:2]" in result


# ---------------------------------------------------------------------------
# line 230: _tool_to_json with constrained_sampling
# ---------------------------------------------------------------------------


def test_tool_to_json_includes_constrained_sampling_when_set() -> None:
    """Line 230: tool.constrained_sampling is not None → included in payload."""
    tool = Tool(
        name="json_tool",
        description="Returns JSON",
        parameters={"type": "object"},
        constrained_sampling={"type": "json_schema", "json_schema": {}},
    )
    payload = _tool_to_json(tool)
    assert "constrainedSampling" in payload
    assert payload["constrainedSampling"] == tool.constrained_sampling


def test_tool_to_json_omits_constrained_sampling_when_none() -> None:
    tool = Tool(name="simple", description="desc", parameters={"type": "object"})
    payload = _tool_to_json(tool)
    assert "constrainedSampling" not in payload


# ---------------------------------------------------------------------------
# line 311, 569-592, 617: deferred stream path + stream_simple
# ---------------------------------------------------------------------------


async def test_stream_simple_delegates_to_stream(monkeypatch) -> None:
    """Line 617: core.stream_simple delegates to core.stream."""
    core = create_faux_core()
    core.set_responses([faux_assistant_message("hello")])
    model = core.get_model()
    context = make_context()
    stream = core.stream_simple(model, context)
    result = await complete(stream)
    assert result.content == [faux_text("hello")]
    assert core.state.call_count == 1


async def test_deferred_stream_produces_deferred_message() -> None:
    """Lines 311, 569-592: stream with deferred=True creates a deferred handle."""
    core = create_faux_core(RegisterFauxProviderOptions(deferred=FauxDeferredOptions(pending_fetches=0)))
    core.set_responses([faux_assistant_message("final answer")])
    model = core.get_model()
    context = make_context()

    stream = core.stream(model, context, StreamOptions(deferred=True))
    events = await drain(stream)
    done_events = [e for e in events if getattr(e, "type", None) == "done"]
    assert done_events, "Expected a done event"
    deferred_msg = done_events[0].message
    assert deferred_msg.stop_reason == "deferred"
    assert deferred_msg.deferred is not None


# ---------------------------------------------------------------------------
# lines 553-557: on_response callback in stream
# ---------------------------------------------------------------------------


async def test_stream_calls_sync_on_response_callback() -> None:
    """Lines 553-557: sync on_response callback is invoked."""
    called: list[tuple] = []

    def on_response(response, model):
        called.append((response.status, model.id))

    core = create_faux_core()
    core.set_responses([faux_assistant_message("hi")])
    model = core.get_model()
    context = make_context()

    stream = core.stream(model, context, StreamOptions(on_response=on_response))
    await drain(stream)
    assert called == [(200, model.id)]


async def test_stream_calls_async_on_response_callback() -> None:
    """Lines 553-557: async on_response callback is awaited."""
    called: list[tuple] = []

    async def on_response(response, model):
        called.append((response.status, model.id))

    core = create_faux_core()
    core.set_responses([faux_assistant_message("hi")])
    model = core.get_model()
    context = make_context()

    stream = core.stream(model, context, StreamOptions(on_response=on_response))
    await drain(stream)
    assert called == [(200, model.id)]


# ---------------------------------------------------------------------------
# lines 624-681: fetch_deferred
# ---------------------------------------------------------------------------


async def test_fetch_deferred_resolves_response() -> None:
    """Lines 624-681: fetch_deferred resolves the deferred step and streams it."""
    core = create_faux_core(RegisterFauxProviderOptions(deferred=FauxDeferredOptions(pending_fetches=0)))
    core.set_responses([faux_assistant_message("deferred answer")])
    model = core.get_model()
    context = make_context()

    # Get the deferred handle
    init_stream = core.stream(model, context, StreamOptions(deferred=True))
    init_events = await drain(init_stream)
    done_evt = next(e for e in init_events if getattr(e, "type", None) == "done")
    handle = done_evt.message.deferred
    assert handle is not None

    # Fetch the deferred result
    fetch_stream = core.fetch_deferred(model, handle)
    fetch_events = await drain(fetch_stream)
    done_evt2 = next(e for e in fetch_events if getattr(e, "type", None) == "done")
    assert done_evt2.message.stop_reason == "stop"
    assert done_evt2.message.content == [faux_text("deferred answer")]
    assert core.state.deferred_fetch_count == 1


async def test_fetch_deferred_with_pending_fetches_returns_deferred_again() -> None:
    """fetch_deferred with pending_fetches > 0 returns another deferred message."""
    core = create_faux_core(RegisterFauxProviderOptions(deferred=FauxDeferredOptions(pending_fetches=1)))
    core.set_responses([faux_assistant_message("answer after wait")])
    model = core.get_model()
    context = make_context()

    # Start deferred
    init_stream = core.stream(model, context, StreamOptions(deferred=True))
    init_events = await drain(init_stream)
    done_evt = next(e for e in init_events if getattr(e, "type", None) == "done")
    handle = done_evt.message.deferred

    # First fetch: pending_fetches=1 → still deferred
    fetch1 = core.fetch_deferred(model, handle)
    fetch1_events = await drain(fetch1)
    done1 = next(e for e in fetch1_events if getattr(e, "type", None) == "done")
    assert done1.message.stop_reason == "deferred"

    # Second fetch: pending_fetches=0 → resolves
    fetch2 = core.fetch_deferred(model, handle)
    fetch2_events = await drain(fetch2)
    done2 = next(e for e in fetch2_events if getattr(e, "type", None) == "done")
    assert done2.message.stop_reason == "stop"


async def test_fetch_deferred_raises_for_unknown_handle() -> None:
    """fetch_deferred with an unknown handle produces an error event."""
    core = create_faux_core()
    model = core.get_model()
    bad_handle = DeferredHandle(provider="faux", model_id=model.id, api="unknown-api", id="nonexistent:123:abc")
    fetch_stream = core.fetch_deferred(model, bad_handle)
    events = await drain(fetch_stream)
    error_events = [e for e in events if getattr(e, "type", None) == "error"]
    assert error_events


async def test_fetch_deferred_raises_for_cancelled_handle() -> None:
    """fetch_deferred on a cancelled entry produces an error event."""
    core = create_faux_core(RegisterFauxProviderOptions(deferred=FauxDeferredOptions(pending_fetches=0)))
    core.set_responses([faux_assistant_message("cancelled")])
    model = core.get_model()
    context = make_context()

    init_stream = core.stream(model, context, StreamOptions(deferred=True))
    init_events = await drain(init_stream)
    done_evt = next(e for e in init_events if getattr(e, "type", None) == "done")
    handle = done_evt.message.deferred

    # Cancel it
    await core.cancel_deferred(model, handle)

    # Fetch after cancel → error
    fetch_stream = core.fetch_deferred(model, handle)
    events = await drain(fetch_stream)
    error_events = [e for e in events if getattr(e, "type", None) == "error"]
    assert error_events


async def test_fetch_deferred_with_on_response_callback() -> None:
    """fetch_deferred calls on_response when provided."""
    called: list[int] = []

    def on_response(resp, model):
        called.append(resp.status)

    core = create_faux_core(RegisterFauxProviderOptions(deferred=FauxDeferredOptions(pending_fetches=0)))
    core.set_responses([faux_assistant_message("x")])
    model = core.get_model()
    context = make_context()

    init_stream = core.stream(model, context, StreamOptions(deferred=True))
    init_events = await drain(init_stream)
    done_evt = next(e for e in init_events if getattr(e, "type", None) == "done")
    handle = done_evt.message.deferred

    from pi_ai.types import StreamOptions as FullStreamOptions

    fetch_stream = core.fetch_deferred(model, handle, FullStreamOptions(on_response=on_response))
    await drain(fetch_stream)
    assert called == [200]


# ---------------------------------------------------------------------------
# lines 688-697: cancel_deferred
# ---------------------------------------------------------------------------


async def test_cancel_deferred_marks_entry_cancelled_and_records_handle() -> None:
    """Lines 688-697: cancel_deferred sets entry.cancelled and appends to state."""
    core = create_faux_core(RegisterFauxProviderOptions(deferred=FauxDeferredOptions(pending_fetches=0)))
    core.set_responses([faux_assistant_message("x")])
    model = core.get_model()
    context = make_context()

    init_stream = core.stream(model, context, StreamOptions(deferred=True))
    init_events = await drain(init_stream)
    done_evt = next(e for e in init_events if getattr(e, "type", None) == "done")
    handle = done_evt.message.deferred

    assert core.state.cancelled_deferred == []
    await core.cancel_deferred(model, handle)
    assert len(core.state.cancelled_deferred) == 1
    assert core.state.cancelled_deferred[0].id == handle.id


async def test_cancel_deferred_calls_on_response_callback() -> None:
    """cancel_deferred invokes the on_response callback from cancel_options."""
    called: list[int] = []

    def on_response(resp, model):
        called.append(resp.status)

    core = create_faux_core(RegisterFauxProviderOptions(deferred=FauxDeferredOptions(pending_fetches=0)))
    core.set_responses([faux_assistant_message("x")])
    model = core.get_model()
    context = make_context()

    init_stream = core.stream(model, context, StreamOptions(deferred=True))
    init_events = await drain(init_stream)
    done_evt = next(e for e in init_events if getattr(e, "type", None) == "done")
    handle = done_evt.message.deferred

    from pi_ai.types import StreamOptions as FullStreamOptions

    await core.cancel_deferred(model, handle, FullStreamOptions(on_response=on_response))
    assert called == [200]


# ---------------------------------------------------------------------------
# line 748: _FauxAuthResolve.__call__
# ---------------------------------------------------------------------------


def test_faux_auth_resolve_returns_auth_result() -> None:
    """Line 748: _FauxAuthResolve().__call__() returns AuthResult."""
    import pi_ai.providers.faux as faux_mod
    from pi_ai.auth.types import AuthResult

    resolver = faux_mod._FauxAuthResolve()
    result = resolver()
    assert isinstance(result, AuthResult)
    assert result.source == "faux"


# ---------------------------------------------------------------------------
# lines 788, 793, 796: _FauxApiModule adapter methods
# ---------------------------------------------------------------------------


async def test_faux_api_module_stream_simple_delegates() -> None:
    """Line 788: _FauxApiModule.stream_simple delegates to core.stream_simple."""
    handle = faux_provider()
    handle.set_responses([faux_assistant_message("via stream_simple")])
    model = handle.get_model()
    context = make_context()

    # Calling provider.stream_simple goes through _FauxApiModule.stream_simple (line 788)
    stream = handle.provider.stream_simple(model, context)
    result = await complete(stream)
    assert result.content == [faux_text("via stream_simple")]


async def test_faux_api_module_fetch_deferred_delegates() -> None:
    """Line 793: _FauxApiModule.fetch_deferred delegates to core.fetch_deferred."""
    handle = faux_provider(RegisterFauxProviderOptions(deferred=FauxDeferredOptions(pending_fetches=0)))
    handle.set_responses([faux_assistant_message("fetched")])
    model = handle.get_model()
    context = make_context()

    init_stream = handle.provider.stream(model, context, StreamOptions(deferred=True))
    init_events = await drain(init_stream)
    done_evt = next(e for e in init_events if getattr(e, "type", None) == "done")
    deferred_handle = done_evt.message.deferred

    # _FauxApiModule.fetch_deferred (line 793)
    fetch_stream = handle.provider.api.fetch_deferred(model, deferred_handle)
    events = await drain(fetch_stream)
    done2 = next(e for e in events if getattr(e, "type", None) == "done")
    assert done2.message.content == [faux_text("fetched")]


async def test_faux_api_module_cancel_deferred_delegates() -> None:
    """Line 796: _FauxApiModule.cancel_deferred delegates to core.cancel_deferred."""
    handle = faux_provider(RegisterFauxProviderOptions(deferred=FauxDeferredOptions(pending_fetches=0)))
    handle.set_responses([faux_assistant_message("x")])
    model = handle.get_model()
    context = make_context()

    init_stream = handle.provider.stream(model, context, StreamOptions(deferred=True))
    init_events = await drain(init_stream)
    done_evt = next(e for e in init_events if getattr(e, "type", None) == "done")
    deferred_handle = done_evt.message.deferred

    # _FauxApiModule.cancel_deferred (line 796)
    await handle.provider.api.cancel_deferred(model, deferred_handle)
    assert handle.state.cancelled_deferred[0].id == deferred_handle.id
