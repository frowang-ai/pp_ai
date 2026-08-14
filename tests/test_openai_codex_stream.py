"""Python port of `packages/ai/test/openai-codex-stream.test.ts` -- not portable.

The 2,622-line TypeScript file tests `src/apis/openai-codex-responses.ts`: the
ChatGPT backend's private Codex protocol. Its 24 `it(...)` cases, in file
order, are:

1. streams SSE responses into AssistantMessageEventStream
2. completes after response.completed even when the SSE body stays open
3. maps response.incomplete to stopReason length even when the SSE body stays
   open
4. aborts SSE fetch after the configured HTTP timeout when response headers do
   not arrive
5. aborts SSE body reads after response headers arrive
6. sets session-id/x-client-request-id headers and prompt_cache_key when
   sessionId is provided
7. omits SSE cache affinity when cacheRetention is none
8. clamps prompt_cache_key to OpenAI's 64-character limit
9. clamps Codex session-id header to 64 characters
10. preserves gpt-5.5 xhigh reasoning effort from simple options
11. forwards required tool choice
12. sets Codex strict mode explicitly and honors constrained sampling
13. does not set session-id/x-client-request-id headers when sessionId is not
    provided
14. forwards auto transport from streamSimple options and uses cached
    websocket context
15. scopes cached websockets to the authenticated account
16. closes one-shot websockets when cacheRetention is none
17. falls back to SSE when websocket connect does not open before the connect
    timeout
18. reconnects once when the websocket connection limit is reached before
    output starts
19. falls back to SSE when a websocket is idle before the first event
20. errors when a websocket is idle after the stream started
21. opens a fresh cached websocket before the backend connection age limit
22. sends only response input deltas in websocket-cached mode
23. zstd-compresses SSE request bodies
24. uses exponential backoff across repeated SSE retries without retry headers

`pi_ai.api.openai_codex_responses` is a **documented placeholder**: its module
docstring records the omission and both entry points raise
`NotImplementedError`. There is no Python code implementing any of the
behaviour above, so every one of the 24 cases would have to invent its subject.

What is asserted here instead is the contract of the omission -- the models
remain discoverable, and both entry points fail loudly with the documented
message rather than silently falling back to a different API.
"""

from __future__ import annotations

import pytest
from pi_ai.api import openai_codex_responses
from pi_ai.providers.all import get_builtin_model, get_builtin_models
from pi_ai.types import Context, SimpleStreamOptions, StreamOptions, UserMessage, now_ms


def test_openai_codex_models_are_registered_under_the_unported_api() -> None:
    models = get_builtin_models("openai-codex")
    assert models
    for model in models:
        assert model.api == "openai-codex-responses"
        assert get_builtin_model("openai-codex", model.id) is not None


@pytest.mark.parametrize("options", [None, StreamOptions(api_key="test"), SimpleStreamOptions(api_key="test")])
def test_stream_entry_points_raise_not_implemented(options: StreamOptions | None) -> None:
    model = get_builtin_models("openai-codex")[0]
    context = Context(system_prompt="sys", messages=[UserMessage(content="hi", timestamp=now_ms())])

    with pytest.raises(NotImplementedError) as stream_error:
        openai_codex_responses.stream(model, context, options)
    with pytest.raises(NotImplementedError) as simple_error:
        openai_codex_responses.stream_simple(model, context, options)

    for error in (stream_error, simple_error):
        assert "not ported to Python" in str(error.value)
        assert "listed for discovery only" in str(error.value)
