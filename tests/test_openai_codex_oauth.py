"""Python port of `packages/ai/test/openai-codex-oauth.test.ts` -- not portable.

The TypeScript file tests `src/auth/oauth/openai-codex.ts`, the ChatGPT/Codex
device-code and browser login flow (login-method selection, 15-minute device
flow timeout, treating 403/404 device-auth responses as pending, and JWT
`chatgpt_account_id` extraction). That module has **no Python counterpart**:
`pi_ai.auth.oauth` ships anthropic, github_copilot, kimi_coding, openrouter,
radius and xai only, and the API it authenticates
(`pi_ai.api.openai_codex_responses`) is a documented placeholder that raises
`NotImplementedError`.

Porting the 8 test cases ("logs in with the OpenAI Codex device code flow",
"offers browser login first and uses the selected OpenAI Codex device code
flow", "cancels when OpenAI Codex login method selection is cancelled",
"cancels the OpenAI Codex device code flow while waiting", "times out the
OpenAI Codex device code flow after 15 minutes", "treats OpenAI Codex device
auth 403 and 404 responses as pending", "includes the response body in OpenAI
Codex device auth poll failures", and "does not write token refresh failures to
stderr") would require first writing the module under test, which is outside
the scope of a verification port, and every assertion would pin behaviour of
code that does not exist. What is asserted here instead is the *contract of the
omission*: `openai-codex` models stay discoverable through the catalog, the
provider exposes no OAuth login, and any attempt to stream fails loudly rather
than silently doing the wrong thing.
"""

from __future__ import annotations

import pytest
from pi_ai.api import openai_codex_responses
from pi_ai.auth import oauth
from pi_ai.providers.all import get_builtin_models
from pi_ai.providers.openai_codex import openai_codex_provider
from pi_ai.types import Context, UserMessage, now_ms


def test_no_openai_codex_oauth_module_exists() -> None:
    assert not hasattr(oauth, "openai_codex")


def test_openai_codex_provider_offers_no_oauth_login() -> None:
    provider = openai_codex_provider()
    assert provider.auth.oauth is None


def test_openai_codex_models_stay_discoverable() -> None:
    models = get_builtin_models("openai-codex")
    assert models
    assert all(model.api == "openai-codex-responses" for model in models)


def test_streaming_an_openai_codex_model_raises_not_implemented() -> None:
    model = get_builtin_models("openai-codex")[0]
    context = Context(messages=[UserMessage(content="hi", timestamp=now_ms())])

    with pytest.raises(NotImplementedError, match=r"openai-codex-responses API is not ported"):
        openai_codex_responses.stream(model, context)
    with pytest.raises(NotImplementedError, match=r"openai-codex-responses API is not ported"):
        openai_codex_responses.stream_simple(model, context)
