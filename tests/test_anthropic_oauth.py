"""Python port of `packages/ai/test/anthropic-oauth.test.ts`.

TypeScript stubs the global `fetch`; this port injects an
`httpx.MockTransport`-backed client into `login_anthropic`/`refresh`, which is
the same seam (both go through the module's single `_post_json` helper).
"""

from __future__ import annotations

import asyncio
import errno
import inspect
import json
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

import httpx
from pi_ai.auth.oauth.anthropic import CALLBACK_PORT, build_anthropic_oauth
from pi_ai.auth.types import AuthEvent, AuthInteraction, AuthPrompt, Credential
from pi_ai.utils.abort import AbortSignal

# TypeScript drives the wired OAuth object (`anthropicOAuth.login(...)`), not the bare
# module function. Going through `build_anthropic_oauth()` means a regression that rewires the
# flow -- pointing `login` at the wrong function, or at one that is not a coroutine
# function -- fails here instead of only in production. Calling `login_x(...)`
# directly would keep passing through such a break.
ANTHROPIC_OAUTH = build_anthropic_oauth()


def test_the_real_oauth_object_wires_coroutine_functions() -> None:
    """Guards the shape the CLI depends on: `provider.auth.oauth.<hook>` must be awaitable."""
    for hook in ("login", "refresh", "to_auth"):
        assert inspect.iscoroutinefunction(getattr(ANTHROPIC_OAUTH, hook)), hook


NEVER_ABORTED = AbortSignal()


@dataclass
class RecordedRequest:
    url: str
    method: str
    body: dict[str, str]


def make_client(requests: list[RecordedRequest], payload: dict[str, object]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(RecordedRequest(url=str(request.url), method=request.method, body=json.loads(request.content)))
        return httpx.Response(200, json=payload, headers={"Content-Type": "application/json"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@dataclass
class RecordingInteraction(AuthInteraction):
    signal: AbortSignal = field(default_factory=lambda: NEVER_ABORTED)
    events: list[AuthEvent] = field(default_factory=list)
    prompts: list[AuthPrompt] = field(default_factory=list)
    answer: str = ""
    answer_from_auth_url: bool = False

    async def prompt(self, prompt: AuthPrompt) -> str:
        self.prompts.append(prompt)
        if prompt.type != "manual_code":
            raise AssertionError(f"Unexpected prompt: {prompt.type}")
        if not self.answer_from_auth_url:
            return self.answer
        auth_url = next(event.url for event in self.events if event.type == "auth_url")
        params = parse_qs(urlparse(auth_url).query)
        state = params["state"][0]
        redirect_uri = params["redirect_uri"][0]
        return f"{redirect_uri}?code=manual-code&state={state}"

    def notify(self, event: AuthEvent) -> None:
        self.events.append(event)


async def _login_on_fixed_callback_port() -> tuple[Credential, list[RecordedRequest]]:
    """Run the default-port login, retrying only a busy-port bind.

    This is the one test that must let `login` bind the real `CALLBACK_PORT`,
    because the behavior under test is precisely that the default flow uses
    that fixed port in its `redirect_uri`. Passing `callback_port=0` would
    bind an ephemeral port and make the port assertion below tautological.

    Binding a fixed port is not hermetic: a second suite running concurrently
    (or a stray earlier run) holds 53692 and this fails with `EADDRINUSE`.
    TypeScript has the identical hazard -- `server.on("error", reject)` at
    `anthropic.ts:153` -- it just never runs two copies at once. Only the
    address collision is retried; every other error propagates.
    """
    last: OSError | None = None
    for attempt in range(10):
        requests: list[RecordedRequest] = []
        client = make_client(
            requests,
            {"access_token": "access-token", "refresh_token": "refresh-token", "expires_in": 3600},
        )
        interaction = RecordingInteraction(answer_from_auth_url=True)
        try:
            return await ANTHROPIC_OAUTH.login(interaction, client=client), requests
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
            last = error
            await asyncio.sleep(0.05 * (attempt + 1))
    raise AssertionError(f"port {CALLBACK_PORT} stayed busy across retries") from last


async def test_keeps_the_localhost_redirect_uri_for_manual_callback_login():
    credentials, requests = await _login_on_fixed_callback_port()

    assert credentials.access == "access-token"
    assert credentials.refresh == "refresh-token"
    assert len(requests) == 1
    assert requests[0].url == "https://platform.claude.com/v1/oauth/token"
    assert requests[0].method == "POST"
    assert requests[0].body["grant_type"] == "authorization_code"
    assert requests[0].body["code"] == "manual-code"
    assert requests[0].body["redirect_uri"] == f"http://localhost:{CALLBACK_PORT}/callback"


async def test_omits_scope_from_refresh_token_requests():
    requests: list[RecordedRequest] = []
    client = make_client(
        requests,
        {"access_token": "new-access-token", "refresh_token": "new-refresh-token", "expires_in": 3600},
    )

    credentials = await ANTHROPIC_OAUTH.refresh(
        Credential(type="oauth", access="old-access-token", refresh="refresh-token", expires=0),
        NEVER_ABORTED,
        client=client,
    )

    assert credentials.access == "new-access-token"
    assert credentials.refresh == "new-refresh-token"
    assert len(requests) == 1
    assert requests[0].url == "https://platform.claude.com/v1/oauth/token"
    assert requests[0].method == "POST"
    assert requests[0].body["grant_type"] == "refresh_token"
    assert requests[0].body["client_id"]
    assert requests[0].body["refresh_token"] == "refresh-token"
    assert "scope" not in requests[0].body


async def test_login_resolves_through_the_manual_code_prompt_and_aborts_it_after_settling():
    requests: list[RecordedRequest] = []
    client = make_client(requests, {"access_token": "access", "refresh_token": "refresh", "expires_in": 3600})
    interaction = RecordingInteraction(answer="the-code")

    credential = await ANTHROPIC_OAUTH.login(interaction, client=client, callback_port=0)

    assert credential.type == "oauth"
    assert credential.access == "access"
    assert any(event.type == "auth_url" for event in interaction.events)
    manual_prompts = [prompt for prompt in interaction.prompts if prompt.type == "manual_code"]
    assert manual_prompts
    # the prompt's signal is aborted once login settles, so UIs can dismiss it
    assert manual_prompts[0].signal is not None
    assert manual_prompts[0].signal.aborted
