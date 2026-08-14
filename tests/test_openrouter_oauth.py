"""Python port of `packages/ai/test/openrouter-oauth.test.ts`.

TypeScript stubs the global `fetch` and drives the real loopback callback
server with `nativeFetch`. This port injects an `httpx.AsyncClient` backed by
`httpx.MockTransport` for the token exchange (the flow accepts one for exactly
this reason) and uses a *real* loopback `httpx.AsyncClient` for the callback
requests, so the one-shot server semantics (200 / 400 / 409 / 502) are
exercised end to end without a network call leaving the machine.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import re
from collections.abc import Awaitable, Callable
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pi_ai.auth.oauth.openrouter import build_openrouter_oauth, login_openrouter
from pi_ai.auth.types import AuthEvent, AuthInteraction, AuthPrompt, Credential, InMemoryCredentialStore
from pi_ai.images_registry import create_images_models
from pi_ai.providers.openrouter import openrouter_provider
from pi_ai.providers.openrouter_images import openrouter_images_provider
from pi_ai.registry import Models
from pi_ai.utils.abort import AbortController, AbortSignal

TOKEN_URL = "https://openrouter.ai/api/v1/auth/keys"
LOGIN_TIMEOUT_S = 5.0


def base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


class RecordingInteraction(AuthInteraction):
    """`AuthInteraction` whose prompt behaviour is supplied per test."""

    def __init__(
        self,
        prompt_impl: Callable[[AuthPrompt], Awaitable[str]] | None = None,
        signal: AbortSignal | None = None,
        on_auth_url: Callable[[str], None] | None = None,
    ) -> None:
        self.signal = signal or AbortController().signal
        self.events: list[AuthEvent] = []
        self.prompts: list[AuthPrompt] = []
        self.auth_url_seen = asyncio.Event()
        self._prompt_impl = prompt_impl
        self._on_auth_url = on_auth_url

    async def prompt(self, prompt: AuthPrompt) -> str:
        self.prompts.append(prompt)
        if self._prompt_impl is None:
            await asyncio.Event().wait()  # never resolves; the flow cancels this task
            raise AssertionError("unreachable")
        return await self._prompt_impl(prompt)

    def notify(self, event: AuthEvent) -> None:
        self.events.append(event)
        if event.type == "auth_url" and event.url:
            self.auth_url_seen.set()
            if self._on_auth_url is not None:
                self._on_auth_url(event.url)

    def auth_url(self) -> str:
        return next(event.url for event in self.events if event.type == "auth_url" and event.url)

    async def wait_for_auth_url(self) -> str:
        """Block until the flow has bound its loopback server and announced the URL.

        The callback server binds and starts its thread before `notify` is
        called, so this is the exact "ready to receive a callback" edge. A
        wall-clock sleep would be a guess, and guesses race under parallel test
        load.
        """
        await asyncio.wait_for(self.auth_url_seen.wait(), timeout=LOGIN_TIMEOUT_S)
        return self.auth_url()


def callback_url_of(authorize_url: str) -> str:
    params = parse_qs(urlparse(authorize_url).query)
    return params["callback_url"][0]


async def get_loopback(url: str, **params: str) -> httpx.Response:
    async with httpx.AsyncClient(timeout=LOGIN_TIMEOUT_S) as client:
        return await client.get(url, params=params)


class ExchangeRecorder:
    """Mock transport for the token endpoint, recording every exchange."""

    def __init__(self, respond: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond

    def client(self) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == TOKEN_URL
            self.requests.append(request)
            return self._respond(request)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def body(self, index: int = 0) -> dict[str, object]:
        return json.loads(self.requests[index].content)


def json_response(body: object, status: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _request: httpx.Response(status, json=body)


async def test_is_exposed_by_both_openrouter_providers_alongside_api_key_auth() -> None:
    for provider in (openrouter_provider(), openrouter_images_provider()):
        assert provider.auth.api_key is not None
        assert provider.auth.oauth is not None
        assert provider.auth.oauth.login_label == "Sign in with OpenRouter"


async def test_resolves_the_same_stored_oauth_key_for_text_and_image_providers() -> None:
    credentials = InMemoryCredentialStore()
    await credentials.set(
        "openrouter",
        Credential(type="oauth", access="sk-or-stored", refresh="", expires=math.inf),
    )

    text_models = Models(credential_store=credentials)
    text_models.add(openrouter_provider())
    image_models = create_images_models(credential_store=credentials)
    image_models.add(openrouter_images_provider())

    text_auth = await text_models.get_auth("openrouter")
    image_auth = await image_models.get_auth("openrouter")
    assert text_auth is not None
    assert image_auth is not None
    assert text_auth.auth.api_key == "sk-or-stored"
    assert image_auth.auth.api_key == "sk-or-stored"


async def test_runs_pkce_on_a_one_shot_loopback_callback_and_exchanges_the_code() -> None:
    exchange = ExchangeRecorder(json_response({"key": "sk-or-test"}))
    callback_response: list[httpx.Response] = []
    manual_prompts: list[AuthPrompt] = []

    async def hanging_prompt(prompt: AuthPrompt) -> str:
        manual_prompts.append(prompt)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    interaction = RecordingInteraction(prompt_impl=hanging_prompt)
    login = asyncio.ensure_future(login_openrouter(interaction, client=exchange.client()))
    await interaction.wait_for_auth_url()

    authorize_url = interaction.auth_url()
    callback = callback_url_of(authorize_url)
    callback_response.append(await get_loopback(callback, code="authorization-code"))

    credential = await asyncio.wait_for(login, timeout=LOGIN_TIMEOUT_S)
    assert credential == Credential(type="oauth", access="sk-or-test", refresh="", expires=math.inf)
    assert callback_response[0].status_code == 200
    assert manual_prompts[0].signal is not None
    assert manual_prompts[0].signal.aborted is True

    parsed = urlparse(authorize_url)
    assert f"{parsed.scheme}://{parsed.netloc}" == "https://openrouter.ai"
    assert parsed.path == "/auth"
    params = parse_qs(parsed.query)
    assert params["code_challenge_method"][0] == "S256"

    parsed_callback = urlparse(callback)
    assert parsed_callback.hostname == "127.0.0.1"
    assert re.fullmatch(r"/oauth/callback/[0-9a-f-]+", parsed_callback.path)
    assert len(parsed_callback.path.removeprefix("/oauth/callback/")) == 36

    body = exchange.body()
    assert body["code"] == "authorization-code"
    assert body["code_challenge_method"] == "S256"
    verifier = body["code_verifier"]
    assert isinstance(verifier, str)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    assert params["code_challenge"][0] == base64url(digest)
    assert len(exchange.requests) == 1


async def test_reports_token_exchange_failures_through_the_callback_page_and_login() -> None:
    exchange = ExchangeRecorder(json_response({"error": {"message": "invalid code"}}, 403))
    interaction = RecordingInteraction()
    login = asyncio.ensure_future(login_openrouter(interaction, client=exchange.client()))
    await interaction.wait_for_auth_url()

    response = await get_loopback(callback_url_of(interaction.auth_url()), code="bad-code")

    with pytest.raises(RuntimeError, match=r"OpenRouter OAuth key exchange failed \(HTTP 403\): invalid code"):
        await asyncio.wait_for(login, timeout=LOGIN_TIMEOUT_S)
    assert response.status_code == 502


async def test_allows_only_one_token_exchange_for_a_callback() -> None:
    release = asyncio.Event()
    started = asyncio.Event()

    async def blocking(_request: httpx.Request) -> httpx.Response:
        started.set()
        await release.wait()
        return httpx.Response(200, json={"key": "sk-or-test"})

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return await blocking(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    interaction = RecordingInteraction()
    login = asyncio.ensure_future(login_openrouter(interaction, client=client))
    await interaction.wait_for_auth_url()

    callback = callback_url_of(interaction.auth_url())
    first = asyncio.ensure_future(get_loopback(callback, code="authorization-code"))
    await asyncio.wait_for(started.wait(), timeout=LOGIN_TIMEOUT_S)

    second = await get_loopback(callback, code="authorization-code")
    assert second.status_code == 409
    assert len(requests) == 1

    release.set()
    credential = await asyncio.wait_for(login, timeout=LOGIN_TIMEOUT_S)
    assert credential.access == "sk-or-test"
    assert (await asyncio.wait_for(first, timeout=LOGIN_TIMEOUT_S)).status_code == 200


async def test_rejects_a_successful_response_that_does_not_contain_a_key() -> None:
    exchange = ExchangeRecorder(json_response({"user_id": "user-1"}))
    interaction = RecordingInteraction()
    login = asyncio.ensure_future(login_openrouter(interaction, client=exchange.client()))
    await interaction.wait_for_auth_url()

    response = await get_loopback(callback_url_of(interaction.auth_url()), code="code-without-key")

    with pytest.raises(RuntimeError, match=r'OpenRouter OAuth response carries no "key"'):
        await asyncio.wait_for(login, timeout=LOGIN_TIMEOUT_S)
    assert response.status_code == 502


async def test_mints_a_key_from_a_pasted_redirect_url_when_the_callback_never_arrives() -> None:
    exchange = ExchangeRecorder(json_response({"key": "sk-or-manual"}))
    callback_holder: list[str] = []

    async def paste_redirect_url(prompt: AuthPrompt) -> str:
        assert prompt.type == "manual_code"
        return f"{callback_holder[0]}?code=manual-code"

    interaction = RecordingInteraction(
        prompt_impl=paste_redirect_url,
        on_auth_url=lambda url: callback_holder.append(callback_url_of(url)),
    )
    credential = await asyncio.wait_for(
        login_openrouter(interaction, client=exchange.client()), timeout=LOGIN_TIMEOUT_S
    )

    assert credential == Credential(type="oauth", access="sk-or-manual", refresh="", expires=math.inf)
    body = exchange.body()
    assert body["code"] == "manual-code"
    assert body["code_challenge_method"] == "S256"
    assert len(exchange.requests) == 1


async def test_accepts_a_bare_authorization_code_from_the_manual_prompt() -> None:
    exchange = ExchangeRecorder(json_response({"key": "sk-or-manual"}))

    async def paste_bare_code(_prompt: AuthPrompt) -> str:
        return "  manual-code  "

    interaction = RecordingInteraction(prompt_impl=paste_bare_code)
    credential = await asyncio.wait_for(
        login_openrouter(interaction, client=exchange.client()), timeout=LOGIN_TIMEOUT_S
    )

    assert credential.access == "sk-or-manual"
    assert exchange.body()["code"] == "manual-code"


async def test_fails_login_when_the_manual_prompt_is_cancelled() -> None:
    exchange = ExchangeRecorder(json_response({"key": "sk-or-unexpected"}))

    async def cancel(_prompt: AuthPrompt) -> str:
        raise RuntimeError("Login cancelled")

    interaction = RecordingInteraction(prompt_impl=cancel)
    with pytest.raises(RuntimeError, match=r"Login cancelled"):
        await asyncio.wait_for(login_openrouter(interaction, client=exchange.client()), timeout=LOGIN_TIMEOUT_S)
    assert exchange.requests == []


async def test_rejects_empty_manual_input_without_exchanging_a_code() -> None:
    exchange = ExchangeRecorder(json_response({"key": "sk-or-unexpected"}))

    async def blank(_prompt: AuthPrompt) -> str:
        return "   "

    interaction = RecordingInteraction(prompt_impl=blank)
    with pytest.raises(RuntimeError, match=r"Missing authorization code"):
        await asyncio.wait_for(login_openrouter(interaction, client=exchange.client()), timeout=LOGIN_TIMEOUT_S)
    assert exchange.requests == []


async def test_closes_the_pending_callback_when_login_is_cancelled() -> None:
    controller = AbortController()
    exchange = ExchangeRecorder(json_response({"key": "sk-or-unexpected"}))
    callback_holder: list[str] = []

    def on_auth_url(url: str) -> None:
        callback_holder.append(callback_url_of(url))
        controller.abort()

    interaction = RecordingInteraction(signal=controller.signal, on_auth_url=on_auth_url)
    with pytest.raises(RuntimeError, match=r"Login cancelled"):
        await asyncio.wait_for(login_openrouter(interaction, client=exchange.client()), timeout=LOGIN_TIMEOUT_S)

    assert callback_holder
    with pytest.raises(httpx.HTTPError):
        await get_loopback(callback_holder[0])


async def test_rejects_before_opening_a_callback_server_when_login_is_already_cancelled() -> None:
    controller = AbortController()
    controller.abort()

    async def blank(_prompt: AuthPrompt) -> str:
        return ""

    class FailingNotify(RecordingInteraction):
        def notify(self, event: AuthEvent) -> None:
            raise AssertionError("Cancelled login must not emit events")

    interaction = FailingNotify(prompt_impl=blank, signal=controller.signal)
    with pytest.raises(RuntimeError, match=r"Login cancelled"):
        await asyncio.wait_for(login_openrouter(interaction), timeout=LOGIN_TIMEOUT_S)


async def test_uses_the_configured_oauth_callback_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_OAUTH_CALLBACK_HOST", "localhost")
    controller = AbortController()
    callback_holder: list[str] = []

    def on_auth_url(url: str) -> None:
        callback_holder.append(callback_url_of(url))
        controller.abort()

    interaction = RecordingInteraction(signal=controller.signal, on_auth_url=on_auth_url)
    with pytest.raises(RuntimeError, match=r"Login cancelled"):
        await asyncio.wait_for(login_openrouter(interaction), timeout=LOGIN_TIMEOUT_S)

    assert urlparse(callback_holder[0]).hostname == "localhost"


def test_build_openrouter_oauth_exposes_the_login_label() -> None:
    built = build_openrouter_oauth()
    assert built.name == "OpenRouter OAuth"
    assert built.login_label == "Sign in with OpenRouter"
