"""Python port of `packages/ai/test/kimi-coding-oauth.test.ts`.

Named with a `_ported` suffix because `tests/test_kimi_coding_oauth.py` already
exists in this repo; this file is the port of the TypeScript test of the same
name.

TypeScript drives the poll timing with `vi.useFakeTimers()`. asyncio has no
equivalent, so every case passes a virtual
:class:`~pi_ai.auth.oauth.device_code.DeviceCodeClock` whose `sleep` advances a
counter instead of really sleeping and whose `monotonic` reads that counter --
the flow's own injection seam, which covers both the device-code polls and the
refresh retry backoff. Every assertion about *when* a request happens is
preserved; only the wall-clock wait is elided.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import time
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from pi_ai.auth.oauth.device_code import DeviceCodeClock
from pi_ai.auth.oauth.kimi_coding import build_kimi_coding_oauth
from pi_ai.auth.types import AuthEvent, AuthInteraction, AuthPrompt, Credential, ResolvedAuth
from pi_ai.utils.abort import AbortController

# TypeScript drives the wired OAuth object (`kimiCodingOAuth.login(...)`), not the bare
# module function. Going through `build_kimi_coding_oauth()` means a regression that rewires the
# flow -- pointing `login` at the wrong function, or at one that is not a coroutine
# function -- fails here instead of only in production. Calling `login_x(...)`
# directly would keep passing through such a break.
KIMI_CODING_OAUTH = build_kimi_coding_oauth()


def test_the_real_oauth_object_wires_coroutine_functions() -> None:
    """Guards the shape the CLI depends on: `provider.auth.oauth.<hook>` must be awaitable."""
    for hook in ("login", "refresh", "to_auth"):
        assert inspect.iscoroutinefunction(getattr(KIMI_CODING_OAUTH, hook)), hook


CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
OAUTH_HOST = "https://auth.kimi.com"


def json_response(body: object, status: int = 200) -> httpx.Response:
    return httpx.Response(status, headers={"Content-Type": "application/json"}, text=json.dumps(body))


def device_authorization_response(**overrides: Any) -> httpx.Response:
    body: dict[str, Any] = {
        "user_code": "ABCD-1234",
        "device_code": "device-code-123",
        "verification_uri": "https://www.kimi.com/code",
        "verification_uri_complete": "https://www.kimi.com/code?user_code=ABCD-1234",
        "interval": 5,
        "expires_in": 600,
    }
    body.update(overrides)
    return json_response(body)


class RecordingInteraction(AuthInteraction):
    def __init__(self, events: list[AuthEvent]) -> None:
        self.signal = AbortController().signal
        self.events = events

    async def prompt(self, prompt: AuthPrompt) -> str:
        raise AssertionError("Kimi Code login should not prompt")

    def notify(self, event: AuthEvent) -> None:
        self.events.append(event)


class VirtualClock:
    """A `DeviceCodeClock` that advances only when the flow sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def as_device_code_clock(self) -> DeviceCodeClock:
        return DeviceCodeClock(monotonic=self.monotonic, sleep=self.sleep)


@pytest.fixture
def virtual_clock() -> VirtualClock:
    return VirtualClock()


@pytest.fixture(autouse=True)
def clear_oauth_host_env() -> Iterator[None]:
    saved = {name: os.environ.pop(name, None) for name in ("KIMI_CODE_OAUTH_HOST", "KIMI_OAUTH_HOST")}
    try:
        yield
    finally:
        for name, value in saved.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value


def form(request: httpx.Request) -> dict[str, str]:
    return {key: values[0] for key, values in parse_qs(request.content.decode()).items()}


async def test_logs_in_with_the_device_authorization_flow(virtual_clock: VirtualClock):
    events: list[AuthEvent] = []
    poll_times: list[float] = []
    device_requests: list[httpx.Request] = []
    token_requests: list[httpx.Request] = []
    poll_responses = [
        json_response({"error": "authorization_pending"}, 400),
        json_response({"access_token": "access-token", "refresh_token": "refresh-token", "expires_in": 3600}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{OAUTH_HOST}/api/oauth/device_authorization":
            device_requests.append(request)
            return device_authorization_response()
        if url == f"{OAUTH_HOST}/api/oauth/token":
            poll_times.append(virtual_clock.now)
            token_requests.append(request)
            if not poll_responses:
                raise AssertionError("Unexpected extra token poll")
            return poll_responses.pop(0)
        raise AssertionError(f"Unexpected request URL: {url}")

    before_ms = time.time() * 1000
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        credential = await KIMI_CODING_OAUTH.login(
            RecordingInteraction(events), client=client, clock=virtual_clock.as_device_code_clock()
        )

    device_request = device_requests[0]
    assert device_request.method == "POST"
    assert device_request.headers["content-type"] == "application/x-www-form-urlencoded"
    assert device_request.headers["accept"] == "application/json"
    assert form(device_request)["client_id"] == CLIENT_ID

    token_form = form(token_requests[0])
    assert token_form["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
    assert token_form["client_id"] == CLIENT_ID
    assert token_form["device_code"] == "device-code-123"

    # TS: expect(events).toEqual([{ type, userCode, verificationUri, intervalSeconds,
    # expiresInSeconds }]) is a full deep-equality check (every AuthEvent field, not just
    # the ones named), so compare the whole dataclass rather than picking fields.
    assert events == [
        AuthEvent(
            type="device_code",
            user_code="ABCD-1234",
            verification_uri="https://www.kimi.com/code?user_code=ABCD-1234",
            interval_seconds=5,
            expires_in_seconds=600,
        )
    ]

    # wait_before_first_poll: the first poll happens after the 5s interval.
    assert poll_times == [5.0, 10.0]

    # TS: expect(credentialPromise).resolves.toEqual({ type, access, refresh, expires:
    # <exact ms> }) pins an exact `expires` value because vi.useFakeTimers() pins
    # Date.now(). This port computes `expires` from the real wall clock
    # (time.time() in kimi_coding.py), which the virtual poll clock does not control,
    # so an exact value can't be reproduced here; a tight wall-clock bound is the
    # closest faithful substitute. Every other field is still compared for full
    # equality, matching toEqual's strictness against unexpected extra fields.
    assert dataclasses.replace(credential, expires=None) == Credential(
        type="oauth", access="access-token", refresh="refresh-token", expires=None
    )
    assert credential.expires is not None
    assert before_ms + 3600 * 1000 <= credential.expires <= time.time() * 1000 + 3600 * 1000


async def test_fails_when_the_device_code_expires(virtual_clock: VirtualClock):
    async with _terminal_error_client("expired_token") as client:
        with pytest.raises(Exception, match="expired"):
            await KIMI_CODING_OAUTH.login(
                RecordingInteraction([]), client=client, clock=virtual_clock.as_device_code_clock()
            )


async def test_fails_when_the_user_denies_the_login(virtual_clock: VirtualClock):
    async with _terminal_error_client("access_denied") as client:
        with pytest.raises(Exception, match="denied"):
            await KIMI_CODING_OAUTH.login(
                RecordingInteraction([]), client=client, clock=virtual_clock.as_device_code_clock()
            )


def _terminal_error_client(error: str) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{OAUTH_HOST}/api/oauth/device_authorization":
            return device_authorization_response()
        if url == f"{OAUTH_HOST}/api/oauth/token":
            return json_response({"error": error}, 400)
        raise AssertionError(f"Unexpected request URL: {url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_honors_the_kimi_code_oauth_host_override(virtual_clock: VirtualClock):
    os.environ["KIMI_CODE_OAUTH_HOST"] = "https://auth.example.com/"
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        urls.append(url)
        if url == "https://auth.example.com/api/oauth/device_authorization":
            return device_authorization_response(interval=1)
        if url == "https://auth.example.com/api/oauth/token":
            return json_response({"access_token": "a", "refresh_token": "r", "expires_in": 60})
        raise AssertionError(f"Unexpected request URL: {url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        credential = await KIMI_CODING_OAUTH.login(
            RecordingInteraction([]), client=client, clock=virtual_clock.as_device_code_clock()
        )

    assert credential.access == "a"
    assert credential.refresh == "r"
    assert urls == [
        "https://auth.example.com/api/oauth/device_authorization",
        "https://auth.example.com/api/oauth/token",
    ]


async def test_refreshes_tokens_and_returns_a_bearer_header_for_requests():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{OAUTH_HOST}/api/oauth/token"
        fields = form(request)
        assert fields["grant_type"] == "refresh_token"
        assert fields["refresh_token"] == "old-refresh"
        assert fields["client_id"] == CLIENT_ID
        return json_response({"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600})

    before = time.time() * 1000
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        credential = await KIMI_CODING_OAUTH.refresh(
            Credential(type="oauth", access="old-access", refresh="old-refresh", expires=before),
            AbortController().signal,
            client=client,
        )

    # TS: expect(credential).toEqual({ type, access, refresh, expires: expect.any(Number) })
    # is full deep equality except `expires`, which TS deliberately leaves as "any number"
    # (real Date.now() is not mocked in this test). Mirror that: pin every other field
    # exactly and only bound-check `expires`.
    assert dataclasses.replace(credential, expires=None) == Credential(
        type="oauth", access="new-access", refresh="new-refresh", expires=None
    )
    assert credential.expires is not None
    assert credential.expires >= before + 3600 * 1000

    resolved = await KIMI_CODING_OAUTH.to_auth(credential)
    # TS: toEqual({ headers: { Authorization: "Bearer new-access" } }) is full equality
    # on the resolved auth object, not just its headers.
    assert resolved == ResolvedAuth(headers={"Authorization": " ".join(["Bearer", "new-access"])})


async def test_retries_refresh_on_429_and_fails_unauthorized_on_invalid_grant(virtual_clock: VirtualClock):
    calls = 0

    def retry_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return json_response({"error": "temporarily_unavailable"}, 429)
        return json_response({"access_token": "a", "refresh_token": "r", "expires_in": 60})

    async with httpx.AsyncClient(transport=httpx.MockTransport(retry_handler)) as client:
        credential = await KIMI_CODING_OAUTH.refresh(
            Credential(type="oauth", access="old", refresh="old", expires=0),
            AbortController().signal,
            client=client,
            clock=virtual_clock.as_device_code_clock(),
        )
    assert credential.access == "a"
    assert calls == 2
    assert virtual_clock.sleeps == [1]

    def invalid_grant_handler(request: httpx.Request) -> httpx.Response:
        return json_response({"error": "invalid_grant"}, 400)

    async with httpx.AsyncClient(transport=httpx.MockTransport(invalid_grant_handler)) as client:
        with pytest.raises(Exception, match="unauthorized"):
            await KIMI_CODING_OAUTH.refresh(
                Credential(type="oauth", access="old", refresh="old", expires=0),
                AbortController().signal,
                client=client,
                clock=virtual_clock.as_device_code_clock(),
            )
