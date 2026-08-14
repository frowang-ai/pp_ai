"""Python port of `packages/ai/test/xai-oauth.test.ts`.

TypeScript stubs the global `fetch` and drives the poll loop with vitest fake
timers. This port injects an `httpx.MockTransport`-backed client and a virtual
`DeviceCodeClock` (the flow accepts both), so the poll *schedule* is asserted
exactly (5s, then 5s, then the slow_down interval) without the test actually
waiting 20 seconds.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from urllib.parse import parse_qs

import httpx
import pytest
from pi_ai.auth.oauth import xai
from pi_ai.auth.oauth.device_code import DeviceCodeClock, DeviceCodeError
from pi_ai.auth.types import AuthEvent, AuthInteraction, AuthPrompt, Credential
from pi_ai.utils.abort import AbortController, AbortSignal

CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
TOKEN_URL = "https://auth.x.ai/oauth2/token"
SCOPE = "openid profile email offline_access grok-cli:access api:access"
TOKEN_LIFETIME_MS = 21_600 * 1000
REFRESH_SKEW_MS = 5 * 60 * 1000


def device_code_response(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "device_code": "device-code",
        "user_code": "ABCD-1234",
        "verification_uri": "https://accounts.x.ai/oauth2/device",
        "expires_in": 900,
        "interval": 5,
    }
    body.update(overrides)
    return body


def token_response(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "expires_in": 21_600,
        "token_type": "Bearer",
    }
    body.update(overrides)
    return {key: value for key, value in body.items() if value is not None}


def request_form(request: httpx.Request) -> dict[str, str]:
    return {key: values[0] for key, values in parse_qs(request.content.decode()).items()}


class VirtualClock:
    """A single virtual time source for the device-code poller.

    This is the `vi.useFakeTimers()` equivalent: the deadline check and the
    inter-poll wait both read from here, so the schedule never depends on how
    long the machine actually took.
    """

    def __init__(self) -> None:
        self.elapsed_ms = 0.0

    def monotonic(self) -> float:
        return self.elapsed_ms / 1000

    async def sleep(self, seconds: float) -> None:
        self.elapsed_ms += seconds * 1000
        await asyncio.sleep(0)

    def device_code_clock(self) -> DeviceCodeClock:
        return DeviceCodeClock(monotonic=self.monotonic, sleep=self.sleep)


class ScriptedInteraction(AuthInteraction):
    def __init__(self, signal: AbortSignal | None = None, on_device_code: Callable[[AuthEvent], None] | None = None):
        self.signal = signal or AbortController().signal
        self.events: list[AuthEvent] = []
        self._on_device_code = on_device_code

    async def prompt(self, prompt: AuthPrompt) -> str:
        raise AssertionError("Unexpected prompt")

    def notify(self, event: AuthEvent) -> None:
        self.events.append(event)
        if event.type == "device_code" and self._on_device_code is not None:
            self._on_device_code(event)

    def device_code_events(self) -> list[AuthEvent]:
        return [event for event in self.events if event.type == "device_code"]


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_uses_the_device_grant_delays_polling_and_handles_pending_and_slow_down() -> None:
    clock = VirtualClock()
    poll_times: list[float] = []
    token_replies: list[httpx.Response] = [
        httpx.Response(400, json={"error": "authorization_pending"}),
        httpx.Response(400, json={"error": "slow_down", "interval": 10}),
        httpx.Response(200, json=token_response()),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        form = request_form(request)
        if url == DEVICE_CODE_URL:
            assert form["client_id"] == CLIENT_ID
            assert form["scope"] == SCOPE
            assert form["referrer"] == "pi"
            return httpx.Response(200, json=device_code_response())
        assert url == TOKEN_URL
        poll_times.append(clock.elapsed_ms)
        assert form["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
        assert form["client_id"] == CLIENT_ID
        assert form["device_code"] == "device-code"
        assert token_replies, "Unexpected token poll"
        return token_replies.pop(0)

    interaction = ScriptedInteraction()
    before_ms = time.time() * 1000
    credential = await xai.login_xai(interaction, client=make_client(handler), clock=clock.device_code_clock())
    after_ms = time.time() * 1000

    events = interaction.device_code_events()
    assert len(events) == 1
    assert events[0].user_code == "ABCD-1234"
    assert events[0].verification_uri == "https://accounts.x.ai/oauth2/device"
    assert events[0].interval_seconds == 5
    assert events[0].expires_in_seconds == 900

    # The first poll waits a full interval, `authorization_pending` keeps the
    # 5s interval, and `slow_down` raises it to the server-reported 10s.
    assert poll_times == [5000, 10_000, 20_000]

    assert credential.type == "oauth"
    assert credential.access == "access-token"
    assert credential.refresh == "refresh-token"
    assert before_ms + TOKEN_LIFETIME_MS - REFRESH_SKEW_MS <= credential.expires
    assert credential.expires <= after_ms + TOKEN_LIFETIME_MS - REFRESH_SKEW_MS


async def test_falls_back_to_the_default_poll_interval_when_interval_is_zero() -> None:
    clock = VirtualClock()
    poll_times: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == DEVICE_CODE_URL:
            return httpx.Response(200, json=device_code_response(interval=0))
        poll_times.append(clock.elapsed_ms)
        return httpx.Response(200, json=token_response())

    await xai.login_xai(ScriptedInteraction(), client=make_client(handler), clock=clock.device_code_clock())
    # RFC 8628 default interval is 5 seconds when the server does not require a wait.
    assert poll_times == [5000]


async def test_prefers_verification_uri_complete_when_the_server_provides_it() -> None:
    clock = VirtualClock()

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == DEVICE_CODE_URL:
            return httpx.Response(
                200,
                json=device_code_response(
                    verification_uri_complete="https://accounts.x.ai/oauth2/device?user_code=ABCD-1234"
                ),
            )
        return httpx.Response(200, json=token_response())

    interaction = ScriptedInteraction()
    await xai.login_xai(interaction, client=make_client(handler), clock=clock.device_code_clock())

    events = interaction.device_code_events()
    assert len(events) == 1
    assert events[0].user_code == "ABCD-1234"
    assert events[0].verification_uri == "https://accounts.x.ai/oauth2/device?user_code=ABCD-1234"
    assert events[0].interval_seconds == 5
    assert events[0].expires_in_seconds == 900


async def test_rejects_a_non_https_verification_uri_complete() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=device_code_response(
                verification_uri_complete="http://accounts.x.ai/oauth2/device?user_code=ABCD-1234"
            ),
        )

    with pytest.raises(RuntimeError, match=r"Untrusted verification URI"):
        await xai.login_xai(ScriptedInteraction(), client=make_client(handler))


@pytest.mark.parametrize(
    "verification_uri",
    ["http://accounts.x.ai/oauth2/device", "file:///etc/passwd", "not a url"],
)
async def test_rejects_a_non_https_verification_uri(verification_uri: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=device_code_response(verification_uri=verification_uri))

    with pytest.raises(RuntimeError, match=r"Untrusted verification URI"):
        await xai.login_xai(ScriptedInteraction(), client=make_client(handler))


@pytest.mark.parametrize("error", ["access_denied", "authorization_denied"])
async def test_fails_when_device_authorization_is_denied(error: str) -> None:
    clock = VirtualClock()
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(200, json=device_code_response(interval=1))
        return httpx.Response(400, json={"error": error})

    with pytest.raises(DeviceCodeError, match=r"xAI device authorization was denied"):
        await xai.login_xai(ScriptedInteraction(), client=make_client(handler), clock=clock.device_code_clock())


async def test_cancels_while_waiting_for_the_first_token_poll() -> None:
    clock = VirtualClock()
    controller = AbortController()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=device_code_response())

    interaction = ScriptedInteraction(signal=controller.signal, on_device_code=lambda _event: controller.abort())
    with pytest.raises(DeviceCodeError, match=r"Login cancelled"):
        await xai.login_xai(interaction, client=make_client(handler), clock=clock.device_code_clock())
    assert len(requests) == 1


async def test_refreshes_tokens_and_preserves_an_unrotated_refresh_token() -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        assert str(request.url) == TOKEN_URL
        form = request_form(request)
        assert form["grant_type"] == "refresh_token"
        assert form["client_id"] == CLIENT_ID
        request_count += 1
        if request_count == 1:
            assert form["refresh_token"] == "old-refresh"
            return httpx.Response(200, json=token_response(access_token="new-access", refresh_token="new-refresh"))
        assert form["refresh_token"] == "keep-refresh"
        return httpx.Response(200, json=token_response(access_token="newer-access", refresh_token=None))

    client = make_client(handler)
    rotated = await xai.refresh(
        Credential(type="oauth", access="old-access", refresh="old-refresh", expires=0),
        AbortController().signal,
        client=client,
    )
    preserved = await xai.refresh(
        Credential(type="oauth", access="old-access", refresh="keep-refresh", expires=0),
        AbortController().signal,
        client=client,
    )

    assert rotated.type == "oauth"
    assert rotated.refresh == "new-refresh"
    assert rotated.access == "new-access"
    assert preserved.refresh == "keep-refresh"
    assert preserved.access == "newer-access"
    assert xai.build_xai_oauth().name == "xAI (Grok/X subscription)"
    assert (await xai.to_auth(preserved)).api_key == "newer-access"


async def test_assumes_a_one_hour_lifetime_when_expires_in_is_missing() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=token_response(expires_in=None))

    before_ms = time.time() * 1000
    credential = await xai.refresh(
        Credential(type="oauth", access="old-access", refresh="old-refresh", expires=0),
        AbortController().signal,
        client=make_client(handler),
    )
    after_ms = time.time() * 1000

    assert before_ms + 3_600_000 - REFRESH_SKEW_MS <= credential.expires
    assert credential.expires <= after_ms + 3_600_000 - REFRESH_SKEW_MS


async def test_rejects_token_responses_with_missing_fields() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=token_response(access_token=None))

    with pytest.raises(RuntimeError, match=r"Invalid xAI OAuth response field: access_token"):
        await xai.refresh(
            Credential(type="oauth", access="old-access", refresh="old-refresh", expires=0),
            AbortController().signal,
            client=make_client(handler),
        )


async def test_surfaces_the_upstream_error_code_and_description_on_refresh_failure() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant", "error_description": "refresh token revoked"})

    with pytest.raises(
        RuntimeError,
        match=r"xAI OAuth token refresh failed \(HTTP 400\): invalid_grant: refresh token revoked",
    ):
        await xai.refresh(
            Credential(type="oauth", access="old-access", refresh="old-refresh", expires=0),
            AbortController().signal,
            client=make_client(handler),
        )
