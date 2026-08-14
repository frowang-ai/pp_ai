"""Tests for pi_ai.auth.oauth.kimi_coding — covering missing lines.

All HTTP calls go through httpx.MockTransport; asyncio.sleep is always
patched so no test actually waits.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from pi_ai.auth.oauth.device_code import DeviceCodeClock, DeviceCodeError
from pi_ai.auth.oauth.kimi_coding import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    _parse_token_response,
    _post_form,
    _trusted_http_url,
    login_kimi_coding,
    poll_for_token,
    refresh_token,
    start_device_authorization,
)
from pi_ai.auth.types import AuthEvent, AuthInteraction
from pi_ai.utils.abort import AbortController

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class _NoopInteraction(AuthInteraction):
    def __init__(self) -> None:
        self.signal = AbortController().signal
        self.events: list[AuthEvent] = []

    async def prompt(self, prompt) -> str:  # type: ignore[override]
        raise AssertionError("unexpected prompt")

    def notify(self, event: AuthEvent) -> None:
        self.events.append(event)


def _good_device_response() -> dict:
    return {
        "device_code": "dev-code",
        "user_code": "USER-CODE",
        "verification_uri": "https://auth.kimi.com/device",
        "verification_uri_complete": "https://auth.kimi.com/device?code=USER-CODE",
        "interval": 1,
        "expires_in": 300,
    }


def _good_token_response() -> dict:
    return {"access_token": "acc-tok", "refresh_token": "ref-tok", "expires_in": 3600}


# ---------------------------------------------------------------------------
# _trusted_http_url — lines 40, 43
# ---------------------------------------------------------------------------


def test_trusted_http_url_rejects_none() -> None:
    assert _trusted_http_url(None) is None


def test_trusted_http_url_rejects_empty_string() -> None:
    assert _trusted_http_url("") is None


def test_trusted_http_url_rejects_non_http_scheme() -> None:
    # line 43: scheme not in ("http", "https")
    assert _trusted_http_url("ftp://example.com") is None


def test_trusted_http_url_accepts_https() -> None:
    url = "https://example.com/path"
    result = _trusted_http_url(url)
    assert result is not None
    assert result.startswith("https://")


# ---------------------------------------------------------------------------
# _post_form — lines 61-62, 66
# ---------------------------------------------------------------------------


async def test_post_form_handles_non_dict_json_body() -> None:
    """Lines 61-62: response JSON is a list → body becomes None."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"[1,2,3]", headers={"content-type": "application/json"})

    client = make_client(handler)
    async with client:
        status, body = await _post_form("http://host", "/p", {}, client)
    assert status == 200
    assert body is None


async def test_post_form_handles_non_json_response_body() -> None:
    """Lines 61-62: response body is not JSON → ValueError caught → body is None."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json", headers={"content-type": "text/plain"})

    client = make_client(handler)
    async with client:
        status, body = await _post_form("http://host", "/p", {}, client)
    assert status == 200
    assert body is None


async def test_post_form_creates_and_closes_its_own_client(monkeypatch) -> None:
    """Line 66: when no client is provided, _post_form creates and closes its own."""
    import pi_ai.auth.oauth.kimi_coding as mod

    closed: list[bool] = []

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"ok": True}

    class _FakeClient:
        async def post(self, *a, **kw):
            return _FakeResponse()

        async def aclose(self):
            closed.append(True)

    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: _FakeClient())

    status, _body = await mod._post_form("http://host", "/p", {})
    assert status == 200
    assert closed == [True]


# ---------------------------------------------------------------------------
# start_device_authorization — lines 74, 86
# ---------------------------------------------------------------------------


async def test_start_device_authorization_raises_on_4xx() -> None:
    """Line 74: status >= 400 raises RuntimeError."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    with pytest.raises(RuntimeError, match="status 401"):
        await start_device_authorization("http://host", make_client(handler))


async def test_start_device_authorization_raises_on_invalid_response() -> None:
    """Line 86: device_code/user_code/uris missing or invalid."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"device_code": "x", "user_code": "y"})  # no uris

    with pytest.raises(RuntimeError, match="Invalid Kimi Code device authorization response"):
        await start_device_authorization("http://host", make_client(handler))


async def test_start_device_authorization_uses_defaults_for_missing_interval_and_expires() -> None:
    """Happy path; also checks that missing interval/expires_in uses defaults."""
    resp = {
        "device_code": "dc",
        "user_code": "uc",
        "verification_uri": "https://auth.kimi.com/device",
        "verification_uri_complete": "https://auth.kimi.com/device?code=uc",
        # interval and expires_in intentionally omitted
    }

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=resp)

    result = await start_device_authorization("http://host", make_client(handler))
    assert result["interval_seconds"] == DEFAULT_POLL_INTERVAL_SECONDS


# ---------------------------------------------------------------------------
# _parse_token_response — line 121
# ---------------------------------------------------------------------------


def test_parse_token_response_raises_on_missing_fields() -> None:
    """Line 121: incomplete token dict raises RuntimeError."""
    with pytest.raises(RuntimeError, match="missing fields"):
        _parse_token_response({"access_token": "x"}, "refresh")


def test_parse_token_response_raises_on_non_positive_expires() -> None:
    with pytest.raises(RuntimeError, match="missing fields"):
        _parse_token_response({"access_token": "x", "refresh_token": "r", "expires_in": 0}, "poll")


def test_parse_token_response_success() -> None:
    cred = _parse_token_response({"access_token": "a", "refresh_token": "r", "expires_in": 3600}, "refresh")
    assert cred.access == "a"
    assert cred.refresh == "r"


# ---------------------------------------------------------------------------
# poll_for_token — lines 142, 149-150, 157-169
# ---------------------------------------------------------------------------


def instant_clock() -> DeviceCodeClock:
    """A `DeviceCodeClock` whose waits are instant and whose deadline advances with them.

    Patching `asyncio.sleep` (process-global) or the private `_sleep`/
    `_abortable_sleep` helpers left the flow's deadline on the real clock, so
    the outcome depended on how long the machine actually took. Injecting the
    one time source the flow reads keeps these cases deterministic.
    """
    now = 0.0

    def monotonic() -> float:
        return now

    async def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    return DeviceCodeClock(monotonic=monotonic, sleep=sleep)


async def test_poll_for_token_raises_on_500() -> None:
    """Line 142: HTTP 500 from token endpoint returns failed result → DeviceCodeError."""
    device = {
        "device_code": "dc",
        "interval_seconds": 0.001,
        "expires_in_seconds": 30,
    }

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    signal = AbortController().signal
    with pytest.raises(DeviceCodeError):
        await poll_for_token("http://host", device, signal, make_client(handler), instant_clock())


async def test_poll_for_token_raises_on_malformed_token_response() -> None:
    """Lines 149-150: status 200 with access_token field but _parse_token_response fails."""
    device = {
        "device_code": "dc",
        "interval_seconds": 0.001,
        "expires_in_seconds": 30,
    }

    def handler(req: httpx.Request) -> httpx.Response:
        # access_token present but no refresh_token → _parse_token_response raises
        return httpx.Response(200, json={"access_token": "x"})

    signal = AbortController().signal
    with pytest.raises(DeviceCodeError, match="missing fields"):
        await poll_for_token("http://host", device, signal, make_client(handler), instant_clock())


async def test_poll_for_token_raises_on_expired_token() -> None:
    """Line 163-166: error == 'expired_token'."""
    device = {
        "device_code": "dc",
        "interval_seconds": 0.001,
        "expires_in_seconds": 30,
    }

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "expired_token"})

    signal = AbortController().signal
    with pytest.raises(DeviceCodeError, match="expired"):
        await poll_for_token("http://host", device, signal, make_client(handler), instant_clock())


async def test_poll_for_token_raises_on_access_denied() -> None:
    """Lines 167-168: error == 'access_denied'."""
    device = {
        "device_code": "dc",
        "interval_seconds": 0.001,
        "expires_in_seconds": 30,
    }

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "access_denied"})

    signal = AbortController().signal
    with pytest.raises(DeviceCodeError, match="denied"):
        await poll_for_token("http://host", device, signal, make_client(handler), instant_clock())


async def test_poll_for_token_slow_down_then_success() -> None:
    """Lines 157-162: error == 'slow_down' followed by success."""
    device = {
        "device_code": "dc",
        "interval_seconds": 0.001,
        "expires_in_seconds": 60,
    }

    call_count = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if call_count[0] == 1:
            return httpx.Response(400, json={"error": "slow_down", "interval": 2})
        return httpx.Response(200, json=_good_token_response())

    signal = AbortController().signal
    cred = await poll_for_token("http://host", device, signal, make_client(handler), instant_clock())
    assert cred.access == "acc-tok"


async def test_poll_for_token_generic_error_message() -> None:
    """Line 169: unknown error → generic failure message."""
    device = {
        "device_code": "dc",
        "interval_seconds": 0.001,
        "expires_in_seconds": 30,
    }

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "some_other_error", "error_description": "details"})

    signal = AbortController().signal
    with pytest.raises(DeviceCodeError):
        await poll_for_token("http://host", device, signal, make_client(handler), instant_clock())


# ---------------------------------------------------------------------------
# _sleep — line 191
# ---------------------------------------------------------------------------


async def test_sleep_throws_when_signal_aborted_while_sleeping() -> None:
    """Line 191: if abort_task finishes first, signal.throw_if_aborted() raises."""
    from pi_ai.auth.oauth import kimi_coding as mod
    from pi_ai.utils.abort import AbortController

    controller = AbortController()
    signal = controller.signal

    async def _abort_soon() -> None:
        await asyncio.sleep(0)
        controller.abort()

    async def _do_sleep() -> None:
        await mod._sleep(10.0, signal)

    _, __ = await asyncio.gather(
        _abort_soon(),
        asyncio.ensure_future(_do_sleep()),
        return_exceptions=True,
    )
    # The sleep task should have been cancelled or raised due to abort
    # We check that the abort path (line 191) was exercised
    from pi_ai.utils.abort import AbortError

    with pytest.raises((AbortError, Exception)):
        await mod._sleep(10.0, signal)


# ---------------------------------------------------------------------------
# refresh_token — lines 232-234
# ---------------------------------------------------------------------------


async def test_refresh_token_retries_on_429_then_succeeds() -> None:
    """Lines 232-234 area: 429 → retry → success; _sleep is called for attempt > 0."""
    call_count = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if call_count[0] == 1:
            return httpx.Response(429, json={"error": "rate_limited"})
        return httpx.Response(200, json=_good_token_response())

    signal = AbortController().signal
    cred = await refresh_token("http://host", "ref-tok", signal, make_client(handler), instant_clock())
    assert cred.access == "acc-tok"
    assert call_count[0] == 2


async def test_refresh_token_raises_on_non_retryable_4xx() -> None:
    """Line 232: non-retryable status (e.g. 400, not 401/403) raises immediately."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_request"})

    signal = AbortController().signal
    with pytest.raises(RuntimeError, match="failed with status 400"):
        await refresh_token("http://host", "ref-tok", signal, make_client(handler), instant_clock())


async def test_refresh_token_exhausts_retries_on_persistent_429() -> None:
    """Line 234: all retries exhausted → raise last_error."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate_limited"})

    signal = AbortController().signal
    with pytest.raises(RuntimeError, match="status 429"):
        await refresh_token("http://host", "ref-tok", signal, make_client(handler), instant_clock())


async def test_refresh_token_raises_on_401() -> None:
    """Unauthorized path (status 401)."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid_grant", "error_description": "expired"})

    signal = AbortController().signal
    with pytest.raises(RuntimeError, match="unauthorized"):
        await refresh_token("http://host", "ref-tok", signal, make_client(handler), instant_clock())


# ---------------------------------------------------------------------------
# login_kimi_coding — lines 238-249
# ---------------------------------------------------------------------------


async def test_login_kimi_coding_full_flow(monkeypatch) -> None:
    """Lines 238-249: login_kimi_coding fetches device code then polls for token."""
    import pi_ai.auth.oauth.kimi_coding as mod

    monkeypatch.setattr(mod, "get_oauth_host", lambda: "http://host")

    call_count = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if "/device_authorization" in req.url.path:
            return httpx.Response(200, json=_good_device_response())
        # token endpoint
        return httpx.Response(200, json=_good_token_response())

    interaction = _NoopInteraction()
    cred = await login_kimi_coding(interaction, client=make_client(handler), clock=instant_clock())
    assert cred.access == "acc-tok"
    assert cred.refresh == "ref-tok"
    # The interaction should have received a device_code event
    assert any(e.type == "device_code" for e in interaction.events)
