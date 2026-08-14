"""End-to-end OAuth login-flow tests: the full `login()` sequence (browser
callback and manual-code-paste races, device-code polling to completion), the
token-refresh path, and expired/invalid refresh tokens surfacing the right
error. Loopback HTTP (127.0.0.1, ephemeral ports) is used for the local
callback servers exactly like `test_auth_oauth.py`'s callback-server tests;
all *provider* API calls go through `httpx.MockTransport`, never the network.
No browser is ever launched: `browser_opener` is always a fake.

Device-code polling is driven by an injected `VirtualClock` rather than by real
`asyncio.sleep`. The flows read their deadline *and* their inter-poll waits
from that one clock, so the poll schedule is asserted exactly and the tests do
not spend real seconds waiting (or become load-dependent when they do).
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
from pi_ai.auth.oauth import anthropic, github_copilot, load, openrouter, radius, xai
from pi_ai.auth.oauth.device_code import DeviceCodeClock, DeviceCodeError
from pi_ai.auth.types import AuthEvent, AuthInteraction, AuthPrompt, Credential
from pi_ai.utils.abort import AbortController

CALLBACK_TIMEOUT_S = 5


def now_ms() -> float:
    return time.time() * 1000


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class VirtualClock:
    """The single time source a device-code flow reads, advanced only by its own waits.

    This is the `vi.useFakeTimers()` equivalent: `monotonic` and `sleep` move
    together, so the deadline can never be decided partly by real elapsed time.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)

    def as_device_code_clock(self) -> DeviceCodeClock:
        return DeviceCodeClock(monotonic=self.monotonic, sleep=self.sleep)


class ScriptedInteraction(AuthInteraction):
    """Fake `AuthInteraction`: prompt() returns scripted answers in order,
    notify() records every event for assertions."""

    def __init__(self, answers: list[str] | None = None) -> None:
        self.signal = AbortController().signal
        self._answers = list(answers or [])
        self.events: list[AuthEvent] = []
        self.prompts: list[AuthPrompt] = []

    async def prompt(self, prompt: AuthPrompt) -> str:
        self.prompts.append(prompt)
        await asyncio.sleep(0)
        if not self._answers:
            raise AssertionError("ScriptedInteraction ran out of scripted prompt answers")
        return self._answers.pop(0)

    def notify(self, event: AuthEvent) -> None:
        self.events.append(event)

    def auth_url(self) -> str:
        return next(e.url for e in self.events if e.type == "auth_url" and e.url)


class HangingInteraction(AuthInteraction):
    """Fake `AuthInteraction` whose prompt() never resolves on its own (used
    to exercise the browser-callback-wins-the-race path without a manual
    paste ever completing)."""

    def __init__(self) -> None:
        self.signal = AbortController().signal
        self.events: list[AuthEvent] = []

    async def prompt(self, prompt: AuthPrompt) -> str:
        await asyncio.Event().wait()  # never resolves; the caller cancels this task.
        raise AssertionError("unreachable")

    def notify(self, event: AuthEvent) -> None:
        self.events.append(event)

    def auth_url(self) -> str:
        return next(e.url for e in self.events if e.type == "auth_url" and e.url)


async def _get_loopback(url: str, **params: str) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        return await client.get(url, params=params)


# --------------------------------------------------------------------------
# anthropic: full login()
# --------------------------------------------------------------------------


async def test_anthropic_login_browser_callback_success():
    interaction = HangingInteraction()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/oauth/token"
        body = json.loads(request.content)
        assert body["grant_type"] == "authorization_code"
        assert body["code"] == "the-code"
        return httpx.Response(200, json={"access_token": "acc", "refresh_token": "ref", "expires_in": 3600})

    login_task = asyncio.ensure_future(
        anthropic.login_anthropic(interaction, client=make_client(handler), callback_port=0)
    )
    await asyncio.sleep(0.05)
    auth_url = interaction.auth_url()
    redirect_uri = auth_url.split("redirect_uri=")[1].split("&")[0]
    from urllib.parse import unquote

    redirect_uri = unquote(redirect_uri)
    state = auth_url.split("state=")[1].split("&")[0]

    response = await _get_loopback(redirect_uri, code="the-code", state=state)
    assert response.status_code == 200

    credential = await asyncio.wait_for(login_task, timeout=CALLBACK_TIMEOUT_S)
    assert credential.access == "acc"
    assert credential.refresh == "ref"


async def test_anthropic_login_browser_callback_state_mismatch_then_success():
    """A stray/incorrect request must not abort the whole login: the server
    keeps waiting (HTTP 400) until a request with the matching state
    arrives."""
    interaction = HangingInteraction()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "acc", "refresh_token": "ref", "expires_in": 3600})

    login_task = asyncio.ensure_future(
        anthropic.login_anthropic(interaction, client=make_client(handler), callback_port=0)
    )
    await asyncio.sleep(0.05)
    auth_url = interaction.auth_url()
    from urllib.parse import unquote

    redirect_uri = unquote(auth_url.split("redirect_uri=")[1].split("&")[0])

    bad_response = await _get_loopback(redirect_uri, code="c1", state="wrong-state")
    assert bad_response.status_code == 400
    assert not login_task.done()

    state = auth_url.split("state=")[1].split("&")[0]
    good_response = await _get_loopback(redirect_uri, code="c1", state=state)
    assert good_response.status_code == 200

    credential = await asyncio.wait_for(login_task, timeout=CALLBACK_TIMEOUT_S)
    assert credential.access == "acc"


async def test_anthropic_login_browser_callback_missing_code_then_success():
    interaction = HangingInteraction()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "acc", "refresh_token": "ref", "expires_in": 3600})

    login_task = asyncio.ensure_future(
        anthropic.login_anthropic(interaction, client=make_client(handler), callback_port=0)
    )
    await asyncio.sleep(0.05)
    auth_url = interaction.auth_url()
    from urllib.parse import unquote

    redirect_uri = unquote(auth_url.split("redirect_uri=")[1].split("&")[0])
    state = auth_url.split("state=")[1].split("&")[0]

    missing_code_response = await _get_loopback(redirect_uri, state=state)
    assert missing_code_response.status_code == 400
    assert not login_task.done()

    good_response = await _get_loopback(redirect_uri, code="c1", state=state)
    assert good_response.status_code == 200
    credential = await asyncio.wait_for(login_task, timeout=CALLBACK_TIMEOUT_S)
    assert credential.access == "acc"


async def test_anthropic_login_browser_callback_error_param_keeps_waiting():
    interaction = HangingInteraction()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "acc", "refresh_token": "ref", "expires_in": 3600})

    login_task = asyncio.ensure_future(
        anthropic.login_anthropic(interaction, client=make_client(handler), callback_port=0)
    )
    await asyncio.sleep(0.05)
    auth_url = interaction.auth_url()
    from urllib.parse import unquote

    redirect_uri = unquote(auth_url.split("redirect_uri=")[1].split("&")[0])
    state = auth_url.split("state=")[1].split("&")[0]

    error_response = await _get_loopback(redirect_uri, error="access_denied")
    assert error_response.status_code == 400
    assert "did not complete" in error_response.text.lower()
    assert not login_task.done()

    good_response = await _get_loopback(redirect_uri, code="c1", state=state)
    assert good_response.status_code == 200
    credential = await asyncio.wait_for(login_task, timeout=CALLBACK_TIMEOUT_S)
    assert credential.access == "acc"

    login_task.cancel()


async def test_anthropic_login_manual_code_paste_success():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["code"] == "pasted-code"
        return httpx.Response(200, json={"access_token": "acc2", "refresh_token": "ref2", "expires_in": 60})

    interaction = ScriptedInteraction()

    async def scripted_prompt(prompt: AuthPrompt) -> str:
        interaction.prompts.append(prompt)
        # The manual code must include the correct state (pkce.verifier) since
        # this path has no HTTP callback to validate it for us.
        auth_url = interaction.auth_url()
        state = auth_url.split("state=")[1].split("&")[0]
        return f"pasted-code#{state}"

    interaction.prompt = scripted_prompt  # type: ignore[method-assign]

    credential = await asyncio.wait_for(
        anthropic.login_anthropic(interaction, client=make_client(handler), callback_port=0),
        timeout=CALLBACK_TIMEOUT_S,
    )
    assert credential.access == "acc2"
    assert credential.refresh == "ref2"


async def test_anthropic_login_manual_code_state_mismatch_raises():
    interaction = ScriptedInteraction(answers=["pasted-code#totally-wrong-state"])

    with pytest.raises(RuntimeError, match="state mismatch"):
        await asyncio.wait_for(
            anthropic.login_anthropic(interaction, client=make_client(lambda r: httpx.Response(200)), callback_port=0),
            timeout=CALLBACK_TIMEOUT_S,
        )


async def test_anthropic_login_manual_code_missing_raises():
    interaction = ScriptedInteraction(answers=[""])

    with pytest.raises(RuntimeError, match="Missing authorization code"):
        await asyncio.wait_for(
            anthropic.login_anthropic(interaction, client=make_client(lambda r: httpx.Response(200)), callback_port=0),
            timeout=CALLBACK_TIMEOUT_S,
        )


async def test_anthropic_login_open_browser_invokes_injected_opener_not_real_browser():
    opened: list[str] = []
    interaction = ScriptedInteraction(answers=[""])

    with pytest.raises(RuntimeError, match="Missing authorization code"):
        await asyncio.wait_for(
            anthropic.login_anthropic(
                interaction,
                open_browser=True,
                browser_opener=lambda url: opened.append(url),
                client=make_client(lambda r: httpx.Response(200)),
                callback_port=0,
            ),
            timeout=CALLBACK_TIMEOUT_S,
        )
    assert len(opened) == 1
    assert opened[0].startswith("https://claude.ai/oauth/authorize")


async def test_anthropic_refresh_missing_token_raises():
    from pi_ai.auth.types import Credential
    from pi_ai.utils.abort import AbortController as AC

    credential = Credential(type="oauth", access="acc", refresh=None)
    with pytest.raises(RuntimeError, match="Missing Anthropic OAuth refresh token"):
        await anthropic.refresh(credential, AC().signal)


def test_anthropic_build_oauth_object():
    built = anthropic.build_anthropic_oauth()
    assert built.name == "Anthropic (Claude Pro/Max)"
    assert built.is_subscription is True
    assert built.login is anthropic.login_anthropic
    assert built.refresh is anthropic.refresh
    assert built.to_auth is anthropic.to_auth


# --------------------------------------------------------------------------
# openrouter: full login()
# --------------------------------------------------------------------------


async def test_openrouter_login_browser_callback_success():
    interaction = HangingInteraction()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/auth/keys"
        body = json.loads(request.content)
        assert body["code"] == "or-code"
        return httpx.Response(200, json={"key": "sk-or-v1-abc"})

    login_task = asyncio.ensure_future(
        openrouter.login_openrouter(interaction, client=make_client(handler), callback_port=0)
    )
    await asyncio.sleep(0.05)
    auth_url = interaction.auth_url()
    from urllib.parse import unquote

    callback_url = unquote(auth_url.split("callback_url=")[1].split("&")[0])

    response = await _get_loopback(callback_url, code="or-code")
    assert response.status_code == 200

    credential = await asyncio.wait_for(login_task, timeout=CALLBACK_TIMEOUT_S)
    assert credential.access == "sk-or-v1-abc"
    assert credential.refresh == ""


async def test_openrouter_login_browser_callback_missing_code_then_success():
    interaction = HangingInteraction()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"key": "sk-or-v1-abc"})

    login_task = asyncio.ensure_future(
        openrouter.login_openrouter(interaction, client=make_client(handler), callback_port=0)
    )
    await asyncio.sleep(0.05)
    auth_url = interaction.auth_url()
    from urllib.parse import unquote

    callback_url = unquote(auth_url.split("callback_url=")[1].split("&")[0])

    missing_code_response = await _get_loopback(callback_url)
    assert missing_code_response.status_code == 400
    assert not login_task.done()

    good_response = await _get_loopback(callback_url, code="or-code")
    assert good_response.status_code == 200
    credential = await asyncio.wait_for(login_task, timeout=CALLBACK_TIMEOUT_S)
    assert credential.access == "sk-or-v1-abc"


async def test_openrouter_login_browser_callback_error_param_fails_immediately():
    interaction = HangingInteraction()

    login_task = asyncio.ensure_future(
        openrouter.login_openrouter(interaction, client=make_client(lambda r: httpx.Response(200)), callback_port=0)
    )
    await asyncio.sleep(0.05)
    auth_url = interaction.auth_url()
    from urllib.parse import unquote

    callback_url = unquote(auth_url.split("callback_url=")[1].split("&")[0])

    response = await _get_loopback(callback_url, error="access_denied", error_description="user said no")
    assert response.status_code == 400

    with pytest.raises(RuntimeError, match="user said no"):
        await asyncio.wait_for(login_task, timeout=CALLBACK_TIMEOUT_S)


@pytest.mark.parametrize(
    "manual_input",
    [
        "bare-code-value",
        "https://example.com/cb?code=bare-code-value&state=x",
        "code=bare-code-value&state=x",
    ],
)
async def test_openrouter_login_manual_code_paste_accepts_multiple_shapes(manual_input):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["code"] == "bare-code-value"
        return httpx.Response(200, json={"key": "sk-manual"})

    interaction = ScriptedInteraction(answers=[manual_input])
    credential = await asyncio.wait_for(
        openrouter.login_openrouter(interaction, client=make_client(handler), callback_port=0),
        timeout=CALLBACK_TIMEOUT_S,
    )
    assert credential.access == "sk-manual"


async def test_openrouter_login_manual_code_missing_raises():
    interaction = ScriptedInteraction(answers=[""])
    with pytest.raises(RuntimeError, match="Missing authorization code"):
        await asyncio.wait_for(
            openrouter.login_openrouter(
                interaction, client=make_client(lambda r: httpx.Response(200)), callback_port=0
            ),
            timeout=CALLBACK_TIMEOUT_S,
        )


async def test_openrouter_login_exchange_failure_detail_from_error_object():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad code"}})

    interaction = ScriptedInteraction(answers=["some-code"])
    with pytest.raises(RuntimeError, match="bad code"):
        await asyncio.wait_for(
            openrouter.login_openrouter(interaction, client=make_client(handler), callback_port=0),
            timeout=CALLBACK_TIMEOUT_S,
        )


def test_openrouter_build_oauth_object():
    built = openrouter.build_openrouter_oauth()
    assert built.name == "OpenRouter OAuth"
    assert built.login_label == "Sign in with OpenRouter"
    assert built.login is openrouter.login_openrouter


# --------------------------------------------------------------------------
# radius: full create_radius_oauth().login() dispatch + login_with_browser/device_code
# --------------------------------------------------------------------------


async def test_radius_login_with_browser_full_success():
    interaction = HangingInteraction()

    def token_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/oauth/token"
        return httpx.Response(
            200, json={"access_token": "racc", "refresh_token": "rref", "expires_in": 3600, "scope": "gateway"}
        )

    login_task = asyncio.ensure_future(
        radius.login_with_browser(
            "https://gateway.example.com",
            "https://gateway.example.com/authorize",
            interaction,
            client=make_client(token_handler),
            callback_port=0,
        )
    )
    await asyncio.sleep(0.05)
    auth_url = interaction.auth_url()
    state = auth_url.split("state=")[1].split("&")[0]

    # The callback path/host come from radius's fixed REDIRECT_URI template but the
    # port is injected via callback_port=0; reconstruct it the same way radius.py does.
    # Discover the ephemeral port from the auth_url's redirect_uri param instead of
    # hardcoding the fixed CALLBACK_PORT (which this test deliberately avoids binding).
    from urllib.parse import unquote

    from pi_ai.auth.oauth.radius import CALLBACK_HOST, CALLBACK_PATH

    redirect_uri = unquote(auth_url.split("redirect_uri=")[1].split("&")[0])
    assert redirect_uri.startswith(f"http://{CALLBACK_HOST}:")
    assert redirect_uri.endswith(CALLBACK_PATH)

    bad = await _get_loopback(redirect_uri, code="c1", state="wrong")
    assert bad.status_code == 400
    assert not login_task.done()

    good = await _get_loopback(redirect_uri, code="c1", state=state)
    assert good.status_code == 200

    credential = await asyncio.wait_for(login_task, timeout=CALLBACK_TIMEOUT_S)
    assert credential.access == "racc"
    assert credential.refresh == "rref"
    assert credential.data == {"scope": "gateway"}


async def test_radius_login_with_browser_error_param_settles_as_incomplete():
    interaction = HangingInteraction()

    login_task = asyncio.ensure_future(
        radius.login_with_browser(
            "https://gateway.example.com",
            "https://gateway.example.com/authorize",
            interaction,
            client=make_client(lambda r: httpx.Response(200)),
            callback_port=0,
        )
    )
    await asyncio.sleep(0.05)
    auth_url = interaction.auth_url()
    from urllib.parse import unquote

    redirect_uri = unquote(auth_url.split("redirect_uri=")[1].split("&")[0])
    state = auth_url.split("state=")[1].split("&")[0]

    # radius.ts checks `state` before `error`: a request with no/wrong state never
    # reaches the error branch and keeps the server waiting, matching the state
    # mismatch behavior above.
    no_state_response = await _get_loopback(redirect_uri, error="access_denied")
    assert no_state_response.status_code == 400
    assert not login_task.done()

    response = await _get_loopback(redirect_uri, error="access_denied", state=state)
    assert response.status_code == 400

    with pytest.raises(RuntimeError, match="OAuth callback did not complete"):
        await asyncio.wait_for(login_task, timeout=CALLBACK_TIMEOUT_S)


async def test_radius_login_with_browser_missing_code_keeps_waiting_then_succeeds():
    interaction = HangingInteraction()

    def token_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "racc", "refresh_token": "rref", "expires_in": 3600})

    login_task = asyncio.ensure_future(
        radius.login_with_browser(
            "https://gateway.example.com",
            "https://gateway.example.com/authorize",
            interaction,
            client=make_client(token_handler),
            callback_port=0,
        )
    )
    await asyncio.sleep(0.05)
    auth_url = interaction.auth_url()
    state = auth_url.split("state=")[1].split("&")[0]
    from urllib.parse import unquote

    redirect_uri = unquote(auth_url.split("redirect_uri=")[1].split("&")[0])

    missing = await _get_loopback(redirect_uri, state=state)
    assert missing.status_code == 400
    assert not login_task.done()

    good = await _get_loopback(redirect_uri, code="c1", state=state)
    assert good.status_code == 200
    credential = await asyncio.wait_for(login_task, timeout=CALLBACK_TIMEOUT_S)
    assert credential.access == "racc"


async def test_radius_login_with_device_code_full_success():
    clock = VirtualClock()
    interaction = ScriptedInteraction()
    calls = {"device": 0, "token": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth/device":
            calls["device"] += 1
            return httpx.Response(
                200,
                json={
                    "device_code": "dc1",
                    "user_code": "USER1",
                    "verification_uri": "https://gateway.example.com/activate",
                    "expires_in": 600,
                    "interval": 1,
                },
            )
        assert request.url.path == "/v1/oauth/token"
        calls["token"] += 1
        if calls["token"] == 1:
            return httpx.Response(400, json={"error": "authorization_pending"})
        return httpx.Response(200, json={"access_token": "dacc", "refresh_token": "dref", "expires_in": 3600})

    credential = await asyncio.wait_for(
        radius.login_with_device_code(
            "https://gateway.example.com",
            interaction,
            client=make_client(handler),
            clock=clock.as_device_code_clock(),
        ),
        timeout=CALLBACK_TIMEOUT_S,
    )
    assert credential.access == "dacc"
    assert calls["token"] == 2
    device_events = [e for e in interaction.events if e.type == "device_code"]
    assert device_events[0].user_code == "USER1"


async def test_radius_create_oauth_login_dispatches_to_device_code(monkeypatch):
    # create_radius_oauth()'s login() doesn't accept an injectable client (it always
    # constructs its own real httpx.AsyncClient for `login_with_device_code`), so this
    # test verifies its *dispatch* wiring in isolation by monkeypatching the
    # module-level `login_with_device_code`/`load_radius_oauth_discovery` symbols
    # `create_radius_oauth` calls -- no network is reached.
    calls: list[str] = []

    async def fake_login_with_device_code(gateway, interaction, *, client=None):
        calls.append(gateway)
        from pi_ai.auth.types import Credential

        return Credential(type="oauth", access="dacc", refresh="dref")

    monkeypatch.setattr(radius, "login_with_device_code", fake_login_with_device_code)
    interaction = ScriptedInteraction(answers=[radius.LOGIN_METHOD_DEVICE_CODE])
    oauth = radius.create_radius_oauth("Test Radius", "https://gateway.example.com/")

    credential = await asyncio.wait_for(oauth.login(interaction), timeout=CALLBACK_TIMEOUT_S)
    assert credential.access == "dacc"
    assert calls == ["https://gateway.example.com"]


async def test_radius_create_oauth_login_dispatches_to_browser(monkeypatch):
    calls: list[str] = []

    async def fake_load_discovery(gateway, client=None):
        calls.append(gateway)
        return {"authorization_endpoint": "https://gateway.example.com/authorize"}

    async def fake_login_with_browser(gateway, authorization_endpoint, interaction, *, client=None, callback_port=None):
        calls.append(authorization_endpoint)
        from pi_ai.auth.types import Credential

        return Credential(type="oauth", access="bacc", refresh="bref")

    monkeypatch.setattr(radius, "load_radius_oauth_discovery", fake_load_discovery)
    monkeypatch.setattr(radius, "login_with_browser", fake_login_with_browser)
    interaction = ScriptedInteraction(answers=[radius.LOGIN_METHOD_BROWSER])
    oauth = radius.create_radius_oauth("Test Radius", "https://gateway.example.com")

    credential = await asyncio.wait_for(oauth.login(interaction), timeout=CALLBACK_TIMEOUT_S)
    assert credential.access == "bacc"
    assert calls == ["https://gateway.example.com", "https://gateway.example.com/authorize"]


async def test_radius_create_oauth_login_unknown_method_raises():
    interaction = ScriptedInteraction(answers=["carrier-pigeon"])
    oauth = radius.create_radius_oauth("Test Radius", "https://gateway.example.com")
    with pytest.raises(RuntimeError, match="Unknown Test Radius sign-in method"):
        await asyncio.wait_for(oauth.login(interaction), timeout=CALLBACK_TIMEOUT_S)


async def test_radius_request_device_authorization_missing_fields_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"device_code": "dc1"})

    with pytest.raises(RuntimeError, match="missing required fields"):
        await radius.request_device_authorization("https://gateway.example.com", make_client(handler))


async def test_radius_load_discovery_invalid_config_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"notTheRightField": "x"})

    with pytest.raises(RuntimeError, match="Invalid Radius OAuth config"):
        await radius.load_radius_oauth_discovery("https://gateway.example.com", make_client(handler))


async def test_radius_load_discovery_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(RuntimeError, match="Could not load Radius OAuth config"):
        await radius.load_radius_oauth_discovery("https://gateway.example.com", make_client(handler))


async def test_radius_device_code_poll_slow_down_then_complete():
    """`slow_down` must back off before polling again.

    The virtual clock records the requested delay instead of waiting it out:
    RFC 8628's +5s increment would otherwise make this test sleep for six real
    seconds, which slows the suite and makes the assertion timing-fragile. The
    recorded delay is the behaviour that matters.
    """
    clock = VirtualClock()
    interaction = ScriptedInteraction()
    calls = {"token": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth/device":
            return httpx.Response(
                200,
                json={
                    "device_code": "dc1",
                    "user_code": "USER1",
                    "verification_uri": "https://gateway.example.com/activate",
                    "expires_in": 600,
                    "interval": 1,
                },
            )
        calls["token"] += 1
        if calls["token"] == 1:
            # Unlike github_copilot.py/xai.py, radius's `OAuthResponseError` carries
            # no numeric interval override for `slow_down`, matching radius.ts (which
            # also never reads one back off the response body): the poller always
            # falls back to RFC 8628's +5s increment, so this test allows enough
            # margin for that ~6s pause between polls.
            return httpx.Response(400, json={"error": "slow_down"})
        return httpx.Response(200, json={"access_token": "dacc", "refresh_token": "dref", "expires_in": 3600})

    credential = await asyncio.wait_for(
        radius.login_with_device_code(
            "https://gateway.example.com",
            interaction,
            client=make_client(handler),
            clock=clock.as_device_code_clock(),
        ),
        timeout=5,
    )
    assert credential.access == "dacc"
    assert calls["token"] == 2
    # Radius polls immediately (it does not set `wait_before_first_poll`), so the only
    # wait is the one after `slow_down`: the 1s server interval plus RFC 8628's +5s.
    assert clock.sleeps == [6.0]


@pytest.mark.parametrize("oauth_error,message_part", [("expired_token", "expired"), ("access_denied", "denied")])
async def test_radius_device_code_poll_terminal_failures(oauth_error, message_part):
    clock = VirtualClock()
    interaction = ScriptedInteraction()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth/device":
            return httpx.Response(
                200,
                json={
                    "device_code": "dc1",
                    "user_code": "USER1",
                    "verification_uri": "https://gateway.example.com/activate",
                    "expires_in": 600,
                    "interval": 1,
                },
            )
        return httpx.Response(400, json={"error": oauth_error})

    with pytest.raises(DeviceCodeError, match=message_part):
        await asyncio.wait_for(
            radius.login_with_device_code(
                "https://gateway.example.com",
                interaction,
                client=make_client(handler),
                clock=clock.as_device_code_clock(),
            ),
            timeout=CALLBACK_TIMEOUT_S,
        )


async def test_radius_device_code_poll_unmapped_error_reraises_oauth_response_error():
    clock = VirtualClock()
    interaction = ScriptedInteraction()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth/device":
            return httpx.Response(
                200,
                json={
                    "device_code": "dc1",
                    "user_code": "USER1",
                    "verification_uri": "https://gateway.example.com/activate",
                    "expires_in": 600,
                    "interval": 1,
                },
            )
        return httpx.Response(400, json={"error": "invalid_grant", "error_description": "gateway rejected"})

    with pytest.raises(radius.OAuthResponseError, match="invalid_grant"):
        await asyncio.wait_for(
            radius.login_with_device_code(
                "https://gateway.example.com",
                interaction,
                client=make_client(handler),
                clock=clock.as_device_code_clock(),
            ),
            timeout=CALLBACK_TIMEOUT_S,
        )


async def test_radius_normalize_gateway_url_strips_trailing_slash():
    assert radius.normalize_radius_gateway_url("https://gw.example.com///") == "https://gw.example.com"
    assert radius.normalize_radius_gateway_url("https://gw.example.com") == "https://gw.example.com"


# --------------------------------------------------------------------------
# load.py
# --------------------------------------------------------------------------


async def test_load_anthropic_oauth():
    built = await load.load_anthropic_oauth()
    assert built.name == "Anthropic (Claude Pro/Max)"


async def test_load_github_copilot_oauth():
    built = await load.load_github_copilot_oauth()
    assert built.name == "GitHub Copilot"


async def test_load_openrouter_oauth():
    built = await load.load_openrouter_oauth()
    assert built.name == "OpenRouter OAuth"


async def test_load_kimi_coding_oauth():
    built = await load.load_kimi_coding_oauth()
    assert built.name


async def test_load_xai_oauth():
    built = await load.load_xai_oauth()
    assert built.name == "xAI (Grok/X subscription)"


async def test_load_radius_oauth():
    built = await load.load_radius_oauth("My Radius", "https://gw.example.com")
    assert built.name == "My Radius"


# --------------------------------------------------------------------------
# github_copilot: full login()
# --------------------------------------------------------------------------


async def test_github_copilot_login_full_flow_default_domain():
    clock = VirtualClock()
    interaction = ScriptedInteraction(answers=[""])
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/login/device/code":
            assert request.url.host == "github.com"
            return httpx.Response(
                200,
                json={
                    "device_code": "devcode",
                    "user_code": "USER-1",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 1,
                    "expires_in": 600,
                },
            )
        if request.url.path == "/login/oauth/access_token":
            return httpx.Response(200, json={"access_token": "gh-access"})
        if request.url.path == "/copilot_internal/v2/token":
            return httpx.Response(
                200,
                json={
                    "token": "tid=x;proxy-ep=proxy.individual.githubcopilot.com;exp=1",
                    "expires_at": int(time.time()) + 1800,
                },
            )
        if request.url.path.endswith("/policy"):
            return httpx.Response(200, json={})
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "gpt-4", "model_picker_enabled": True, "policy": {"state": "enabled"}}]},
            )
        raise AssertionError(f"unexpected request {request.url}")

    credential = await asyncio.wait_for(
        github_copilot.login_github_copilot(
            interaction,
            client=make_client(handler),
            clock=clock.as_device_code_clock(),
        ),
        timeout=CALLBACK_TIMEOUT_S,
    )
    assert credential.access.startswith("tid=x;")
    assert credential.refresh == "gh-access"
    assert credential.data["availableModelIds"] == ["gpt-4"]
    assert any(url.endswith("/models/claude-opus-4.5/policy") for url in calls)
    device_events = [e for e in interaction.events if e.type == "device_code"]
    assert device_events[0].user_code == "USER-1"


async def test_github_copilot_login_full_flow_enterprise_domain():
    clock = VirtualClock()
    interaction = ScriptedInteraction(answers=["acme.ghe.com"])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/login/device/code":
            assert request.url.host == "acme.ghe.com"
            return httpx.Response(
                200,
                json={
                    "device_code": "devcode",
                    "user_code": "USER-2",
                    "verification_uri": "https://acme.ghe.com/login/device",
                    "interval": 1,
                    "expires_in": 600,
                },
            )
        if request.url.path == "/login/oauth/access_token":
            assert request.url.host == "acme.ghe.com"
            return httpx.Response(200, json={"access_token": "gh-access-2"})
        if request.url.path == "/copilot_internal/v2/token":
            assert request.url.host == "api.acme.ghe.com"
            return httpx.Response(200, json={"token": "copilot-token-2", "expires_at": int(time.time()) + 1800})
        if request.url.path == "/models":
            assert request.url.host == "copilot-api.acme.ghe.com"
            return httpx.Response(200, json={"data": []})
        if request.url.path.endswith("/policy"):
            assert request.url.host == "copilot-api.acme.ghe.com"
            return httpx.Response(200, json={})
        raise AssertionError(f"unexpected request {request.url}")

    credential = await asyncio.wait_for(
        github_copilot.login_github_copilot(
            interaction, client=make_client(handler), clock=clock.as_device_code_clock()
        ),
        timeout=CALLBACK_TIMEOUT_S,
    )
    assert credential.access == "copilot-token-2"
    assert credential.data["enterprise_url"] == "acme.ghe.com"
    assert credential.data["availableModelIds"] == []


async def test_github_copilot_login_bounds_policy_update_concurrency():
    clock = VirtualClock()
    interaction = ScriptedInteraction(answers=[""])
    active = 0
    peak = 0
    policy_requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak, policy_requests
        if request.url.path == "/login/device/code":
            return httpx.Response(
                200,
                json={
                    "device_code": "devcode",
                    "user_code": "USER-3",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 1,
                    "expires_in": 600,
                },
            )
        if request.url.path == "/login/oauth/access_token":
            return httpx.Response(200, json={"access_token": "gh-access"})
        if request.url.path == "/copilot_internal/v2/token":
            return httpx.Response(200, json={"token": "copilot-token", "expires_at": int(time.time()) + 1800})
        if request.url.path == "/models":
            return httpx.Response(200, json={"data": []})
        if request.url.path.endswith("/policy"):
            policy_requests += 1
            active += 1
            peak = max(peak, active)
            # Yield so any sibling request in the same batch can start before
            # this one finishes; without that the peak is always 1.
            await asyncio.sleep(0.005)
            active -= 1
            return httpx.Response(200, json={})
        raise AssertionError(f"unexpected request {request.url}")

    await asyncio.wait_for(
        github_copilot.login_github_copilot(
            interaction, client=make_client(handler), clock=clock.as_device_code_clock()
        ),
        timeout=CALLBACK_TIMEOUT_S,
    )
    assert policy_requests > 4
    assert peak == 4
    assert github_copilot.COPILOT_POLICY_CONCURRENCY == 4


async def test_github_copilot_login_invalid_enterprise_domain_raises():
    interaction = ScriptedInteraction(answers=["http://"])
    with pytest.raises(RuntimeError, match="Invalid GitHub Enterprise URL"):
        await asyncio.wait_for(
            github_copilot.login_github_copilot(interaction, client=make_client(lambda r: httpx.Response(200))),
            timeout=CALLBACK_TIMEOUT_S,
        )


async def test_github_copilot_login_cancelled_raises():
    class AbortingInteraction(AuthInteraction):
        def __init__(self) -> None:
            self.signal = AbortController().signal
            self._controller = self.signal

        async def prompt(self, prompt: AuthPrompt) -> str:
            # Simulate the signal getting aborted concurrently while the domain
            # prompt was in flight (e.g. the user hit Ctrl+C).
            controller = AbortController()
            controller.abort()
            self.signal = controller.signal
            return ""

        def notify(self, event: AuthEvent) -> None:
            pass

    interaction = AbortingInteraction()
    with pytest.raises(RuntimeError, match="Login cancelled"):
        await asyncio.wait_for(
            github_copilot.login_github_copilot(interaction, client=make_client(lambda r: httpx.Response(200))),
            timeout=CALLBACK_TIMEOUT_S,
        )


@pytest.mark.parametrize(
    "error,message_part",
    [
        ("expired_token", "Device flow failed"),
        ("access_denied", "Device flow failed"),
    ],
)
async def test_github_copilot_poll_for_access_token_terminal_failures(error, message_part):
    clock = VirtualClock()
    device = {"device_code": "dc1", "interval": 1, "expires_in": 60}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": error, "error_description": "nope"})

    with pytest.raises(DeviceCodeError, match=message_part):
        await asyncio.wait_for(
            github_copilot.poll_for_github_access_token(
                "github.com", device, AbortController().signal, make_client(handler), clock.as_device_code_clock()
            ),
            timeout=CALLBACK_TIMEOUT_S,
        )


async def test_github_copilot_poll_for_access_token_pending_then_slow_down_then_complete():
    clock = VirtualClock()
    device = {"device_code": "dc1", "interval": 1, "expires_in": 60}
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"error": "authorization_pending"})
        if calls["n"] == 2:
            return httpx.Response(200, json={"error": "slow_down", "interval": 1})
        return httpx.Response(200, json={"access_token": "gh-final"})

    token = await asyncio.wait_for(
        github_copilot.poll_for_github_access_token(
            "github.com", device, AbortController().signal, make_client(handler), clock.as_device_code_clock()
        ),
        timeout=8,
    )
    assert token == "gh-final"
    assert calls["n"] == 3


async def test_github_copilot_poll_for_access_token_invalid_response_fails():
    clock = VirtualClock()
    device = {"device_code": "dc1", "interval": 1, "expires_in": 60}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    with pytest.raises(DeviceCodeError, match="Invalid device token response"):
        await asyncio.wait_for(
            github_copilot.poll_for_github_access_token(
                "github.com", device, AbortController().signal, make_client(handler), clock.as_device_code_clock()
            ),
            timeout=CALLBACK_TIMEOUT_S,
        )


def test_github_copilot_parse_model_ids_skips_tool_calls_false():
    raw = {
        "data": [
            {
                "id": "model-a",
                "model_picker_enabled": True,
                "policy": {"state": "enabled"},
                "capabilities": {"supports": {"tool_calls": False}},
            },
            {"id": "model-b", "model_picker_enabled": True, "policy": {"state": "enabled"}},
        ]
    }
    assert github_copilot.parse_available_copilot_model_ids(raw, allow_policy_fallback=False) == ["model-b"]


def test_github_copilot_parse_model_ids_policy_fallback_for_individual_accounts():
    # Individual accounts sometimes report every `model_picker_enabled` flag as
    # false despite an explicit enabled policy; the fallback list still surfaces
    # those models when `allow_policy_fallback` is set (limited to the
    # individual-account base URL by `fetch_available_github_copilot_model_ids`).
    raw = {
        "data": [
            {"id": "model-a", "model_picker_enabled": False, "policy": {"state": "enabled"}},
            {"id": "model-b", "model_picker_enabled": False, "policy": {"state": "disabled"}},
        ]
    }
    assert github_copilot.parse_available_copilot_model_ids(raw, allow_policy_fallback=True) == ["model-a"]
    assert github_copilot.parse_available_copilot_model_ids(raw, allow_policy_fallback=False) == []


def test_github_copilot_parse_model_ids_invalid_response_raises():
    with pytest.raises(RuntimeError, match="Invalid Copilot models response"):
        github_copilot.parse_available_copilot_model_ids({"data": "not-a-list"}, allow_policy_fallback=False)


async def test_github_copilot_enable_model_network_error_returns_false():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    async with make_client(handler) as client:
        enabled = await github_copilot.enable_github_copilot_model("tok", "model-x", None, client)
    assert enabled is False


async def test_github_copilot_fetch_json_error_status_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    async with make_client(handler) as client:
        with pytest.raises(RuntimeError, match="403"):
            await github_copilot.fetch_available_github_copilot_model_ids("tok", None, client)


def test_github_copilot_enterprise_domain_none_when_no_data():
    credential = Credential(type="oauth", access="x")
    assert github_copilot.copilot_enterprise_domain(credential) is None


def test_github_copilot_enterprise_domain_normalizes_stored_url():
    credential = Credential(type="oauth", access="x", data={"enterprise_url": "https://acme.ghe.com/"})
    assert github_copilot.copilot_enterprise_domain(credential) == "acme.ghe.com"


def test_github_copilot_build_oauth_object():
    built = github_copilot.build_github_copilot_oauth()
    assert built.name == "GitHub Copilot"
    assert built.is_subscription is True


# --------------------------------------------------------------------------
# xai: full login()
# --------------------------------------------------------------------------


async def test_xai_login_full_flow_uses_verification_uri_complete():
    clock = VirtualClock()
    interaction = ScriptedInteraction()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/device/code":
            body = dict(x.split("=") for x in request.content.decode().split("&"))
            assert body["client_id"] == xai.CLIENT_ID
            return httpx.Response(
                200,
                json={
                    "device_code": "dc1",
                    "user_code": "USER-X",
                    "verification_uri": "https://auth.x.ai/activate",
                    "verification_uri_complete": "https://auth.x.ai/activate?user_code=USER-X",
                    "interval": 1,
                    "expires_in": 60,
                },
            )
        assert request.url.path == "/oauth2/token"
        return httpx.Response(200, json={"access_token": "xacc", "refresh_token": "xref", "expires_in": 3600})

    credential = await asyncio.wait_for(
        xai.login_xai(interaction, client=make_client(handler), clock=clock.as_device_code_clock()),
        timeout=CALLBACK_TIMEOUT_S,
    )
    assert credential.access == "xacc"
    assert credential.refresh == "xref"
    device_events = [e for e in interaction.events if e.type == "device_code"]
    assert device_events[0].verification_uri == "https://auth.x.ai/activate?user_code=USER-X"


async def test_xai_login_pending_then_complete():
    clock = VirtualClock()
    interaction = ScriptedInteraction()
    calls = {"token": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/device/code":
            return httpx.Response(
                200,
                json={
                    "device_code": "dc1",
                    "user_code": "USER-X",
                    "verification_uri": "https://auth.x.ai/activate",
                    "interval": 1,
                    "expires_in": 60,
                },
            )
        calls["token"] += 1
        if calls["token"] == 1:
            return httpx.Response(400, json={"error": "authorization_pending"})
        return httpx.Response(200, json={"access_token": "xacc", "refresh_token": "xref", "expires_in": 3600})

    credential = await asyncio.wait_for(
        xai.login_xai(interaction, client=make_client(handler), clock=clock.as_device_code_clock()),
        timeout=CALLBACK_TIMEOUT_S,
    )
    assert credential.access == "xacc"
    assert calls["token"] == 2


@pytest.mark.parametrize("error", ["access_denied", "authorization_denied", "expired_token"])
async def test_xai_login_poll_terminal_failures(error):
    clock = VirtualClock()
    interaction = ScriptedInteraction()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/device/code":
            return httpx.Response(
                200,
                json={
                    "device_code": "dc1",
                    "user_code": "USER-X",
                    "verification_uri": "https://auth.x.ai/activate",
                    "interval": 1,
                    "expires_in": 60,
                },
            )
        return httpx.Response(400, json={"error": error})

    with pytest.raises(DeviceCodeError):
        await asyncio.wait_for(
            xai.login_xai(interaction, client=make_client(handler), clock=clock.as_device_code_clock()),
            timeout=CALLBACK_TIMEOUT_S,
        )


async def test_xai_login_poll_slow_down_then_complete():
    clock = VirtualClock()
    interaction = ScriptedInteraction()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/device/code":
            return httpx.Response(
                200,
                json={
                    "device_code": "dc1",
                    "user_code": "USER-X",
                    "verification_uri": "https://auth.x.ai/activate",
                    "interval": 1,
                    "expires_in": 60,
                },
            )
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(400, json={"error": "slow_down", "interval": 1})
        return httpx.Response(200, json={"access_token": "xacc", "refresh_token": "xref", "expires_in": 3600})

    credential = await asyncio.wait_for(
        xai.login_xai(interaction, client=make_client(handler), clock=clock.as_device_code_clock()), timeout=8
    )
    assert credential.access == "xacc"


def test_xai_parse_device_code_interval_zero_falls_back_to_default():
    body = {
        "device_code": "dc1",
        "user_code": "u",
        "verification_uri": "https://auth.x.ai/activate",
        "interval": 0,
        "expires_in": 60,
    }
    device = xai._parse_device_code(body)
    assert device["interval_seconds"] is None


def test_xai_parse_device_code_rejects_non_https_verification_uri_complete():
    body = {
        "device_code": "dc1",
        "user_code": "u",
        "verification_uri": "https://auth.x.ai/activate",
        "verification_uri_complete": "http://auth.x.ai/activate",
        "expires_in": 60,
    }
    with pytest.raises(RuntimeError, match="Untrusted verification URI"):
        xai._parse_device_code(body)


def test_xai_credential_from_token_response_reuses_refresh_token_when_omitted_on_refresh():
    body = {"access_token": "new-access", "expires_in": 100}
    credential = xai._credential_from_token_response(body, previous_refresh_token="old-refresh")
    assert credential.refresh == "old-refresh"
    assert credential.access == "new-access"


def test_xai_credential_from_token_response_requires_refresh_token_without_previous():
    body = {"access_token": "new-access", "expires_in": 100}
    with pytest.raises(RuntimeError, match="refresh_token"):
        xai._credential_from_token_response(body, previous_refresh_token=None)


def test_xai_credential_from_token_response_default_expires_in_when_missing():
    body = {"access_token": "acc", "refresh_token": "ref"}
    before = now_ms()
    credential = xai._credential_from_token_response(body)
    # DEFAULT_TOKEN_LIFETIME_SECONDS (3600s) minus the 5 minute refresh skew.
    assert credential.expires - before == pytest.approx((3600 - 300) * 1000, abs=2000)


async def test_xai_refresh_failure_formats_error_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant", "error_description": "token revoked"})

    with pytest.raises(RuntimeError, match="token revoked"):
        async with make_client(handler) as client:
            await xai.refresh(
                Credential(type="oauth", access="a", refresh="r"), AbortController().signal, client=client
            )


def test_xai_build_oauth_object():
    built = xai.build_xai_oauth()
    assert built.name == "xAI (Grok/X subscription)"
    assert built.is_subscription is True
    assert built.login_label == "Sign in with SuperGrok or X Premium"
