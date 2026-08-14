"""Python port of `packages/ai/test/github-copilot-oauth.test.ts`.

TypeScript drives the device-flow timing cases with `vi.useFakeTimers()` and
`vi.advanceTimersByTimeAsync`. asyncio has no equivalent, so the poll-timing
cases pass a virtual :class:`~pi_ai.auth.oauth.device_code.DeviceCodeClock`
whose `sleep` advances a counter instead of really sleeping and whose
`monotonic` reads that counter. Every assertion about *when* a poll happens
(and about the resulting interval growth after `slow_down`) is preserved
exactly; only the wall-clock wait is elided.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import httpx
import pytest
from pi_ai.auth.oauth.device_code import DeviceCodeClock
from pi_ai.auth.oauth.github_copilot import build_github_copilot_oauth
from pi_ai.auth.types import AuthEvent, AuthInteraction, AuthPrompt, Credential, InMemoryCredentialStore
from pi_ai.providers.github_copilot import github_copilot_provider
from pi_ai.registry import Models
from pi_ai.utils.abort import AbortController, AbortSignal

# TypeScript drives the wired OAuth object (`githubCopilotOAuth.login(...)`), not the bare
# module function. Going through `build_github_copilot_oauth()` means a regression that rewires the
# flow -- pointing `login` at the wrong function, or at one that is not a coroutine
# function -- fails here instead of only in production. Calling `login_x(...)`
# directly would keep passing through such a break.
GITHUB_COPILOT_OAUTH = build_github_copilot_oauth()


def test_the_real_oauth_object_wires_coroutine_functions() -> None:
    """Guards the shape the CLI depends on: `provider.auth.oauth.<hook>` must be awaitable."""
    for hook in ("login", "refresh", "to_auth"):
        assert inspect.iscoroutinefunction(getattr(GITHUB_COPILOT_OAUTH, hook)), hook


NEVER_ABORTED = AbortController().signal


def json_response(body: object, status: int = 200) -> httpx.Response:
    return httpx.Response(status, headers={"Content-Type": "application/json"}, text=json.dumps(body))


class RecordingInteraction(AuthInteraction):
    def __init__(
        self,
        signal: AbortSignal | None = None,
        prompt_reply: str = "",
    ) -> None:
        self.signal = signal or NEVER_ABORTED
        self._prompt_reply = prompt_reply
        self.prompts: list[AuthPrompt] = []
        self.device_codes: list[AuthEvent] = []
        self.progress: list[str] = []

    async def prompt(self, prompt: AuthPrompt) -> str:
        if prompt.type != "text":
            raise AssertionError(f"Unexpected prompt: {prompt.type}")
        self.prompts.append(prompt)
        return self._prompt_reply

    def notify(self, event: AuthEvent) -> None:
        if event.type == "device_code":
            self.device_codes.append(event)
        if event.type == "progress" and event.message is not None:
            self.progress.append(event.message)


# --- virtual clock ---------------------------------------------------------


class VirtualClock:
    """A `DeviceCodeClock` that advances only when the flow sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds

    def as_device_code_clock(self) -> DeviceCodeClock:
        return DeviceCodeClock(monotonic=self.monotonic, sleep=self.sleep)


@pytest.fixture
def virtual_clock() -> VirtualClock:
    return VirtualClock()


# --- refresh / model filtering --------------------------------------------


def make_refresh_client(
    data: list[dict[str, Any]], proxy_host: str = "proxy.individual.githubcopilot.com"
) -> httpx.AsyncClient:
    access_token = f"tid=test;exp=9999999999;proxy-ep={proxy_host};"
    models_url = f"https://{proxy_host.replace('proxy.', 'api.', 1)}/models"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/copilot_internal/v2/token" in url:
            return json_response({"token": access_token, "expires_at": 9999999999})
        if url == models_url:
            scheme, _, token = request.headers["authorization"].partition(" ")
            assert scheme == "Bearer"
            assert token == access_token
            return json_response({"data": data})
        raise AssertionError(f"Unexpected request URL: {url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def refresh_github_copilot_models_for_test(
    data: list[dict[str, Any]], proxy_host: str = "proxy.individual.githubcopilot.com"
) -> Credential:
    async with make_refresh_client(data, proxy_host) as client:
        return await GITHUB_COPILOT_OAUTH.refresh(
            Credential(type="oauth", access="old-access-token", refresh="ghu_refresh_token", expires=0),
            NEVER_ABORTED,
            client=client,
        )


async def available_model_ids_through_registry(credential: Credential) -> list[str]:
    store = InMemoryCredentialStore()
    await store.set("github-copilot", credential)
    models = Models(credential_store=store)
    models.add(github_copilot_provider())
    return [model.id for model in await models.get_available("github-copilot")]


async def test_filters_models_to_the_authenticated_account_picker_catalog():
    credential = await refresh_github_copilot_models_for_test(
        [
            {"id": "gpt-4.1", "model_picker_enabled": True, "capabilities": {"supports": {"tool_calls": True}}},
            {
                "id": "claude-opus-4.7",
                "model_picker_enabled": True,
                "policy": {"state": "disabled"},
                "capabilities": {"supports": {"tool_calls": True}},
            },
            {
                "id": "gpt-5.4-nano",
                "model_picker_enabled": False,
                "policy": {"state": "enabled"},
                "capabilities": {"supports": {"tool_calls": True}},
            },
        ]
    )
    assert credential.data["availableModelIds"] == ["gpt-4.1"]
    assert await available_model_ids_through_registry(credential) == ["gpt-4.1"]


async def test_falls_back_to_explicitly_enabled_policy_models_when_the_picker_catalog_is_empty():
    credential = await refresh_github_copilot_models_for_test(
        [
            {
                "id": "gpt-4.1",
                "model_picker_enabled": False,
                "policy": {"state": "enabled"},
                "capabilities": {"supports": {"tool_calls": True}},
            },
            {
                "id": "claude-opus-4.7",
                "model_picker_enabled": False,
                "policy": {"state": "disabled"},
                "capabilities": {"supports": {"tool_calls": True}},
            },
            {
                "id": "gpt-5.4-nano",
                "model_picker_enabled": False,
                "capabilities": {"supports": {"tool_calls": True}},
            },
            {
                "id": "gpt-4o",
                "model_picker_enabled": False,
                "policy": {"state": "enabled"},
                "capabilities": {"supports": {"tool_calls": False}},
            },
        ]
    )
    assert credential.data["availableModelIds"] == ["gpt-4.1"]
    assert await available_model_ids_through_registry(credential) == ["gpt-4.1"]


async def test_does_not_fall_back_to_policy_models_for_non_individual_accounts():
    credential = await refresh_github_copilot_models_for_test(
        [
            {
                "id": "gpt-4.1",
                "model_picker_enabled": False,
                "policy": {"state": "enabled"},
                "capabilities": {"supports": {"tool_calls": True}},
            }
        ],
        "proxy.business.githubcopilot.com",
    )
    assert credential.data["availableModelIds"] == []


# --- device flow -----------------------------------------------------------


def make_login_client(
    device_code_body: dict[str, Any],
    access_token_responses: list[httpx.Response] | None = None,
    poll_times: list[float] | None = None,
    clock: VirtualClock | None = None,
    device_code_requests: list[httpx.Request] | None = None,
    access_token_requests: list[httpx.Request] | None = None,
) -> httpx.AsyncClient:
    pending = list(access_token_responses or [json_response({"access_token": "ghu_refresh_token"})])

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/login/device/code"):
            if device_code_requests is not None:
                device_code_requests.append(request)
            return json_response(device_code_body)
        if url.endswith("/login/oauth/access_token"):
            if poll_times is not None and clock is not None:
                poll_times.append(clock.now)
            if access_token_requests is not None:
                access_token_requests.append(request)
            if not pending:
                raise AssertionError("Unexpected extra access token poll")
            return pending.pop(0)
        if "/copilot_internal/v2/token" in url:
            return json_response(
                {
                    "token": "tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;",
                    "expires_at": 9999999999,
                }
            )
        if url.endswith("/models"):
            return json_response({"data": []})
        if "/models/" in url and url.endswith("/policy"):
            return httpx.Response(200, text="")
        raise AssertionError(f"Unexpected request URL: {url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_reports_device_code_details_through_the_device_code_event(virtual_clock: VirtualClock):
    interaction = RecordingInteraction()
    client = make_login_client(
        {
            "device_code": "device-code",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://github.com/login/device",
            "interval": 1,
            "expires_in": 900,
        }
    )
    async with client:
        await GITHUB_COPILOT_OAUTH.login(interaction, client=client, clock=virtual_clock.as_device_code_clock())

    assert len(interaction.device_codes) == 1
    event = interaction.device_codes[0]
    assert event.user_code == "ABCD-EFGH"
    assert event.verification_uri == "https://github.com/login/device"
    assert event.interval_seconds == 1
    assert event.expires_in_seconds == 900


async def test_rejects_a_non_http_verification_uri_before_it_reaches_the_device_code_event():
    # A malicious enterprise OAuth server could return a verification_uri that
    # the browser launcher would otherwise hand to the OS. Ensure such values
    # are rejected at the deserialization boundary.
    interaction = RecordingInteraction()
    client = make_login_client(
        {
            "device_code": "device-code",
            "user_code": "ABCD-EFGH",
            "verification_uri": "$(id>/dev/null)",
            "interval": 1,
            "expires_in": 900,
        }
    )
    async with client:
        with pytest.raises(Exception, match="Untrusted verification_uri"):
            await GITHUB_COPILOT_OAUTH.login(interaction, client=client)

    assert interaction.device_codes == []


async def test_normalizes_verification_uri_before_it_reaches_the_device_code_event(
    virtual_clock: VirtualClock,
):
    raw_verification_uri = "https://github.com/login/\x1b]8;;evil"
    interaction = RecordingInteraction()
    client = make_login_client(
        {
            "device_code": "device-code",
            "user_code": "ABCD-EFGH",
            "verification_uri": raw_verification_uri,
            "interval": 1,
            "expires_in": 900,
        }
    )
    async with client:
        await GITHUB_COPILOT_OAUTH.login(interaction, client=client, clock=virtual_clock.as_device_code_clock())

    assert len(interaction.device_codes) == 1
    event = interaction.device_codes[0]
    assert event.user_code == "ABCD-EFGH"
    # `new URL(raw).href` in TypeScript; the ESC byte is percent-encoded.
    assert event.verification_uri == "https://github.com/login/%1B]8;;evil"
    assert event.interval_seconds == 1
    assert event.expires_in_seconds == 900
    assert event.verification_uri != raw_verification_uri


async def test_waits_before_polling_and_increases_the_interval_after_slow_down(
    virtual_clock: VirtualClock,
):
    poll_times: list[float] = []
    device_code_requests: list[httpx.Request] = []
    access_token_requests: list[httpx.Request] = []
    client = make_login_client(
        {
            "device_code": "device-code",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://github.com/login/device",
            "interval": 5,
            "expires_in": 900,
        },
        access_token_responses=[
            json_response({"error": "authorization_pending", "error_description": "pending"}),
            json_response({"error": "slow_down", "error_description": "slow down", "interval": 7}),
            json_response({"access_token": "ghu_refresh_token"}),
        ],
        poll_times=poll_times,
        clock=virtual_clock,
        device_code_requests=device_code_requests,
        access_token_requests=access_token_requests,
    )
    async with client:
        await GITHUB_COPILOT_OAUTH.login(
            RecordingInteraction(), client=client, clock=virtual_clock.as_device_code_clock()
        )

    device_request = device_code_requests[0]
    assert device_request.method == "POST"
    assert device_request.headers["accept"] == "application/json"
    assert device_request.headers["content-type"] == "application/x-www-form-urlencoded"
    assert "client_id=" in device_request.content.decode()
    assert "scope=read%3Auser" in device_request.content.decode()

    token_request = access_token_requests[0]
    assert token_request.method == "POST"
    assert token_request.headers["accept"] == "application/json"
    assert token_request.headers["content-type"] == "application/x-www-form-urlencoded"
    body = token_request.content.decode()
    assert "client_id=" in body
    assert "device_code=device-code" in body
    assert "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code" in body

    # 5s wait before the first poll, 5s to the second, then the server-provided
    # 7s slow_down interval to the third.
    assert poll_times == [5.0, 10.0, 17.0]


async def test_times_out_after_repeated_slow_down_responses(virtual_clock: VirtualClock):
    poll_times: list[float] = []
    client = make_login_client(
        {
            "device_code": "device-code",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://github.com/login/device",
            "interval": 5,
            "expires_in": 25,
        },
        access_token_responses=[
            json_response({"error": "slow_down", "error_description": "slow down"}),
            json_response({"error": "slow_down", "error_description": "still too fast"}),
            json_response({"error": "authorization_pending", "error_description": "pending"}),
        ],
        poll_times=poll_times,
        clock=virtual_clock,
    )
    async with client:
        with pytest.raises(Exception, match="Device flow timed out after one or more slow_down responses"):
            await GITHUB_COPILOT_OAUTH.login(
                RecordingInteraction(), client=client, clock=virtual_clock.as_device_code_clock()
            )

    assert poll_times == [5.0, 15.0]
