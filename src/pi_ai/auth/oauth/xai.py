"""xAI OAuth device-code flow.

Python port of `packages/ai/src/auth/oauth/xai.ts`.
"""

from __future__ import annotations

import math
import time
from urllib.parse import urlparse

import httpx

from ...utils.abort import AbortSignal
from ...utils.url import normalize_http_url
from ..types import AuthEvent, AuthInteraction, Credential, OAuthAuth, ResolvedAuth
from .device_code import REAL_CLOCK, DeviceCodeClock, DeviceCodePollResult, poll_oauth_device_code_flow

CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
SCOPE = "openid profile email offline_access grok-cli:access api:access"
DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
TOKEN_URL = "https://auth.x.ai/oauth2/token"
# Refresh slightly before the reported expiry to avoid using a token that dies mid-request.
REFRESH_SKEW_MS = 5 * 60 * 1000
DEFAULT_TOKEN_LIFETIME_SECONDS = 3600
REQUEST_TIMEOUT_S = 30.0


def _required_string(body: dict[str, object], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Invalid xAI OAuth response field: {field}")
    return value


def _positive_number(body: dict[str, object], field: str) -> float:
    value = body.get(field)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"Invalid xAI OAuth response field: {field}")
    return value


def _validate_verification_uri(raw: str) -> str:
    """The verification URI is opened in the user's browser; only https is trusted.

    Normalized like TypeScript's `new URL(raw).href` so control characters
    cannot reach the terminal or the OS `open` launcher verbatim.
    """
    try:
        normalized = normalize_http_url(raw)
    except ValueError as error:
        raise RuntimeError("Untrusted verification URI in xAI OAuth response") from error
    if urlparse(normalized).scheme != "https":
        raise RuntimeError("Untrusted verification URI in xAI OAuth response")
    return normalized


async def _post_form(
    url: str, fields: dict[str, str], client: httpx.AsyncClient | None = None
) -> tuple[bool, int, dict[str, object]]:
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
    try:
        response = await http_client.post(
            url,
            headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            data=fields,
        )
        try:
            body = response.json()
            if not isinstance(body, dict):
                body = {}
        except ValueError:
            body = {}
        return response.status_code < 400, response.status_code, body
    finally:
        if owns_client:
            await http_client.aclose()


def _request_failure(action: str, status: int, body: dict[str, object]) -> RuntimeError:
    error = body.get("error") if isinstance(body.get("error"), str) else None
    description = body.get("error_description") if isinstance(body.get("error_description"), str) else None
    detail = ": ".join(filter(None, [error, description]))
    suffix = f": {detail}" if detail else ""
    return RuntimeError(f"xAI OAuth {action} failed (HTTP {status}){suffix}")


def _parse_device_code(body: dict[str, object]) -> dict[str, object]:
    # RFC 8628 allows interval 0 (no minimum wait); fall back to the poller's
    # default instead of failing on non-positive or malformed values.
    interval = body.get("interval")
    interval_seconds = (
        interval if isinstance(interval, (int, float)) and math.isfinite(interval) and interval > 0 else None
    )
    verification_uri_complete = body.get("verification_uri_complete")
    verification_uri_complete = (
        _validate_verification_uri(verification_uri_complete)
        if isinstance(verification_uri_complete, str) and verification_uri_complete
        else None
    )
    return {
        "device_code": _required_string(body, "device_code"),
        "user_code": _required_string(body, "user_code"),
        "verification_uri": _validate_verification_uri(_required_string(body, "verification_uri")),
        "verification_uri_complete": verification_uri_complete,
        "interval_seconds": interval_seconds,
        "expires_in_seconds": _positive_number(body, "expires_in"),
    }


def _credential_from_token_response(body: dict[str, object], previous_refresh_token: str | None = None) -> Credential:
    access = _required_string(body, "access_token")
    # xAI may omit refresh_token on refresh when the token is not rotated.
    refresh = (
        previous_refresh_token
        if body.get("refresh_token") is None and previous_refresh_token
        else _required_string(body, "refresh_token")
    )
    expires_in_seconds = (
        DEFAULT_TOKEN_LIFETIME_SECONDS if body.get("expires_in") is None else _positive_number(body, "expires_in")
    )
    return Credential(
        type="oauth",
        access=access,
        refresh=refresh,
        expires=time.time() * 1000 + expires_in_seconds * 1000 - REFRESH_SKEW_MS,
    )


async def request_device_code(client: httpx.AsyncClient | None = None) -> dict[str, object]:
    ok, status, body = await _post_form(
        DEVICE_CODE_URL, {"client_id": CLIENT_ID, "scope": SCOPE, "referrer": "pi"}, client
    )
    if not ok:
        raise _request_failure("device authorization", status, body)
    return _parse_device_code(body)


async def poll_for_tokens(
    device: dict[str, object],
    signal: AbortSignal,
    client: httpx.AsyncClient | None = None,
    clock: DeviceCodeClock = REAL_CLOCK,
) -> Credential:
    async def poll() -> DeviceCodePollResult[Credential]:
        ok, status, body = await _post_form(
            TOKEN_URL,
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": CLIENT_ID,
                "device_code": str(device["device_code"]),
            },
            client,
        )
        if ok:
            return DeviceCodePollResult(status="complete", value=_credential_from_token_response(body))

        error = body.get("error")
        if error == "authorization_pending":
            return DeviceCodePollResult(status="pending")
        if error == "slow_down":
            interval = body.get("interval")
            return DeviceCodePollResult(
                status="slow_down", interval_seconds=interval if isinstance(interval, (int, float)) else None
            )
        if error in ("access_denied", "authorization_denied"):
            return DeviceCodePollResult(status="failed", message="xAI device authorization was denied")
        if error == "expired_token":
            return DeviceCodePollResult(status="failed", message="xAI device code expired")
        return DeviceCodePollResult(
            status="failed", message=_request_failure("device token polling", status, body).args[0]
        )

    return await poll_oauth_device_code_flow(
        poll,
        signal,
        interval_seconds=device.get("interval_seconds"),
        expires_in_seconds=device.get("expires_in_seconds"),
        wait_before_first_poll=True,
        clock=clock,
    )


async def login_xai(
    interaction: AuthInteraction,
    *,
    client: httpx.AsyncClient | None = None,
    clock: DeviceCodeClock = REAL_CLOCK,
) -> Credential:
    device = await request_device_code(client)
    interaction.notify(
        AuthEvent(
            type="device_code",
            user_code=str(device["user_code"]),
            verification_uri=str(device.get("verification_uri_complete") or device["verification_uri"]),
            interval_seconds=device.get("interval_seconds"),
            expires_in_seconds=device.get("expires_in_seconds"),
        )
    )
    return await poll_for_tokens(device, interaction.signal, client, clock)


async def refresh(
    credential: Credential, signal: AbortSignal, *, client: httpx.AsyncClient | None = None
) -> Credential:
    ok, status, body = await _post_form(
        TOKEN_URL,
        {"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": credential.refresh or ""},
        client,
    )
    if not ok:
        raise _request_failure("token refresh", status, body)
    return _credential_from_token_response(body, credential.refresh)


async def to_auth(credential: Credential) -> ResolvedAuth:
    return ResolvedAuth(api_key=credential.access)


def build_xai_oauth() -> OAuthAuth:
    return OAuthAuth(
        name="xAI (Grok/X subscription)",
        is_subscription=True,
        login_label="Sign in with SuperGrok or X Premium",
        login=login_xai,
        refresh=refresh,
        to_auth=to_auth,
    )
