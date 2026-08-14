"""Kimi Code (subscription) OAuth flow.

Python port of `packages/ai/src/auth/oauth/kimi-coding.ts`.

RFC 8628 device authorization grant against ``https://auth.kimi.com`` with
JSON responses. The access token authenticates requests to
``https://api.kimi.com/coding`` as an ``Authorization: Bearer`` header.
"""

from __future__ import annotations

import asyncio
import math
import time
from urllib.parse import urlparse

import httpx

from ...utils.abort import AbortSignal
from ...utils.provider_env import get_provider_env_value
from ..types import AuthEvent, AuthInteraction, Credential, OAuthAuth, ResolvedAuth
from .device_code import REAL_CLOCK, DeviceCodeClock, DeviceCodePollResult, poll_oauth_device_code_flow

CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
DEFAULT_OAUTH_HOST = "https://auth.kimi.com"
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
DEFAULT_POLL_INTERVAL_SECONDS = 5
REQUEST_TIMEOUT_S = 30.0
REFRESH_MAX_RETRIES = 3


def get_oauth_host() -> str:
    override = get_provider_env_value("KIMI_CODE_OAUTH_HOST") or get_provider_env_value("KIMI_OAUTH_HOST")
    return (override or DEFAULT_OAUTH_HOST).rstrip("/")


def _trusted_http_url(value: object) -> str | None:
    """The verification URI is opened in the user's browser; only http(s) URLs are trusted."""
    if not isinstance(value, str) or not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        return None
    return parsed.geturl()


async def _post_form(
    oauth_host: str, path: str, fields: dict[str, str], client: httpx.AsyncClient | None = None
) -> tuple[int, dict[str, object] | None]:
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
    try:
        response = await http_client.post(
            f"{oauth_host}{path}",
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            data=fields,
        )
        try:
            body = response.json()
            body = body if isinstance(body, dict) else None
        except ValueError:
            body = None
        return response.status_code, body
    finally:
        if owns_client:
            await http_client.aclose()


async def start_device_authorization(oauth_host: str, client: httpx.AsyncClient | None = None) -> dict[str, object]:
    status, json_body = await _post_form(
        oauth_host, "/api/oauth/device_authorization", {"client_id": CLIENT_ID}, client
    )
    if status >= 400:
        raise RuntimeError(f"Kimi Code device authorization failed with status {status}")

    device_code = json_body.get("device_code") if json_body else None
    user_code = json_body.get("user_code") if json_body else None
    verification_uri = json_body.get("verification_uri") if json_body else None
    verification_uri_complete = json_body.get("verification_uri_complete") if json_body else None
    if (
        not isinstance(device_code, str)
        or not isinstance(user_code, str)
        or not _trusted_http_url(verification_uri)
        or not _trusted_http_url(verification_uri_complete)
    ):
        raise RuntimeError(f"Invalid Kimi Code device authorization response: {json_body}")

    interval = json_body.get("interval") if json_body else None
    expires_in = json_body.get("expires_in") if json_body else None
    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": verification_uri,
        "verification_uri_complete": verification_uri_complete,
        "interval_seconds": (
            interval
            if isinstance(interval, (int, float)) and math.isfinite(interval) and interval > 0
            else DEFAULT_POLL_INTERVAL_SECONDS
        ),
        "expires_in_seconds": (
            expires_in
            if isinstance(expires_in, (int, float)) and math.isfinite(expires_in) and expires_in > 0
            else DEVICE_CODE_TIMEOUT_SECONDS
        ),
    }


def _parse_token_response(json_body: dict[str, object] | None, operation: str) -> Credential:
    access_token = json_body.get("access_token") if json_body else None
    refresh_token = json_body.get("refresh_token") if json_body else None
    expires_in = json_body.get("expires_in") if json_body else None
    if (
        not isinstance(access_token, str)
        or not access_token
        or not isinstance(refresh_token, str)
        or not refresh_token
        or not isinstance(expires_in, (int, float))
        or not math.isfinite(expires_in)
        or expires_in <= 0
    ):
        raise RuntimeError(f"Kimi Code token {operation} response missing fields: {json_body}")
    return Credential(
        type="oauth", access=access_token, refresh=refresh_token, expires=time.time() * 1000 + expires_in * 1000
    )


async def poll_for_token(
    oauth_host: str,
    device: dict[str, object],
    signal: AbortSignal,
    client: httpx.AsyncClient | None = None,
    clock: DeviceCodeClock = REAL_CLOCK,
) -> Credential:
    async def poll() -> DeviceCodePollResult[Credential]:
        status, json_body = await _post_form(
            oauth_host,
            "/api/oauth/token",
            {
                "client_id": CLIENT_ID,
                "device_code": str(device["device_code"]),
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            client,
        )
        if status >= 500:
            return DeviceCodePollResult(
                status="failed", message=f"Kimi Code device token request failed with status {status}"
            )

        if status < 400 and json_body and isinstance(json_body.get("access_token"), str):
            try:
                return DeviceCodePollResult(status="complete", value=_parse_token_response(json_body, "poll"))
            except RuntimeError as error:
                return DeviceCodePollResult(status="failed", message=str(error))

        error = json_body.get("error") if json_body else None
        description = json_body.get("error_description") if json_body else None
        suffix = f": {description}" if isinstance(description, str) else ""
        if error == "authorization_pending":
            return DeviceCodePollResult(status="pending")
        if error == "slow_down":
            interval = json_body.get("interval") if json_body else None
            return DeviceCodePollResult(
                status="slow_down",
                interval_seconds=interval if isinstance(interval, (int, float)) and interval > 0 else None,
            )
        if error == "expired_token":
            return DeviceCodePollResult(
                status="failed", message="Kimi Code device authorization expired. Please restart login."
            )
        if error == "access_denied":
            return DeviceCodePollResult(status="failed", message="Kimi Code login was denied.")
        return DeviceCodePollResult(
            status="failed",
            message=f"Kimi Code device token request failed (status {status})"
            + (f": {error}{suffix}" if isinstance(error, str) else ""),
        )

    return await poll_oauth_device_code_flow(
        poll,
        signal,
        interval_seconds=device.get("interval_seconds"),
        expires_in_seconds=device.get("expires_in_seconds"),
        wait_before_first_poll=True,
        clock=clock,
    )


async def _sleep(seconds: float, signal: AbortSignal, clock: DeviceCodeClock = REAL_CLOCK) -> None:
    signal.throw_if_aborted()
    sleep_task: asyncio.Task[None] = asyncio.ensure_future(clock.sleep(seconds))
    abort_task: asyncio.Task[None] = asyncio.ensure_future(signal.wait())
    try:
        done, _pending = await asyncio.wait({sleep_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
        if abort_task in done:
            signal.throw_if_aborted()
    finally:
        for task in (sleep_task, abort_task):
            if not task.done():
                task.cancel()


def _is_retryable_refresh_failure(status: int) -> bool:
    return status == 429 or status >= 500


async def refresh_token(
    oauth_host: str,
    refresh_token_value: str,
    signal: AbortSignal,
    client: httpx.AsyncClient | None = None,
    clock: DeviceCodeClock = REAL_CLOCK,
) -> Credential:
    last_error: Exception | None = None
    for attempt in range(REFRESH_MAX_RETRIES + 1):
        if attempt > 0:
            await _sleep(2 ** (attempt - 1), signal, clock)
        signal.throw_if_aborted()

        status, json_body = await _post_form(
            oauth_host,
            "/api/oauth/token",
            {"client_id": CLIENT_ID, "grant_type": "refresh_token", "refresh_token": refresh_token_value},
            client,
        )

        if status < 400:
            return _parse_token_response(json_body, "refresh")

        # Unauthorized: the stored credential is dead; the caller clears it and prompts re-login.
        error = json_body.get("error") if json_body else None
        if status in (401, 403) or error == "invalid_grant":
            description = json_body.get("error_description") if json_body else None
            suffix = f": {description}" if isinstance(description, str) else ""
            raise RuntimeError(f"Kimi Code token refresh unauthorized (status {status}){suffix}")

        if _is_retryable_refresh_failure(status) and attempt < REFRESH_MAX_RETRIES:
            last_error = RuntimeError(f"Kimi Code token refresh failed with status {status}")
            continue

        raise RuntimeError(f"Kimi Code token refresh failed with status {status}: {json_body}")

    raise last_error or RuntimeError("Kimi Code token refresh failed")


async def login_kimi_coding(
    interaction: AuthInteraction,
    *,
    client: httpx.AsyncClient | None = None,
    clock: DeviceCodeClock = REAL_CLOCK,
) -> Credential:
    oauth_host = get_oauth_host()
    device = await start_device_authorization(oauth_host, client)
    interaction.notify(
        AuthEvent(
            type="device_code",
            user_code=str(device["user_code"]),
            verification_uri=str(device["verification_uri_complete"]),
            interval_seconds=device.get("interval_seconds"),
            expires_in_seconds=device.get("expires_in_seconds"),
        )
    )
    return await poll_for_token(oauth_host, device, interaction.signal, client, clock)


async def refresh(
    credential: Credential,
    signal: AbortSignal,
    *,
    client: httpx.AsyncClient | None = None,
    clock: DeviceCodeClock = REAL_CLOCK,
) -> Credential:
    return await refresh_token(get_oauth_host(), credential.refresh or "", signal, client, clock)


async def to_auth(credential: Credential) -> ResolvedAuth:
    return ResolvedAuth(headers={"Authorization": f"Bearer {credential.access}"})


def build_kimi_coding_oauth() -> OAuthAuth:
    return OAuthAuth(
        name="Kimi Code (subscription)",
        is_subscription=True,
        login_label="Sign in with Kimi Code",
        login=login_kimi_coding,
        refresh=refresh,
        to_auth=to_auth,
    )
