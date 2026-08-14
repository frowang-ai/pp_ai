"""Radius gateway OAuth flow.

Python port of `packages/ai/src/auth/oauth/radius.ts`.

Radius is a pi-messages gateway. OAuth client APIs live on the configured
gateway; only the interactive browser authorization endpoint is discovered.
Model catalog loading is owned by the Radius provider (out of scope here).
"""

from __future__ import annotations

import json as json_module
import time
import uuid
from urllib.parse import urlencode, urljoin

import httpx

from ...utils.abort import AbortSignal
from ..types import AuthEvent, AuthInteraction, AuthPrompt, Credential, OAuthAuth, ResolvedAuth
from .device_code import REAL_CLOCK, DeviceCodeClock, DeviceCodePollResult, poll_oauth_device_code_flow
from .oauth_page import CallbackResult, OAuthCallbackServer, oauth_error_html
from .pkce import generate_pkce

CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 1456
CALLBACK_PATH = "/oauth/callback"
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
TOKEN_EXPIRY_SKEW_MS = 60_000
LOGIN_METHOD_BROWSER = "browser"
LOGIN_METHOD_DEVICE_CODE = "device-code"
OAUTH_CLIENT_ID = "pi-gateway"
OAUTH_SCOPE = "gateway offline_access"
OAUTH_DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
REQUEST_TIMEOUT_S = 30.0


def normalize_radius_gateway_url(gateway: str) -> str:
    """Minimal equivalent of `normalizeRadiusGatewayUrl`: strip trailing slashes."""
    return gateway.rstrip("/")


class OAuthResponseError(RuntimeError):
    def __init__(self, status: int, oauth_error: str | None, description: str | None, message: str) -> None:
        if oauth_error:
            detail = f"{oauth_error}: {description}" if description else oauth_error
        else:
            detail = description or str(status)
        super().__init__(f"{message}: {detail}")
        self.status = status
        self.oauth_error = oauth_error


async def _read_oauth_response_error(response: httpx.Response, message: str) -> OAuthResponseError:
    text = response.text
    oauth_error: str | None = None
    description: str | None = None
    if text:
        try:
            data = json_module.loads(text)
            if isinstance(data, dict):
                oauth_error = data.get("error") if isinstance(data.get("error"), str) else None
                description = data.get("error_description") if isinstance(data.get("error_description"), str) else None
        except ValueError:
            description = text
    return OAuthResponseError(response.status_code, oauth_error, description, message)


async def load_radius_oauth_discovery(gateway: str, client: httpx.AsyncClient | None = None) -> dict[str, str]:
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
    try:
        response = await http_client.get(urljoin(gateway + "/", "v1/oauth"), headers={"accept": "application/json"})
        if response.status_code >= 400:
            raise RuntimeError(
                f"Could not load Radius OAuth config from {gateway}: {response.status_code} {response.text}"
            )
        discovery = response.json()
        endpoint = discovery.get("authorizationEndpoint") if isinstance(discovery, dict) else None
        if not isinstance(endpoint, str):
            raise RuntimeError(f"Invalid Radius OAuth config from {gateway}")
        return {"authorization_endpoint": endpoint}
    finally:
        if owns_client:
            await http_client.aclose()


async def request_oauth_token(
    gateway: str, body: dict[str, str], client: httpx.AsyncClient | None = None
) -> Credential:
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
    try:
        response = await http_client.post(
            urljoin(gateway + "/", "v1/oauth/token"),
            headers={"accept": "application/json", "content-type": "application/x-www-form-urlencoded"},
            data=body,
        )
        if response.status_code >= 400:
            raise await _read_oauth_response_error(response, "Radius OAuth token request failed")

        data = response.json()
        return Credential(
            type="oauth",
            access=data["access_token"],
            refresh=data["refresh_token"],
            expires=time.time() * 1000 + float(data["expires_in"]) * 1000 - TOKEN_EXPIRY_SKEW_MS,
            data={"scope": data["scope"]} if data.get("scope") else {},
        )
    finally:
        if owns_client:
            await http_client.aclose()


async def request_device_authorization(gateway: str, client: httpx.AsyncClient | None = None) -> dict[str, object]:
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
    try:
        response = await http_client.post(
            urljoin(gateway + "/", "v1/oauth/device"),
            headers={"accept": "application/json", "content-type": "application/x-www-form-urlencoded"},
            data={"client_id": OAUTH_CLIENT_ID, "scope": OAUTH_SCOPE},
        )
        if response.status_code >= 400:
            raise await _read_oauth_response_error(response, "Radius OAuth device authorization failed")

        data = response.json()
        if (
            not data.get("device_code")
            or not data.get("user_code")
            or not data.get("verification_uri")
            or not data.get("expires_in")
        ):
            raise RuntimeError("Radius OAuth device authorization response is missing required fields")
        return {
            "device_code": data["device_code"],
            "user_code": data["user_code"],
            "verification_uri": data["verification_uri"],
            "expires_in": data["expires_in"],
            "interval": data.get("interval"),
        }
    finally:
        if owns_client:
            await http_client.aclose()


async def login_with_browser(
    gateway: str,
    authorization_endpoint: str,
    interaction: AuthInteraction,
    *,
    client: httpx.AsyncClient | None = None,
    callback_port: int = CALLBACK_PORT,
) -> Credential:
    pkce = generate_pkce()
    state = str(uuid.uuid4())

    def _validate_callback(result: CallbackResult) -> tuple[int, str] | tuple[int, str, bool] | None:
        # Mirrors radius.ts's inline http.createServer callback: state mismatches and
        # missing codes reject the request (400) without settling, so the server keeps
        # waiting for a subsequent correct request; an explicit `error` param settles
        # with no code, surfacing "OAuth callback did not complete." to the caller.
        if result.params.get("state") != state:
            return 400, oauth_error_html("OAuth state mismatch.")
        if result.params.get("error"):
            return 400, oauth_error_html(result.params.get("error_description") or result.params["error"]), True
        if not result.params.get("code"):
            return 400, oauth_error_html("Missing authorization code.")
        return None

    server = OAuthCallbackServer(CALLBACK_PATH, host=CALLBACK_HOST, port=callback_port, on_callback=_validate_callback)
    # Built from the server's actual bound port (equal to REDIRECT_URI for the real
    # default `callback_port=CALLBACK_PORT`, but must track an injected ephemeral
    # port in tests so the advertised URL matches where the server is listening).
    redirect_uri = f"http://{CALLBACK_HOST}:{server.port}{CALLBACK_PATH}"
    params = {
        "response_type": "code",
        "client_id": OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": OAUTH_SCOPE,
        "code_challenge": pkce.challenge,
        "code_challenge_method": "S256",
        "handoff": "url",
        "state": state,
    }
    authorize_url = f"{authorization_endpoint}?{urlencode(params)}"
    interaction.notify(AuthEvent(type="progress", message=f"Listening for OAuth callback on {redirect_uri}"))
    interaction.notify(AuthEvent(type="auth_url", url=authorize_url, instructions="Continue in your browser."))

    try:
        callback = await server.wait_for_callback()
        # `_validate_callback` above only settles this future when the request either
        # carries a valid, state-matching code (default 200 path) or an explicit
        # `error` param (settled with the errored params instead of a code); treat
        # both a `None` callback and one carrying no code the same as TS's `!code`.
        if callback is None or not callback.params.get("code"):
            if interaction.signal.aborted:
                raise RuntimeError("Login cancelled")
            raise RuntimeError("OAuth callback did not complete.")
        code = callback.params["code"]

        return await request_oauth_token(
            gateway,
            {
                "grant_type": "authorization_code",
                "client_id": OAUTH_CLIENT_ID,
                "redirect_uri": redirect_uri,
                "code": code,
                "code_verifier": pkce.verifier,
            },
            client,
        )
    finally:
        server.close()


async def login_with_device_code(
    gateway: str,
    interaction: AuthInteraction,
    *,
    client: httpx.AsyncClient | None = None,
    clock: DeviceCodeClock = REAL_CLOCK,
) -> Credential:
    device = await request_device_authorization(gateway, client)
    interaction.notify(
        AuthEvent(
            type="device_code",
            user_code=str(device["user_code"]),
            verification_uri=str(device["verification_uri"]),
            interval_seconds=device.get("interval"),
            expires_in_seconds=device.get("expires_in"),
        )
    )

    async def poll() -> DeviceCodePollResult[Credential]:
        try:
            credential = await request_oauth_token(
                gateway,
                {
                    "grant_type": OAUTH_DEVICE_CODE_GRANT_TYPE,
                    "client_id": OAUTH_CLIENT_ID,
                    "device_code": str(device["device_code"]),
                },
                client,
            )
            return DeviceCodePollResult(status="complete", value=credential)
        except OAuthResponseError as error:
            if error.oauth_error == "authorization_pending":
                return DeviceCodePollResult(status="pending")
            if error.oauth_error == "slow_down":
                return DeviceCodePollResult(status="slow_down")
            if error.oauth_error == "expired_token":
                return DeviceCodePollResult(status="failed", message="Device authorization expired.")
            if error.oauth_error == "access_denied":
                return DeviceCodePollResult(status="failed", message="Device authorization was denied.")
            raise

    return await poll_oauth_device_code_flow(
        poll,
        interaction.signal,
        interval_seconds=device.get("interval"),
        expires_in_seconds=device.get("expires_in"),
        clock=clock,
    )


def create_radius_oauth(name: str, gateway: str) -> OAuthAuth:
    normalized_gateway = normalize_radius_gateway_url(gateway)

    async def login(interaction: AuthInteraction) -> Credential:
        login_method = await interaction.prompt(
            AuthPrompt(
                type="select",
                message=f"Sign in to {name}:",
                options=(
                    {"id": LOGIN_METHOD_BROWSER, "label": "Sign in with browser (recommended)"},
                    {
                        "id": LOGIN_METHOD_DEVICE_CODE,
                        "label": "Sign in with device code (when signing in from another device)",
                    },
                ),
            )
        )

        if login_method == LOGIN_METHOD_DEVICE_CODE:
            return await login_with_device_code(normalized_gateway, interaction)
        if login_method == LOGIN_METHOD_BROWSER:
            discovery = await load_radius_oauth_discovery(normalized_gateway)
            return await login_with_browser(normalized_gateway, discovery["authorization_endpoint"], interaction)
        raise RuntimeError(f"Unknown {name} sign-in method: {login_method}")

    async def refresh(
        credential: Credential, signal: AbortSignal, *, client: httpx.AsyncClient | None = None
    ) -> Credential:
        return await request_oauth_token(
            normalized_gateway,
            {"grant_type": "refresh_token", "client_id": OAUTH_CLIENT_ID, "refresh_token": credential.refresh or ""},
            client,
        )

    async def to_auth(credential: Credential) -> ResolvedAuth:
        return ResolvedAuth(api_key=credential.access)

    return OAuthAuth(name=name, login=login, refresh=refresh, to_auth=to_auth)
