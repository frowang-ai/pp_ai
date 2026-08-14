"""Anthropic OAuth flow (Claude Pro/Max).

Python port of `packages/ai/src/auth/oauth/anthropic.ts`.

The TypeScript flow starts a `node:http` callback server on a fixed loopback
port and races it against a pasted "manual code" prompt for headless/remote
sessions. This port keeps the same two-path shape (`login()` races the local
callback server against `interaction.prompt()`) but the callback server is
the shared :class:`~pi_ai.auth.oauth.oauth_page.OAuthCallbackServer` instead
of a bespoke one, and its port is injectable for tests.
"""

from __future__ import annotations

import asyncio
import base64
import time
import webbrowser
from collections.abc import Callable
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from ...utils.abort import AbortController, AbortSignal
from ..types import AuthEvent, AuthInteraction, AuthPrompt, Credential, OAuthAuth, ResolvedAuth
from .oauth_page import CallbackResult, OAuthCallbackServer, oauth_error_html
from .pkce import generate_pkce

CLIENT_ID = base64.b64decode("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl").decode("ascii")
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 53692
CALLBACK_PATH = "/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"
SCOPES = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
# Refresh slightly before the reported expiry to avoid using a token that dies mid-request.
REFRESH_SKEW_MS = 5 * 60 * 1000
REQUEST_TIMEOUT_S = 30.0


def _parse_authorization_input(value: str) -> tuple[str | None, str | None]:
    value = value.strip()
    if not value:
        return None, None

    parsed = urlparse(value)
    if parsed.scheme and parsed.query:
        params = parse_qs(parsed.query)
        return params.get("code", [None])[0], params.get("state", [None])[0]

    if "#" in value:
        code, _, state = value.partition("#")
        return code, state

    if "code=" in value:
        params = parse_qs(value)
        return params.get("code", [None])[0], params.get("state", [None])[0]

    return value, None


async def _post_json(
    url: str, body: dict[str, str | int], client: httpx.AsyncClient | None = None
) -> dict[str, object]:
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
    try:
        response = await http_client.post(
            url, json=body, headers={"Content-Type": "application/json", "Accept": "application/json"}
        )
        text = response.text
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP request failed. status={response.status_code}; url={url}; body={text}")
        return response.json()
    finally:
        if owns_client:
            await http_client.aclose()


def _credential_from_token_response(data: dict[str, object]) -> Credential:
    return Credential(
        type="oauth",
        refresh=str(data["refresh_token"]),
        access=str(data["access_token"]),
        expires=time.time() * 1000 + float(data["expires_in"]) * 1000 - REFRESH_SKEW_MS,
    )


async def exchange_authorization_code(
    code: str,
    state: str,
    verifier: str,
    redirect_uri: str = REDIRECT_URI,
    client: httpx.AsyncClient | None = None,
) -> Credential:
    data = await _post_json(
        TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "state": state,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        },
        client,
    )
    return _credential_from_token_response(data)


async def refresh_anthropic_token(refresh_token: str, client: httpx.AsyncClient | None = None) -> Credential:
    data = await _post_json(
        TOKEN_URL,
        {"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": refresh_token},
        client,
    )
    return _credential_from_token_response(data)


def build_authorize_url(challenge: str, state: str, redirect_uri: str = REDIRECT_URI) -> str:
    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


async def login_anthropic(
    interaction: AuthInteraction,
    *,
    open_browser: bool = False,
    browser_opener: Callable[[str], object] = webbrowser.open,
    client: httpx.AsyncClient | None = None,
    callback_port: int = CALLBACK_PORT,
) -> Credential:
    """Run the interactive Anthropic OAuth login.

    Races the local callback server against `interaction.prompt()` for a
    pasted authorization code/URL, exactly like the TypeScript flow. The
    callback server binds to ``callback_port`` (injectable for tests; ``0``
    picks an ephemeral port, which real logins cannot use because Anthropic's
    registered redirect URI is the fixed ``CALLBACK_PORT``).
    """
    pkce = generate_pkce()

    def _validate_callback(result: CallbackResult) -> tuple[int, str] | None:
        # Matches the TypeScript callback server: a request with a bad/missing
        # code, state, or an `error` param gets a 400 and the server keeps
        # waiting for a valid one instead of settling on the first request.
        error = result.params.get("error")
        if error:
            return 400, oauth_error_html("Anthropic authentication did not complete.", f"Error: {error}")
        code = result.params.get("code")
        state = result.params.get("state")
        if not code or not state:
            return 400, oauth_error_html("Missing code or state parameter.")
        if state != pkce.verifier:
            return 400, oauth_error_html("State mismatch.")
        return None

    server = OAuthCallbackServer(CALLBACK_PATH, host=CALLBACK_HOST, port=callback_port, on_callback=_validate_callback)
    redirect_uri = f"http://localhost:{server.port}{CALLBACK_PATH}"
    auth_url = build_authorize_url(pkce.challenge, pkce.verifier, redirect_uri)
    # Handed to the manual_code prompt and aborted once login settles, so a UI
    # showing the paste field can dismiss it (TypeScript's `manualAbort`).
    manual_abort = AbortController()

    async def _cancel_wait_on_abort() -> None:
        await interaction.signal.wait()
        server.cancel()

    # TypeScript's `interaction.signal.addEventListener("abort", () => server.cancelWait())`.
    abort_watcher: asyncio.Task[None] = asyncio.ensure_future(_cancel_wait_on_abort())

    try:
        interaction.notify(
            AuthEvent(
                type="auth_url",
                url=auth_url,
                instructions=(
                    "Complete login in your browser. If the browser is on another machine, "
                    "paste the final redirect URL here."
                ),
            )
        )
        if open_browser:
            browser_opener(auth_url)

        manual_task: asyncio.Task[str] = asyncio.ensure_future(
            interaction.prompt(
                AuthPrompt(
                    type="manual_code",
                    message="Complete login in your browser, or paste the authorization code / redirect URL here:",
                    placeholder=redirect_uri,
                    signal=manual_abort.signal,
                )
            )
        )
        # Mirror the TypeScript `.then()/.catch()` that calls `cancelWait()` as soon as
        # the manual prompt settles: without this, `wait_for_callback()` below never
        # observes the manual answer and hangs forever when the browser can't reach
        # the loopback server (exactly the scenario manual entry exists for).
        manual_task.add_done_callback(lambda _task: server.cancel())

        callback = await server.wait_for_callback()
        code: str | None = None
        state: str | None = None
        if callback is not None:
            # `_validate_callback` above already guarantees code/state are present
            # and state matches before the server settles this future.
            code = callback.params.get("code")
            state = callback.params.get("state")
            manual_task.cancel()
        else:
            manual_input = await manual_task
            parsed_code, parsed_state = _parse_authorization_input(manual_input)
            if parsed_state and parsed_state != pkce.verifier:
                raise RuntimeError("OAuth state mismatch")
            code = parsed_code
            state = parsed_state or pkce.verifier

        if not code:
            raise RuntimeError("Missing authorization code")
        if not state:
            raise RuntimeError("Missing OAuth state")

        interaction.notify(AuthEvent(type="progress", message="Exchanging authorization code for tokens..."))
        return await exchange_authorization_code(code, state, pkce.verifier, redirect_uri, client)
    finally:
        abort_watcher.cancel()
        manual_abort.abort()
        server.close()


async def refresh(
    credential: Credential, signal: AbortSignal, *, client: httpx.AsyncClient | None = None
) -> Credential:
    if not credential.refresh:
        raise RuntimeError("Missing Anthropic OAuth refresh token")
    return await refresh_anthropic_token(credential.refresh, client)


async def to_auth(credential: Credential) -> ResolvedAuth:
    return ResolvedAuth(api_key=credential.access)


def build_anthropic_oauth() -> OAuthAuth:
    return OAuthAuth(
        name="Anthropic (Claude Pro/Max)",
        is_subscription=True,
        login=login_anthropic,
        refresh=refresh,
        to_auth=to_auth,
    )
