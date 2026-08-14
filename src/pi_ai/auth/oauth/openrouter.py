"""OpenRouter OAuth PKCE flow.

Python port of `packages/ai/src/auth/oauth/openrouter.ts`.

OpenRouter exchanges an authorization code for a permanent, user-controlled
API key rather than an expiring access/refresh token pair, so `refresh()` is
a no-op that returns the stored credential unchanged. The callback is handled
by the shared :class:`~pi_ai.auth.oauth.oauth_page.OAuthCallbackServer` on an
ephemeral port, raced against a manual prompt exactly like the TypeScript
one-shot loopback server.

Unlike `anthropic.py`/`radius.py`, this flow runs the token exchange *inside*
the callback request handler (as the TypeScript original does), so the browser
page itself reports the exchange result: 200 on success, 502 when the exchange
fails, 409 for a second request against an already-claimed callback. That
cannot be expressed with the shared
:class:`~pi_ai.auth.oauth.oauth_page.OAuthCallbackServer`, whose `on_callback`
hook is synchronous and one-shot, so this module keeps its own server exactly
like `openrouter.ts` does.
"""

from __future__ import annotations

import asyncio
import math
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from ...utils.abort import AbortController, AbortSignal
from ...utils.provider_env import get_provider_env_value
from ..types import AuthEvent, AuthInteraction, AuthPrompt, Credential, OAuthAuth, ResolvedAuth
from .oauth_page import oauth_error_html, oauth_success_html
from .pkce import generate_pkce

AUTHORIZE_URL = "https://openrouter.ai/auth"
TOKEN_URL = "https://openrouter.ai/api/v1/auth/keys"
TOKEN_EXCHANGE_TIMEOUT_S = 30.0
LOGIN_TIMEOUT_S = 5 * 60.0


def get_callback_host() -> str:
    """Loopback host the callback server binds to (`PI_OAUTH_CALLBACK_HOST`)."""
    return get_provider_env_value("PI_OAUTH_CALLBACK_HOST") or "127.0.0.1"


def _parse_authorization_input(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme and parsed.query:
        return parse_qs(parsed.query).get("code", [None])[0]
    if "code=" in value:
        return parse_qs(value).get("code", [None])[0]
    return value


def _error_detail(body: dict[str, object]) -> str | None:
    if isinstance(body.get("error_description"), str):
        return body["error_description"]
    if isinstance(body.get("message"), str):
        return body["message"]
    if isinstance(body.get("error"), str):
        return body["error"]
    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return None


async def exchange_authorization_code(
    code: str,
    verifier: str,
    client: httpx.AsyncClient | None = None,
    signal: AbortSignal | None = None,
) -> Credential:
    if signal is not None and signal.aborted:
        raise RuntimeError("Login cancelled")
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=TOKEN_EXCHANGE_TIMEOUT_S)
    try:
        response = await http_client.post(
            TOKEN_URL,
            headers={"accept": "application/json", "content-type": "application/json"},
            json={"code": code, "code_verifier": verifier, "code_challenge_method": "S256"},
        )
        try:
            body = response.json()
            if not isinstance(body, dict):
                body = {}
        except ValueError:
            body = {}

        if response.status_code >= 400:
            detail = _error_detail(body)
            raise RuntimeError(
                f"OpenRouter OAuth key exchange failed (HTTP {response.status_code})"
                + (f": {detail}" if detail else "")
            )
        key = body.get("key")
        if not isinstance(key, str) or not key:
            raise RuntimeError('OpenRouter OAuth response carries no "key"')
        return Credential(type="oauth", access=key, refresh="", expires=math.inf)
    finally:
        if owns_client:
            await http_client.aclose()


def build_authorize_url(callback_url: str, challenge: str) -> str:
    params = {"callback_url": callback_url, "code_challenge": challenge, "code_challenge_method": "S256"}
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


class OpenRouterCallbackServer:
    """One-shot loopback server that exchanges the callback code itself.

    Mirrors `startCallbackServer` in `openrouter.ts`: the browser request that
    carries the authorization code is held open while the key exchange runs, so
    the page reports the real outcome (200 / 502) and a second request against
    an already-claimed callback gets 409 instead of starting a second exchange.
    """

    def __init__(
        self,
        callback_path: str,
        verifier: str,
        signal: AbortSignal,
        *,
        client: httpx.AsyncClient | None = None,
        port: int = 0,
    ) -> None:
        if signal.aborted:
            raise RuntimeError("Login cancelled")
        self._loop = asyncio.get_running_loop()
        self._future: asyncio.Future[Credential | None] = self._loop.create_future()
        self._callback_path = callback_path
        self._verifier = verifier
        self._client = client
        self._signal = signal
        self._claimed = False
        self._closed = False
        server_self = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass  # Never log request details; they may carry a code secret.

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                params = {key: values[0] for key, values in parse_qs(parsed.query).items()}
                if parsed.path != server_self._callback_path:
                    self._respond(404, oauth_error_html("OAuth callback route not found."))
                    return
                if server_self._claimed or server_self._future.done():
                    self._respond(409, oauth_error_html("This OAuth callback has already been used."))
                    return

                error = params.get("error")
                if error:
                    description = params.get("error_description") or error
                    self._respond(400, oauth_error_html("OpenRouter authorization was denied.", description))
                    server_self._fail(RuntimeError(f"OpenRouter authorization failed: {description}"))
                    return

                code = params.get("code")
                if not code:
                    self._respond(400, oauth_error_html("OpenRouter returned no authorization code."))
                    return
                server_self._claimed = True

                try:
                    credential = asyncio.run_coroutine_threadsafe(
                        exchange_authorization_code(
                            code, server_self._verifier, server_self._client, server_self._signal
                        ),
                        server_self._loop,
                    ).result()
                except BaseException as error:
                    message = str(error) or "Unknown token exchange error"
                    self._respond(502, oauth_error_html("OpenRouter key exchange failed.", message))
                    server_self._fail(error if isinstance(error, Exception) else RuntimeError(message))
                    return
                self._respond(200, oauth_success_html("Signed in to OpenRouter. You may now close this page."))
                server_self._settle(credential)

            def _respond(self, status: int, body: str) -> None:
                encoded = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        host = get_callback_host()
        # `daemon_threads` keeps `close()` from joining a request thread that is
        # still blocked on the exchange running on this event loop.
        server_class = type("_OpenRouterHTTPServer", (ThreadingHTTPServer,), {"daemon_threads": True})
        self._httpd = server_class((host, port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        self.callback_url = f"http://{host}:{self._httpd.server_address[1]}{callback_path}"

        self._timeout = self._loop.call_later(
            LOGIN_TIMEOUT_S, lambda: self._fail(RuntimeError("OpenRouter OAuth login timed out"))
        )
        self._abort_watcher: asyncio.Task[None] = asyncio.ensure_future(self._fail_on_abort())

    async def _fail_on_abort(self) -> None:
        await self._signal.wait()
        self._fail(RuntimeError("Login cancelled"))

    def _settle(self, credential: Credential | None) -> None:
        self._loop.call_soon_threadsafe(self._settle_now, credential)

    def _settle_now(self, credential: Credential | None) -> None:
        if not self._future.done():
            self._future.set_result(credential)

    def _fail(self, error: BaseException) -> None:
        self._loop.call_soon_threadsafe(self._fail_now, error)

    def _fail_now(self, error: BaseException) -> None:
        if not self._future.done():
            self._future.set_exception(error)

    def cancel_wait(self) -> None:
        """Hand the login to manual entry unless a callback already claimed it."""
        if not self._claimed:
            self._settle_now(None)

    async def wait_for_credential(self) -> Credential | None:
        return await self._future

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._timeout.cancel()
        self._abort_watcher.cancel()
        if not self._future.done():
            self._future.cancel()
        self._httpd.shutdown()
        self._httpd.server_close()


async def login_openrouter(
    interaction: AuthInteraction,
    *,
    client: httpx.AsyncClient | None = None,
    callback_port: int = 0,
) -> Credential:
    pkce = generate_pkce()
    callback_path = f"/oauth/callback/{uuid.uuid4()}"
    server = OpenRouterCallbackServer(
        callback_path, pkce.verifier, interaction.signal, client=client, port=callback_port
    )
    # Handed to the manual_code prompt and aborted once login settles, so a UI
    # showing the paste field can dismiss it (TypeScript's `manualAbort`).
    manual_abort = AbortController()
    manual_input: str | None = None
    manual_error: BaseException | None = None
    manual_task: asyncio.Task[str] | None = None

    try:
        authorize_url = build_authorize_url(server.callback_url, pkce.challenge)
        interaction.notify(
            AuthEvent(type="progress", message=f"Listening for OpenRouter OAuth callback on {server.callback_url}")
        )
        interaction.notify(
            AuthEvent(
                type="auth_url",
                url=authorize_url,
                instructions=(
                    "Complete sign-in in your browser. If the browser is on another machine, "
                    "paste the final redirect URL here."
                ),
            )
        )

        manual_task = asyncio.ensure_future(
            interaction.prompt(
                AuthPrompt(
                    type="manual_code",
                    message="Complete sign-in in your browser, or paste the authorization code / redirect URL here:",
                    placeholder=server.callback_url,
                    signal=manual_abort.signal,
                )
            )
        )

        def _on_manual_settled(task: asyncio.Task[str]) -> None:
            # Mirrors the TypeScript `.then()/.catch()` that calls `cancelWait()` as
            # soon as the manual prompt settles: without this, the callback wait
            # below never observes the manual answer and hangs forever when the
            # browser cannot reach the loopback server.
            nonlocal manual_input, manual_error
            if task.cancelled():
                return
            error = task.exception()
            if error is not None:
                manual_error = error
            else:
                manual_input = task.result()
            server.cancel_wait()

        manual_task.add_done_callback(_on_manual_settled)

        credential = await server.wait_for_credential()
        if manual_error is not None:
            raise manual_error
        if credential is not None:
            return credential

        if manual_error is None and not manual_task.done():
            await asyncio.wait([manual_task])
        if manual_error is not None:
            raise manual_error
        code = _parse_authorization_input(manual_input) if manual_input else None
        if not code:
            raise RuntimeError("Missing authorization code")
        interaction.notify(AuthEvent(type="progress", message="Exchanging authorization code for an API key..."))
        return await exchange_authorization_code(code, pkce.verifier, client, interaction.signal)
    finally:
        manual_abort.abort()
        if manual_task is not None and not manual_task.done():
            manual_task.cancel()
        server.close()


async def refresh(credential: Credential, signal: AbortSignal) -> Credential:
    return credential


async def to_auth(credential: Credential) -> ResolvedAuth:
    return ResolvedAuth(api_key=credential.access)


def build_openrouter_oauth() -> OAuthAuth:
    return OAuthAuth(
        name="OpenRouter OAuth",
        login_label="Sign in with OpenRouter",
        login=login_openrouter,
        refresh=refresh,
        to_auth=to_auth,
    )
