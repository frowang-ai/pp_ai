"""Local OAuth redirect/callback page.

Python port of `packages/ai/src/auth/oauth/oauth-page.ts` plus the loopback
callback server that every browser-based provider flow
(`anthropic.ts`/`openrouter.ts`/`radius.ts`) starts inline with `node:http`.
Rather than duplicating that server three times, this module exposes one
reusable :class:`OAuthCallbackServer` built on `http.server`, bound to
``127.0.0.1`` on an ephemeral port by default (the port is injectable so tests
never depend on a real network binding beyond loopback).
"""

from __future__ import annotations

import asyncio
import html
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800" aria-hidden="true">'
    '<path fill="#fff" fill-rule="evenodd" d="M165.29 165.29 H517.36 V400 H400 V517.36 H282.65 V634.72 H165.29 Z '
    'M282.65 282.65 V400 H400 V282.65 Z"/><path fill="#fff" d="M517.36 400 H634.72 V634.72 H517.36 Z"/></svg>'
)


def _render_page(title: str, heading: str, message: str, details: str | None = None) -> str:
    escaped_title = html.escape(title)
    escaped_heading = html.escape(heading)
    escaped_message = html.escape(message)
    details_html = f'<div class="details">{html.escape(details)}</div>' if details else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escaped_title}</title>
  <style>
    :root {{
      --text: #fafafa;
      --text-dim: #a1a1aa;
      --page-bg: #09090b;
    }}
    * {{ box-sizing: border-box; }}
    html {{ color-scheme: dark; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: var(--page-bg);
      color: var(--text);
      text-align: center;
    }}
    main {{ width: 100%; max-width: 560px; display: flex; flex-direction: column; align-items: center; }}
    .logo {{ width: 72px; height: 72px; display: block; margin-bottom: 24px; }}
    h1 {{ margin: 0 0 10px; font-size: 28px; line-height: 1.15; font-weight: 650; color: var(--text); }}
    p {{ margin: 0; line-height: 1.7; color: var(--text-dim); font-size: 15px; }}
    .details {{ margin-top: 16px; font-size: 13px; color: var(--text-dim); white-space: pre-wrap; word-break: break-word; }}
  </style>
</head>
<body>
  <main>
    <div class="logo">{_LOGO_SVG}</div>
    <h1>{escaped_heading}</h1>
    <p>{escaped_message}</p>
    {details_html}
  </main>
</body>
</html>"""


def oauth_success_html(message: str) -> str:
    return _render_page("Authentication successful", "Authentication successful", message)


def oauth_error_html(message: str, details: str | None = None) -> str:
    return _render_page("Authentication failed", "Authentication failed", message, details)


@dataclass
class CallbackResult:
    """Query parameters from a single-shot OAuth redirect callback."""

    path: str
    params: dict[str, str]


class OAuthCallbackServer:
    """A one-shot loopback HTTP server that waits for an OAuth redirect.

    Mirrors the inline `node:http` server every browser-based provider flow
    starts: bind to a loopback host, serve exactly one successful callback
    (or keep rejecting mismatched/invalid ones until it gets one, or is
    closed), and hand the result back to the caller as an ``asyncio.Future``.

    ``host``/``port`` are injectable so tests can pin ``port=0`` (ephemeral,
    the default) and read back the bound port via :attr:`port`.
    """

    def __init__(
        self,
        callback_path: str,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        on_callback: Callable[[CallbackResult], tuple[int, str] | tuple[int, str, bool] | None] | None = None,
    ) -> None:
        self._callback_path = callback_path
        self._on_callback = on_callback
        self._loop = asyncio.get_event_loop()
        self._future: asyncio.Future[CallbackResult | None] = self._loop.create_future()

        server_self = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass  # Never log request details; they may carry a code/state secret.

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                params = {key: values[0] for key, values in parse_qs(parsed.query).items()}
                if parsed.path != server_self._callback_path:
                    self._respond(404, oauth_error_html("Callback route not found."))
                    return

                result = CallbackResult(path=parsed.path, params=params)
                status, body = (200, oauth_success_html("Authentication completed. You can close this window."))
                settle = True
                if server_self._on_callback is not None:
                    override = server_self._on_callback(result)
                    if override is not None:
                        if len(override) == 3:
                            status, body, settle = override
                        else:
                            status, body = override
                            settle = status == 200
                self._respond(status, body)
                if settle:
                    server_self._loop.call_soon_threadsafe(server_self._settle, result)

            def _respond(self, status: int, body: str) -> None:
                encoded = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self._httpd = HTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def host(self) -> str:
        return self._httpd.server_address[0]

    def _settle(self, result: CallbackResult | None) -> None:
        if not self._future.done():
            self._future.set_result(result)

    def cancel(self) -> None:
        """Stop waiting for a callback without an error (a manual code won instead)."""
        self._loop.call_soon_threadsafe(self._settle, None)

    async def wait_for_callback(self) -> CallbackResult | None:
        return await self._future

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
