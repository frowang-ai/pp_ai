"""HTTP transport for provider APIs.

The TypeScript package uses each vendor's official SDK. The Python port talks
to the OpenAI-compatible and Anthropic HTTP APIs directly with ``httpx``, so
this module owns the shared pieces: request execution and Server-Sent Events
decoding.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from .abort import AbortSignal
from .node_http_proxy import resolve_http_proxy_url_for_target
from .provider_retry import ProviderRetryOptions, retry_provider_request

DEFAULT_TIMEOUT_MS = 600_000

# TypeScript passes `AbortSignal.timeout(timeoutMs)` to `fetch`, which rejects
# with "The operation was aborted due to timeout". `httpx` raises
# `TimeoutException` with an *empty* message, and that text is both surfaced to
# the user as `errorMessage` and matched by the retry classifier, so normalize
# it to the TypeScript wording.
TIMEOUT_ERROR_MESSAGE = "The operation was aborted due to timeout"

# Idle (between-reads) timeout applied to every provider request, mirroring the
# undici `bodyTimeout`/`headersTimeout` the TypeScript CLI installs globally via
# `core/http-dispatcher.ts`. `None` means "leave httpx defaults alone";
# `0` disables the idle timeout entirely.
_idle_timeout_ms: int | None = None


def set_idle_timeout_ms(timeout_ms: int | None) -> None:
    """Install the global idle timeout used by :func:`build_timeout`."""
    global _idle_timeout_ms
    _idle_timeout_ms = timeout_ms


def get_idle_timeout_ms() -> int | None:
    return _idle_timeout_ms


@dataclass
class SseEvent:
    """One decoded ``text/event-stream`` message."""

    data: str
    event: str | None = None
    id: str | None = None


class ProviderHttpError(Exception):
    """Non-2xx response from a provider endpoint."""

    def __init__(self, status: int, body: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"{status}: {body}" if body else f"{status} status code (no body)")
        self.status = status
        self.status_code = status
        self.body = body
        self.headers = headers or {}
        self.error: Any = None
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            self.error = parsed.get("error", parsed)


@dataclass
class HttpRequest:
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    json_body: dict[str, Any] = field(default_factory=dict)
    timeout_ms: int | None = None
    method: str = "POST"
    proxy_env: dict[str, str] | None = None
    """Provider-scoped env overrides consulted before `os.environ` for proxy vars."""
    signal: AbortSignal | None = None
    """Cancels an in-flight request or stream, like the `signal` TypeScript passes to `fetch`."""


def build_timeout(timeout_ms: int | None) -> httpx.Timeout:
    seconds = (timeout_ms if timeout_ms is not None else DEFAULT_TIMEOUT_MS) / 1000
    read = seconds
    if _idle_timeout_ms is not None:
        read = None if _idle_timeout_ms == 0 else _idle_timeout_ms / 1000
    return httpx.Timeout(seconds, connect=min(seconds, 30.0), read=read)


def build_client(request: HttpRequest) -> httpx.AsyncClient:
    """An `httpx` client for ``request``, honoring the proxy environment.

    TypeScript installs a proxy-aware undici dispatcher globally; the same
    `http_proxy`/`https_proxy`/`no_proxy` resolution lives in
    :mod:`pi_ai.utils.node_http_proxy` and is applied per client here.
    """
    return httpx.AsyncClient(
        timeout=build_timeout(request.timeout_ms),
        proxy=resolve_http_proxy_url_for_target(request.url, request.proxy_env),
    )


async def stream_sse(
    request: HttpRequest,
    client: httpx.AsyncClient | None = None,
    on_response: Any = None,
) -> AsyncIterator[SseEvent]:
    """POST ``request`` and yield decoded SSE events.

    Raises :class:`ProviderHttpError` before yielding anything when the response
    status is not 2xx, so callers see provider errors as exceptions rather than
    as an empty stream. ``request.signal``, when set, interrupts the request
    while it is waiting for the next chunk, the way TypeScript's `fetch(url,
    { signal })` tears the connection down.
    """
    owns_client = client is None
    http_client = client or build_client(request)
    try:
        async with http_client.stream(
            request.method,
            request.url,
            headers=request.headers,
            json=request.json_body,
            timeout=build_timeout(request.timeout_ms),
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise ProviderHttpError(response.status_code, body, dict(response.headers))
            if on_response is not None:
                await _call_on_response(on_response, response)
            async for event in _abortable(_decode_sse(response.aiter_lines()), request.signal):
                yield event
    except httpx.TimeoutException as error:
        raise _named_timeout_error(error) from error
    finally:
        if owns_client:
            await http_client.aclose()


def _named_timeout_error(error: httpx.TimeoutException) -> httpx.TimeoutException:
    """Give an `httpx` timeout the message TypeScript's `AbortSignal.timeout()` produces.

    The exception type is preserved so existing `httpx.TimeoutException`
    handlers keep working; only the (empty) message is filled in.
    """
    if str(error):
        return error
    try:
        request = error.request
    except RuntimeError:
        request = None
    return type(error)(TIMEOUT_ERROR_MESSAGE, request=request)


async def _abortable(iterator: AsyncIterator[Any], signal: AbortSignal | None) -> AsyncIterator[Any]:
    """Yield from ``iterator``, raising as soon as ``signal`` aborts.

    Without this a stream that never produces another chunk keeps the adapter
    waiting forever after an abort, whereas TypeScript's fetch rejects at once.
    """
    if signal is None:
        async for item in iterator:
            yield item
        return

    signal.throw_if_aborted()
    abort_task = asyncio.ensure_future(signal.wait())
    try:
        while True:
            next_task = asyncio.ensure_future(iterator.__anext__())
            done, _pending = await asyncio.wait({next_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
            if next_task not in done:
                next_task.cancel()
                signal.throw_if_aborted()
                continue
            try:
                yield next_task.result()
            except StopAsyncIteration:
                return
    finally:
        if not abort_task.done():
            abort_task.cancel()


async def stream_sse_with_retry(
    request: HttpRequest,
    client: httpx.AsyncClient | None = None,
    on_response: Any = None,
    retry: ProviderRetryOptions | None = None,
) -> AsyncIterator[SseEvent]:
    """:func:`stream_sse`, retrying retryable provider errors before the first event.

    TypeScript wraps the provider SDK call that resolves once response headers
    have arrived (`client.chat.completions.create(...).withResponse()`) in
    `retryProviderRequest`, so a 429/5xx status is retried but a mid-stream
    failure is not. `stream_sse` raises :class:`ProviderHttpError` before it
    yields anything, so pulling the first event inside the retried coroutine
    reproduces exactly that boundary.
    """
    if retry is None or retry.max_retries <= 0:
        async for event in stream_sse(request, client=client, on_response=on_response):
            yield event
        return

    async def open_stream() -> tuple[AsyncIterator[SseEvent], list[SseEvent]]:
        iterator = stream_sse(request, client=client, on_response=on_response).__aiter__()
        try:
            return iterator, [await iterator.__anext__()]
        except StopAsyncIteration:
            return iterator, []

    iterator, buffered = await retry_provider_request(open_stream, retry)
    for event in buffered:
        yield event
    async for event in iterator:
        yield event


async def _call_on_response(on_response: Any, response: httpx.Response) -> None:
    from ..types import ProviderResponse

    result = on_response(ProviderResponse(status=response.status_code, headers=dict(response.headers)))
    if hasattr(result, "__await__"):
        await result


async def _decode_sse(lines: AsyncIterator[str]) -> AsyncIterator[SseEvent]:
    data_lines: list[str] = []
    event_name: str | None = None
    event_id: str | None = None

    async for raw_line in lines:
        line = raw_line.rstrip("\r")
        if line == "":
            if data_lines:
                yield SseEvent(data="\n".join(data_lines), event=event_name, id=event_id)
            data_lines = []
            event_name = None
            event_id = None
            continue
        if line.startswith(":"):
            continue
        field_name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field_name == "data":
            data_lines.append(value)
        elif field_name == "event":
            event_name = value
        elif field_name == "id":
            event_id = value

    if data_lines:
        yield SseEvent(data="\n".join(data_lines), event=event_name, id=event_id)
