"""Transport tests for pi_ai.utils.http.

These run against a real HTTP server on a loopback socket rather than a mock
transport, so the SSE decoding, chunk boundaries and error paths are exercised
end to end.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest
from pi_ai.utils.http import (
    HttpRequest,
    ProviderHttpError,
    SseEvent,
    build_timeout,
    stream_sse,
)


class _Server:
    """A minimal HTTP/1.1 server that replays a canned response."""

    def __init__(self, response: bytes, delay: float = 0.0) -> None:
        self.response = response
        self.delay = delay
        self.requests: list[bytes] = []
        self._server: asyncio.AbstractServer | None = None
        self.port = 0

    async def __aenter__(self) -> _Server:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        header = await reader.readuntil(b"\r\n\r\n")
        length = 0
        for line in header.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":")[1])
        body = await reader.readexactly(length) if length else b""
        self.requests.append(header + body)

        if self.delay:
            await asyncio.sleep(self.delay)
        writer.write(self.response)
        await writer.drain()
        writer.close()


def sse_response(chunks: list[str], status: int = 200) -> bytes:
    body = "".join(chunks).encode()
    reason = {200: "OK", 401: "Unauthorized", 500: "Internal Server Error"}.get(status, "Status")
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        "content-type: text/event-stream\r\n"
        f"content-length: {len(body)}\r\n"
        "x-request-id: req-42\r\n"
        "\r\n"
    ).encode()
    return head + body


async def collect(request: HttpRequest, **kwargs) -> list[SseEvent]:
    return [event async for event in stream_sse(request, **kwargs)]


async def test_decodes_multiple_sse_events_over_a_real_socket():
    chunks = [
        'data: {"n": 1}\n\n',
        'data: {"n": 2}\n\n',
        "data: [DONE]\n\n",
    ]
    async with _Server(sse_response(chunks)) as server:
        events = await collect(HttpRequest(url=server.url, json_body={"hello": "world"}))

    assert [event.data for event in events] == ['{"n": 1}', '{"n": 2}', "[DONE]"]
    assert json.loads(events[0].data) == {"n": 1}


async def test_sends_the_json_body_and_headers():
    async with _Server(sse_response(["data: x\n\n"])) as server:
        await collect(
            HttpRequest(
                url=server.url,
                headers={"authorization": "Bearer secret", "x-custom": "1"},
                json_body={"model": "m"},
            )
        )
        raw = server.requests[0].decode()

    assert "POST / HTTP/1.1" in raw
    assert "authorization: Bearer secret" in raw
    assert "x-custom: 1" in raw
    assert '{"model":"m"}' in raw


async def test_parses_named_events_and_ids():
    chunks = ["event: message_start\nid: 7\ndata: {}\n\n", "event: ping\ndata: {}\n\n"]
    async with _Server(sse_response(chunks)) as server:
        events = await collect(HttpRequest(url=server.url))

    assert events[0].event == "message_start"
    assert events[0].id == "7"
    assert events[1].event == "ping"
    assert events[1].id is None


async def test_joins_multi_line_data_fields():
    async with _Server(sse_response(["data: line one\ndata: line two\n\n"])) as server:
        events = await collect(HttpRequest(url=server.url))
    assert events[0].data == "line one\nline two"


async def test_ignores_comment_lines():
    async with _Server(sse_response([": keep-alive\n\n", "data: real\n\n"])) as server:
        events = await collect(HttpRequest(url=server.url))
    assert [event.data for event in events] == ["real"]


async def test_emits_a_trailing_event_without_a_blank_line():
    async with _Server(sse_response(["data: last"])) as server:
        events = await collect(HttpRequest(url=server.url))
    assert [event.data for event in events] == ["last"]


async def test_strips_only_one_leading_space_after_the_colon():
    async with _Server(sse_response(["data:  two spaces\n\n"])) as server:
        events = await collect(HttpRequest(url=server.url))
    assert events[0].data == " two spaces"


async def test_raises_provider_http_error_before_yielding_on_error_status():
    body = '{"error": {"message": "bad key", "type": "auth"}}'
    async with _Server(sse_response([body], status=401)) as server:
        with pytest.raises(ProviderHttpError) as excinfo:
            await collect(HttpRequest(url=server.url))

    error = excinfo.value
    assert error.status == 401
    assert error.status_code == 401
    assert "bad key" in error.body
    assert error.error == {"message": "bad key", "type": "auth"}
    assert error.headers["x-request-id"] == "req-42"


async def test_provider_http_error_without_a_json_body():
    async with _Server(sse_response(["upstream exploded"], status=500)) as server:
        with pytest.raises(ProviderHttpError) as excinfo:
            await collect(HttpRequest(url=server.url))
    assert excinfo.value.error is None
    assert "upstream exploded" in str(excinfo.value)


async def test_on_response_hook_receives_status_and_headers():
    seen: list = []

    async def on_response(response):
        seen.append((response.status, response.headers.get("x-request-id")))

    async with _Server(sse_response(["data: x\n\n"])) as server:
        await collect(HttpRequest(url=server.url), on_response=on_response)

    assert seen == [(200, "req-42")]


async def test_on_response_is_not_called_for_an_error_status():
    seen: list = []

    async def on_response(response):
        seen.append(response.status)

    async with _Server(sse_response(["nope"], status=500)) as server:
        with pytest.raises(ProviderHttpError):
            await collect(HttpRequest(url=server.url), on_response=on_response)

    assert seen == []


async def test_reuses_an_injected_client_without_closing_it():
    async with _Server(sse_response(["data: x\n\n", "data: y\n\n"])) as server:
        client = httpx.AsyncClient()
        try:
            first = await collect(HttpRequest(url=server.url), client=client)
            second = await collect(HttpRequest(url=server.url), client=client)
            assert not client.is_closed
        finally:
            await client.aclose()

    assert [e.data for e in first] == ["x", "y"]
    assert [e.data for e in second] == ["x", "y"]


async def test_timeout_is_honoured():
    async with _Server(sse_response(["data: x\n\n"]), delay=0.5) as server:
        with pytest.raises(httpx.TimeoutException):
            await collect(HttpRequest(url=server.url, timeout_ms=50))


def test_build_timeout_defaults_and_caps_connect():
    default = build_timeout(None)
    assert default.read == 600.0
    assert default.connect == 30.0

    short = build_timeout(5_000)
    assert short.read == 5.0
    assert short.connect == 5.0


async def test_streaming_is_incremental_not_buffered():
    """Events must be delivered as they arrive, not only after the body ends."""
    received: list[str] = []
    started = asyncio.Event()

    class SlowServer(_Server):
        async def _handle(self, reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
            await writer.drain()
            for payload in (b"data: first\n\n", b"data: second\n\n"):
                writer.write(hex(len(payload))[2:].encode() + b"\r\n" + payload + b"\r\n")
                await writer.drain()
                await started.wait()
                started.clear()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
            writer.close()

    async def consume(server: SlowServer) -> None:
        async for event in stream_sse(HttpRequest(url=server.url)):
            received.append(event.data)
            started.set()

    async with SlowServer(b"") as server:
        await asyncio.wait_for(consume(server), timeout=5)

    assert received == ["first", "second"]


async def _drain(iterator: AsyncIterator[SseEvent]) -> list[SseEvent]:
    return [event async for event in iterator]
