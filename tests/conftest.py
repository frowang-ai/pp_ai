"""Shared fixtures for the `pi_ai` test package."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pi_ai.utils import http as pi_http


@pytest.fixture(autouse=True)
def reset_http_idle_timeout() -> Iterator[None]:
    """Run every `pi_ai` test with no global HTTP idle-timeout override installed.

    `set_idle_timeout_ms` mirrors TypeScript's process-wide undici dispatcher, so
    it is module state shared by everything running in the same process. Under
    `pytest-xdist` a worker can run `pi-coding-agent` tests (which install an idle
    timeout through `configure_http_dispatcher`) in the same process as `pi-ai`
    tests, and any leak makes `build_timeout` return the leaked read timeout
    instead of the caller's `timeout_ms`. That is what intermittently broke
    `test_mistral_http_transport.py::test_applies_the_request_timeout_while_waiting_for_an_sse_chunk`
    (it saw `read == 300.0` rather than `0.005`). Pinning the precondition here
    keeps timeout assertions independent of test ordering.
    """
    previous = pi_http.get_idle_timeout_ms()
    pi_http.set_idle_timeout_ms(None)
    try:
        yield
    finally:
        pi_http.set_idle_timeout_ms(previous)
