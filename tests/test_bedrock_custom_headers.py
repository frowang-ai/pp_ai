"""Python port of `packages/ai/test/bedrock-custom-headers.test.ts` — not portable.

`packages/ai/src/api/bedrock-converse-stream.ts` is a documented omission of
this port (see the repository README and
:mod:`pi_ai.api.bedrock_converse_stream`).

Beyond the missing adapter, this file is specifically about the AWS SDK's
middleware stack: it asserts that a `build`-step middleware named
`pi-ai-custom-headers` (the `MIDDLEWARE_NAME` constant in the TS test, matching
the `name: "pi-ai-custom-headers"` passed to `client.middlewareStack.add` in
`addCustomHeadersMiddleware`) is registered on `client.middlewareStack`, and
inspects the middleware function by invoking it with a synthetic Smithy
request object. There is no middleware stack in `pi-ai`, whose requests are
plain `httpx` calls, so the mechanism itself has no Python analogue.

Behaviors left uncovered by the Python port:

- a build-step middleware named `pi-ai-custom-headers` with `priority: "low"`
  injects caller-supplied headers into the signed request, calling `next`
  exactly once with the same args object (VC1);
- reserved headers (`host`, `authorization`, `x-amz-*`, matched
  case-insensitively per `isReservedHeader`) are skipped even when the caller
  supplies mixed-case variants, while non-reserved headers (`x-allowed`) are
  still applied and the request's pre-existing header keys are otherwise
  unchanged (VC2);
- no middleware is registered when `headers` is absent or an empty object
  (VC3, two sub-cases);
- the middleware handler passes both a request object with no `headers`
  property and a request of `undefined` through unchanged (calling `next`
  with the identical args object) rather than throwing (VC3 structural
  guard);
- `streamSimple` forwards `headers` end to end to the same middleware
  registration and injection behavior as `stream` (VC4 regression guard).
"""

from __future__ import annotations

import pytest

_REASON = "bedrock-converse-stream is a documented omission of this port, and this case additionally asserts on the AWS SDK's middleware stack (a build-step middleware invoked with a synthetic Smithy request). pi-ai issues plain httpx requests and has no middleware stack, so the mechanism itself has no Python analogue."


@pytest.mark.skip(reason=_REASON)
def test_vc1_registers_a_build_step_middleware_that_injects_the_caller_header_happy_path() -> None:
    """`it("VC1: registers a build-step middleware that injects the caller header (happy path)")`:
    with `headers: { "x-custom": "v" }`, exactly one middleware registration named
    `pi-ai-custom-headers` is found, with `opts.step === "build"` and `opts.priority ===
    "low"`. Invoking `reg.handler(nextSpy)({ request: { headers: {} } })` sets
    `request.headers["x-custom"] === "v"` and calls `nextSpy` exactly once with the
    unmodified args object.
    """


@pytest.mark.skip(reason=_REASON)
def test_vc2_skips_reserved_headers_case_insensitively_while_applying_allowed_ones() -> None:
    """`it("VC2: skips reserved headers case-insensitively while applying allowed ones")`: caller
    headers include `authorization`, `x-amz-date`, `x-allowed`, and mixed-case duplicates
    `Authorization`, `X-Amz-Date`, `HOST` (all with value `"evil"`/`"evil2"`/`"evil3"`).
    Invoking the middleware against a fake request already carrying `authorization:
    "real-auth"`, `x-amz-date: "real-date"`, `host: "real-host"` asserts those three
    existing values are left untouched, `x-allowed` becomes `"ok"`, no mixed-case reserved
    key (`Authorization`, `X-Amz-Date`, `HOST`) leaks in as a *new* key, the final header
    key set is exactly `{authorization, host, x-allowed, x-amz-date}`, and `nextSpy` is
    called exactly once.
    """


@pytest.mark.skip(reason=_REASON)
def test_vc3_registers_no_middleware_when_headers_is_undefined() -> None:
    """`it("VC3: registers no middleware when headers is undefined")`: with no `headers` option
    at all, `middlewareStack` carries zero registrations named `pi-ai-custom-headers`.
    """


@pytest.mark.skip(reason=_REASON)
def test_vc3_registers_no_middleware_when_headers_is_empty() -> None:
    """`it("VC3: registers no middleware when headers is empty")`: with `headers: {}` (present
    but empty), `middlewareStack` still carries zero registrations named
    `pi-ai-custom-headers`.
    """


@pytest.mark.skip(reason=_REASON)
def test_vc3_structural_guard_passes_through_unchanged_when_the_request_has_no_headers() -> None:
    """`it("VC3 (structural guard): passes through unchanged when the request has no headers")`:
    with `headers: { "x-custom": "v" }` registering the middleware, invoking its handler on
    (1) `{ request: {} }` (no `headers` key on `request`) resolves without throwing and
    calls `nextSpy` with that exact args object, and (2) `{ request: undefined }` likewise
    resolves without throwing and calls `nextSpy` with that exact args object -- `nextSpy`
    is called exactly twice in total, once per sub-case.
    """


@pytest.mark.skip(reason=_REASON)
def test_vc4_streamsimplebedrock_forwards_headers_end_to_end_regression_guard() -> None:
    """`it("VC4: streamSimpleBedrock forwards headers end-to-end (regression guard)")`: driving
    `streamSimple` (not `stream`) with `{ headers: { "x-custom": "v" } }` still registers
    exactly one `pi-ai-custom-headers` middleware with `opts.step === "build"`, and
    invoking its handler against `{ request: { headers: {} } }` sets
    `request.headers["x-custom"] === "v"` -- confirming `streamSimple` wires headers
    through the same code path as `stream`, not just the `stream` code path tested above.
    """
