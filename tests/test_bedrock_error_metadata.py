"""Python port of `packages/ai/test/bedrock-error-metadata.test.ts` — not portable.

`packages/ai/src/api/bedrock-converse-stream.ts` is a documented omission of
this port (see the repository README and
:mod:`pi_ai.api.bedrock_converse_stream`), so there is no Python code path that
produces these diagnostics.

Every case here is expressed in terms of AWS SDK error shapes that only exist
in the JavaScript SDK: `BedrockRuntimeServiceException` instances carrying
`$metadata.httpStatusCode`/`$metadata.requestId`, the SDK's `"Unknown"`
placeholder for a missing `x-amzn-errortype` response header, and mid-stream
errors that surface either a bare object literal or an `Error` renamed after
the event frame's `:error-code`.

All cases pin the `bedrock_response_failure` diagnostic appended by
`appendBedrockFailureDiagnostic` in `bedrock-converse-stream.ts` (the TS test's
own `DIAGNOSTIC_TYPE` constant is `"bedrock_response_failure"`, not a generic
`provider_request_failed` label). That diagnostic's `details` field is built
from at most three keys -- `status` (from `$metadata.httpStatusCode`),
`errorCode` (only for `Error`s whose `.name` ends in `"Exception"`, via
`extractBedrockErrorCode`), and `requestId` (from `$metadata.requestId` or the
stream's fallback request id) -- each independently omitted, never guessed,
when unavailable, over 200 characters, or blank after trimming
(`normalizeDiagnosticValue`).

Behaviors left uncovered by the Python port (all about that diagnostic):

- a non-2xx from `client.send()` records `status`, `errorCode` and
  `requestId`, the diagnostic carries no `error` field, and its own key set is
  exactly `{details, timestamp, type}`;
- `errorMessage` is left untouched (still `"Validation error: <message>"`) so
  retry classification is unaffected by the diagnostic changes;
- a modeled mid-stream exception thrown as a bare object literal (no `.name`)
  reports only `requestId` (no `errorCode` is available, since a bare object
  is not an `Error` and has no `.name` at all);
- a mid-stream `Error` whose `.name` ends in `"Exception"` (an unmodeled AWS
  error frame) reports that name as `errorCode` plus `requestId`;
- a mid-stream `Error` whose `.name` does NOT end in `"Exception"`
  (`"TimeoutError"`, a transport failure name) is never surfaced as
  `errorCode` -- only `requestId` is reported;
- when the rejection carries no `$metadata` and no `.name` ending in
  `"Exception"` at all (`errorMessage: "socket hang up"`), no diagnostic is
  appended, `stopReason` is still `"error"`, and `errorMessage` equals the raw
  message;
- when the turn is aborted before the send rejects, `stopReason` is
  `"aborted"` and no diagnostic is appended even though the underlying error
  otherwise carries full status/errorCode/requestId metadata;
- a 5000-character error name (`errorCode`) and a 5000-character request id
  are both dropped for exceeding `normalizeDiagnosticValue`'s 200-character
  bound, leaving only `{status: 400}` in `details`;
- the SDK's `"Unknown"` error-name placeholder (its fallback when the
  response carried no `x-amzn-errortype` header) does not end in
  `"Exception"`, so it is omitted rather than reported as `errorCode`, leaving
  only `{status, requestId}`.
"""

from __future__ import annotations

import pytest

_REASON = "bedrock-converse-stream is a documented omission of this port: it needs the AWS SDK's SigV4 signer, credential-provider chain and binary event-stream framing, none of which are in pi-ai's dependency set. The TypeScript case asserts on a mocked @aws-sdk/client-bedrock-runtime command, so there is no Python code path to call."


@pytest.mark.skip(reason=_REASON)
def test_records_status_error_code_and_request_id_for_a_non_2xx_from_client_send() -> None:
    """`it("records status, error code and request id for a non-2xx from client.send()")`:
    a `client.send()` rejection carrying `$metadata: { httpStatusCode: 400, requestId }`
    and `.name === "ValidationException"` produces `message.stopReason === "error"` and a
    `bedrock_response_failure` diagnostic whose `details` deep-equals `{ status: 400,
    errorCode: "ValidationException", requestId }`. The diagnostic has no `error` field
    (`diagnostic.error === undefined`) and its own keys, sorted, are exactly
    `["details", "timestamp", "type"]`.
    """


@pytest.mark.skip(reason=_REASON)
def test_leaves_errormessage_untouched_so_retry_classification_is_unaffected() -> None:
    """`it("leaves errorMessage untouched so retry classification is unaffected")`: for the
    same `ValidationException` rejection (now also carrying a `$response.body` stream
    stub), `message.errorMessage === "Validation error: The provided model identifier is
    invalid."` -- the SDK-formatted message used for retry/error-category classification
    is unchanged by adding the `bedrock_response_failure` diagnostic alongside it.
    """


@pytest.mark.skip(reason=_REASON)
def test_reports_only_the_request_id_for_a_modeled_mid_stream_exception() -> None:
    """`it("reports only the request id for a modeled mid-stream exception")`: the mocked
    stream yields `messageStart` then throws a bare object literal `{ message: "Too many
    requests, please wait." }` (not an `Error`, so it has no `.name`). `stopReason ===
    "error"` and the diagnostic's `details` deep-equals `{ requestId }` only -- no
    `errorCode`, since a bare object literal can never produce one.
    """


@pytest.mark.skip(reason=_REASON)
def test_captures_the_error_code_for_an_unmodeled_mid_stream_error() -> None:
    """`it("captures the error code for an unmodeled mid-stream error")`: the mocked stream
    throws a real `Error("Model stream terminated unexpectedly.")` with `.name =
    "ModelStreamErrorException"` (ends in `"Exception"`, so it counts as modeled). The
    diagnostic's `details` deep-equals `{ errorCode: "ModelStreamErrorException",
    requestId }`.
    """


@pytest.mark.skip(reason=_REASON)
def test_does_not_report_a_transport_failure_name_as_a_provider_error_code() -> None:
    """`it("does not report a transport failure name as a provider error code")`: the mocked
    stream throws a real `Error("Connection timed out after 1000 ms")` with `.name =
    "TimeoutError"` (does NOT end in `"Exception"`). The diagnostic's `details`
    deep-equals `{ requestId }` only -- `"TimeoutError"` is never surfaced as `errorCode`
    even though it is a real `Error` with an informative `.name`.
    """


@pytest.mark.skip(reason=_REASON)
def test_emits_no_diagnostic_when_the_failure_carries_no_provider_metadata() -> None:
    """`it("emits no diagnostic when the failure carries no provider metadata")`: `client.send()`
    rejects with a plain `new Error("socket hang up")` (no `$metadata`, `.name` does not
    end in `"Exception"`). `message.stopReason === "error"`, `message.errorMessage ===
    "socket hang up"`, and no `bedrock_response_failure` diagnostic is attached at all
    (since `details` would end up empty, `appendBedrockFailureDiagnostic` appends
    nothing).
    """


@pytest.mark.skip(reason=_REASON)
def test_emits_no_diagnostic_for_an_aborted_turn() -> None:
    """`it("emits no diagnostic for an aborted turn")`: the same fully-populated
    `ValidationException` rejection as the first case (status 400, errorCode, requestId
    all available) is used, but the run is driven with an already-aborted
    `AbortController.signal`. `message.stopReason === "aborted"` and no diagnostic is
    attached -- an aborted turn suppresses the diagnostic even when the underlying error
    would otherwise produce a fully-populated one.
    """


@pytest.mark.skip(reason=_REASON)
def test_drops_header_derived_values_that_exceed_the_length_bound() -> None:
    """`it("drops header-derived values that exceed the length bound")`: the rejection's
    error name is `"E".repeat(5000) + "Exception"` (5009 chars, still ends in
    `"Exception"`) and `$metadata.requestId` is `"R".repeat(5000)`; `$metadata.httpStatusCode
    === 400`. The diagnostic's `details` deep-equals `{ status: 400 }` only -- both the
    over-long `errorCode` and the over-long `requestId` are dropped by the 200-character
    bound, while the numeric `status` (not string-length-bounded) survives.
    """


@pytest.mark.skip(reason=_REASON)
def test_omits_the_sdk_s_unknown_placeholder_instead_of_reporting_it_as_a_code() -> None:
    """`it("omits the SDK's Unknown placeholder instead of reporting it as a code")`: the
    rejection's error name is exactly `"Unknown"` (the SDK's own fallback name when the
    response carried no `x-amzn-errortype` header) with `$metadata: { httpStatusCode: 403,
    requestId }`. Since `"Unknown"` does not end in `"Exception"`, the diagnostic's
    `details` deep-equals `{ status: 403, requestId }` -- no `errorCode` key at all, not
    even `errorCode: "Unknown"`.
    """
