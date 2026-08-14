"""Python port of `packages/ai/test/bedrock-raw-stop-reason.test.ts` — not portable.

`packages/ai/src/api/bedrock-converse-stream.ts` is a documented omission of
this port (see the repository README and
:mod:`pi_ai.api.bedrock_converse_stream`): it depends on the AWS SDK's SigV4
signer, credential-provider chain and binary event-stream framing, none of
which are in `pi-ai`'s dependency set. The TypeScript test drives a mocked
`@aws-sdk/client-bedrock-runtime` `ConverseStreamCommand` response, so there is
nothing for a Python counterpart to call.

Behaviors left uncovered by the Python port:

- a successful Bedrock stop keeps `rawStopReason` (`end_turn`) alongside the
  mapped `stopReason` (`stop`) and leaves `errorMessage` unset;
- an unmapped Bedrock stop reason (`guardrail_intervened`) maps to
  `stopReason: "error"` while keeping the raw value and setting
  `errorMessage` to "Provider stopped with: guardrail_intervened".

The identical raw-stop-reason contract *is* covered for the ported providers:
see `tests/test_mistral_raw_stop_reason.py`, `tests/test_google_raw_stop_reason.py`
and `tests/test_openai_completions_raw_stop_reason.py`.
"""

from __future__ import annotations

import pytest

_REASON = "bedrock-converse-stream is a documented omission of this port: it needs the AWS SDK's SigV4 signer, credential-provider chain and binary event-stream framing, none of which are in pi-ai's dependency set. The TypeScript case asserts on a mocked @aws-sdk/client-bedrock-runtime command, so there is no Python code path to call."


@pytest.mark.skip(reason=_REASON)
def test_preserves_raw_bedrock_stop_reasons_for_successful_stops() -> None:
    """`it("preserves raw Bedrock stop reasons for successful stops")` asserts that rawStopReason is 'end_turn', stopReason is 'stop' and errorMessage is unset."""


@pytest.mark.skip(reason=_REASON)
def test_preserves_raw_bedrock_stop_reasons_for_provider_error_stops() -> None:
    """`it("preserves raw Bedrock stop reasons for provider error stops")` asserts that an unmapped 'guardrail_intervened' keeps rawStopReason, maps stopReason to 'error' and sets errorMessage to 'Provider stopped with: guardrail_intervened'."""
