"""Python port of `packages/ai/test/bedrock-endpoint-resolution.test.ts`.

Only the first case is portable. The rest assert on the config object handed to
the AWS SDK's `BedrockRuntimeClient` constructor, and
`packages/ai/src/api/bedrock-converse-stream.ts` is a documented omission of
this port (see the repository README and
:mod:`pi_ai.api.bedrock_converse_stream`): it depends on the AWS SDK's SigV4
signer, credential-provider chain and binary event-stream framing, none of
which are in `pi-ai`'s dependency set.

Behaviors left uncovered by the Python port:

- `AWS_REGION` wins over a pinned standard endpoint (`region` set, `endpoint`
  left unset);
- both `endpoint` and `region` are derived from a built-in EU endpoint when
  neither a region nor a profile is configured;
- an explicit `profile` stream option and a credential-scoped `AWS_PROFILE`
  keep the derived endpoint and region, but an *ambient* `AWS_PROFILE` (set
  directly on `process.env`, not passed as an option) instead clears both --
  three sub-cases with two different outcomes, not one uniform outcome;
- a custom `baseUrl` is passed through to the SDK client unchanged, and
  `region` still falls back to `AWS_REGION` since the custom endpoint carries
  no region to derive;
- the region is extracted from a standard or GovCloud inference-profile ARN
  regardless of `AWS_REGION`;
- ambient `AWS_PROFILE` auth survives `compat` dispatch for custom model ids,
  while bearer-token fields (`token`/`authSchemePreference`) stay unset;
- the generic `apiKey` option is used as a Bedrock bearer token, setting both
  `config.token` and `config.authSchemePreference`.
"""

from __future__ import annotations

import pytest
from pi_ai.providers.all import get_builtin_model


def test_assigns_eu_central_1_runtime_urls_to_built_in_eu_inference_profiles() -> None:
    model = get_builtin_model("amazon-bedrock", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")

    assert model.base_url == "https://bedrock-runtime.eu-central-1.amazonaws.com"


_REASON = "bedrock-converse-stream is a documented omission of this port: it needs the AWS SDK's SigV4 signer, credential-provider chain and binary event-stream framing, none of which are in pi-ai's dependency set. The TypeScript case asserts on a mocked @aws-sdk/client-bedrock-runtime command, so there is no Python code path to call."


@pytest.mark.skip(reason=_REASON)
def test_does_not_pin_standard_aws_endpoints_when_aws_region_is_configured() -> None:
    """`it("does not pin standard AWS endpoints when AWS_REGION is configured")` asserts that the SDK client config has `region` set from AWS_REGION and `endpoint` left undefined."""


@pytest.mark.skip(reason=_REASON)
def test_derives_region_from_a_built_in_eu_endpoint_when_no_region_or_profile_is_configured() -> None:
    """`it("derives region from a built-in EU endpoint when no region or profile is configured")`
    asserts BOTH that `config.endpoint` is pinned to the built-in EU base URL
    (`https://bedrock-runtime.eu-central-1.amazonaws.com`) AND that `config.region`
    is parsed out of that same URL as `"eu-central-1"`, when neither `AWS_REGION`/
    `AWS_DEFAULT_REGION` nor an ambient `AWS_PROFILE` is configured.
    """


@pytest.mark.skip(reason=_REASON)
def test_handles_missing_regions_for_explicit_scoped_and_ambient_profiles() -> None:
    """`it("handles missing regions for explicit, scoped, and ambient profiles")` runs three
    sub-cases against the same EU inference-profile model and asserts they behave
    *differently*:
    1. an explicit `{ profile: "bedrock-profile" }` stream option keeps the derived
       endpoint/region (`config.profile === "bedrock-profile"`, `config.endpoint ===
       "https://bedrock-runtime.eu-central-1.amazonaws.com"`, `config.region ===
       "eu-central-1"`);
    2. a credential-scoped `{ env: { AWS_PROFILE: "scoped-bedrock-profile" } }` option
       behaves the same way (`config.profile === "scoped-bedrock-profile"`, same
       endpoint/region);
    3. an *ambient* `process.env.AWS_PROFILE = "ambient-bedrock-profile"` (no explicit
       option) instead clears the pinned endpoint and region entirely:
       `config.profile === "ambient-bedrock-profile"` but `config.endpoint` and
       `config.region` are both `undefined` — the opposite of cases 1 and 2, not the
       same "keeps the derived endpoint and region" outcome.
    """


@pytest.mark.skip(reason=_REASON)
def test_still_passes_custom_bedrock_endpoints_through_to_the_sdk_client() -> None:
    """`it("still passes custom Bedrock endpoints through to the SDK client")` asserts BOTH
    that a custom (non-`bedrock-runtime.*.amazonaws.com`) `baseUrl` reaches
    `config.endpoint` unchanged AND that `config.region` still comes from
    `AWS_REGION` (`"us-west-2"`) since the custom endpoint carries no region of its
    own to derive from.
    """


@pytest.mark.skip(reason=_REASON)
def test_extracts_region_from_inference_profile_arn_regardless_of_aws_region() -> None:
    """`it("extracts region from inference profile ARN regardless of AWS_REGION")` asserts that the region comes from the ARN, not from AWS_REGION."""


@pytest.mark.skip(reason=_REASON)
def test_extracts_region_from_govcloud_inference_profile_arn() -> None:
    """`it("extracts region from GovCloud inference profile ARN")` asserts that a us-gov-* region is parsed out of the ARN."""


@pytest.mark.skip(reason=_REASON)
def test_preserves_ambient_aws_auth_for_custom_model_ids_through_compat_dispatch() -> None:
    """`it("preserves ambient AWS auth for custom model IDs through compat dispatch")`, with
    an ambient `process.env.AWS_PROFILE = "bedrock-profile"` (not a stream option) and a
    model whose `id` is overridden to a custom inference-profile ARN, dispatches through
    the generic `stream()` compat entry point (not calling the Bedrock stream function
    directly) and asserts `config.profile === "bedrock-profile"` while `config.token` and
    `config.authSchemePreference` both stay `undefined` — i.e. ambient AWS-profile auth is
    preserved and no bearer-token auth path is triggered, even when routed through the
    generic dispatcher for a non-catalog model id.
    """


@pytest.mark.skip(reason=_REASON)
def test_uses_the_generic_api_key_option_as_a_bedrock_bearer_token() -> None:
    """`it("uses the generic API key option as a Bedrock bearer token")` passes the generic
    `{ apiKey: "bedrock-api-key" }` stream option (not a Bedrock-specific `bearerToken`
    option) and asserts BOTH `config.token` deep-equals `{ token: "bedrock-api-key" }` AND
    `config.authSchemePreference` deep-equals `["httpBearerAuth"]` — i.e. the generic
    `apiKey` option is accepted as the Bedrock bearer token and switches auth scheme
    preference to bearer auth.
    """
