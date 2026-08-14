"""Python port of `packages/ai/test/bedrock-credentials.test.ts` — not portable.

`packages/ai/src/api/bedrock-converse-stream.ts` is a documented omission of
this port (see the repository README and
:mod:`pi_ai.api.bedrock_converse_stream`), so the credential wiring these cases
pin does not exist in Python.

The TypeScript test mocks only `@aws-sdk/client-bedrock-runtime` and asserts on
the `credentials`/`profile` fields of the config object passed to the
`BedrockRuntimeClient` constructor. Per the current
`bedrock-converse-stream.ts` source (`getConfiguredBedrockCredentials`,
`optionsProfile`): when an *option*-level profile is configured (`options.profile`
or `options.env.AWS_PROFILE`), `config.credentials` is left `undefined` (letting
the SDK's own chain resolve the named profile) and `config.profile` carries that
profile name. When no option-level profile is configured, `config.credentials`
is instead populated directly from ambient `AWS_ACCESS_KEY_ID`/
`AWS_SECRET_ACCESS_KEY`, *even if* a purely-ambient `process.env.AWS_PROFILE`
(not passed through an option) is also set -- `config.profile` still reflects
that ambient value, but it does not suppress the access-key credentials the way
an option-level profile does.

Behaviors left uncovered by the Python port:

- an explicit `profile` stream option and a credential-scoped `env.AWS_PROFILE`
  option both leave `config.credentials` `undefined` (SDK resolves the named
  profile itself) while `config.profile` carries the given name;
- ambient `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` populate
  `config.credentials` directly, and `config.profile` is `undefined`, when no
  option-level profile is configured;
- a purely-ambient `AWS_PROFILE` (set on `process.env`, not passed as an
  option) does NOT suppress the ambient access-key credentials: `config.profile`
  reflects the ambient value but `config.credentials` still equals the ambient
  keys object, unlike the option-level profile cases above.

Note that the *provider-level* half of this behavior — which credential source
`amazon-bedrock` reports as available — is ported and covered:
`pi_ai.providers.amazon_bedrock._resolve_bedrock_auth` implements the same
precedence order (stored key, `AWS_BEARER_TOKEN_BEDROCK`, stored/ambient
`AWS_PROFILE`, access keys, ECS task role, web identity token).
"""

from __future__ import annotations

import pytest

_REASON = "bedrock-converse-stream is a documented omission of this port: it needs the AWS SDK's SigV4 signer, credential-provider chain and binary event-stream framing, none of which are in pi-ai's dependency set. The TypeScript case asserts on a mocked @aws-sdk/client-bedrock-runtime command, so there is no Python code path to call."


@pytest.mark.skip(reason=_REASON)
def test_prefers_explicit_and_scoped_profiles_over_ambient_aws_access_keys() -> None:
    """`it("prefers explicit and scoped profiles over ambient AWS access keys")`, with ambient
    `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` stubbed, runs two sub-cases: (1) an explicit
    `{ profile: "explicit-profile" }` stream option gives `config.profile ===
    "explicit-profile"` and `config.credentials === undefined`; (2) a credential-scoped
    `{ env: { AWS_PROFILE: "scoped-profile" } }` option gives `config.profile ===
    "scoped-profile"` and `config.credentials === undefined` -- in both cases the ambient
    access keys are NOT placed on `config.credentials` because an option-level profile is
    configured.
    """


@pytest.mark.skip(reason=_REASON)
def test_uses_ambient_aws_access_keys_when_no_profile_is_configured() -> None:
    """`it("uses ambient AWS access keys when no profile is configured")`, with ambient
    `AWS_ACCESS_KEY_ID="AKIAEXAMPLE"`/`AWS_SECRET_ACCESS_KEY="secretexample"` stubbed and no
    profile option, asserts `config.profile === undefined` AND `config.credentials`
    deep-equals `{ accessKeyId: "AKIAEXAMPLE", secretAccessKey: "secretexample" }`.
    """


@pytest.mark.skip(reason=_REASON)
def test_uses_ambient_aws_access_keys_when_only_an_ambient_profile_is_set() -> None:
    """`it("uses ambient AWS access keys when only an ambient profile is set")`, with ambient
    `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` AND a purely-ambient `process.env.AWS_PROFILE
    = "ambient-profile"` all stubbed (no `profile` stream option), asserts
    `config.profile === "ambient-profile"` (the ambient value is still reported) BUT
    `config.credentials` still deep-equals `{ accessKeyId: "AKIAEXAMPLE", secretAccessKey:
    "secretexample" }` -- an ambient-only profile does not suppress the access-key
    credentials the way an option-level profile does in the previous two cases.
    """
