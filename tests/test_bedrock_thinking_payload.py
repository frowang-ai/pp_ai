"""Python port of `packages/ai/test/bedrock-thinking-payload.test.ts` — not portable.

`packages/ai/src/api/bedrock-converse-stream.ts` is a documented omission of
this port (see the repository README and
:mod:`pi_ai.api.bedrock_converse_stream`): it depends on the AWS SDK's SigV4
signer, credential-provider chain and binary event-stream framing, none of
which are in `pi-ai`'s dependency set. Every case here asserts on the
`additionalModelRequestFields` of the object handed to a mocked
`ConverseStreamCommand`, so there is nothing for a Python counterpart to call.

Behaviors left uncovered by the Python port (Bedrock-specific thinking payload
construction, keyed off the Bedrock model id / ARN / `model.name` rather than
off model metadata):

- adaptive thinking (`thinking: { type: "adaptive", display: "summarized" }`
  plus `output_config: { effort: "high" }` and no `anthropic_beta`) for Claude
  Opus 4.8, Fable 5, Sonnet 5 and Opus 5 when reasoning is enabled;
- an `xhigh` reasoning level maps to `output_config: { effort: "xhigh" }` (the
  `thinking` block itself is unaffected, still `{ type: "adaptive", display:
  "summarized" }`) for Opus 4.8, Opus 5 and Fable 5;
- `thinking.display` is omitted (only `{ type: "enabled", budget_tokens: 16384
  }`) for GovCloud model ids on non-adaptive Claude thinking, and
  `anthropic_beta` is `["interleaved-thinking-2025-05-14"]`;
- `thinking.display` is omitted (only `{ type: "adaptive" }`, `output_config`
  still `{ effort: "high" }`, `anthropic_beta` still unset) for GovCloud
  regions on adaptive Claude thinking;
- a real end-to-end streaming call against the model's `maxTokens` cap
  produces more than 4096 output tokens (proving Bedrock's own 4096-token
  default was overridden), gated on real AWS credentials;
- adaptive thinking is selected via `model.name` (with the same `{ type:
  "adaptive", display: "summarized" }` / `{ effort: "high" }` shape) when the
  model id is an application-inference-profile ARN that carries no model name
  of its own;
- cache points are injected (a second `system` block and the last user
  message's last content block both carry a `cachePoint` key) when
  `model.name` identifies a supported Claude model, even though the ARN
  itself does not;
- non-adaptive Claude identified only via `model.name` falls back to
  fixed-budget thinking (`{ type: "enabled", budget_tokens: <a number> }`,
  matched via `toMatchObject` so the exact budget is unconstrained) with
  `anthropic_beta: ["interleaved-thinking-2025-05-14"]`.

The provider-agnostic half of this behavior — which models advertise adaptive
thinking and how `xhigh` is clamped — is covered by
`tests/test_anthropic_adaptive_thinking_models.py`,
`tests/test_anthropic_force_adaptive_thinking.py` and
`tests/test_ai_max_thinking.py`.
"""

from __future__ import annotations

import pytest

_REASON = "bedrock-converse-stream is a documented omission of this port: it needs the AWS SDK's SigV4 signer, credential-provider chain and binary event-stream framing, none of which are in pi-ai's dependency set. The TypeScript case asserts on a mocked @aws-sdk/client-bedrock-runtime command, so there is no Python code path to call."


@pytest.mark.skip(reason=_REASON)
def test_uses_adaptive_thinking_for_claude_opus_4_8_when_reasoning_is_enabled() -> None:
    """`it("uses adaptive thinking for Claude Opus 4.8 when reasoning is enabled")`, on a
    model with id overridden to `global.anthropic.claude-opus-4-8-v1`, asserts
    `additionalModelRequestFields.thinking` deep-equals `{ type: "adaptive", display:
    "summarized" }`, `additionalModelRequestFields.output_config` deep-equals `{ effort:
    "high" }`, and `additionalModelRequestFields.anthropic_beta` is `undefined`.
    """


@pytest.mark.skip(reason=_REASON)
def test_maps_xhigh_reasoning_to_effort_xhigh_for_claude_opus_4_8() -> None:
    """`it("maps xhigh reasoning to effort=xhigh for Claude Opus 4.8")`, with `{ reasoning:
    "xhigh" }`, asserts `additionalModelRequestFields.thinking` is still `{ type:
    "adaptive", display: "summarized" }` (unaffected by the reasoning level) while
    `additionalModelRequestFields.output_config` deep-equals `{ effort: "xhigh" }` (the
    `effort` key lives on `output_config`, not nested under `thinking`), and
    `anthropic_beta` is `undefined`.
    """


@pytest.mark.skip(reason=_REASON)
def test_uses_adaptive_thinking_for_claude_fable_5_when_reasoning_is_enabled() -> None:
    """`it("uses adaptive thinking for Claude Fable 5 when reasoning is enabled")` (catalog
    model `global.anthropic.claude-fable-5`, unmodified) asserts `thinking` deep-equals
    `{ type: "adaptive", display: "summarized" }`, `output_config` deep-equals `{ effort:
    "high" }`, and `anthropic_beta` is `undefined`.
    """


@pytest.mark.skip(reason=_REASON)
def test_uses_adaptive_thinking_for_claude_sonnet_5_when_reasoning_is_enabled() -> None:
    """`it("uses adaptive thinking for Claude Sonnet 5 when reasoning is enabled")` (catalog
    model `global.anthropic.claude-sonnet-5`, unmodified) asserts `thinking` deep-equals
    `{ type: "adaptive", display: "summarized" }`, `output_config` deep-equals `{ effort:
    "high" }`, and `anthropic_beta` is `undefined`.
    """


@pytest.mark.skip(reason=_REASON)
def test_uses_adaptive_thinking_for_claude_opus_5_when_reasoning_is_enabled() -> None:
    """`it("uses adaptive thinking for Claude Opus 5 when reasoning is enabled")` (catalog
    model `global.anthropic.claude-opus-5`, unmodified) asserts `thinking` deep-equals
    `{ type: "adaptive", display: "summarized" }`, `output_config` deep-equals `{ effort:
    "high" }`, and `anthropic_beta` is `undefined`.
    """


@pytest.mark.skip(reason=_REASON)
def test_maps_xhigh_reasoning_to_effort_xhigh_for_claude_opus_5() -> None:
    """`it("maps xhigh reasoning to effort=xhigh for Claude Opus 5")`, with `{ reasoning:
    "xhigh" }` on `global.anthropic.claude-opus-5`, asserts `thinking` deep-equals
    `{ type: "adaptive", display: "summarized" }`, `output_config` deep-equals `{ effort:
    "xhigh" }`, and `anthropic_beta` is `undefined`.
    """


@pytest.mark.skip(reason=_REASON)
def test_maps_xhigh_reasoning_to_effort_xhigh_for_claude_fable_5() -> None:
    """`it("maps xhigh reasoning to effort=xhigh for Claude Fable 5")`, with `{ reasoning:
    "xhigh" }` on `global.anthropic.claude-fable-5`, asserts `thinking` deep-equals
    `{ type: "adaptive", display: "summarized" }` and `output_config` deep-equals
    `{ effort: "xhigh" }` (this case does not additionally assert on `anthropic_beta`).
    """


@pytest.mark.skip(reason=_REASON)
def test_omits_display_for_govcloud_model_ids_on_non_adaptive_claude_thinking() -> None:
    """`it("omits display for GovCloud model ids on non-adaptive Claude thinking")`, on a
    model with id overridden to `us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0` (a
    non-adaptive Claude model, so fixed-budget thinking applies), asserts `thinking`
    deep-equals `{ type: "enabled", budget_tokens: 16384 }` (no `display` key at all) and
    `anthropic_beta` deep-equals `["interleaved-thinking-2025-05-14"]`.
    """


@pytest.mark.skip(reason=_REASON)
def test_omits_display_for_govcloud_regions_on_adaptive_claude_thinking() -> None:
    """`it("omits display for GovCloud regions on adaptive Claude thinking")`, on the same
    Opus-4.8 model as the first case but with `{ region: "us-gov-west-1" }`, asserts
    `thinking` deep-equals `{ type: "adaptive" }` (no `display` key, unlike the non-GovCloud
    adaptive cases above which include `display: "summarized"`), `output_config`
    deep-equals `{ effort: "high" }`, and `anthropic_beta` is `undefined`.
    """


@pytest.mark.skip(reason=_REASON)
def test_uses_the_model_maxtokens_cap_instead_of_bedrock_s_4096_token_default_for_adaptive_claude_models() -> None:
    """`it("uses the model maxTokens cap instead of Bedrock's 4096-token default for adaptive
    Claude models")` is gated on real AWS credentials (`describe.skipIf(!hasBedrockCredentials())`)
    and makes an actual network call: with `model.maxTokens` overridden to 6000 and a
    prompt asking for 5200 repetitions of a token, it asserts `response.stopReason !==
    "error"` and `response.usage.output > 4096` -- proving the model's own `maxTokens`
    (not Bedrock's 4096-token API default) bounded the response. This is a live e2e check,
    not just a payload-construction assertion; it cannot be ported regardless of the
    adapter gap since it requires real Bedrock access.
    """


@pytest.mark.skip(reason=_REASON)
def test_uses_adaptive_thinking_when_model_name_contains_the_model_name_but_arn_does_not() -> None:
    """`it("uses adaptive thinking when model.name contains the model name but ARN does not")`,
    on a model whose `id` is overridden to the ARN
    `arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile` (no
    model name embedded) and whose `name` is set to `"Claude Opus 4.6"`, asserts `thinking`
    deep-equals `{ type: "adaptive", display: "summarized" }` and `output_config`
    deep-equals `{ effort: "high" }` -- the adaptive-thinking decision falls back to
    `model.name` when the ARN carries no model name to match against.
    """


@pytest.mark.skip(reason=_REASON)
def test_injects_cache_points_when_model_name_identifies_a_supported_claude_model() -> None:
    """`it("injects cache points when model.name identifies a supported Claude model")`, on a
    model with the same application-inference-profile ARN id but `name: "Claude Sonnet
    4.6"`, asserts the captured payload's `system` array has length 2 with a `cachePoint`
    key present on `system[1]` (the system prompt plus an appended cache-point marker),
    and that the last content block of the last message in `messages` also has a
    `cachePoint` key -- cache points are injected based on `model.name`, not the ARN.
    """


@pytest.mark.skip(reason=_REASON)
def test_falls_back_to_fixed_budget_thinking_for_non_adaptive_claude_via_model_name() -> None:
    """`it("falls back to fixed-budget thinking for non-adaptive Claude via model.name")`, on a
    model with an application-inference-profile ARN id but `name: "Claude Sonnet 4.5"` (a
    non-adaptive Claude), asserts `thinking` matches (via `toMatchObject`, not exact
    equality) `{ type: "enabled", budget_tokens: expect.any(Number) }` -- the exact budget
    value is unconstrained, only its presence and type -- and `anthropic_beta` deep-equals
    `["interleaved-thinking-2025-05-14"]`.
    """
