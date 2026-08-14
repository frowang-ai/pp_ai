"""Python port of `packages/ai/test/bedrock-models.test.ts`.

The TypeScript suite additionally makes one real `complete()` call per model,
gated on AWS credentials plus `BEDROCK_EXTENSIVE_MODEL_TEST`. Those cases are
not ported: they are live-network e2e checks, and `bedrock-converse-stream` is
not runnable in this port at all (it needs SigV4 and the Smithy stack). The
offline catalog assertions below are the part that pins model identifiers.
"""

from __future__ import annotations

import pytest
from pi_ai.providers.all import get_builtin_models

MODELS = get_builtin_models("amazon-bedrock")


def test_should_get_all_available_bedrock_models():
    assert len(MODELS) > 0


def test_exposes_claude_opus_5_through_an_inference_profile_only():
    assert any(model.id == "global.anthropic.claude-opus-5" for model in MODELS)
    assert not any(model.id == "anthropic.claude-opus-5" for model in MODELS)


@pytest.mark.skip(
    reason="Live AWS call: TypeScript gates it on AWS credentials plus BEDROCK_EXTENSIVE_MODEL_TEST, "
    "and bedrock-converse-stream is not ported (needs SigV4 and the Smithy stack)."
)
def test_should_make_a_simple_request_with_each_model():
    """`it(\\`should make a simple request with ${model.id}\\`)`, once per catalog model.

    Asserts the reply is an assistant message with non-empty content, that
    `usage.input + usage.cacheRead > 0` and `usage.output > 0`, that
    `errorMessage` is falsy, and that the joined text blocks are non-empty.
    """
