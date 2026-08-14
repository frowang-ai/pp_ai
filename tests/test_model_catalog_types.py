"""Python port of `packages/ai/test/model-catalog-types.test.ts`.

The TypeScript file uses `expectTypeOf` to pin the literal types derived from
the grouped generated model data (`XAI_MODELS["grok-4.5"].api` is exactly
`"openai-responses"`, etc.). Python has no compile-time type assertion, so
those assertions are ported as the equivalent runtime value checks against the
same catalog entries -- the catalog data they read is identical.
"""

from __future__ import annotations

from pi_ai.providers.all import get_builtin_model


def test_derives_model_api_id_and_provider_from_grouped_model_data() -> None:
    grok_45 = get_builtin_model("xai", "grok-4.5")
    assert grok_45 is not None
    assert grok_45.api == "openai-responses"
    assert grok_45.id == "grok-4.5"
    assert grok_45.provider == "xai"

    grok_43 = get_builtin_model("xai", "grok-4.3")
    assert grok_43 is not None
    assert grok_43.api == "openai-completions"


def test_routes_github_copilot_grok_45_through_the_responses_api() -> None:
    model = get_builtin_model("github-copilot", "grok-4.5")
    assert model is not None
    assert model.api == "openai-responses"
