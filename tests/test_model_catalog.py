"""Tests for the generated model catalog loader.

Covers `pi_ai.model_catalog` (port of `packages/ai/src/model-catalog.ts`) and
`pi_ai.providers.all` (port of `packages/ai/src/providers/all.ts`). No network:
everything reads the JSON shards committed under `pi_ai/providers/data/`.
"""

import json

import pytest
from pi_ai.model_catalog import (
    DATA_DIR,
    MODEL_DATA_MANIFEST_FILE,
    MODEL_DATA_SCHEMA_VERSION,
    flatten_model_catalog,
    get_model_data_generated_at,
    get_model_data_provider_ids,
    load_model_catalog,
    load_model_groups,
    load_models,
    model_from_data,
    read_model_data_manifest,
)
from pi_ai.providers import (
    all_providers,
    builtin_models,
    builtin_providers,
    get_builtin_model,
    get_builtin_model_data_generated_at,
    get_builtin_models,
    get_builtin_providers,
)
from pi_ai.types import Model

# --------------------------------------------------------------------------
# shards and manifest
# --------------------------------------------------------------------------


def test_data_directory_is_committed_and_hydrated():
    provider_ids = get_model_data_provider_ids()
    assert len(provider_ids) >= 30
    assert provider_ids == sorted(provider_ids)
    assert "openai" in provider_ids
    assert "anthropic" in provider_ids
    # The manifest is not a provider shard.
    assert MODEL_DATA_MANIFEST_FILE not in provider_ids


def test_manifest_lists_every_shard_with_its_sha256():
    import hashlib

    manifest = read_model_data_manifest()
    assert manifest is not None
    assert manifest.schema_version == MODEL_DATA_SCHEMA_VERSION
    assert manifest.generated_at
    assert len(manifest.structure_hash) == 64

    for provider_id in get_model_data_provider_ids():
        filename = f"{provider_id}.json"
        content = (DATA_DIR / filename).read_bytes()
        assert manifest.files[filename] == hashlib.sha256(content).hexdigest()


def test_generated_at_is_a_millisecond_timestamp():
    generated_at = get_model_data_generated_at()
    assert generated_at is not None
    # Well after 2020 and expressed in milliseconds, matching `Date.parse`.
    assert generated_at > 1_600_000_000_000
    assert get_builtin_model_data_generated_at() == generated_at


def test_generated_at_is_none_for_an_unhydrated_directory(tmp_path):
    assert read_model_data_manifest(tmp_path) is None
    assert get_model_data_generated_at(tmp_path) is None
    assert get_model_data_provider_ids(tmp_path / "missing") == []


# --------------------------------------------------------------------------
# flattening
# --------------------------------------------------------------------------


def test_flatten_model_catalog_merges_every_api_group():
    groups = {
        "openai-completions": {"a": {"id": "a", "api": "openai-completions"}},
        "anthropic-messages": {"b": {"id": "b", "api": "anthropic-messages"}},
    }
    catalog = flatten_model_catalog("test", groups)
    assert sorted(catalog) == ["a", "b"]
    assert catalog["b"].api == "anthropic-messages"


def test_model_from_data_maps_camel_case_keys():
    model = model_from_data(
        {
            "id": "demo",
            "name": "Demo",
            "api": "openai-responses",
            "provider": "demo-provider",
            "baseUrl": "https://demo.invalid/v1",
            "reasoning": True,
            "thinkingLevelMap": {"off": None, "high": "high"},
            "input": ["text", "image"],
            "cost": {
                "input": 1.25,
                "output": 10,
                "cacheRead": 0.125,
                "cacheWrite": 0,
                "tiers": [{"inputTokensAbove": 272000, "input": 2.5, "output": 15, "cacheRead": 0.25, "cacheWrite": 0}],
            },
            "contextWindow": 400000,
            "maxTokens": 128000,
            "headers": {"X-Demo": "1"},
            "compat": {"supportsStrictMode": True},
        }
    )
    assert model.base_url == "https://demo.invalid/v1"
    assert model.context_window == 400000
    assert model.max_tokens == 128000
    assert model.thinking_level_map == {"off": None, "high": "high"}
    assert model.headers == {"X-Demo": "1"}
    assert model.compat == {"supportsStrictMode": True}
    assert model.cost.cache_read == 0.125
    assert model.cost.tiers[0].input_tokens_above == 272000
    assert model.cost.tiers[0].output == 15


def test_model_from_data_defaults_missing_optional_fields():
    model = model_from_data({"id": "bare"})
    assert model.name == "bare"
    assert model.api == "openai-completions"
    assert model.input == ["text"]
    assert model.cost.input == 0
    assert model.cost.tiers == []
    assert model.thinking_level_map == {}


def test_load_model_catalog_of_an_unknown_provider_is_empty():
    assert load_model_catalog("does-not-exist") == {}
    assert load_models("does-not-exist") == []
    assert load_model_groups("does-not-exist") == {}


def test_load_models_returns_fresh_objects_each_call():
    first = load_models("groq")
    second = load_models("groq")
    assert [m.id for m in first] == [m.id for m in second]
    assert first[0] is not second[0]


def test_shard_groups_agree_with_each_model_api():
    for provider_id in get_model_data_provider_ids():
        groups = load_model_groups(provider_id)
        assert groups, provider_id
        for api, models in groups.items():
            for model_id, data in models.items():
                assert data["api"] == api
                assert data["id"] == model_id
                assert data["provider"] == provider_id


# --------------------------------------------------------------------------
# cost tiers
# --------------------------------------------------------------------------


def test_openai_long_context_models_carry_a_second_cost_tier():
    model = get_builtin_model("openai", "gpt-5.6-sol")
    assert model is not None
    assert model.cost.input > 0
    tier = model.cost.tiers[0]
    assert tier.input_tokens_above == 272000
    # The long-context tier doubles input/cache and multiplies output by 1.5.
    assert tier.input == pytest.approx(model.cost.input * 2)
    assert tier.output == pytest.approx(model.cost.output * 1.5)
    assert tier.cache_read == pytest.approx(model.cost.cache_read * 2)


def test_azure_clones_drop_the_openai_cost_tiers():
    openai_model = get_builtin_model("openai", "gpt-5.6-sol")
    azure_model = get_builtin_model("azure-openai-responses", "gpt-5.6-sol")
    assert openai_model is not None and azure_model is not None
    assert openai_model.cost.tiers
    assert azure_model.cost.tiers == []
    assert azure_model.cost.input == openai_model.cost.input
    # Azure resolves the resource host per deployment, so no base URL is baked in.
    assert azure_model.base_url == ""


def test_every_cost_tier_starts_above_zero_tokens():
    for provider_id in get_model_data_provider_ids():
        for model in get_builtin_models(provider_id):
            for tier in model.cost.tiers:
                assert tier.input_tokens_above > 0, f"{provider_id}/{model.id}"


# --------------------------------------------------------------------------
# thinking level maps
# --------------------------------------------------------------------------


def test_openai_responses_gpt5_models_map_off_to_no_thinking():
    model = get_builtin_model("openai", "gpt-5.6-sol")
    assert model is not None
    assert model.thinking_level_map["off"] == "none"
    assert model.thinking_level_map["xhigh"] == "xhigh"
    assert model.thinking_level_map["max"] == "max"


def test_thinking_level_maps_only_use_known_levels():
    levels = {"off", "minimal", "low", "medium", "high", "xhigh", "max"}
    for provider_id in get_model_data_provider_ids():
        for model in get_builtin_models(provider_id):
            assert set(model.thinking_level_map) <= levels, f"{provider_id}/{model.id}"


def test_github_copilot_gpt5_maps_minimal_onto_low():
    # Copilot rejects `reasoning_effort: "minimal"`, so the generator remaps it.
    model = get_builtin_model("github-copilot", "gpt-5-mini")
    assert model is not None
    assert model.thinking_level_map["minimal"] == "low"
    assert model.thinking_level_map["off"] is None


def test_anthropic_adaptive_thinking_levels_follow_the_model_generation():
    # "max" is available on every adaptive-thinking Claude model; "xhigh" only
    # on Opus 4.7/4.8/5, Sonnet 5 and Fable 5.
    sonnet_5 = get_builtin_model("anthropic", "claude-sonnet-5")
    sonnet_4_6 = get_builtin_model("anthropic", "claude-sonnet-4-6")
    haiku = get_builtin_model("anthropic", "claude-haiku-4-5")
    assert sonnet_5 is not None and sonnet_4_6 is not None and haiku is not None
    assert sonnet_5.thinking_level_map == {"xhigh": "xhigh", "max": "max"}
    assert sonnet_4_6.thinking_level_map == {"max": "max"}
    assert haiku.thinking_level_map == {}


def test_effort_based_models_map_the_three_native_levels():
    model = get_builtin_model("groq", "openai/gpt-oss-120b")
    assert model is not None
    assert model.thinking_level_map == {
        "off": None,
        "minimal": None,
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": None,
        "max": None,
    }


# --------------------------------------------------------------------------
# built-in providers
# --------------------------------------------------------------------------


def test_builtin_providers_cover_every_generated_shard():
    providers = builtin_providers()
    provider_ids = {provider.id for provider in providers}
    assert set(get_builtin_providers()) <= provider_ids
    # `radius` is dynamic and has no generated shard.
    assert provider_ids - set(get_builtin_providers()) == {"radius"}
    assert all_providers.__doc__  # the alias is documented
    assert {p.id for p in all_providers()} == provider_ids


def test_every_builtin_provider_resolves_its_models():
    for provider in builtin_providers():
        if provider.id == "radius":
            continue
        models = provider.get_models()
        assert models, provider.id
        for model in models:
            assert model.provider == provider.id
            assert model.id
            assert model.context_window > 0, f"{provider.id}/{model.id}"
            assert model.max_tokens > 0, f"{provider.id}/{model.id}"
            assert provider.get_model(model.id) is not None


def test_multi_api_providers_dispatch_on_the_model_api():
    provider = next(p for p in builtin_providers() if p.id == "fireworks")
    assert isinstance(provider.api, dict)
    for model in provider.get_models():
        assert provider.api_for(model) is provider.api[model.api]


def test_dispatch_fails_loudly_for_an_unsupported_api():
    from pi_ai.models import ModelsError

    provider = next(p for p in builtin_providers() if p.id == "fireworks")
    with pytest.raises(ModelsError):
        provider.api_for(Model(id="nope", api="pi-messages"))


def test_builtin_models_registry_finds_a_model_by_reference():
    models = builtin_models()
    assert models.find_model("openai/gpt-5.6-sol") is not None
    assert models.get_model("anthropic", "claude-sonnet-5") is not None
    assert models.get_model("anthropic", "not-a-model") is None


def test_get_builtin_model_and_models_agree_with_the_shards():
    for provider_id in get_model_data_provider_ids():
        models = get_builtin_models(provider_id)
        assert models
        first = models[0]
        assert get_builtin_model(provider_id, first.id).id == first.id
    assert get_builtin_model("openai", "not-a-model") is None
    assert get_builtin_models("not-a-provider") == []


def test_model_ids_are_unique_within_a_provider():
    for provider_id in get_model_data_provider_ids():
        ids = [model.id for model in get_builtin_models(provider_id)]
        assert len(ids) == len(set(ids)), provider_id


def test_shards_are_serialized_without_trailing_whitespace():
    for provider_id in get_model_data_provider_ids():
        text = (DATA_DIR / f"{provider_id}.json").read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert json.loads(text)
