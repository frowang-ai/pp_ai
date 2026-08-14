"""Tests for the model-catalog generator's correction logic.

`packages/pi-ai/scripts/generate_models.py` is the port of
`packages/ai/scripts/generate-models.ts`. Its value is the per-provider
corrections it applies on top of models.dev, so those are what is tested here:
compat detection, the thinking-level maps, the cost helpers and the manifest.

Nothing here touches the network: the fetchers are never called, and each test
builds its own model dicts.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_models as gen  # noqa: E402
import model_data  # noqa: E402
from models_dev_reasoning_options import THINKING_LEVELS, get_effort_thinking_level_map  # noqa: E402


def model(**overrides):
    base = {
        "id": "test-model",
        "name": "Test Model",
        "api": "openai-completions",
        "provider": "test",
        "baseUrl": "https://example.invalid/v1",
        "reasoning": False,
        "input": ["text"],
        "cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 128000,
        "maxTokens": 8192,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# cost helpers
# --------------------------------------------------------------------------


def test_round_cost_matches_six_decimal_rounding():
    assert gen.round_cost(1.23456749) == 1.234567
    assert gen.round_cost(1 / 3) == 0.333333
    assert gen.round_cost(2.5) == 2.5


def test_long_context_pricing_adds_a_single_tier():
    cost = gen.with_openai_long_context_pricing({"input": 5, "output": 30, "cacheRead": 0.5, "cacheWrite": 6.25})
    assert cost["input"] == 5
    assert len(cost["tiers"]) == 1
    tier = cost["tiers"][0]
    assert tier["inputTokensAbove"] == gen.OPENAI_LONG_CONTEXT_INPUT_THRESHOLD
    assert tier["input"] == 10
    assert tier["output"] == 45
    assert tier["cacheRead"] == 1
    assert tier["cacheWrite"] == 12.5


def test_long_context_pricing_does_not_mutate_the_input_cost():
    original = {"input": 5, "output": 30, "cacheRead": 0.5, "cacheWrite": 0}
    gen.with_openai_long_context_pricing(original)
    assert "tiers" not in original


def test_make_model_omits_absent_optional_keys():
    built = gen.make_model(
        id="m",
        name="",
        api="openai-completions",
        provider="p",
        base_url="https://p.invalid",
        reasoning=False,
        input=["text"],
        cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        context_window=1,
        max_tokens=1,
    )
    assert built["name"] == "m"  # falls back to the id
    assert "thinkingLevelMap" not in built
    assert "compat" not in built
    assert "headers" not in built


def test_make_model_deep_copies_shared_cost_objects():
    shared = {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0}
    first = gen.make_model(
        id="a",
        name="A",
        api="openai-completions",
        provider="p",
        base_url="https://p.invalid",
        reasoning=False,
        input=["text"],
        cost=shared,
        context_window=1,
        max_tokens=1,
    )
    first["cost"]["input"] = 99
    assert shared["input"] == 1


# --------------------------------------------------------------------------
# compat detection
# --------------------------------------------------------------------------


def test_deepseek_gets_the_deepseek_thinking_format():
    compat = gen.detect_openai_completions_compat(model(provider="deepseek", baseUrl="https://api.deepseek.com"))
    assert compat["thinkingFormat"] == "deepseek"
    assert compat["maxTokensField"] == "max_tokens"
    assert compat["supportsStore"] is False


def test_zai_is_detected_by_provider_or_base_url():
    by_provider = gen.detect_openai_completions_compat(model(provider="zai"))
    by_url = gen.detect_openai_completions_compat(model(provider="other", baseUrl="https://api.z.ai/api/paas/v4"))
    assert by_provider["thinkingFormat"] == "zai"
    assert by_url["thinkingFormat"] == "zai"


def test_ant_ling_is_detected_by_base_url():
    compat = gen.detect_openai_completions_compat(model(provider="other", baseUrl="https://api.ant-ling.com/v1"))
    assert compat["thinkingFormat"] == "ant-ling"
    assert compat["maxTokensField"] == "max_tokens"


def test_openrouter_anthropic_models_use_anthropic_cache_control():
    compat = gen.detect_openai_completions_compat(model(provider="openrouter", id="anthropic/claude-sonnet-4.5"))
    assert compat["cacheControlFormat"] == "anthropic"
    openai_model = gen.detect_openai_completions_compat(model(provider="openrouter", id="openai/gpt-4o"))
    assert openai_model.get("cacheControlFormat") is None


def test_openrouter_vendor_prefixed_models_keep_the_developer_role():
    anthropic_model = gen.detect_openai_completions_compat(model(provider="openrouter", id="anthropic/claude-opus-4"))
    other_model = gen.detect_openai_completions_compat(model(provider="openrouter", id="qwen/qwen3-coder"))
    assert anthropic_model["supportsDeveloperRole"] is True
    assert other_model["supportsDeveloperRole"] is False


def test_a_plain_openai_compatible_endpoint_keeps_the_defaults():
    compat = gen.detect_openai_completions_compat(model(provider="unknown", baseUrl="https://vendor.invalid/v1"))
    delta = gen.openai_completions_compat_delta(compat)
    assert delta == {}


def test_compat_delta_only_emits_non_default_keys():
    compat = gen.detect_openai_completions_compat(model(provider="deepseek", baseUrl="https://api.deepseek.com"))
    delta = gen.openai_completions_compat_delta(compat)
    assert "thinkingFormat" in delta
    assert "supportsStore" in delta
    # Unchanged defaults are dropped rather than repeated in every shard.
    assert set(delta) < set(compat)


def test_apply_compat_metadata_drops_an_empty_compat_key():
    candidate = model(provider="unknown", baseUrl="https://vendor.invalid/v1")
    gen.apply_openai_completions_compat_metadata(candidate)
    assert "compat" not in candidate


def test_apply_compat_metadata_lets_explicit_overrides_win():
    candidate = model(provider="deepseek", baseUrl="https://api.deepseek.com", compat={"thinkingFormat": "openai"})
    gen.apply_openai_completions_compat_metadata(candidate)
    assert candidate["compat"]["thinkingFormat"] == "openai"


def test_apply_compat_metadata_skips_non_completions_apis():
    candidate = model(api="anthropic-messages", provider="anthropic")
    gen.apply_openai_completions_compat_metadata(candidate)
    assert "compat" not in candidate


# --------------------------------------------------------------------------
# thinking level maps
# --------------------------------------------------------------------------


def test_gpt5_responses_models_get_an_off_level():
    # `gpt-5` predates the native `reasoning.effort: "none"` value, so "off"
    # means "send no effort at all"; `gpt-5.1` and later map it to "none".
    plain = model(id="gpt-5", api="openai-responses", provider="openai", reasoning=True)
    native_none = model(id="gpt-5.1", api="openai-responses", provider="openai", reasoning=True)
    gen.apply_thinking_level_metadata(plain)
    gen.apply_thinking_level_metadata(native_none)
    assert plain["thinkingLevelMap"]["off"] is None
    assert native_none["thinkingLevelMap"]["off"] == "none"


def test_github_copilot_gpt5_remaps_minimal_to_low():
    candidate = model(id="gpt-5-mini", api="openai-responses", provider="github-copilot", reasoning=True)
    gen.apply_thinking_level_metadata(candidate)
    assert candidate["thinkingLevelMap"]["minimal"] == "low"


def test_anthropic_adaptive_thinking_unlocks_max_and_xhigh():
    sonnet_5 = model(
        id="claude-sonnet-5",
        api="anthropic-messages",
        provider="anthropic",
        reasoning=True,
        compat={"forceAdaptiveThinking": True},
    )
    sonnet_4_6 = model(
        id="claude-sonnet-4-6",
        api="anthropic-messages",
        provider="anthropic",
        reasoning=True,
        compat={"forceAdaptiveThinking": True},
    )
    gen.apply_thinking_level_metadata(sonnet_5)
    gen.apply_thinking_level_metadata(sonnet_4_6)
    assert sonnet_5["thinkingLevelMap"] == {"xhigh": "xhigh", "max": "max"}
    assert sonnet_4_6["thinkingLevelMap"] == {"max": "max"}


def test_merge_thinking_level_map_layers_onto_existing_entries():
    candidate = model(thinkingLevelMap={"low": "low"})
    gen.merge_thinking_level_map(candidate, {"high": "high", "low": None})
    assert candidate["thinkingLevelMap"] == {"low": None, "high": "high"}


def test_effort_options_become_a_full_thinking_level_map():
    level_map = get_effort_thinking_level_map([{"type": "effort", "values": ["low", "medium", "high"]}])
    assert level_map is not None
    assert set(level_map) == {"off", *THINKING_LEVELS}
    assert level_map["low"] == "low"
    assert level_map["xhigh"] is None
    assert level_map["off"] is None


def test_a_native_none_effort_becomes_the_off_level():
    level_map = get_effort_thinking_level_map([{"type": "effort", "values": ["none", "low", "high"]}])
    assert level_map is not None
    assert level_map["off"] == "none"
    assert level_map["medium"] is None


def test_effort_options_without_an_effort_list_are_ignored():
    assert get_effort_thinking_level_map(None) is None
    assert get_effort_thinking_level_map([]) is None
    assert get_effort_thinking_level_map([{"type": "budget", "values": ["1024"]}]) is None
    # `default` has no pi equivalent, so a map made only of it is dropped.
    assert get_effort_thinking_level_map([{"type": "effort", "values": ["default"]}]) is None


# --------------------------------------------------------------------------
# other metadata passes
# --------------------------------------------------------------------------


def test_strict_tool_support_is_provider_specific():
    openai_model = model(api="openai-responses", provider="openai")
    anthropic_model = model(api="anthropic-messages", provider="anthropic")
    other_model = model(api="openai-responses", provider="github-copilot")
    for candidate in (openai_model, anthropic_model, other_model):
        gen.apply_strict_tool_compat_metadata(candidate)
    assert openai_model["compat"] == {"supportsStrictMode": True}
    assert anthropic_model["compat"] == {"supportsStrictTools": True}
    assert "compat" not in other_model


def test_grammar_tools_require_a_gpt5_or_newer_id():
    new_model = model(id="gpt-5.1", api="openai-responses", provider="openai")
    old_model = model(id="gpt-4o", api="openai-responses", provider="openai")
    gen.apply_openai_grammar_tool_compat_metadata(new_model)
    gen.apply_openai_grammar_tool_compat_metadata(old_model)
    assert new_model["compat"]["supportsOpenAIGrammarTools"] is True
    assert "compat" not in old_model


def test_explicit_prompt_cache_mode_follows_a_priced_cache_write():
    priced = model(
        id="gpt-5.6-sol",
        api="openai-responses",
        provider="openai",
        cost={"input": 5, "output": 30, "cacheRead": 0.5, "cacheWrite": 6.25},
    )
    free = model(id="gpt-5.1", api="openai-responses", provider="openai")
    gen.apply_openai_explicit_prompt_cache_metadata(priced)
    gen.apply_openai_explicit_prompt_cache_metadata(free)
    assert priced["compat"]["supportsExplicitPromptCacheMode"] is True
    assert "compat" not in free


def test_direct_reasoning_effort_needs_openai_thinking_format():
    responses_model = model(api="openai-responses", provider="openai")
    deepseek_model = model(provider="deepseek", baseUrl="https://api.deepseek.com")
    adaptive_model = model(api="anthropic-messages", compat={"forceAdaptiveThinking": True})
    plain_anthropic = model(api="anthropic-messages")
    assert gen.supports_direct_reasoning_effort(responses_model) is True
    assert gen.supports_direct_reasoning_effort(deepseek_model) is False
    assert gen.supports_direct_reasoning_effort(adaptive_model) is True
    assert gen.supports_direct_reasoning_effort(plain_anthropic) is False


# --------------------------------------------------------------------------
# serialization and manifest
# --------------------------------------------------------------------------


def test_integral_floats_serialize_as_integers():
    text = gen.serialize_json({"input": 5.0, "nested": [1.0, 2.5], "flag": True})
    assert text == '{"input":5,"nested":[1,2.5],"flag":true}\n'


def test_manifest_records_a_sha256_per_file():
    structure = {"demo": {"model-a": "openai-completions"}}
    contents = {"demo.json": '{"openai-completions":{"model-a":{}}}\n'}
    manifest = model_data.create_model_data_manifest(structure, contents, "2026-01-01T00:00:00.000Z")
    assert manifest["schemaVersion"] == model_data.MODEL_DATA_SCHEMA_VERSION
    assert manifest["generatedAt"] == "2026-01-01T00:00:00.000Z"
    assert manifest["files"]["demo.json"] == model_data.sha256(contents["demo.json"])
    assert len(manifest["structureHash"]) == 64


def test_structure_hash_is_order_independent():
    first = model_data.model_data_structure_hash({"a": {"x": "openai-completions", "y": "openai-responses"}})
    second = model_data.model_data_structure_hash({"a": {"y": "openai-responses", "x": "openai-completions"}})
    assert first == second


def test_structure_hash_changes_when_a_model_moves_api():
    first = model_data.model_data_structure_hash({"a": {"x": "openai-completions"}})
    second = model_data.model_data_structure_hash({"a": {"x": "openai-responses"}})
    assert first != second


def write_shard(directory, provider_id, models):
    groups: dict[str, dict[str, dict]] = {}
    structure: dict[str, str] = {}
    for entry in models:
        groups.setdefault(entry["api"], {})[entry["id"]] = entry
        structure[entry["id"]] = entry["api"]
    filename = f"{provider_id}.json"
    content = gen.serialize_json(groups)
    (directory / filename).write_text(content, encoding="utf-8")
    full_structure = {provider_id: structure}
    manifest = model_data.create_model_data_manifest(full_structure, {filename: content}, "2026-01-01T00:00:00.000Z")
    (directory / model_data.MODEL_DATA_MANIFEST_FILE).write_text(gen.serialize_json(manifest), encoding="utf-8")
    return full_structure


def test_validate_model_data_directory_accepts_a_matching_shard(tmp_path):
    structure = write_shard(tmp_path, "demo", [model(id="model-a", provider="demo")])
    model_data.validate_model_data_directory(structure, tmp_path)


def test_validate_model_data_directory_rejects_a_missing_model(tmp_path):
    structure = write_shard(tmp_path, "demo", [model(id="model-a", provider="demo")])
    structure["demo"]["model-b"] = "openai-completions"
    with pytest.raises(ValueError, match="model IDs do not match"):
        model_data.validate_model_data_directory(structure, tmp_path)


def test_validate_model_data_directory_rejects_an_edited_shard(tmp_path):
    structure = write_shard(tmp_path, "demo", [model(id="model-a", provider="demo")])
    path = tmp_path / "demo.json"
    path.write_text(path.read_text(encoding="utf-8").replace("Test Model", "Edited"), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its manifest hash"):
        model_data.validate_model_data_directory(structure, tmp_path)


def test_the_committed_data_directory_validates():
    model_data.validate_generated_model_data(model_data.DATA_DIR)
