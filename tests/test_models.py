import pytest
from pi_ai.models import (
    ModelsError,
    calculate_cost,
    clamp_thinking_level,
    complete,
    get_supported_thinking_levels,
    has_api,
    models_are_equal,
)
from pi_ai.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    ModelCostTier,
    Usage,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream


def make_model(**overrides) -> Model:
    defaults = dict(
        id="model-a",
        name="Model A",
        api="openai-completions",
        provider="openai",
        base_url="https://example.test",
        reasoning=False,
        cost=ModelCost(input=1.0, output=2.0, cache_read=0.5, cache_write=1.5),
    )
    defaults.update(overrides)
    return Model(**defaults)


# --------------------------------------------------------------------------
# calculate_cost
# --------------------------------------------------------------------------


def test_calculate_cost_uses_base_rates():
    model = make_model()
    usage = Usage(input=1000, output=500, cache_read=200, cache_write=100)
    cost = calculate_cost(model, usage)
    assert cost.input == pytest.approx(1000 * 1.0 / 1_000_000)
    assert cost.output == pytest.approx(500 * 2.0 / 1_000_000)
    assert cost.cache_read == pytest.approx(200 * 0.5 / 1_000_000)
    assert cost.cache_write == pytest.approx(100 * 1.5 / 1_000_000)
    assert cost.total == pytest.approx(cost.input + cost.output + cost.cache_read + cost.cache_write)


def test_calculate_cost_selects_highest_matching_tier():
    model = make_model(
        cost=ModelCost(
            input=1.0,
            output=2.0,
            cache_read=0.5,
            cache_write=1.5,
            tiers=[
                ModelCostTier(input=2.0, output=4.0, cache_read=1.0, cache_write=3.0, input_tokens_above=1000),
                ModelCostTier(input=3.0, output=6.0, cache_read=1.5, cache_write=4.5, input_tokens_above=200_000),
            ],
        )
    )
    # 2000 total input tokens exceeds the 1000 threshold but not 200_000, so the
    # first tier's rates apply.
    usage = Usage(input=2000, output=0, cache_read=0, cache_write=0)
    cost = calculate_cost(model, usage)
    assert cost.input == pytest.approx(2000 * 2.0 / 1_000_000)

    # 300_000 total input tokens exceeds both thresholds; the higher tier wins.
    usage_high = Usage(input=300_000, output=0, cache_read=0, cache_write=0)
    cost_high = calculate_cost(model, usage_high)
    assert cost_high.input == pytest.approx(300_000 * 3.0 / 1_000_000)


def test_calculate_cost_tier_not_selected_when_input_tokens_at_or_below_threshold():
    model = make_model(
        cost=ModelCost(
            input=1.0,
            output=2.0,
            cache_read=0.5,
            cache_write=1.5,
            tiers=[ModelCostTier(input=2.0, output=4.0, cache_read=1.0, cache_write=3.0, input_tokens_above=1000)],
        )
    )
    usage = Usage(input=1000, output=0, cache_read=0, cache_write=0)
    cost = calculate_cost(model, usage)
    assert cost.input == pytest.approx(1000 * 1.0 / 1_000_000)


def test_calculate_cost_charges_2x_base_input_for_cache_write_1h():
    model = make_model(cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75))
    usage = Usage(input=0, output=0, cache_read=0, cache_write=1000, cache_write_1h=400)
    cost = calculate_cost(model, usage)
    short_write = 1000 - 400
    expected_cache_write = (3.75 * short_write + 3.0 * 2 * 400) / 1_000_000
    assert cost.cache_write == pytest.approx(expected_cache_write)


def test_calculate_cost_total_is_sum_of_parts():
    model = make_model(cost=ModelCost(input=1.0, output=2.0, cache_read=0.5, cache_write=1.5))
    usage = Usage(input=100, output=200, cache_read=50, cache_write=75, cache_write_1h=25)
    cost = calculate_cost(model, usage)
    assert cost.total == pytest.approx(cost.input + cost.output + cost.cache_read + cost.cache_write)


def test_calculate_cost_mutates_and_returns_usage_cost():
    model = make_model()
    usage = Usage(input=10, output=10, cache_read=0, cache_write=0)
    result = calculate_cost(model, usage)
    assert result is usage.cost


# --------------------------------------------------------------------------
# get_supported_thinking_levels
# --------------------------------------------------------------------------


def test_get_supported_thinking_levels_non_reasoning_model():
    model = make_model(reasoning=False)
    assert get_supported_thinking_levels(model) == ["off"]


def test_get_supported_thinking_levels_default_reasoning_model_includes_all_but_xhigh_max():
    model = make_model(reasoning=True)
    assert get_supported_thinking_levels(model) == ["off", "minimal", "low", "medium", "high"]


def test_get_supported_thinking_levels_excludes_level_mapped_to_none():
    model = make_model(reasoning=True, thinking_level_map={"medium": None})
    levels = get_supported_thinking_levels(model)
    assert "medium" not in levels
    assert levels == ["off", "minimal", "low", "high"]


def test_get_supported_thinking_levels_requires_explicit_entry_for_xhigh_and_max():
    model = make_model(reasoning=True, thinking_level_map={"xhigh": "xhigh-level"})
    levels = get_supported_thinking_levels(model)
    assert "xhigh" in levels
    assert "max" not in levels


# --------------------------------------------------------------------------
# clamp_thinking_level
# --------------------------------------------------------------------------


def test_clamp_thinking_level_exact_match():
    model = make_model(reasoning=True)
    assert clamp_thinking_level(model, "medium") == "medium"


def test_clamp_thinking_level_falls_back_upward():
    model = make_model(reasoning=True, thinking_level_map={"medium": None})
    # "medium" is unsupported; the next supported level upward is "high".
    assert clamp_thinking_level(model, "medium") == "high"


def test_clamp_thinking_level_falls_back_downward_when_nothing_higher_supported():
    model = make_model(reasoning=True)  # supports up to "high", not xhigh/max
    assert clamp_thinking_level(model, "xhigh") == "high"


def test_clamp_thinking_level_unknown_level_returns_first_available():
    model = make_model(reasoning=False)
    assert clamp_thinking_level(model, "off") == "off"


def test_clamp_thinking_level_value_outside_known_levels_returns_fallback():
    model = make_model(reasoning=True)
    # "bogus" is not a member of THINKING_LEVELS at all, so no index lookup is
    # possible; the function must fall back to the first available level.
    assert clamp_thinking_level(model, "bogus") == "off"


def test_clamp_thinking_level_returns_fallback_when_nothing_is_available():
    model = make_model(
        reasoning=True,
        thinking_level_map={"off": None, "minimal": None, "low": None, "medium": None, "high": None},
    )
    assert get_supported_thinking_levels(model) == []
    assert clamp_thinking_level(model, "medium") == "off"


# --------------------------------------------------------------------------
# models_are_equal / has_api
# --------------------------------------------------------------------------


def test_models_are_equal_compares_id_and_provider():
    a = make_model(id="m1", provider="p1")
    b = make_model(id="m1", provider="p1")
    c = make_model(id="m1", provider="p2")
    assert models_are_equal(a, b) is True
    assert models_are_equal(a, c) is False


def test_models_are_equal_returns_false_for_none():
    a = make_model()
    assert models_are_equal(None, a) is False
    assert models_are_equal(a, None) is False
    assert models_are_equal(None, None) is False


def test_has_api_matches_model_api_field():
    model = make_model(api="anthropic-messages")
    assert has_api(model, "anthropic-messages") is True
    assert has_api(model, "openai-completions") is False


# --------------------------------------------------------------------------
# ModelsError
# --------------------------------------------------------------------------


def test_models_error_carries_code_and_message():
    error = ModelsError("auth", "credential store read failed")
    assert error.code == "auth"
    assert str(error) == "credential store read failed"


# --------------------------------------------------------------------------
# complete()
# --------------------------------------------------------------------------


async def test_complete_drains_stream_and_returns_final_message():
    stream = AssistantMessageEventStream()
    message = AssistantMessage(content=[], stop_reason="stop")
    stream.push(DoneEvent(reason="stop", message=message))

    result = await complete(stream)
    assert result is message
