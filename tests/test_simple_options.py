from pi_ai.api.simple_options import (
    MIN_ANSWER_TOKENS,
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_max_tokens_to_context,
    clamp_reasoning,
)
from pi_ai.types import Context, Model, SimpleStreamOptions, ThinkingBudgets, UserMessage
from pi_ai.utils.abort import AbortSignal


def _model(**overrides) -> Model:
    defaults = dict(id="m", context_window=1000, max_tokens=500)
    defaults.update(overrides)
    return Model(**defaults)


def test_clamp_max_tokens_to_context_returns_requested_when_context_window_is_zero():
    model = _model(context_window=0)
    context = Context()
    assert clamp_max_tokens_to_context(model, context, 5000) == 5000


def test_clamp_max_tokens_to_context_floors_at_min_max_tokens():
    model = _model(context_window=0)
    context = Context()
    assert clamp_max_tokens_to_context(model, context, -10) == 1


def test_clamp_max_tokens_to_context_reduces_when_context_usage_leaves_less_room():
    model = _model(context_window=100)
    # A very long system prompt eats most of the context window.
    context = Context(messages=[], system_prompt="x" * 400)
    result = clamp_max_tokens_to_context(model, context, 1000)
    # available = 100 - ~100 tokens - 4096 safety => clamped to MIN_MAX_TOKENS (1)
    assert result == 1


def test_clamp_max_tokens_to_context_does_not_exceed_requested_max_tokens():
    model = _model(context_window=1_000_000)
    context = Context(messages=[UserMessage(content="hi")])
    result = clamp_max_tokens_to_context(model, context, 200)
    assert result == 200


def test_build_base_options_with_no_options_uses_model_defaults():
    model = _model(max_tokens=500, context_window=1_000_000)
    context = Context()
    result = build_base_options(model, context)

    assert result.max_tokens == 500
    assert result.api_key is None
    assert result.temperature is None
    assert result.sampling_params == {}
    assert result.headers == {}
    assert result.env == {}
    assert result.metadata == {}
    assert result.signal is None
    assert result.telemetry_context is None


def test_build_base_options_merges_model_and_options_sampling_params():
    model = _model(sampling_params={"top_p": 0.9, "seed": 1})
    context = Context()
    options = SimpleStreamOptions(sampling_params={"seed": 2, "top_k": 5})

    result = build_base_options(model, context, options)

    assert result.sampling_params == {"top_p": 0.9, "seed": 2, "top_k": 5}


def test_build_base_options_uses_explicit_api_key_over_options_api_key():
    model = _model()
    context = Context()
    options = SimpleStreamOptions(api_key="from-options")

    result = build_base_options(model, context, options, api_key="explicit")

    assert result.api_key == "explicit"


def test_build_base_options_falls_back_to_options_api_key():
    model = _model()
    context = Context()
    options = SimpleStreamOptions(api_key="from-options")

    result = build_base_options(model, context, options)

    assert result.api_key == "from-options"


def test_build_base_options_passes_through_signal_and_telemetry_context():
    model = _model()
    context = Context()
    signal = AbortSignal()
    telemetry = {"trace_id": "abc"}
    options = SimpleStreamOptions(signal=signal, telemetry_context=telemetry)

    result = build_base_options(model, context, options)

    assert result.signal is signal
    assert result.telemetry_context is telemetry


def test_build_base_options_clamps_max_tokens_using_options_override():
    model = _model(max_tokens=500, context_window=0)
    context = Context()
    options = SimpleStreamOptions(max_tokens=10_000)

    result = build_base_options(model, context, options)

    assert result.max_tokens == 10_000


def test_clamp_reasoning_maps_xhigh_and_max_to_high():
    assert clamp_reasoning("xhigh") == "high"
    assert clamp_reasoning("max") == "high"


def test_clamp_reasoning_passes_through_other_levels():
    assert clamp_reasoning("low") == "low"
    assert clamp_reasoning("medium") == "medium"
    assert clamp_reasoning(None) is None


def test_adjust_max_tokens_for_thinking_uses_model_cap_when_no_explicit_base():
    result = adjust_max_tokens_for_thinking(None, 10_000, "medium")
    assert result.max_tokens == 10_000
    assert result.thinking_budget == 8192


def test_adjust_max_tokens_for_thinking_adds_budget_to_explicit_base_capped_by_model():
    result = adjust_max_tokens_for_thinking(1000, 10_000, "low")
    # 1000 + 2048 = 3048, below the model cap of 10_000.
    assert result.max_tokens == 3048
    assert result.thinking_budget == 2048


def test_adjust_max_tokens_for_thinking_caps_at_model_max_tokens():
    result = adjust_max_tokens_for_thinking(9000, 10_000, "high")
    # 9000 + 16384 exceeds the model cap, so max_tokens clamps to 10_000; since
    # max_tokens then falls at/below the thinking budget, the budget shrinks to
    # leave room for the answer.
    assert result.max_tokens == 10_000
    assert result.thinking_budget == 10_000 - MIN_ANSWER_TOKENS


def test_adjust_max_tokens_for_thinking_shrinks_budget_when_max_tokens_leaves_no_room_for_answer():
    # max_tokens (10_000) <= thinking_budget requires shrinking the budget so at
    # least MIN_ANSWER_TOKENS remain for the answer.
    result = adjust_max_tokens_for_thinking(None, 10_000, "max", ThinkingBudgets(high=20_000))
    assert result.max_tokens == 10_000
    assert result.thinking_budget == 10_000 - MIN_ANSWER_TOKENS


def test_adjust_max_tokens_for_thinking_uses_custom_budgets_override():
    custom = ThinkingBudgets(medium=100)
    result = adjust_max_tokens_for_thinking(None, 10_000, "medium", custom)
    assert result.thinking_budget == 100


def test_adjust_max_tokens_for_thinking_custom_budgets_only_overrides_given_levels():
    custom = ThinkingBudgets(medium=100)
    result = adjust_max_tokens_for_thinking(1000, 10_000, "low", custom)
    # "low" was not overridden, so the default (2048) still applies.
    assert result.thinking_budget == 2048
