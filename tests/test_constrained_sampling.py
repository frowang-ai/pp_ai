import pytest
from pi_ai.api.constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    create_grammar_tool_input_properties,
    get_grammar_tool_input,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from pi_ai.types import (
    GrammarConstrainedSampling as GrammarConstrainedSamplingConfig,
)
from pi_ai.types import (
    JsonSchemaConstrainedSampling,
    Tool,
)


def _object_tool(required_prop: str = "input", prop_type: str = "string", **overrides) -> Tool:
    parameters = {
        "type": "object",
        "properties": {required_prop: {"type": prop_type}},
        "required": [required_prop],
    }
    return Tool(name="t", description="d", parameters=parameters, **overrides)


def test_get_grammar_tool_input_returns_string_argument():
    assert get_grammar_tool_input("tool", {"input": "hello"}, "input") == "hello"


def test_get_grammar_tool_input_raises_when_argument_missing():
    with pytest.raises(ValueError, match='requires argument "input" to be a string'):
        get_grammar_tool_input("tool", {}, "input")


def test_get_grammar_tool_input_raises_when_argument_not_a_string():
    with pytest.raises(ValueError, match='requires argument "input" to be a string'):
        get_grammar_tool_input("tool", {"input": 5}, "input")


def test_append_grammar_tool_input_json_delta_starts_and_appends():
    buffer = GrammarToolInputJsonBuffer(input="")
    delta1 = append_grammar_tool_input_json_delta(buffer, "code", "print(", False)
    assert delta1 == '{"code":"print('
    assert buffer.input == "print("
    assert buffer.started is True

    delta2 = append_grammar_tool_input_json_delta(buffer, "code", 'print("hi")', False)
    assert delta2 == '\\"hi\\")'
    assert buffer.input == 'print("hi")'


def test_append_grammar_tool_input_json_delta_closes_buffer():
    buffer = GrammarToolInputJsonBuffer(input="abc", started=True)
    delta = append_grammar_tool_input_json_delta(buffer, "code", "abc", True)
    assert delta == '"}'
    assert buffer.closed is True


def test_append_grammar_tool_input_json_delta_no_delta_and_not_closing_returns_none():
    buffer = GrammarToolInputJsonBuffer(input="abc", started=True)
    assert append_grammar_tool_input_json_delta(buffer, "code", "abc", False) is None


def test_append_grammar_tool_input_json_delta_escapes_special_characters():
    buffer = GrammarToolInputJsonBuffer(input="", started=False)
    delta = append_grammar_tool_input_json_delta(buffer, "code", 'line\nwith"quote', False)
    assert delta == '{"code":"line\\nwith\\"quote'


def test_append_grammar_tool_input_json_delta_raises_on_non_monotonic_change():
    buffer = GrammarToolInputJsonBuffer(input="abc", started=True)
    with pytest.raises(ValueError, match="changed non-monotonically"):
        append_grammar_tool_input_json_delta(buffer, "code", "xyz", False)


def test_append_grammar_tool_input_json_delta_raises_after_closed_with_different_input():
    buffer = GrammarToolInputJsonBuffer(input="abc", started=True, closed=True)
    with pytest.raises(ValueError, match="changed after it was closed"):
        append_grammar_tool_input_json_delta(buffer, "code", "abcd", True)


def test_append_grammar_tool_input_json_delta_repeat_close_with_same_input_is_noop():
    buffer = GrammarToolInputJsonBuffer(input="abc", started=True, closed=True)
    assert append_grammar_tool_input_json_delta(buffer, "code", "abc", True) is None


def test_resolve_json_schema_strict_sampling_returns_none_without_config():
    tool = Tool(name="t", description="d")
    assert resolve_json_schema_strict_sampling(tool, True) is None


def test_resolve_json_schema_strict_sampling_returns_true_when_supported():
    tool = Tool(
        name="t",
        description="d",
        constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
    )
    assert resolve_json_schema_strict_sampling(tool, True) is True


def test_resolve_json_schema_strict_sampling_returns_none_when_prefer_and_unsupported():
    tool = Tool(
        name="t",
        description="d",
        constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
    )
    assert resolve_json_schema_strict_sampling(tool, False) is None


def test_resolve_json_schema_strict_sampling_raises_when_require_and_unsupported():
    tool = Tool(
        name="my_tool",
        description="d",
        constrained_sampling=JsonSchemaConstrainedSampling(strict="require"),
    )
    with pytest.raises(ValueError, match='"my_tool" requires JSON-schema constrained sampling'):
        resolve_json_schema_strict_sampling(tool, False)


def test_resolve_json_schema_strict_sampling_ignores_grammar_config():
    tool = Tool(
        name="t",
        description="d",
        constrained_sampling=GrammarConstrainedSamplingConfig(variants={"openai_lark": "grammar"}),
    )
    assert resolve_json_schema_strict_sampling(tool, True) is None


def test_resolve_grammar_constrained_sampling_returns_none_without_config():
    tool = _object_tool()
    assert resolve_grammar_constrained_sampling(tool, True) is None


def test_resolve_grammar_constrained_sampling_returns_none_when_unsupported():
    tool = _object_tool(constrained_sampling=GrammarConstrainedSamplingConfig(variants={"openai_lark": "x -> y"}))
    assert resolve_grammar_constrained_sampling(tool, False) is None


def test_resolve_grammar_constrained_sampling_prefers_lark_over_regex():
    tool = _object_tool(
        constrained_sampling=GrammarConstrainedSamplingConfig(
            variants={"openai_lark": "lark def", "openai_regex": "regex def"}
        )
    )
    result = resolve_grammar_constrained_sampling(tool, True)
    assert result is not None
    assert result.format == "lark"
    assert result.definition == "lark def"
    assert result.input_property == "input"


def test_resolve_grammar_constrained_sampling_falls_back_to_regex():
    tool = _object_tool(constrained_sampling=GrammarConstrainedSamplingConfig(variants={"openai_regex": "regex def"}))
    result = resolve_grammar_constrained_sampling(tool, True)
    assert result is not None
    assert result.format == "regex"
    assert result.definition == "regex def"


def test_resolve_grammar_constrained_sampling_raises_when_no_variant_provided():
    tool = _object_tool(constrained_sampling=GrammarConstrainedSamplingConfig(variants={}))
    with pytest.raises(ValueError, match="no supported grammar variant was provided"):
        resolve_grammar_constrained_sampling(tool, True)


def test_resolve_grammar_constrained_sampling_raises_when_variant_is_blank():
    tool = _object_tool(constrained_sampling=GrammarConstrainedSamplingConfig(variants={"openai_lark": "   "}))
    with pytest.raises(ValueError, match="no supported grammar variant was provided"):
        resolve_grammar_constrained_sampling(tool, True)


def test_resolve_grammar_constrained_sampling_raises_when_schema_is_not_object():
    tool = Tool(
        name="my_tool",
        description="d",
        parameters={"type": "string"},
        constrained_sampling=GrammarConstrainedSamplingConfig(variants={"openai_lark": "x"}),
    )
    with pytest.raises(ValueError, match='"my_tool" cannot use grammar constrained sampling'):
        resolve_grammar_constrained_sampling(tool, True)


def test_resolve_grammar_constrained_sampling_raises_when_multiple_required_properties():
    tool = Tool(
        name="my_tool",
        description="d",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a", "b"],
        },
        constrained_sampling=GrammarConstrainedSamplingConfig(variants={"openai_lark": "x"}),
    )
    with pytest.raises(ValueError, match="exactly one required string property"):
        resolve_grammar_constrained_sampling(tool, True)


def test_resolve_grammar_constrained_sampling_raises_when_required_property_missing_from_properties():
    tool = Tool(
        name="my_tool",
        description="d",
        parameters={"type": "object", "properties": {}, "required": ["input"]},
        constrained_sampling=GrammarConstrainedSamplingConfig(variants={"openai_lark": "x"}),
    )
    with pytest.raises(ValueError, match="requires a properties entry for input"):
        resolve_grammar_constrained_sampling(tool, True)


def test_resolve_grammar_constrained_sampling_raises_when_required_property_not_string_type():
    tool = _object_tool(
        prop_type="number",
        constrained_sampling=GrammarConstrainedSamplingConfig(variants={"openai_lark": "x"}),
    )
    with pytest.raises(ValueError, match="must have type string"):
        resolve_grammar_constrained_sampling(tool, True)


def test_create_grammar_tool_input_properties_collects_only_grammar_tools():
    grammar_tool = _object_tool(constrained_sampling=GrammarConstrainedSamplingConfig(variants={"openai_lark": "x"}))
    json_tool = Tool(name="json_tool", description="d", constrained_sampling=JsonSchemaConstrainedSampling())
    plain_tool = Tool(name="plain_tool", description="d")

    properties = create_grammar_tool_input_properties([grammar_tool, json_tool, plain_tool], True)

    assert properties == {"t": "input"}


def test_create_grammar_tool_input_properties_handles_none_tools():
    assert create_grammar_tool_input_properties(None, True) == {}
