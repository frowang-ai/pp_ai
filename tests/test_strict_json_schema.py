"""Strict constrained sampling must narrow the tool schema, not pass it through.

Port of `packages/ai/src/api/constrained-sampling.ts`'s `makeStrictJsonSchema`
and `getJsonSchemaToolParameters`.

Neither had been ported. Every provider that supports strict tool sampling --
Anthropic, OpenAI Responses, Google, Mistral, Bedrock -- resolved the strict
flag correctly and then sent the model the *raw* `tool.parameters`, so a schema
the provider rejects under strict mode went out unchanged. The flag was
computed and then thrown away at the point it mattered.

The transform is not cosmetic: strict mode requires every property to appear in
`required` and forbids `additionalProperties`, so an optional property has to be
re-expressed as a nullable one.
"""

from __future__ import annotations

import pytest
from pi_ai.api.constrained_sampling import (
    UnsupportedStrictJsonSchemaError,
    get_json_schema_tool_parameters,
    make_strict_json_schema,
)
from pi_ai.types import Tool


def _tool(parameters: dict) -> Tool:
    return Tool(name="t", description="d", parameters=parameters)


def test_optional_properties_become_nullable_and_required():
    """Strict mode has no notion of an optional property.

    Upstream keeps the field usable by widening its type to include `null`
    rather than dropping it, so the model can still omit it semantically.
    """
    result = make_strict_json_schema(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "number"}},
            "required": ["a"],
        }
    )

    assert result["required"] == ["a", "b"]
    assert result["additionalProperties"] is False
    assert result["properties"]["a"] == {"type": "string"}
    assert result["properties"]["b"] == {"anyOf": [{"type": "number"}, {"type": "null"}]}


def test_a_property_that_already_allows_null_is_left_alone():
    result = make_strict_json_schema(
        {"type": "object", "properties": {"a": {"type": ["string", "null"]}}, "required": []}
    )

    assert result["properties"]["a"] == {"type": ["string", "null"]}


def test_nested_objects_are_narrowed_too():
    result = make_strict_json_schema(
        {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"inner": {"type": "string"}},
                    "required": [],
                }
            },
            "required": ["outer"],
        }
    )

    outer = result["properties"]["outer"]
    assert outer["additionalProperties"] is False
    assert outer["required"] == ["inner"]


def test_the_input_schema_is_not_mutated():
    """The same `Tool` object is reused across requests.

    Upstream deep-clones for this reason; mutating in place would make the
    second request see a schema already narrowed by the first.
    """
    original = {"type": "object", "properties": {"a": {"type": "string"}}, "required": []}
    snapshot = {"type": "object", "properties": {"a": {"type": "string"}}, "required": []}

    make_strict_json_schema(original)

    assert original == snapshot


@pytest.mark.parametrize(
    "schema, reason",
    [
        ({"type": "string"}, "root schema must have type object"),
        ({"type": "object", "properties": {"a": {"$ref": "#/x"}}, "required": ["a"]}, "$ref"),
        ({"type": "object", "properties": {"a": {"allOf": []}}, "required": ["a"]}, "allOf"),
        (
            {"type": "object", "additionalProperties": {"type": "string"}, "properties": {}},
            "additionalProperties",
        ),
        ({"type": "object", "properties": {}, "required": ["missing"]}, "unknown property"),
    ],
)
def test_unsupported_schemas_raise(schema, reason):
    with pytest.raises(UnsupportedStrictJsonSchemaError, match=reason.replace("$", r"\$")):
        make_strict_json_schema(schema)


def test_tool_parameters_are_only_transformed_under_strict():
    """`strict` is a tri-state: only `True` narrows.

    `None` means "the provider decides", and must behave like off here.
    """
    params = {"type": "object", "properties": {"a": {"type": "string"}}, "required": []}
    tool = _tool(params)

    assert get_json_schema_tool_parameters(tool, None) is params
    assert get_json_schema_tool_parameters(tool, False) is params
    assert get_json_schema_tool_parameters(tool, True) is not params
    assert get_json_schema_tool_parameters(tool, True)["required"] == ["a"]
