"""Tests for `pi_ai.utils.validation`.

Includes the Python port of `packages/ai/test/validation.test.ts`.
"""

import json

import pytest
from pi_ai.types import Tool, ToolCall
from pi_ai.utils.validation import (
    ToolValidationError,
    _format_validation_path,
    _get_sub_schema_validator,
    _matches_json_type,
    _missing_property,
    _normalize_tuple_items,
    string_enum,
    validate_tool_arguments,
    validate_tool_call,
)


def make_tool(parameters: dict, name: str = "my_tool", description: str = "A tool") -> Tool:
    return Tool(name=name, description=description, parameters=parameters)


def make_call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(id="call-1", name=name, arguments=arguments)


# --------------------------------------------------------------------------
# validate_tool_call
# --------------------------------------------------------------------------


def test_validate_tool_call_finds_tool_by_name():
    tools = [
        make_tool({"type": "object", "properties": {}}, name="other"),
        make_tool({"type": "object", "properties": {"x": {"type": "integer"}}}, name="target"),
    ]
    call = make_call("target", {"x": 5})
    assert validate_tool_call(tools, call) == {"x": 5}


def test_validate_tool_call_raises_for_unknown_name():
    tools = [make_tool({"type": "object", "properties": {}}, name="known")]
    call = make_call("unknown", {})
    with pytest.raises(ToolValidationError, match='Tool "unknown" not found'):
        validate_tool_call(tools, call)


# --------------------------------------------------------------------------
# validate_tool_arguments: basic pass-through / coercion
# --------------------------------------------------------------------------


def test_validate_tool_arguments_returns_coerced_args():
    tool = make_tool({"type": "object", "properties": {"count": {"type": "integer"}}})
    call = make_call("my_tool", {"count": "5"})
    assert validate_tool_arguments(tool, call) == {"count": 5}


@pytest.mark.parametrize(
    ("schema_type", "raw", "expected"),
    [
        ("integer", "5", 5),
        ("number", "5.5", 5.5),
        ("boolean", "true", True),
        ("boolean", "false", False),
        ("string", 5, "5"),
        ("string", True, "true"),
    ],
)
def test_string_and_primitive_coercion(schema_type, raw, expected):
    tool = make_tool({"type": "object", "properties": {"v": {"type": schema_type}}})
    call = make_call("my_tool", {"v": raw})
    assert validate_tool_arguments(tool, call) == {"v": expected}


def test_null_coercion_from_empty_string_zero_and_false():
    tool = make_tool({"type": "object", "properties": {"v": {"type": "null"}}})
    for raw in ("", 0, False):
        call = make_call("my_tool", {"v": raw})
        assert validate_tool_arguments(tool, call) == {"v": None}


# --------------------------------------------------------------------------
# nested object and array coercion
# --------------------------------------------------------------------------


def test_nested_object_coercion():
    tool = make_tool(
        {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                }
            },
        }
    )
    call = make_call("my_tool", {"inner": {"x": "42"}})
    assert validate_tool_arguments(tool, call) == {"inner": {"x": 42}}


def test_array_coercion_with_uniform_items_schema():
    tool = make_tool(
        {
            "type": "object",
            "properties": {"values": {"type": "array", "items": {"type": "integer"}}},
        }
    )
    call = make_call("my_tool", {"values": ["1", "2", "3"]})
    assert validate_tool_arguments(tool, call) == {"values": [1, 2, 3]}


def test_array_coercion_with_per_index_items_schemas():
    tool = make_tool(
        {
            "type": "object",
            "properties": {
                "pair": {
                    "type": "array",
                    "items": [{"type": "integer"}, {"type": "string"}],
                }
            },
        }
    )
    call = make_call("my_tool", {"pair": ["5", 5]})
    assert validate_tool_arguments(tool, call) == {"pair": [5, "5"]}


def test_additional_properties_schema_coerces_undeclared_keys():
    tool = make_tool(
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": {"type": "integer"},
        }
    )
    call = make_call("my_tool", {"a": "1", "b": "2"})
    assert validate_tool_arguments(tool, call) == {"a": 1, "b": 2}


# --------------------------------------------------------------------------
# anyOf / oneOf / allOf coercion
# --------------------------------------------------------------------------


def test_any_of_coerces_to_first_matching_member():
    tool = make_tool(
        {
            "type": "object",
            "properties": {"v": {"anyOf": [{"type": "integer"}, {"type": "boolean"}]}},
        }
    )
    call = make_call("my_tool", {"v": "5"})
    assert validate_tool_arguments(tool, call) == {"v": 5}


def test_one_of_coerces_to_matching_member():
    tool = make_tool(
        {
            "type": "object",
            "properties": {"v": {"oneOf": [{"type": "integer"}, {"type": "boolean"}]}},
        }
    )
    call = make_call("my_tool", {"v": "true"})
    assert validate_tool_arguments(tool, call) == {"v": True}


def test_all_of_applies_each_nested_schema_coercion_in_sequence():
    tool = make_tool(
        {
            "type": "object",
            "properties": {"v": {"allOf": [{"type": "integer"}]}},
        }
    )
    call = make_call("my_tool", {"v": "7"})
    assert validate_tool_arguments(tool, call) == {"v": 7}


def test_union_type_list_leaves_value_alone_when_already_matching_member():
    tool = make_tool(
        {
            "type": "object",
            "properties": {"v": {"type": ["string", "integer"]}},
        }
    )
    call = make_call("my_tool", {"v": "hello"})
    assert validate_tool_arguments(tool, call) == {"v": "hello"}


# --------------------------------------------------------------------------
# error message format
# --------------------------------------------------------------------------


def test_error_message_format_includes_path_message_and_received_arguments():
    tool = make_tool({"type": "object", "properties": {"count": {"type": "integer"}}})
    call = make_call("my_tool", {"count": "not-a-number"})

    with pytest.raises(ToolValidationError) as exc_info:
        validate_tool_arguments(tool, call)

    message = str(exc_info.value)
    lines = message.splitlines()
    assert lines[0] == 'Validation failed for tool "my_tool":'
    assert any(line.startswith("  - count:") for line in lines)
    assert "Received arguments:" in message
    received_json = message.split("Received arguments:\n", 1)[1]
    assert json.loads(received_json) == {"count": "not-a-number"}


def test_required_property_error_names_missing_property_in_path():
    tool = make_tool(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    )
    call = make_call("my_tool", {})

    with pytest.raises(ToolValidationError) as exc_info:
        validate_tool_arguments(tool, call)

    message = str(exc_info.value)
    assert "  - name:" in message


def test_required_property_error_names_nested_missing_property_path():
    tool = make_tool(
        {
            "type": "object",
            "properties": {
                "sub": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                }
            },
            "required": ["sub"],
        }
    )
    call = make_call("my_tool", {"sub": {}})

    with pytest.raises(ToolValidationError) as exc_info:
        validate_tool_arguments(tool, call)

    message = str(exc_info.value)
    assert "  - sub.id:" in message


# --------------------------------------------------------------------------
# _matches_json_type branches (exercised through "type": [...] unions)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("schema_types", "value"),
    [
        (["boolean", "number"], 5),  # boolean check false, then number check true
        (["number", "integer"], 5),  # number check true first (integer never reached)
        (["string", "integer"], 5),  # string false, integer true
        (["integer", "boolean"], True),  # integer excludes bool, boolean true
        (["number", "string"], "s"),  # number false, string true
        (["string", "null"], None),  # string false, null true
        (["null", "array"], []),  # null false, array true
        (["array", "object"], {}),  # array false, object true
    ],
)
def test_union_type_matching_exercises_all_matches_json_type_branches(schema_types, value):
    tool = make_tool({"type": "object", "properties": {"v": {"type": schema_types}}})
    call = make_call("my_tool", {"v": value})
    assert validate_tool_arguments(tool, call) == {"v": value}


def test_matches_json_type_returns_false_for_unrecognized_type():
    assert _matches_json_type("anything", "unknown-type") is False


# --------------------------------------------------------------------------
# primitive coercion edge cases
# --------------------------------------------------------------------------


def test_number_and_integer_coerce_none_to_zero():
    tool = make_tool({"type": "object", "properties": {"v": {"type": "integer"}}})
    call = make_call("my_tool", {"v": None})
    assert validate_tool_arguments(tool, call) == {"v": 0}

    tool_num = make_tool({"type": "object", "properties": {"v": {"type": "number"}}})
    call_num = make_call("my_tool", {"v": None})
    assert validate_tool_arguments(tool_num, call_num) == {"v": 0}


def test_number_and_integer_coerce_bool_to_one_or_zero():
    tool = make_tool({"type": "object", "properties": {"v": {"type": "integer"}}})
    assert validate_tool_arguments(tool, make_call("my_tool", {"v": True})) == {"v": 1}
    assert validate_tool_arguments(tool, make_call("my_tool", {"v": False})) == {"v": 0}


def test_boolean_coerces_none_to_false():
    tool = make_tool({"type": "object", "properties": {"v": {"type": "boolean"}}})
    call = make_call("my_tool", {"v": None})
    assert validate_tool_arguments(tool, call) == {"v": False}


def test_boolean_coerces_numeric_one_and_zero():
    tool = make_tool({"type": "object", "properties": {"v": {"type": "boolean"}}})
    assert validate_tool_arguments(tool, make_call("my_tool", {"v": 1})) == {"v": True}
    assert validate_tool_arguments(tool, make_call("my_tool", {"v": 0})) == {"v": False}


def test_string_coerces_none_to_empty_string():
    tool = make_tool({"type": "object", "properties": {"v": {"type": "string"}}})
    call = make_call("my_tool", {"v": None})
    assert validate_tool_arguments(tool, call) == {"v": ""}


def test_string_coerces_float_using_number_to_string():
    tool = make_tool({"type": "object", "properties": {"v": {"type": "string"}}})
    call = make_call("my_tool", {"v": 5.5})
    assert validate_tool_arguments(tool, call) == {"v": "5.5"}


def test_string_coerces_integer_valued_float_without_decimal_point():
    tool = make_tool({"type": "object", "properties": {"v": {"type": "string"}}})
    call = make_call("my_tool", {"v": 5.0})
    assert validate_tool_arguments(tool, call) == {"v": "5"}


def test_boolean_leaves_non_zero_one_number_unchanged_and_fails_validation():
    tool = make_tool({"type": "object", "properties": {"v": {"type": "boolean"}}})
    call = make_call("my_tool", {"v": 5})
    with pytest.raises(ToolValidationError):
        validate_tool_arguments(tool, call)


def test_boolean_leaves_non_primitive_value_unchanged_and_fails_validation():
    tool = make_tool({"type": "object", "properties": {"v": {"type": "boolean"}}})
    call = make_call("my_tool", {"v": [1, 2]})
    with pytest.raises(ToolValidationError):
        validate_tool_arguments(tool, call)


def test_string_leaves_already_string_value_unchanged():
    tool = make_tool({"type": "object", "properties": {"v": {"type": "string"}}})
    call = make_call("my_tool", {"v": "already"})
    assert validate_tool_arguments(tool, call) == {"v": "already"}


def test_null_leaves_non_falsy_value_unchanged_and_fails_validation():
    tool = make_tool({"type": "object", "properties": {"v": {"type": "null"}}})
    call = make_call("my_tool", {"v": "not-null"})
    with pytest.raises(ToolValidationError):
        validate_tool_arguments(tool, call)


# --------------------------------------------------------------------------
# nested array coercion edge cases
# --------------------------------------------------------------------------


def test_array_items_beyond_tuple_length_are_left_untouched():
    tool = make_tool(
        {
            "type": "object",
            "properties": {"pair": {"type": "array", "items": [{"type": "integer"}]}},
        }
    )
    call = make_call("my_tool", {"pair": ["5", "extra"]})
    assert validate_tool_arguments(tool, call) == {"pair": [5, "extra"]}


def test_array_per_index_non_dict_item_schema_is_skipped():
    tool = make_tool(
        {
            "type": "object",
            "properties": {"pair": {"type": "array", "items": [{"type": "integer"}, True]}},
        }
    )
    call = make_call("my_tool", {"pair": ["5", "ignored"]})
    assert validate_tool_arguments(tool, call) == {"pair": [5, "ignored"]}


def test_array_without_items_schema_is_left_untouched():
    tool = make_tool({"type": "object", "properties": {"values": {"type": "array"}}})
    call = make_call("my_tool", {"values": ["a", "b"]})
    assert validate_tool_arguments(tool, call) == {"values": ["a", "b"]}


def test_object_without_properties_still_applies_additional_properties_schema():
    tool = make_tool({"type": "object", "additionalProperties": {"type": "integer"}})
    call = make_call("my_tool", {"a": "1", "b": "2"})
    assert validate_tool_arguments(tool, call) == {"a": 1, "b": 2}


def test_boolean_property_schema_leaves_value_unchanged():
    tool = make_tool({"type": "object", "properties": {"v": True}})
    call = make_call("my_tool", {"v": "anything"})
    assert validate_tool_arguments(tool, call) == {"v": "anything"}


# --------------------------------------------------------------------------
# union coercion edge cases
# --------------------------------------------------------------------------


def test_any_of_leaves_value_unchanged_when_already_valid_for_a_member():
    tool = make_tool({"type": "object", "properties": {"v": {"anyOf": [{"type": "integer"}, {"type": "string"}]}}})
    call = make_call("my_tool", {"v": 5})
    assert validate_tool_arguments(tool, call) == {"v": 5}


def test_any_of_skips_non_dict_member_schemas():
    tool = make_tool({"type": "object", "properties": {"v": {"anyOf": [True, {"type": "integer"}]}}})
    call = make_call("my_tool", {"v": "5"})
    assert validate_tool_arguments(tool, call) == {"v": 5}


def test_any_of_leaves_value_unchanged_when_no_member_can_be_coerced():
    tool = make_tool({"type": "object", "properties": {"v": {"anyOf": [{"type": "integer"}]}}})
    call = make_call("my_tool", {"v": "not-a-number"})
    with pytest.raises(ToolValidationError):
        validate_tool_arguments(tool, call)


# --------------------------------------------------------------------------
# private helpers exercised directly for defensive branches unreachable
# through the public API with well-formed schemas
# --------------------------------------------------------------------------


def test_get_sub_schema_validator_returns_none_for_non_dict_schema():
    assert _get_sub_schema_validator(True) is None
    assert _get_sub_schema_validator("not-a-schema") is None


def test_get_sub_schema_validator_returns_none_when_validator_construction_raises(monkeypatch):
    import pi_ai.utils.validation as validation_module

    def exploding_get_validator(schema):
        raise RuntimeError("boom")

    monkeypatch.setattr(validation_module, "_get_validator", exploding_get_validator)
    assert _get_sub_schema_validator({"type": "object"}) is None


def test_normalize_tuple_items_recurses_into_top_level_list():
    normalized = _normalize_tuple_items([{"type": "integer"}, {"items": [{"type": "string"}]}])
    assert normalized[0] == {"type": "integer"}
    assert normalized[1] == {"prefixItems": [{"type": "string"}]}


def test_normalize_tuple_items_returns_non_dict_non_list_schema_unchanged():
    assert _normalize_tuple_items(True) is True
    assert _normalize_tuple_items(False) is False


def test_format_validation_path_falls_back_to_instance_path_when_message_has_no_missing_property():
    class FakeError:
        validator = "required"
        instance_path = ()
        absolute_path = ()
        message = "required property is missing"

    assert _missing_property(FakeError()) is None
    assert _format_validation_path(FakeError()) == "root"


def test_string_enum_builds_a_plain_enum_schema():
    assert string_enum(["add", "subtract"]) == {"type": "string", "enum": ["add", "subtract"]}


def test_string_enum_carries_an_optional_description_and_default():
    assert string_enum(["add", "subtract"], description="The operation", default="add") == {
        "type": "string",
        "enum": ["add", "subtract"],
        "description": "The operation",
        "default": "add",
    }


def test_string_enum_omits_empty_metadata():
    schema = string_enum(["add"], description="", default="")
    assert "description" not in schema
    assert "default" not in schema


def test_string_enum_copies_the_values():
    values = ["add", "subtract"]
    schema = string_enum(values)
    values.append("multiply")
    assert schema["enum"] == ["add", "subtract"]


# --------------------------------------------------------------------------
# Ported from `packages/ai/test/validation.test.ts`
#
# Not ported: "still validates when Function constructor is unavailable". That
# case stubs `globalThis.Function` to force TypeBox off its codegen path onto
# its interpreted fallback under a CSP-style policy. The port validates with
# `jsonschema`, which never generates code from strings, so there is no
# equivalent fallback to select.
#
# Also not ported: the `new Function(Compile(...).Code())` assertion inside
# "accepts null for nullable array schemas with items" -- same reason (it
# exists only to exercise TypeBox's *generated* validator after the CSP test
# has globally selected the interpreted one).
# --------------------------------------------------------------------------


def _plain_schema_tool_call(schema: dict, value) -> tuple[Tool, ToolCall]:
    tool = Tool(
        name="echo",
        description="Echo tool",
        parameters={"type": "object", "properties": {"value": schema}, "required": ["value"]},
    )
    return tool, ToolCall(id="tool-1", name="echo", arguments={"value": value})


@pytest.mark.parametrize(
    ("schema", "raw", "expected"),
    [
        ({"type": "number"}, "42", 42),
        ({"type": "number"}, True, 1),
        ({"type": "number"}, None, 0),
        ({"type": "integer"}, "42", 42),
        ({"type": "boolean"}, "true", True),
        ({"type": "boolean"}, "false", False),
        ({"type": "boolean"}, 1, True),
        ({"type": "boolean"}, 0, False),
        ({"type": "string"}, None, ""),
        ({"type": "string"}, True, "true"),
        ({"type": "null"}, "", None),
        ({"type": "null"}, 0, None),
        ({"type": "null"}, False, None),
        ({"type": ["number", "string"]}, "1", "1"),
        ({"type": ["boolean", "number"]}, "1", 1),
    ],
)
def test_ts_coerces_serialized_plain_json_schemas_with_ajv_compatible_primitive_rules(schema, raw, expected):
    tool, tool_call = _plain_schema_tool_call(schema, raw)
    assert validate_tool_arguments(tool, tool_call) == {"value": expected}


def test_ts_preserves_a_value_that_already_matches_a_nullable_union_arm():
    # TypeBox `Type.Union([Type.Number(), Type.Null()])` serializes to `anyOf`.
    tool, tool_call = _plain_schema_tool_call({"anyOf": [{"type": "number"}, {"type": "null"}]}, None)
    assert validate_tool_arguments(tool, tool_call) == {"value": None}


def test_ts_preserves_a_value_that_already_matches_a_one_of_nullable_union_arm():
    tool, tool_call = _plain_schema_tool_call({"oneOf": [{"type": "number"}, {"type": "null"}]}, None)
    assert validate_tool_arguments(tool, tool_call) == {"value": None}


def test_ts_still_coerces_nullable_unions_when_the_original_value_does_not_match_any_arm():
    tool, tool_call = _plain_schema_tool_call({"anyOf": [{"type": "number"}, {"type": "null"}]}, "42")
    assert validate_tool_arguments(tool, tool_call) == {"value": 42}


def test_ts_accepts_null_for_nullable_array_schemas_with_items():
    tool, tool_call = _plain_schema_tool_call({"type": ["array", "null"], "items": {"type": "string"}}, None)
    assert validate_tool_arguments(tool, tool_call) == {"value": None}


@pytest.mark.parametrize(
    ("schema", "raw"),
    [
        ({"type": "boolean"}, "1"),
        ({"type": "boolean"}, "0"),
        ({"type": "null"}, "null"),
        ({"type": "integer"}, "42.1"),
    ],
)
def test_ts_rejects_invalid_coercions_for_serialized_plain_json_schemas(schema, raw):
    tool, tool_call = _plain_schema_tool_call(schema, raw)
    with pytest.raises(ToolValidationError, match="Validation failed"):
        validate_tool_arguments(tool, tool_call)
