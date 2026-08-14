"""Constrained sampling helpers: JSON-schema strict mode and OpenAI grammar tools.

Python port of `packages/ai/src/api/constrained-sampling.ts`.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast

from ..types import GrammarConstrainedSampling as GrammarConstrainedSamplingConfig
from ..types import Tool


@dataclass
class GrammarConstrainedSampling:
    format: Literal["lark", "regex"]
    definition: str
    input_property: str


@dataclass
class GrammarToolInputJsonBuffer:
    input: str
    started: bool = False
    closed: bool = False


def get_grammar_tool_input(tool_name: str, arguments: dict[str, Any], input_property: str) -> str:
    value = arguments.get(input_property)
    if not isinstance(value, str):
        raise ValueError(f'Grammar tool call "{tool_name}" requires argument "{input_property}" to be a string.')
    return value


def _json_string_body(text: str) -> str:
    """`JSON.stringify(text).slice(1, -1)`: the escaped contents without quotes."""
    return json.dumps(text)[1:-1]


def append_grammar_tool_input_json_delta(
    buffer: GrammarToolInputJsonBuffer,
    input_property: str,
    next_input: str,
    close: bool,
) -> str | None:
    if buffer.closed:
        if close and next_input == buffer.input:
            return None
        raise ValueError(f'grammar tool input for property "{input_property}" changed after it was closed')
    if not next_input.startswith(buffer.input):
        raise ValueError(f'grammar tool input for property "{input_property}" changed non-monotonically')

    input_delta = next_input[len(buffer.input) :]
    if not close and len(input_delta) == 0:
        return None

    delta = ""
    if not buffer.started:
        delta += f'{{{json.dumps(input_property)}:"'
        buffer.started = True
    delta += _json_string_body(input_delta)
    buffer.input = next_input

    if close:
        delta += '"}'
        buffer.closed = True
    return delta


def _infer_grammar_input_property(tool: Tool) -> str:
    schema = tool.parameters
    if schema.get("type") != "object":
        raise ValueError("grammar constrained sampling requires an object parameter schema")

    required = schema.get("required")
    if not isinstance(required, list) or len(required) != 1 or not isinstance(required[0], str):
        raise ValueError("grammar constrained sampling requires exactly one required string property")

    input_property = required[0]
    properties = schema.get("properties") or {}
    property_schema = properties.get(input_property)
    if not property_schema:
        raise ValueError(f"grammar constrained sampling requires a properties entry for {input_property}")
    if not isinstance(property_schema, dict) or property_schema.get("type") != "string":
        raise ValueError(f"grammar constrained sampling property {input_property} must have type string")
    return input_property


def resolve_json_schema_strict_sampling(tool: Tool, supports_strict_mode: bool) -> bool | None:
    config = tool.constrained_sampling
    if not config or config.type != "json_schema":
        return None

    if supports_strict_mode:
        return True
    if config.strict == "require":
        raise ValueError(
            f'Tool "{tool.name}" requires JSON-schema constrained sampling, but strict tools are unsupported.'
        )
    return None


def resolve_grammar_constrained_sampling(
    tool: Tool, supports_openai_grammar_tools: bool
) -> GrammarConstrainedSampling | None:
    config = tool.constrained_sampling
    if not config or config.type != "grammar":
        return None

    if not supports_openai_grammar_tools:
        return None

    grammar_config = cast(GrammarConstrainedSamplingConfig, config)
    lark_definition = grammar_config.variants.get("openai_lark")
    regex_definition = grammar_config.variants.get("openai_regex")
    has_lark_definition = isinstance(lark_definition, str) and lark_definition.strip() != ""
    has_regex_definition = isinstance(regex_definition, str) and regex_definition.strip() != ""
    if not has_lark_definition and not has_regex_definition:
        raise ValueError(
            f'Tool "{tool.name}" cannot use grammar constrained sampling: no supported grammar variant was provided.'
        )

    try:
        return GrammarConstrainedSampling(
            format="lark" if has_lark_definition else "regex",
            definition=lark_definition if has_lark_definition else regex_definition,  # type: ignore[arg-type]
            input_property=_infer_grammar_input_property(tool),
        )
    except ValueError as error:
        raise ValueError(f'Tool "{tool.name}" cannot use grammar constrained sampling: {error}.') from error


def create_grammar_tool_input_properties(
    tools: list[Tool] | None, supports_openai_grammar_tools: bool
) -> dict[str, str]:
    properties: dict[str, str] = {}
    for tool in tools or []:
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar is not None:
            properties[tool.name] = grammar.input_property
    return properties


_MISSING = object()


class UnsupportedStrictJsonSchemaError(Exception):
    """A tool schema that cannot be expressed in the strict subset."""


_UNSUPPORTED_STRICT_SCHEMA_KEYS = (
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
)


def _is_json_schema_object(value: Any) -> bool:
    return isinstance(value, dict)


def _is_structured_schema(schema: Any) -> bool:
    if not _is_json_schema_object(schema):
        return False
    raw = schema.get("type")
    types = [raw] if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
    return "object" in types or "array" in types or "properties" in schema or "items" in schema


def _schema_allows_null(schema: Any) -> bool:
    if not _is_json_schema_object(schema):
        return False
    raw = schema.get("type")
    if raw == "null" or (isinstance(raw, list) and "null" in raw):
        return True
    if schema.get("const", _MISSING) is None:
        return True
    enum = schema.get("enum")
    if isinstance(enum, list) and None in enum:
        return True
    any_of = schema.get("anyOf")
    return isinstance(any_of, list) and any(_schema_allows_null(v) for v in any_of)


def _make_json_schema_node_strict(schema: Any) -> None:
    if not _is_json_schema_object(schema):
        raise UnsupportedStrictJsonSchemaError("boolean schemas are unsupported")
    for key in _UNSUPPORTED_STRICT_SCHEMA_KEYS:
        if schema.get(key) is not None:
            raise UnsupportedStrictJsonSchemaError(f"{key} schemas are unsupported")

    any_of = schema.get("anyOf")
    if any_of is not None:
        if not isinstance(any_of, list) or not any_of:
            raise UnsupportedStrictJsonSchemaError("anyOf must contain at least one schema")
        for variant in any_of:
            if _is_structured_schema(variant):
                raise UnsupportedStrictJsonSchemaError("object and array unions are unsupported")
            _make_json_schema_node_strict(variant)

    items = schema.get("items")
    if items is not None:
        if isinstance(items, list):
            raise UnsupportedStrictJsonSchemaError("tuple schemas are unsupported")
        _make_json_schema_node_strict(items)

    is_object_schema = schema.get("type") == "object"
    if schema.get("properties") is not None and not is_object_schema:
        raise UnsupportedStrictJsonSchemaError("properties require type object")
    if not is_object_schema:
        return
    additional = schema.get("additionalProperties", _MISSING)
    if additional is not _MISSING and additional is not False:
        raise UnsupportedStrictJsonSchemaError("schema-valued or true additionalProperties is unsupported")
    properties = schema.get("properties")
    if properties is not None and not _is_json_schema_object(properties):
        raise UnsupportedStrictJsonSchemaError("object properties must be a schema map")
    required_raw = schema.get("required")
    if required_raw is not None and (
        not isinstance(required_raw, list) or any(not isinstance(k, str) for k in required_raw)
    ):
        raise UnsupportedStrictJsonSchemaError("object required must be a string array")

    properties = properties if properties is not None else {}
    property_names = list(properties.keys())
    required = set(required_raw) if isinstance(required_raw, list) else set()
    if any(key not in property_names for key in required):
        raise UnsupportedStrictJsonSchemaError("required contains an unknown property")
    for key, prop in list(properties.items()):
        _make_json_schema_node_strict(prop)
        if key not in required and not _schema_allows_null(prop):
            # Optional properties become nullable instead of optional, because
            # strict mode requires every property to be listed in `required`.
            properties[key] = {"anyOf": [prop, {"type": "null"}]}
    schema["required"] = property_names
    schema["additionalProperties"] = False


def make_strict_json_schema(schema: Any) -> dict[str, Any]:
    """Convert a tool schema to the strict subset provider constrained sampling expects.

    Port of `makeStrictJsonSchema` (`api/constrained-sampling.ts:117`). Works on
    a deep copy: the tool's own `parameters` must not be mutated, since the same
    `Tool` object is reused across requests.
    """
    cloned = deepcopy(schema)
    if not _is_json_schema_object(cloned):
        raise UnsupportedStrictJsonSchemaError("root schema must have type object")
    _make_json_schema_node_strict(cloned)
    if cloned.get("type") != "object":
        raise UnsupportedStrictJsonSchemaError("root schema must have type object")
    return cloned


def get_json_schema_tool_parameters(tool: Tool, strict: bool | None) -> Any:
    """Port of `getJsonSchemaToolParameters`. Only `strict is True` transforms."""
    return make_strict_json_schema(tool.parameters) if strict is True else tool.parameters
