"""Tool-argument validation and JSON Schema coercion.

Python port of `packages/ai/src/utils/validation.ts`. TypeScript compiles the
tool schema with TypeBox; here the schema is plain JSON Schema validated with
:mod:`jsonschema`, preceded by the same best-effort coercion pass so that
models that emit ``"3"`` for an integer field still succeed.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from ..types import Tool, ToolCall

_validator_cache: dict[int, tuple[dict[str, Any], Draft202012Validator]] = {}


class ToolValidationError(ValueError):
    """Raised when tool call arguments do not satisfy the tool schema."""


def _get_schema_types(schema: dict[str, Any]) -> list[str]:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return [schema_type]
    if isinstance(schema_type, list):
        return [t for t in schema_type if isinstance(t, str)]
    return []


def _matches_json_type(value: Any, json_type: str) -> bool:
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "null":
        return value is None
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    return False


def _coerce_primitive_by_type(value: Any, json_type: str) -> Any:
    if json_type in ("number", "integer"):
        if value is None:
            return 0
        if isinstance(value, str) and value.strip():
            try:
                parsed = float(value)
            except ValueError:
                return value
            if json_type == "integer":
                return int(parsed) if parsed.is_integer() else value
            return int(parsed) if parsed.is_integer() and "." not in value and "e" not in value.lower() else parsed
        if isinstance(value, bool):
            return 1 if value else 0
        return value
    if json_type == "boolean":
        if value is None:
            return False
        if value == "true":
            return True
        if value == "false":
            return False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
        return value
    if json_type == "string":
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return _number_to_string(value)
        return value
    if json_type == "null":
        if value == "" or value == 0 or value is False:
            return None
        return value
    return value


def _number_to_string(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _get_sub_schema_validator(schema: Any) -> Draft202012Validator | None:
    if not isinstance(schema, dict):
        return None
    try:
        return _get_validator(schema)
    except Exception:
        return None


def _apply_schema_object_coercion(value: dict[str, Any], schema: dict[str, Any]) -> None:
    properties = schema.get("properties")
    defined_keys = set(properties.keys()) if isinstance(properties, dict) else set()

    if isinstance(properties, dict):
        for key, property_schema in properties.items():
            if key not in value:
                continue
            value[key] = _coerce_with_json_schema(value[key], property_schema)

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        for key in list(value.keys()):
            if key in defined_keys:
                continue
            value[key] = _coerce_with_json_schema(value[key], additional)


def _apply_schema_array_coercion(value: list[Any], schema: dict[str, Any]) -> None:
    items = schema.get("items")
    if isinstance(items, list):
        for index in range(len(value)):
            if index >= len(items):
                continue
            item_schema = items[index]
            if not isinstance(item_schema, dict):
                continue
            value[index] = _coerce_with_json_schema(value[index], item_schema)
        return

    if isinstance(items, dict):
        for index in range(len(value)):
            value[index] = _coerce_with_json_schema(value[index], items)


def _coerce_with_union_schema(value: Any, schemas: list[Any]) -> Any:
    for schema in schemas:
        validator = _get_sub_schema_validator(schema)
        if validator is not None and validator.is_valid(value):
            return value

    for schema in schemas:
        if not isinstance(schema, dict):
            continue
        coerced = _coerce_with_json_schema(copy.deepcopy(value), schema)
        validator = _get_sub_schema_validator(schema)
        if validator is not None and validator.is_valid(coerced):
            return coerced
    return value


def _coerce_with_json_schema(value: Any, schema: Any) -> Any:
    if not isinstance(schema, dict):
        return value

    next_value = value

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for nested in all_of:
            next_value = _coerce_with_json_schema(next_value, nested)

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        next_value = _coerce_with_union_schema(next_value, any_of)

    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        next_value = _coerce_with_union_schema(next_value, one_of)

    schema_types = _get_schema_types(schema)
    matches_union_member = len(schema_types) > 1 and any(
        _matches_json_type(next_value, schema_type) for schema_type in schema_types
    )
    if schema_types and not matches_union_member:
        for schema_type in schema_types:
            candidate = _coerce_primitive_by_type(next_value, schema_type)
            if candidate is not next_value:
                next_value = candidate
                break

    if "object" in schema_types and isinstance(next_value, dict):
        _apply_schema_object_coercion(next_value, schema)

    if "array" in schema_types and isinstance(next_value, list):
        _apply_schema_array_coercion(next_value, schema)

    return next_value


def _normalize_tuple_items(schema: Any) -> Any:
    """Translate legacy tuple-style ``items`` arrays into Draft 2020-12 ``prefixItems``.

    Tool schemas may use ``items: [schemaA, schemaB, ...]`` to validate a fixed-length
    tuple (as produced by TypeBox's ``Type.Tuple`` on the TypeScript side, whose
    ``Compile`` validator accepts this form). ``jsonschema``'s ``Draft202012Validator``
    only understands the 2020-12 keywords, where a per-index ``items`` array must be
    expressed as ``prefixItems`` and ``items`` (if present) applies to any elements
    beyond the tuple; passing the legacy array form straight through raises
    ``AttributeError`` deep inside the validator instead of validating. Additional
    items beyond the tuple length were implicitly allowed under the legacy form, so
    the ``items`` keyword is dropped rather than set to ``false``.
    """
    if isinstance(schema, list):
        return [_normalize_tuple_items(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    normalized = dict(schema)
    items = normalized.get("items")
    if isinstance(items, list):
        normalized["prefixItems"] = [_normalize_tuple_items(item) for item in items]
        del normalized["items"]
    elif items is not None:
        normalized["items"] = _normalize_tuple_items(items)

    for key in ("properties", "patternProperties"):
        value = normalized.get(key)
        if isinstance(value, dict):
            normalized[key] = {k: _normalize_tuple_items(v) for k, v in value.items()}

    additional = normalized.get("additionalProperties")
    if isinstance(additional, dict):
        normalized["additionalProperties"] = _normalize_tuple_items(additional)

    for key in ("allOf", "anyOf", "oneOf"):
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = [_normalize_tuple_items(item) for item in value]

    return normalized


def _get_validator(schema: dict[str, Any]) -> Draft202012Validator:
    # The cache keeps a reference to the schema so its id() cannot be reused.
    key = id(schema)
    cached = _validator_cache.get(key)
    if cached is not None and cached[0] is schema:
        return cached[1]
    validator = Draft202012Validator(_normalize_tuple_items(schema))
    _validator_cache[key] = (schema, validator)
    return validator


def _format_validation_path(error: ValidationError) -> str:
    parts = [str(part) for part in error.absolute_path]
    if error.validator == "required":
        missing = _missing_property(error)
        if missing:
            parts.append(missing)
    return ".".join(parts) if parts else "root"


def _missing_property(error: ValidationError) -> str | None:
    message = error.message
    if "'" in message:
        return message.split("'")[1]
    return None


def validate_tool_call(tools: list[Tool], tool_call: ToolCall) -> dict[str, Any]:
    """Find ``tool_call.name`` in ``tools`` and validate its arguments."""
    tool = next((t for t in tools if t.name == tool_call.name), None)
    if tool is None:
        raise ToolValidationError(f'Tool "{tool_call.name}" not found')
    return validate_tool_arguments(tool, tool_call)


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> dict[str, Any]:
    """Validate and coerce ``tool_call.arguments`` against ``tool.parameters``."""
    args = copy.deepcopy(tool_call.arguments)
    coerced = _coerce_with_json_schema(args, tool.parameters)
    if isinstance(coerced, dict):
        args = coerced

    validator = _get_validator(tool.parameters)
    if validator.is_valid(args):
        return args

    errors = sorted(validator.iter_errors(args), key=lambda e: list(e.absolute_path))
    details = "\n".join(f"  - {_format_validation_path(error)}: {error.message}" for error in errors)
    if not details:
        details = "Unknown validation error"
    received = json.dumps(tool_call.arguments, indent=2, ensure_ascii=False)
    raise ToolValidationError(
        f'Validation failed for tool "{tool_call.name}":\n{details}\n\nReceived arguments:\n{received}'
    )


def string_enum(
    values: Sequence[str],
    description: str | None = None,
    default: str | None = None,
) -> dict[str, Any]:
    """A string-enum tool-parameter schema.

    Python port of `StringEnum` in `packages/ai/src/utils/typebox-helpers.ts`.
    TypeScript needs `Type.Unsafe` to make TypeBox emit a plain
    ``{"type": "string", "enum": [...]}`` schema instead of the ``anyOf``/
    ``const`` form that Google's API and several other providers reject. Python
    tool schemas are already plain dicts, so this just builds that shape
    directly, keeping the helper's name and provider-compatibility guarantee.
    """
    schema: dict[str, Any] = {"type": "string", "enum": list(values)}
    if description:
        schema["description"] = description
    if default:
        schema["default"] = default
    return schema
