"""Python port of `packages/ai/test/mistral-tool-schema.test.ts`.

The TypeScript test builds the tool schema with TypeBox, whose schema objects
carry extra `Symbol` keys that the Mistral SDK rejects; `toFunctionTools`
strips them with `stripSymbolKeys`. Python dicts cannot have symbol keys, so
the three `Object.getOwnPropertySymbols(...)` assertions have no analogue and
are skipped. Everything else the test pins -- one serialized tool, `strict`
true from `constrainedSampling: { strict: "require" }`, the nested schema
surviving serialization, and a transport failure surfacing as an `error`
assistant message rather than a schema validation error -- is ported.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from pi_ai.compat import complete
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import (
    Context,
    JsonSchemaConstrainedSampling,
    StreamOptions,
    Tool,
    UserMessage,
    now_ms,
)


async def test_serializes_nested_tool_schemas_and_the_strict_flag() -> None:
    catalog_model = get_builtin_model("mistral", "devstral-medium-latest")
    assert catalog_model is not None
    # Port 9 is the discard port: the connection is refused immediately, so no
    # request leaves the machine (the TypeScript test uses the same address).
    model = dataclasses.replace(catalog_model, base_url="http://127.0.0.1:9")

    parameters = {
        "type": "object",
        "properties": {"nested": {"type": "object", "properties": {"value": {"type": "string"}}}},
        "required": ["nested"],
    }
    context = Context(
        messages=[UserMessage(content="Hi", timestamp=now_ms())],
        tools=[
            Tool(
                name="inspect_schema",
                description="Inspect the schema",
                parameters=parameters,
                constrained_sampling=JsonSchemaConstrainedSampling(strict="require"),
            )
        ],
    )

    captured: dict[str, Any] = {}

    def on_payload(payload: dict[str, Any], _model: object) -> dict[str, Any]:
        captured["payload"] = payload
        return payload

    response = await complete(model, context, StreamOptions(api_key="fake-key", on_payload=on_payload))

    payload = captured.get("payload")
    assert payload is not None
    tools = payload["tools"]
    assert len(tools) == 1
    assert tools[0]["function"]["strict"] is True
    payload_parameters = tools[0]["function"]["parameters"]
    assert payload_parameters is not None
    # TS asserts `Object.getOwnPropertySymbols(payloadParameters).toHaveLength(0)` here,
    # and again for `properties` and `properties.nested`. Skipped: those pin that
    # `stripSymbolKeys` removed TypeBox's `Symbol` keys, and Python dict keys can never
    # be symbols, so the property holds by construction with nothing to assert against.
    properties = payload_parameters["properties"]
    assert properties
    assert properties["nested"]

    assert response.stop_reason == "error"
    assert "Input validation failed" not in (response.error_message or "")
