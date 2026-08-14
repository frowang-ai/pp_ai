"""Python port of `packages/ai/test/google-shared-gemini3-unsigned-tool-call.test.ts`."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pi_ai.api.google_shared import convert_messages, requires_tool_call_id
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def make_gemini3_model(api: str, provider: str, model_id: str = "gemini-3-pro-preview") -> Model:
    return Model(
        id=model_id,
        name="Gemini 3 Pro Preview",
        api=api,
        provider=provider,
        base_url="https://example.com",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=8192,
    )


def make_context(api: str, provider: str, model_id: str, thought_signature: str | None = None) -> Context:
    return Context(
        messages=[
            UserMessage(content="Hi"),
            AssistantMessage(
                content=[
                    ToolCall(
                        id="call_1",
                        name="bash",
                        arguments={"command": "echo hi"},
                        thought_signature=thought_signature,
                    ),
                    ToolCall(id="call_2", name="bash", arguments={"command": "ls -la"}),
                ],
                api=api,
                provider=provider,
                model=model_id,
                usage=Usage(),
                stop_reason="toolUse",
            ),
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="bash",
                content=[TextContent(text="hi")],
                is_error=False,
            ),
            ToolResultMessage(
                tool_call_id="call_2",
                tool_name="bash",
                content=[TextContent(text="files")],
                is_error=False,
            ),
        ]
    )


def all_parts(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [part for content in contents for part in content.get("parts", [])]


@pytest.mark.parametrize(
    ("api", "provider", "model_id"),
    [
        ("google-generative-ai", "google", "gemini-3-pro-preview"),
        ("google-generative-ai", "google", "gemini-3.6-flash"),
        ("google-vertex", "google-vertex", "gemini-3-pro-preview"),
    ],
)
def test_preserves_tool_call_ids(api: str, provider: str, model_id: str):
    model = make_gemini3_model(api, provider, model_id)
    contents = convert_messages(model, make_context(api, provider, model_id))
    parts = all_parts(contents)

    function_call_ids = [part["functionCall"]["id"] for part in parts if part.get("functionCall", {}).get("id")]
    function_response_ids = [
        part["functionResponse"]["id"] for part in parts if part.get("functionResponse", {}).get("id")
    ]

    assert function_call_ids == ["call_1", "call_2"]
    assert function_response_ids == ["call_1", "call_2"]


def test_no_skip_thought_signature_validator_for_unsigned_google_gen_ai_tool_calls():
    model = make_gemini3_model("google-generative-ai", "google")
    contents = convert_messages(model, make_context("google-generative-ai", "google", "other-model"))

    model_turn = next(content for content in contents if content["role"] == "model")
    function_call_parts = [part for part in model_turn.get("parts", []) if "functionCall" in part]
    assert len(function_call_parts) == 2
    assert function_call_parts[0].get("thoughtSignature") is None
    assert function_call_parts[1].get("thoughtSignature") is None
    assert "skip_thought_signature_validator" not in json.dumps(model_turn)

    text_parts = [part for part in model_turn.get("parts", []) if "text" in part]
    assert [part for part in text_parts if "Historical context" in part["text"]] == []


def test_no_skip_thought_signature_validator_for_unsigned_vertex_tool_calls():
    model = make_gemini3_model("google-vertex", "google-vertex")
    contents = convert_messages(model, make_context("google-vertex", "google-vertex", "gemini-3-pro-preview"))

    model_turn = next(content for content in contents if content["role"] == "model")
    function_call_parts = [part for part in model_turn.get("parts", []) if "functionCall" in part]
    assert len(function_call_parts) == 2
    assert function_call_parts[0].get("thoughtSignature") is None
    assert function_call_parts[1].get("thoughtSignature") is None
    assert "skip_thought_signature_validator" not in json.dumps(model_turn)


def test_preserves_valid_thought_signature_for_the_same_provider_and_model():
    model = make_gemini3_model("google-generative-ai", "google")
    valid_sig = "AAAAAAAAAAAAAAAAAAAAAA=="
    contents = convert_messages(
        model,
        make_context("google-generative-ai", "google", "gemini-3-pro-preview", valid_sig),
    )

    model_turn = next(content for content in contents if content["role"] == "model")
    function_call_parts = [part for part in model_turn.get("parts", []) if "functionCall" in part]
    assert len(function_call_parts) == 2
    assert function_call_parts[0]["thoughtSignature"] == valid_sig
    assert function_call_parts[1].get("thoughtSignature") is None


def test_does_not_add_a_thought_signature_for_non_gemini_3_models():
    model = make_gemini3_model("google-generative-ai", "google", "gemini-2.5-flash")
    contents = convert_messages(model, make_context("google-generative-ai", "google", "other-model"))

    model_turn = next(content for content in contents if content["role"] == "model")
    function_call_parts = [part for part in model_turn.get("parts", []) if "functionCall" in part]
    function_response_parts = [part for part in all_parts(contents) if "functionResponse" in part]

    assert len(function_call_parts) == 2
    assert all(part["functionCall"].get("id") is None for part in function_call_parts)
    assert all(part.get("thoughtSignature") is None for part in function_call_parts)
    assert len(function_response_parts) == 2
    assert all(part["functionResponse"].get("id") is None for part in function_response_parts)


@pytest.mark.parametrize(
    ("expected", "model_id"),
    [
        (False, "gemini-2.5-flash"),
        (True, "gemini-3.6-flash"),
        (True, "claude-sonnet-4-5"),
        (True, "gpt-oss-120b"),
    ],
)
def test_requires_tool_call_id(expected: bool, model_id: str):
    assert requires_tool_call_id(model_id) is expected
