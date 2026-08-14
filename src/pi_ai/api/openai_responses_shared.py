"""Shared OpenAI Responses API conversion and streaming logic.

Python port of `packages/ai/src/api/openai-responses-shared.ts`. Used by both
`openai_responses.py` (OpenAI, and any Responses-compatible provider) and
`azure_openai_responses.py`. Owns input-item conversion, tool conversion, the
streaming event state machine, usage parsing and stop-reason mapping.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Literal

from ..models import calculate_cost
from ..types import (
    AssistantMessage,
    Context,
    Model,
    StopReason,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.hash import short_hash
from ..utils.json_parse import parse_streaming_json
from ..utils.json_stringify import json_stringify
from ..utils.sanitize_unicode import sanitize_surrogates
from .constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    get_grammar_tool_input,
    get_json_schema_tool_parameters,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from .transform_messages import transform_messages

# =============================================================================
# Utilities
# =============================================================================


def _encode_text_signature_v1(item_id: str, phase: str | None = None) -> str:
    payload: dict[str, Any] = {"v": 1, "id": item_id}
    if phase:
        payload["phase"] = phase
    return json_stringify(payload)


def _parse_text_signature(signature: str | None) -> tuple[str, str | None] | None:
    if not signature:
        return None
    if signature.startswith("{"):
        try:
            parsed = json.loads(signature)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("v") == 1 and isinstance(parsed.get("id"), str):
            phase = parsed.get("phase")
            if phase in ("commentary", "final_answer"):
                return parsed["id"], phase
            return parsed["id"], None
    return signature, None


def _model_supports_developer_role(model: Model) -> bool:
    """Whether `model.compat` opts out of the `developer` role. Default: true."""
    for key in ("supportsDeveloperRole", "supports_developer_role"):
        if key in model.compat:
            return model.compat[key] is not False
    return True


def _convert_tool_result_output(model: Model, content: list[Any]) -> str | list[dict[str, Any]]:
    text_result = "\n".join(block.text for block in content if block.type == "text")
    images = [block for block in content if block.type == "image"]
    has_text = len(text_result) > 0

    if not images or "image" not in model.input:
        if has_text:
            return sanitize_surrogates(text_result)
        if images:
            return sanitize_surrogates("(see attached image)")
        return sanitize_surrogates("(no tool output)")

    output: list[dict[str, Any]] = []
    if has_text:
        output.append({"type": "input_text", "text": sanitize_surrogates(text_result)})
    for image in images:
        output.append(
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:{image.mime_type};base64,{image.data}",
            }
        )
    return output


@dataclass
class ConvertResponsesToolsOptions:
    strict: bool | None = None
    supports_strict_mode: bool = True
    supports_openai_grammar_tools: bool = False
    defer_loading: bool = False


@dataclass
class ConvertResponsesMessagesOptions:
    include_system_prompt: bool = True
    grammar_tool_input_properties: dict[str, str] | None = None
    deferred_tools: dict[str, Tool] | None = None
    deferred_tools_mode: Literal["additional-tools", "tool-search"] | None = None
    tool_options: ConvertResponsesToolsOptions | None = None


# =============================================================================
# Message conversion
# =============================================================================

_NON_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def _normalize_id_part(part: str) -> str:
    sanitized = _NON_ID_CHARS.sub("_", part)
    normalized = sanitized[:64]
    return normalized.rstrip("_")


def _build_foreign_responses_item_id(item_id: str) -> str:
    normalized = f"fc_{short_hash(item_id)}"
    return normalized[:64]


def convert_responses_messages(
    model: Model,
    context: Context,
    allowed_tool_call_providers: set[str],
    options: ConvertResponsesMessagesOptions | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    loaded_tool_names: set[str] = set()

    def normalize_tool_call_id(tool_call_id: str, _target_model: Model, source: AssistantMessage) -> str:
        if model.provider not in allowed_tool_call_providers:
            return _normalize_id_part(tool_call_id)
        if "|" not in tool_call_id:
            return _normalize_id_part(tool_call_id)
        call_id, item_id = tool_call_id.split("|", 1)
        normalized_call_id = _normalize_id_part(call_id)
        is_foreign_tool_call = source.provider != model.provider or source.api != model.api
        normalized_item_id = (
            _build_foreign_responses_item_id(item_id) if is_foreign_tool_call else _normalize_id_part(item_id)
        )
        # OpenAI Responses API requires item id to start with "fc"
        if not normalized_item_id.startswith("fc_"):
            normalized_item_id = _normalize_id_part(f"fc_{normalized_item_id}")
        return f"{normalized_call_id}|{normalized_item_id}"

    transformed_messages = transform_messages(context.messages, model, normalize_tool_call_id)

    include_system_prompt = options.include_system_prompt if options is not None else True
    if include_system_prompt and context.system_prompt:
        role = "developer" if model.reasoning and _model_supports_developer_role(model) else "system"
        messages.append({"role": role, "content": sanitize_surrogates(context.system_prompt)})

    grammar_tool_input_properties = options.grammar_tool_input_properties if options is not None else None
    deferred_tools = options.deferred_tools if options is not None else None
    deferred_tools_mode = options.deferred_tools_mode if options is not None else None
    tool_options = options.tool_options if options is not None else None

    msg_index = 0
    for msg in transformed_messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": sanitize_surrogates(msg.content)}],
                    }
                )
            else:
                content: list[dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        content.append({"type": "input_text", "text": sanitize_surrogates(item.text)})
                    else:
                        content.append(
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": f"data:{item.mime_type};base64,{item.data}",
                            }
                        )
                if not content:
                    msg_index += 1
                    continue
                messages.append({"role": "user", "content": content})
        elif msg.role == "assistant":
            output: list[dict[str, Any]] = []
            is_same_provider_and_api = msg.provider == model.provider and msg.api == model.api
            is_same_model = is_same_provider_and_api and msg.model == model.id
            is_different_model = is_same_provider_and_api and msg.model != model.id
            text_block_index = 0

            for block in msg.content:
                if block.type == "thinking":
                    if block.thinking_signature:
                        output.append(json.loads(block.thinking_signature))
                elif block.type == "text":
                    parsed_signature = _parse_text_signature(block.text_signature)
                    fallback_message_id = (
                        f"msg_pi_{msg_index}" if text_block_index == 0 else f"msg_pi_{msg_index}_{text_block_index}"
                    )
                    text_block_index += 1
                    msg_id = parsed_signature[0] if parsed_signature else None
                    phase = parsed_signature[1] if parsed_signature else None
                    if not msg_id:
                        msg_id = fallback_message_id
                    elif len(msg_id) > 64:
                        msg_id = f"msg_{short_hash(msg_id)}"
                    item: dict[str, Any] = {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": sanitize_surrogates(block.text), "annotations": []}
                        ],
                        "status": "completed",
                        "id": msg_id,
                    }
                    if phase:
                        item["phase"] = phase
                    output.append(item)
                elif block.type == "toolCall":
                    if "|" in block.id:
                        call_id, item_id = block.id.split("|", 1)
                    else:
                        call_id, item_id = block.id, None
                    custom_input_property = (
                        grammar_tool_input_properties.get(block.name) if grammar_tool_input_properties else None
                    )

                    # For different-model messages, drop id to avoid pairing validation.
                    # OpenAI tracks which fc_xxx IDs were paired with rs_xxx reasoning items.
                    # By omitting the id, we avoid triggering that validation (like cross-provider does).
                    # When replaying custom-tool calls as a function_call, also drop non-fc_* ids such as
                    # ctc_* custom-tool ids because function_call item ids must be fc_*.
                    if (is_different_model and item_id is not None and item_id.startswith("fc_")) or (
                        custom_input_property is None and not (item_id is not None and item_id.startswith("fc_"))
                    ):
                        item_id = None

                    can_replay_namespace = is_same_model or (
                        deferred_tools is not None and block.name in deferred_tools
                    )

                    if custom_input_property is not None:
                        tool_item: dict[str, Any] = {
                            "type": "custom_tool_call",
                            "call_id": call_id,
                            "name": block.name,
                            "input": sanitize_surrogates(
                                get_grammar_tool_input(block.name, block.arguments, custom_input_property)
                            ),
                        }
                        if item_id is not None:
                            tool_item["id"] = item_id
                        if can_replay_namespace and block.namespace is not None:
                            tool_item["namespace"] = block.namespace
                        output.append(tool_item)
                    else:
                        function_item: dict[str, Any] = {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": block.name,
                            "arguments": json_stringify(block.arguments),
                        }
                        if item_id is not None:
                            function_item["id"] = item_id
                        if can_replay_namespace and block.namespace is not None:
                            function_item["namespace"] = block.namespace
                        output.append(function_item)

            if not output:
                msg_index += 1
                continue
            messages.extend(output)
        elif msg.role == "toolResult":
            call_id = msg.tool_call_id.split("|", 1)[0] if "|" in msg.tool_call_id else msg.tool_call_id
            tool_result_output = _convert_tool_result_output(model, msg.content)

            if grammar_tool_input_properties and msg.tool_name in grammar_tool_input_properties:
                messages.append({"type": "custom_tool_call_output", "call_id": call_id, "output": tool_result_output})
            else:
                messages.append({"type": "function_call_output", "call_id": call_id, "output": tool_result_output})

            deferred_tool_list: list[Tool] = []
            for name in msg.added_tool_names or []:
                tool = deferred_tools.get(name) if deferred_tools else None
                if not tool or name in loaded_tool_names:
                    continue
                loaded_tool_names.add(name)
                deferred_tool_list.append(tool)

            if deferred_tool_list and deferred_tools_mode == "additional-tools":
                messages.append(
                    {
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": convert_responses_tools(deferred_tool_list, tool_options),
                    }
                )
            elif deferred_tool_list and deferred_tools_mode == "tool-search":
                names = [tool.name for tool in deferred_tool_list]
                joined_names = ",".join(names)
                search_call_id = f"pi_tool_load_{short_hash(f'{msg.tool_call_id}:{joined_names}')}"
                messages.append(
                    {
                        "type": "tool_search_call",
                        "call_id": search_call_id,
                        "execution": "client",
                        "status": "completed",
                        "arguments": {"query": " ".join(names), "limit": len(names)},
                    }
                )
                deferred_options = ConvertResponsesToolsOptions(
                    strict=tool_options.strict if tool_options else None,
                    supports_strict_mode=tool_options.supports_strict_mode if tool_options else True,
                    supports_openai_grammar_tools=tool_options.supports_openai_grammar_tools if tool_options else False,
                    defer_loading=True,
                )
                messages.append(
                    {
                        "type": "tool_search_output",
                        "call_id": search_call_id,
                        "execution": "client",
                        "status": "completed",
                        "tools": convert_responses_tools(deferred_tool_list, deferred_options),
                    }
                )
        msg_index += 1

    return messages


# =============================================================================
# Tool conversion
# =============================================================================


def convert_responses_tools(
    tools: list[Tool], options: ConvertResponsesToolsOptions | None = None
) -> list[dict[str, Any]]:
    options = options or ConvertResponsesToolsOptions()
    default_strict = False if options.strict is None else options.strict
    supports_strict_mode = options.supports_strict_mode
    supports_openai_grammar_tools = options.supports_openai_grammar_tools

    converted: list[dict[str, Any]] = []
    for tool in tools:
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar is not None:
            entry: dict[str, Any] = {
                "type": "custom",
                "name": tool.name,
                "description": tool.description,
                "format": {"type": "grammar", "syntax": grammar.format, "definition": grammar.definition},
            }
            if options.defer_loading:
                entry["defer_loading"] = True
            converted.append(entry)
            continue

        constrained_strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        function_tool: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            # `getJsonSchemaToolParameters(tool, strict === true)` upstream: the
            # strict subset, not the raw schema.
            "parameters": get_json_schema_tool_parameters(tool, constrained_strict is True),
        }
        if options.defer_loading:
            function_tool["defer_loading"] = True
        if supports_strict_mode:
            function_tool["strict"] = constrained_strict if constrained_strict is not None else default_strict
        converted.append(function_tool)
    return converted


# =============================================================================
# Stream processing
# =============================================================================


@dataclass
class _CustomToolCallInput:
    property: str
    json_buffer: GrammarToolInputJsonBuffer


@dataclass
class _StreamingToolCall:
    """Scratch state for a streaming tool call, kept out of the replayed `ToolCall`."""

    block: ToolCall
    partial_json: str | None = None
    custom_input: _CustomToolCallInput | None = None


@dataclass
class _ThinkingSlot:
    block: ThinkingContent
    content_index: int
    type: Literal["thinking"] = "thinking"


@dataclass
class _TextSlot:
    block: TextContent
    content_index: int
    type: Literal["text"] = "text"


@dataclass
class _ToolCallSlot:
    entry: _StreamingToolCall
    content_index: int
    type: Literal["toolCall"] = "toolCall"


ResponsesOutputSlot = _ThinkingSlot | _TextSlot | _ToolCallSlot


def _get_custom_tool_call_input(entry: _StreamingToolCall) -> str:
    if entry.custom_input is None:
        return ""
    value = entry.block.arguments.get(entry.custom_input.property)
    return value if isinstance(value, str) else ""


def _append_custom_tool_call_input(entry: _StreamingToolCall, next_input: str, close: bool) -> str | None:
    custom_input = entry.custom_input
    if custom_input is None:
        return None
    delta = append_grammar_tool_input_json_delta(custom_input.json_buffer, custom_input.property, next_input, close)
    entry.block.arguments = {custom_input.property: next_input}
    return delta


@dataclass
class OpenAIResponsesStreamOptions:
    service_tier: str | None = None
    grammar_tool_input_properties: dict[str, str] | None = None
    resolve_service_tier: Callable[[str | None, str | None], str | None] | None = None
    apply_service_tier_pricing: Callable[[Usage, str | None], None] | None = None


async def process_responses_stream(
    openai_stream: AsyncIterator[dict[str, Any]],
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    options: OpenAIResponsesStreamOptions | None = None,
) -> None:
    saw_terminal_response_event = False
    output_slots: dict[int, ResponsesOutputSlot] = {}
    reasoning_blocks_by_id: dict[str, ThinkingContent] = {}

    def apply_message_phase_stop_reason(item: dict[str, Any]) -> None:
        if item.get("type") == "message" and item.get("phase") == "final_answer":
            output.stop_reason = "stop"

    def get_slot(output_index: int, slot_type: str) -> ResponsesOutputSlot | None:
        slot = output_slots.get(output_index)
        return slot if slot is not None and slot.type == slot_type else None

    def push_tool_call_delta(slot: _ToolCallSlot, delta: str | None) -> None:
        if delta is None:
            return
        stream.push(ToolCallDeltaEvent(content_index=slot.content_index, delta=delta, partial=output))

    def create_slot(output_index: int, item: dict[str, Any]) -> ResponsesOutputSlot | None:
        item_type = item.get("type")
        if item_type == "reasoning":
            block = ThinkingContent(thinking="")
            output.content.append(block)
            slot: ResponsesOutputSlot = _ThinkingSlot(block=block, content_index=len(output.content) - 1)
            output_slots[output_index] = slot
            stream.push(ThinkingStartEvent(content_index=slot.content_index, partial=output))
            return slot
        if item_type == "message":
            apply_message_phase_stop_reason(item)
            text_block = TextContent(text="")
            output.content.append(text_block)
            slot = _TextSlot(block=text_block, content_index=len(output.content) - 1)
            output_slots[output_index] = slot
            stream.push(TextStartEvent(content_index=slot.content_index, partial=output))
            return slot
        if item_type == "function_call":
            tool_call = ToolCall(
                id=f"{item.get('call_id')}|{item.get('id')}",
                name=item.get("name", ""),
                arguments={},
                namespace=item.get("namespace"),
            )
            entry = _StreamingToolCall(block=tool_call, partial_json=item.get("arguments") or "")
            output.content.append(tool_call)
            slot = _ToolCallSlot(entry=entry, content_index=len(output.content) - 1)
            output_slots[output_index] = slot
            stream.push(ToolCallStartEvent(content_index=slot.content_index, partial=output))
            return slot
        if item_type == "custom_tool_call":
            input_property = (
                (options.grammar_tool_input_properties or {}).get(item.get("name", "")) if options else None
            ) or "input"
            input_value = item.get("input") or ""
            tool_call = ToolCall(
                id=f"{item.get('call_id')}|{item.get('id')}",
                name=item.get("name", ""),
                arguments={input_property: input_value},
                namespace=item.get("namespace"),
            )
            entry = _StreamingToolCall(
                block=tool_call,
                custom_input=_CustomToolCallInput(
                    property=input_property,
                    json_buffer=GrammarToolInputJsonBuffer(input="", started=False, closed=False),
                ),
            )
            output.content.append(tool_call)
            slot = _ToolCallSlot(entry=entry, content_index=len(output.content) - 1)
            output_slots[output_index] = slot
            stream.push(ToolCallStartEvent(content_index=slot.content_index, partial=output))
            return slot
        return None

    def get_or_create_slot(output_index: int, item: dict[str, Any]) -> ResponsesOutputSlot | None:
        return output_slots.get(output_index) or create_slot(output_index, item)

    def backfill_reasoning_signatures(response_output: list[dict[str, Any]]) -> None:
        # Azure OpenAI can omit reasoning.encrypted_content from response.output_item.done
        # and provide it only in response.completed.response.output. Backfill the
        # persisted reasoning signature from the terminal response to keep store:false
        # multi-turn replay stateless. See https://github.com/earendil-works/pi/issues/6409.
        for item in response_output:
            if item.get("type") != "reasoning" or not item.get("encrypted_content"):
                continue
            block = reasoning_blocks_by_id.get(item.get("id"))
            if block is None or not block.thinking_signature:
                continue
            stored_item = json.loads(block.thinking_signature)
            if stored_item.get("encrypted_content"):
                continue
            stored_item["encrypted_content"] = item["encrypted_content"]
            block.thinking_signature = json_stringify(stored_item)

    def finalize_response(response: dict[str, Any]) -> None:
        nonlocal saw_terminal_response_event
        saw_terminal_response_event = True
        backfill_reasoning_signatures(response.get("output") or [])
        if response.get("id"):
            output.response_id = response["id"]

        usage = response.get("usage")
        if usage:
            input_details = usage.get("input_tokens_details") or {}
            cached_tokens = input_details.get("cached_tokens") or 0
            cache_write_tokens = input_details.get("cache_write_tokens") or 0
            output.usage = Usage(
                # OpenAI includes cached and cache-write tokens in input_tokens, so subtract both.
                input=max(0, (usage.get("input_tokens") or 0) - cached_tokens - cache_write_tokens),
                output=usage.get("output_tokens") or 0,
                cache_read=cached_tokens,
                cache_write=cache_write_tokens,
                reasoning=(usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0,
                total_tokens=usage.get("total_tokens") or 0,
            )
        calculate_cost(model, output.usage)
        if options is not None and options.apply_service_tier_pricing is not None:
            if options.resolve_service_tier is not None:
                service_tier = options.resolve_service_tier(response.get("service_tier"), options.service_tier)
            else:
                service_tier = response.get("service_tier") or options.service_tier
            options.apply_service_tier_pricing(output.usage, service_tier)

        # Map status to stop reason. For incomplete responses, retain the provider's
        # specific reason so max-output truncation and content filtering stay distinct.
        status = response.get("status")
        incomplete_details = response.get("incomplete_details") or {}
        incomplete_reason = (
            incomplete_details.get("reason") if isinstance(incomplete_details.get("reason"), str) else None
        )
        output.raw_stop_reason = f"{status}.{incomplete_reason}" if incomplete_reason else status
        mapped_stop_reason, error_message = map_stop_reason(status, incomplete_reason)
        output.stop_reason = mapped_stop_reason
        output.error_message = error_message
        if any(block.type == "toolCall" for block in output.content) and output.stop_reason == "stop":
            output.stop_reason = "toolUse"

    async for event in openai_stream:
        event_type = event.get("type")
        if event_type == "response.created":
            output.response_id = event["response"]["id"]
        elif event_type == "response.output_item.added":
            create_slot(event["output_index"], event["item"])
        elif event_type == "response.reasoning_summary_text.delta":
            slot = get_slot(event["output_index"], "thinking")
            if slot is None:
                continue
            slot.block.thinking += event["delta"]
            stream.push(ThinkingDeltaEvent(content_index=slot.content_index, delta=event["delta"], partial=output))
        elif event_type == "response.reasoning_summary_part.done":
            slot = get_slot(event["output_index"], "thinking")
            if slot is None:
                continue
            slot.block.thinking += "\n\n"
            stream.push(ThinkingDeltaEvent(content_index=slot.content_index, delta="\n\n", partial=output))
        elif event_type == "response.reasoning_text.delta":
            slot = get_slot(event["output_index"], "thinking")
            if slot is None:
                continue
            slot.block.thinking += event["delta"]
            stream.push(ThinkingDeltaEvent(content_index=slot.content_index, delta=event["delta"], partial=output))
        elif event_type == "response.output_text.delta" or event_type == "response.refusal.delta":
            slot = get_slot(event["output_index"], "text")
            if slot is None:
                continue
            slot.block.text += event["delta"]
            stream.push(TextDeltaEvent(content_index=slot.content_index, delta=event["delta"], partial=output))
        elif event_type == "response.function_call_arguments.delta":
            slot = get_slot(event["output_index"], "toolCall")
            if slot is None or slot.entry.partial_json is None:
                continue
            slot.entry.partial_json += event["delta"]
            slot.entry.block.arguments = parse_streaming_json(slot.entry.partial_json)
            push_tool_call_delta(slot, event["delta"])
        elif event_type == "response.function_call_arguments.done":
            slot = get_slot(event["output_index"], "toolCall")
            if slot is None or slot.entry.partial_json is None:
                continue
            previous_partial_json = slot.entry.partial_json
            slot.entry.partial_json = event["arguments"]
            slot.entry.block.arguments = parse_streaming_json(slot.entry.partial_json)
            if event["arguments"].startswith(previous_partial_json):
                delta = event["arguments"][len(previous_partial_json) :]
                if delta:
                    push_tool_call_delta(slot, delta)
        elif event_type == "response.custom_tool_call_input.delta":
            slot = get_slot(event["output_index"], "toolCall")
            if slot is None or slot.entry.custom_input is None:
                continue
            push_tool_call_delta(
                slot,
                _append_custom_tool_call_input(
                    slot.entry, _get_custom_tool_call_input(slot.entry) + event["delta"], False
                ),
            )
        elif event_type == "response.custom_tool_call_input.done":
            slot = get_slot(event["output_index"], "toolCall")
            if slot is None or slot.entry.custom_input is None:
                continue
            push_tool_call_delta(slot, _append_custom_tool_call_input(slot.entry, event["input"], True))
        elif event_type == "response.output_item.done":
            item = event["item"]
            apply_message_phase_stop_reason(item)
            slot = get_or_create_slot(event["output_index"], item)
            item_type = item.get("type")

            if item_type == "reasoning" and isinstance(slot, _ThinkingSlot):
                summary_text = "\n\n".join(s.get("text", "") for s in (item.get("summary") or []))
                content_text = "\n\n".join(c.get("text", "") for c in (item.get("content") or []))
                slot.block.thinking = summary_text or content_text or slot.block.thinking
                slot.block.thinking_signature = json_stringify(item)
                reasoning_blocks_by_id[item["id"]] = slot.block
                stream.push(
                    ThinkingEndEvent(content_index=slot.content_index, content=slot.block.thinking, partial=output)
                )
                del output_slots[event["output_index"]]
            elif item_type == "message" and isinstance(slot, _TextSlot):
                parts = []
                for content_item in item.get("content") or []:
                    if content_item.get("type") == "output_text":
                        parts.append(content_item.get("text", ""))
                    else:
                        parts.append(content_item.get("refusal", "") or "")
                slot.block.text = "".join(parts)
                slot.block.text_signature = _encode_text_signature_v1(item["id"], item.get("phase"))
                stream.push(TextEndEvent(content_index=slot.content_index, content=slot.block.text, partial=output))
                del output_slots[event["output_index"]]
            elif (
                item_type == "function_call" and isinstance(slot, _ToolCallSlot) and slot.entry.partial_json is not None
            ):
                slot.entry.block.arguments = parse_streaming_json(
                    item.get("arguments") or slot.entry.partial_json or "{}"
                )
                if item.get("namespace") is not None:
                    slot.entry.block.namespace = item["namespace"]
                # Finalize in-place and drop the scratch buffer so replay only
                # carries parsed arguments.
                slot.entry.partial_json = None
                stream.push(
                    ToolCallEndEvent(content_index=slot.content_index, tool_call=slot.entry.block, partial=output)
                )
                del output_slots[event["output_index"]]
            elif (
                item_type == "custom_tool_call"
                and isinstance(slot, _ToolCallSlot)
                and slot.entry.custom_input is not None
            ):
                push_tool_call_delta(
                    slot,
                    _append_custom_tool_call_input(
                        slot.entry, item.get("input") or _get_custom_tool_call_input(slot.entry), True
                    ),
                )
                if item.get("namespace") is not None:
                    slot.entry.block.namespace = item["namespace"]
                slot.entry.custom_input = None
                stream.push(
                    ToolCallEndEvent(content_index=slot.content_index, tool_call=slot.entry.block, partial=output)
                )
                del output_slots[event["output_index"]]
        elif event_type in ("response.completed", "response.incomplete"):
            finalize_response(event["response"])
        elif event_type == "error":
            raise RuntimeError(f"Error Code {event.get('code')}: {event.get('message')}")
        elif event_type == "response.failed":
            saw_terminal_response_event = True
            response = event.get("response") or {}
            output.raw_stop_reason = response.get("status")
            error = response.get("error")
            details = response.get("incomplete_details")
            if error:
                message = f"{error.get('code') or 'unknown'}: {error.get('message') or 'no message'}"
            elif details and details.get("reason"):
                message = f"incomplete: {details['reason']}"
            else:
                message = "Unknown error (no error details in response)"
            raise RuntimeError(message)

    if not saw_terminal_response_event:
        raise RuntimeError("OpenAI Responses stream ended before a terminal response event")


def map_stop_reason(status: str | None, incomplete_reason: str | None = None) -> tuple[StopReason, str | None]:
    if not status:
        return "stop", None
    if status == "completed":
        return "stop", None
    if status == "incomplete":
        if incomplete_reason == "max_output_tokens":
            return "length", None
        return "error", (
            f"Response incomplete: {incomplete_reason}"
            if incomplete_reason
            else "Response incomplete without a provider reason"
        )
    if status in ("failed", "cancelled"):
        return "error", None
    # These two are wonky ...
    if status in ("in_progress", "queued"):
        return "stop", None
    raise ValueError(f"Unhandled stop reason: {status}")
