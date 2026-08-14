"""Core data types for pi_ai.

Python port of `packages/ai/src/types.ts`. TypeScript discriminated unions are
modelled as dataclasses carrying a literal ``type``/``role`` tag so that
``isinstance`` and tag checks both work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .utils.abort import AbortSignal

JsonValue = str | int | float | bool | None | list[Any] | dict[str, Any]

KnownApi = Literal[
    "openai-completions",
    "mistral-conversations",
    "openai-responses",
    "azure-openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "bedrock-converse-stream",
    "google-generative-ai",
    "google-vertex",
    "pi-messages",
]
Api = str
ImagesApi = str
ProviderId = str
ImagesProviderId = str

ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
ModelThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]
CacheRetention = Literal["none", "short", "long"]
Transport = Literal["sse", "websocket", "websocket-cached", "auto"]
SessionAffinityFormat = Literal["openai", "openai-nosession", "openrouter"]
StopReason = Literal["pending", "stop", "length", "toolUse", "error", "aborted", "deferred"]
ImagesStopReason = Literal["stop", "error", "aborted"]
GrammarFormat = Literal["openai_lark", "openai_regex"]

THINKING_LEVELS: tuple[ModelThinkingLevel, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


def now_ms() -> int:
    """Unix timestamp in milliseconds, matching JavaScript's ``Date.now()``."""
    return int(time.time() * 1000)


@dataclass
class ThinkingBudgets:
    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


# --------------------------------------------------------------------------
# Content blocks
# --------------------------------------------------------------------------


@dataclass
class TextContent:
    text: str
    text_signature: str | None = None
    type: Literal["text"] = "text"


@dataclass
class ThinkingContent:
    thinking: str
    thinking_signature: str | None = None
    redacted: bool | None = None
    type: Literal["thinking"] = "thinking"


@dataclass
class ImageContent:
    data: str
    """Base64-encoded image bytes."""
    mime_type: str
    type: Literal["image"] = "image"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    thought_signature: str | None = None
    namespace: str | None = None
    type: Literal["toolCall"] = "toolCall"


Content = TextContent | ThinkingContent | ImageContent | ToolCall
AssistantContent = TextContent | ThinkingContent | ToolCall
UserContent = TextContent | ImageContent


@dataclass
class TextSignatureV1:
    id: str
    phase: Literal["commentary", "final_answer"] | None = None
    v: Literal[1] = 1


# --------------------------------------------------------------------------
# Usage and cost
# --------------------------------------------------------------------------


@dataclass
class Cost:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int | None = None
    reasoning: int | None = None
    total_tokens: int = 0
    cost: Cost = field(default_factory=Cost)


@dataclass
class ModelCostRates:
    input: float = 0.0
    """USD per million tokens."""
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass
class ModelCostTier(ModelCostRates):
    input_tokens_above: int = 0
    """This tier applies when total input usage exceeds this token count."""


@dataclass
class ModelCost(ModelCostRates):
    tiers: list[ModelCostTier] = field(default_factory=list)


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------


@dataclass
class UserMessage:
    content: str | list[UserContent] = ""
    timestamp: int = field(default_factory=now_ms)
    role: Literal["user"] = "user"


@dataclass
class DeferredHandle:
    provider: str
    model_id: str
    api: str
    id: str
    expires_at: int | None = None
    poll_after_ms: int | None = None
    data: JsonValue = None


@dataclass
class AssistantMessageDiagnostic:
    kind: str
    message: str
    detail: dict[str, Any] | None = None
    timestamp: int = field(default_factory=now_ms)


@dataclass
class AssistantMessage:
    api: Api = ""
    provider: ProviderId = ""
    model: str = ""
    content: list[AssistantContent] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "pending"
    response_model: str | None = None
    response_id: str | None = None
    diagnostics: list[AssistantMessageDiagnostic] = field(default_factory=list)
    deferred: DeferredHandle | None = None
    error_message: str | None = None
    raw_stop_reason: str | None = None
    end_turn: bool | None = None
    timestamp: int = field(default_factory=now_ms)
    role: Literal["assistant"] = "assistant"


@dataclass
class ToolResultMessage:
    tool_call_id: str = ""
    tool_name: str = ""
    content: list[UserContent] = field(default_factory=list)
    details: Any = None
    usage: Usage | None = None
    added_tool_names: list[str] | None = None
    is_error: bool = False
    timestamp: int = field(default_factory=now_ms)
    role: Literal["toolResult"] = "toolResult"


Message = UserMessage | AssistantMessage | ToolResultMessage


# --------------------------------------------------------------------------
# Tools and request context
# --------------------------------------------------------------------------


@dataclass
class JsonSchemaConstrainedSampling:
    strict: Literal["prefer", "require"] = "prefer"
    type: Literal["json_schema"] = "json_schema"


@dataclass
class GrammarConstrainedSampling:
    variants: dict[str, str] = field(default_factory=dict)
    type: Literal["grammar"] = "grammar"


ConstrainedSamplingConfig = JsonSchemaConstrainedSampling | GrammarConstrainedSampling


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    """JSON Schema for the tool arguments."""
    constrained_sampling: ConstrainedSamplingConfig | Literal[False] | None = None


@dataclass
class Context:
    messages: list[Message] = field(default_factory=list)
    system_prompt: str | None = None
    tools: list[Tool] | None = None


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@dataclass
class Model:
    id: str
    name: str = ""
    api: Api = "openai-completions"
    provider: ProviderId = ""
    base_url: str = ""
    reasoning: bool = False
    thinking_level_map: dict[str, str | None] = field(default_factory=dict)
    input: list[Literal["text", "image"]] = field(default_factory=lambda: ["text"])
    cost: ModelCost = field(default_factory=ModelCost)
    context_window: int = 0
    max_tokens: int = 0
    sampling_params: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    compat: dict[str, Any] = field(default_factory=dict)
    """Compatibility overrides. Keys mirror the ``*Compat`` interfaces in TypeScript."""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.id


@dataclass
class ImagesModel:
    id: str
    name: str = ""
    api: ImagesApi = ""
    provider: ImagesProviderId = ""
    base_url: str = ""
    input: list[Literal["text", "image"]] = field(default_factory=lambda: ["text"])
    output: list[Literal["text", "image"]] = field(default_factory=lambda: ["image"])
    cost: ModelCost = field(default_factory=ModelCost)
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.id


@dataclass
class ImagesContext:
    input: list[UserContent] = field(default_factory=list)


@dataclass
class AssistantImages:
    api: ImagesApi = ""
    provider: ImagesProviderId = ""
    model: str = ""
    output: list[UserContent] = field(default_factory=list)
    response_id: str | None = None
    usage: Usage | None = None
    stop_reason: ImagesStopReason = "stop"
    error_message: str | None = None
    timestamp: int = field(default_factory=now_ms)


# --------------------------------------------------------------------------
# Request options
# --------------------------------------------------------------------------


@dataclass
class ProviderResponse:
    status: int
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class StreamOptions:
    """Auth, HTTP transport and sampling options shared by provider requests."""

    api_key: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str | None] = field(default_factory=dict)
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = 60_000
    temperature: float | None = None
    sampling_params: dict[str, Any] = field(default_factory=dict)
    max_tokens: int | None = None
    transport: Transport | None = None
    cache_retention: CacheRetention | None = None
    session_id: str | None = None
    websocket_connect_timeout_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    """Provider-specific options that have no dedicated field."""

    on_payload: Any = None
    """Optional ``(payload, model) -> payload | None`` hook applied before sending."""
    on_response: Any = None
    """Optional ``(ProviderResponse, model) -> None`` hook."""
    signal: AbortSignal | None = None
    """Optional cooperative cancellation signal for the request."""
    telemetry_context: Any = None
    """Opaque telemetry context threaded through to provider instrumentation."""


@dataclass
class SimpleStreamOptions(StreamOptions):
    reasoning: ThinkingLevel | None = None
    deferred: bool | dict[str, str] | None = None
    thinking_budgets: ThinkingBudgets | None = None


@dataclass
class ImagesOptions:
    """Auth and HTTP transport options for an image-generation request.

    TypeScript declares `ImagesOptions extends ProviderRequestOptions` — the
    same base :class:`StreamOptions` extends — but image generation is a single
    non-streaming request, so the sampling and transport fields of
    :class:`StreamOptions` have no meaning here and are left out rather than
    inherited and ignored. `ProviderImagesOptions` (`ImagesOptions & Record<string,
    unknown>`) is the open-ended form provider entry points accept; ``metadata``
    already carries anything a specific provider wants, so it has no separate
    type here.
    """

    api_key: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str | None] = field(default_factory=dict)
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = 60_000
    metadata: dict[str, Any] = field(default_factory=dict)
    signal: AbortSignal | None = None
    telemetry_context: Any = None
    """Opaque telemetry context threaded through to provider instrumentation."""

    on_payload: Any = None
    """Optional ``(payload, model) -> payload | None`` hook applied before sending."""
    on_response: Any = None
    """Optional ``(ProviderResponse, model) -> None`` hook."""


# --------------------------------------------------------------------------
# Stream events
# --------------------------------------------------------------------------


@dataclass
class StartEvent:
    partial: AssistantMessage
    type: Literal["start"] = "start"


@dataclass
class TextStartEvent:
    content_index: int
    partial: AssistantMessage
    type: Literal["text_start"] = "text_start"


@dataclass
class TextDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["text_delta"] = "text_delta"


@dataclass
class TextEndEvent:
    content_index: int
    content: str
    partial: AssistantMessage
    type: Literal["text_end"] = "text_end"


@dataclass
class ThinkingStartEvent:
    content_index: int
    partial: AssistantMessage
    type: Literal["thinking_start"] = "thinking_start"


@dataclass
class ThinkingDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass
class ThinkingEndEvent:
    content_index: int
    content: str
    partial: AssistantMessage
    type: Literal["thinking_end"] = "thinking_end"


@dataclass
class ToolCallStartEvent:
    content_index: int
    partial: AssistantMessage
    type: Literal["toolcall_start"] = "toolcall_start"


@dataclass
class ToolCallDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["toolcall_delta"] = "toolcall_delta"


@dataclass
class ToolCallEndEvent:
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage
    type: Literal["toolcall_end"] = "toolcall_end"


@dataclass
class DoneEvent:
    reason: Literal["stop", "length", "toolUse", "deferred"]
    message: AssistantMessage
    type: Literal["done"] = "done"


@dataclass
class ErrorEvent:
    reason: Literal["aborted", "error"]
    error: AssistantMessage
    type: Literal["error"] = "error"


AssistantMessageEvent = (
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent
)
