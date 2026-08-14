"""Python port of `packages/ai/test/bedrock-convert-messages.test.ts` — not portable.

`packages/ai/src/api/bedrock-converse-stream.ts` is a documented omission of
this port (see the repository README and
:mod:`pi_ai.api.bedrock_converse_stream`): it is built on the AWS SDK's SigV4
signer, credential-provider chain and binary event-stream framing, none of
which exist in `pi-ai`'s dependency set (`httpx`, `jsonschema`). There is
therefore no Python function for these cases to exercise.

The TypeScript test replaces `@aws-sdk/client-bedrock-runtime` with
`vi.mock(...)` and asserts on the object handed to `ConverseStreamCommand`, so
even the pure message-conversion half of the file is written against the AWS
SDK's command shape rather than against a provider-agnostic payload.

Behaviors left uncovered by the Python port (all message conversion for the
Bedrock Converse request):

- native strict tool use gated by model capability;
- unknown user/assistant content blocks skipped instead of throwing;
- a user message left with no content (all-unknown blocks, blank string
  content, or content emptied by surrogate sanitization) replaced by a
  placeholder block;
- blank user text blocks filtered when other content remains;
- assistant text blocks emptied by surrogate sanitization skipped;
- blank tool-result content replaced by a placeholder;
- assistant messages consisting only of unknown blocks dropped entirely.

Restoring coverage requires porting the Bedrock adapter first, which in turn
requires adding `botocore`/`boto3` to `pi-ai` — a dependency decision outside
the scope of porting these tests.
"""

from __future__ import annotations

import pytest

_REASON = "bedrock-converse-stream is a documented omission of this port: it needs the AWS SDK's SigV4 signer, credential-provider chain and binary event-stream framing, none of which are in pi-ai's dependency set. The TypeScript case asserts on a mocked @aws-sdk/client-bedrock-runtime command, so there is no Python code path to call."


@pytest.mark.skip(reason=_REASON)
def test_gates_native_strict_tool_use_by_model_capability() -> None:
    """`it("gates native strict tool use by model capability")` runs two sub-cases against
    the same tool call: (1) with `constrainedSampling: { type: "json_schema", strict:
    "require" }` on `us.anthropic.claude-sonnet-4-5-20250929-v1:0`, asserts
    `toolConfig.tools[0].toolSpec.strict === true`; (2) with `strict: "prefer"` (a lower
    bar than "require") on `amazon.nova-lite-v1:0` (a model that does not support native
    strict tool use), asserts `toolConfig.tools[0].toolSpec.strict === undefined` -- i.e.
    the `strict` field is populated only when both the request demands it and the model
    supports it.
    """


@pytest.mark.skip(reason=_REASON)
def test_skips_unknown_user_content_blocks_instead_of_throwing() -> None:
    """`it("skips unknown user content blocks instead of throwing")`: a user message with
    content `[{ type: "text", text: "hello" }, { type: "unknown", data: "foo" }]` converts
    without throwing to exactly one message whose `content` array has length 1 and whose
    sole block deep-equals `{ text: "hello" }` -- the unknown block is silently dropped,
    not just tolerated alongside a rejected message.
    """


@pytest.mark.skip(reason=_REASON)
def test_skips_unknown_assistant_content_blocks_instead_of_throwing() -> None:
    """`it("skips unknown assistant content blocks instead of throwing")`: the same
    text+unknown content pair, but as an assistant message (full `AssistantMessage` shape
    with `usage`/`stopReason`/etc.), converts to exactly one message whose single content
    block deep-equals `{ text: "hello" }`.
    """


@pytest.mark.skip(reason=_REASON)
def test_replaces_user_messages_with_only_unknown_content_blocks_with_a_placeholder() -> None:
    """`it("replaces user messages with only unknown content blocks with a placeholder")`:
    a user message whose only content block is `{ type: "unknown", data: "foo" }` still
    converts to exactly one message, whose `content` deep-equals the single-element array
    `[{ text: "<empty>" }]` -- Bedrock rejects messages with zero content blocks, so an
    all-unknown message is replaced with the literal `"<empty>"` placeholder rather than
    being dropped like the analogous all-unknown assistant case below.
    """


@pytest.mark.skip(reason=_REASON)
def test_replaces_blank_user_string_content_with_a_placeholder() -> None:
    """`it("replaces blank user string content with a placeholder")`: a user message whose
    `content` is the literal string `"   "` (whitespace only, not a content-block array)
    converts to one message whose `content` deep-equals `[{ text: "<empty>" }]`.
    """


@pytest.mark.skip(reason=_REASON)
def test_filters_blank_user_text_blocks_when_other_content_remains() -> None:
    """`it("filters blank user text blocks when other content remains")`: a user message
    with content blocks `[{ type: "text", text: "" }, { type: "text", text: "hello" }]`
    converts to one message whose `content` deep-equals `[{ text: "hello" }]` -- the blank
    block is filtered out but, unlike the all-blank case, no placeholder is inserted
    because a non-blank block still remains.
    """


@pytest.mark.skip(reason=_REASON)
def test_replaces_user_content_emptied_by_surrogate_sanitization_with_a_placeholder() -> None:
    """`it("replaces user content emptied by surrogate sanitization with a placeholder")`:
    a user message whose `content` is the single lone UTF-16 high surrogate
    `String.fromCharCode(0xd83d)` (an unpaired surrogate with no visible text) is
    sanitized down to nothing, and converts to one message whose `content` deep-equals
    `[{ text: "<empty>" }]` -- exercising the string-content path (as opposed to the
    content-block-array path in the previous case).
    """


@pytest.mark.skip(reason=_REASON)
def test_skips_assistant_text_blocks_emptied_by_surrogate_sanitization() -> None:
    """`it("skips assistant text blocks emptied by surrogate sanitization")`: an assistant
    message whose sole content block is `{ type: "text", text: <lone high surrogate
    0xd83d> }` sanitizes to an empty string and, unlike the user-message case, is NOT
    replaced with a placeholder -- the whole converted `messages` array has length 0,
    i.e. the entire assistant message is omitted (assistant messages never get the
    `"<empty>"` placeholder; only user/tool-result messages do).
    """


@pytest.mark.skip(reason=_REASON)
def test_replaces_blank_tool_result_content_with_a_placeholder() -> None:
    """`it("replaces blank tool result content with a placeholder")`: a `toolResult`
    message with content `[{ type: "text", text: "" }]` converts to one message whose
    `content[0].toolResult.content` deep-equals `[{ text: "<empty>" }]` -- the placeholder
    behavior extends to tool-result content, not just user message content.
    """


@pytest.mark.skip(reason=_REASON)
def test_skips_assistant_messages_with_only_unknown_content_blocks() -> None:
    """`it("skips assistant messages with only unknown content blocks")`: an assistant
    message whose only content block is `{ type: "unknown", data: "foo" }` converts to a
    `messages` array of length 0 -- the entire assistant message is dropped, in contrast
    to the analogous all-unknown *user* message, which is kept and given a placeholder.
    """
