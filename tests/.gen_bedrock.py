from __future__ import annotations

import pathlib
import re

REASON_SDK = (
    "bedrock-converse-stream is a documented omission of this port: it needs the AWS SDK's "
    "SigV4 signer, credential-provider chain and binary event-stream framing, none of which "
    "are in pi-ai's dependency set. The TypeScript case asserts on a mocked "
    "@aws-sdk/client-bedrock-runtime command, so there is no Python code path to call."
)
REASON_MIDDLEWARE = (
    "bedrock-converse-stream is a documented omission of this port, and this case additionally "
    "asserts on the AWS SDK's middleware stack (a build-step middleware invoked with a synthetic "
    "Smithy request). pi-ai issues plain httpx requests and has no middleware stack, so the "
    "mechanism itself has no Python analogue."
)

CASES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "test_bedrock_thinking_payload.py": (
        REASON_SDK,
        [
            (
                "uses adaptive thinking for Claude Opus 4.8 when reasoning is enabled",
                "additionalModelRequestFields.thinking is {type: 'adaptive'} and carries no budget_tokens.",
            ),
            (
                "maps xhigh reasoning to effort=xhigh for Claude Opus 4.8",
                "additionalModelRequestFields.thinking.effort is 'xhigh'.",
            ),
            (
                "uses adaptive thinking for Claude Fable 5 when reasoning is enabled",
                "additionalModelRequestFields.thinking is {type: 'adaptive'}.",
            ),
            (
                "uses adaptive thinking for Claude Sonnet 5 when reasoning is enabled",
                "additionalModelRequestFields.thinking is {type: 'adaptive'}.",
            ),
            (
                "uses adaptive thinking for Claude Opus 5 when reasoning is enabled",
                "additionalModelRequestFields.thinking is {type: 'adaptive'}.",
            ),
            (
                "maps xhigh reasoning to effort=xhigh for Claude Opus 5",
                "additionalModelRequestFields.thinking.effort is 'xhigh'.",
            ),
            (
                "maps xhigh reasoning to effort=xhigh for Claude Fable 5",
                "additionalModelRequestFields.thinking.effort is 'xhigh'.",
            ),
            (
                "omits display for GovCloud model ids on non-adaptive Claude thinking",
                "the fixed-budget thinking block has no `display` key when the model id is a GovCloud id.",
            ),
            (
                "omits display for GovCloud regions on adaptive Claude thinking",
                "the adaptive thinking block has no `display` key when the resolved region is GovCloud.",
            ),
            (
                "uses the model maxTokens cap instead of Bedrock's 4096-token default for adaptive Claude models",
                "inferenceConfig.maxTokens equals the model's own maxTokens rather than the SDK default of 4096.",
            ),
            (
                "uses adaptive thinking when model.name contains the model name but ARN does not",
                "the adaptive decision falls back to model.name when the ARN carries no model name.",
            ),
            (
                "injects cache points when model.name identifies a supported Claude model",
                "cachePoint blocks are appended to the system and message blocks.",
            ),
            (
                "falls back to fixed-budget thinking for non-adaptive Claude via model.name",
                "thinking is {type: 'enabled', budget_tokens: n} for a model identified only by model.name.",
            ),
        ],
    ),
    "test_bedrock_convert_messages.py": (
        REASON_SDK,
        [
            (
                "gates native strict tool use by model capability",
                "toolConfig carries the strict/native tool spec only when the model advertises the capability.",
            ),
            (
                "skips unknown user content blocks instead of throwing",
                "an unrecognized user content block is dropped and the remaining blocks are converted.",
            ),
            (
                "skips unknown assistant content blocks instead of throwing",
                "an unrecognized assistant content block is dropped and the remaining blocks are converted.",
            ),
            (
                "replaces user messages with only unknown content blocks with a placeholder",
                "the converted user message carries a single placeholder text block.",
            ),
            (
                "replaces blank user string content with a placeholder",
                "a user message whose string content is blank becomes a placeholder text block.",
            ),
            (
                "filters blank user text blocks when other content remains",
                "blank text blocks are removed while the non-blank blocks survive.",
            ),
            (
                "replaces user content emptied by surrogate sanitization with a placeholder",
                "content reduced to nothing by lone-surrogate stripping becomes a placeholder.",
            ),
            (
                "skips assistant text blocks emptied by surrogate sanitization",
                "such assistant blocks are dropped rather than sent as empty text.",
            ),
            (
                "replaces blank tool result content with a placeholder",
                "a tool-result message with blank content gets a placeholder block.",
            ),
            (
                "skips assistant messages with only unknown content blocks",
                "the assistant message is omitted from the converted message list entirely.",
            ),
        ],
    ),
    "test_bedrock_error_metadata.py": (
        REASON_SDK,
        [
            (
                "records status, error code and request id for a non-2xx from client.send()",
                "the provider_request_failed diagnostic carries status, errorCode and requestId and nothing else.",
            ),
            (
                "leaves errorMessage untouched so retry classification is unaffected",
                "errorMessage keeps the original SDK text.",
            ),
            (
                "reports only the request id for a modeled mid-stream exception",
                "the diagnostic has requestId but no errorCode, because none is available.",
            ),
            (
                "captures the error code for an unmodeled mid-stream error",
                "the diagnostic reports the event frame's :error-code plus requestId.",
            ),
            (
                "does not report a transport failure name as a provider error code",
                "a TimeoutError name is not surfaced as errorCode.",
            ),
            (
                "emits no diagnostic when the failure carries no provider metadata",
                "no provider_request_failed diagnostic is attached.",
            ),
            (
                "emits no diagnostic for an aborted turn",
                "no provider_request_failed diagnostic is attached when the turn was aborted.",
            ),
            (
                "drops header-derived values that exceed the length bound",
                "over-long header-derived values are omitted from the diagnostic.",
            ),
            (
                "omits the SDK's Unknown placeholder instead of reporting it as a code",
                "an x-amzn-errortype of 'Unknown' yields no errorCode.",
            ),
        ],
    ),
    "test_bedrock_endpoint_resolution.py": (
        REASON_SDK,
        [
            (
                "does not pin standard AWS endpoints when AWS_REGION is configured",
                "the SDK client config has `region` set from AWS_REGION and `endpoint` left undefined.",
            ),
            (
                "derives region from a built-in EU endpoint when no region or profile is configured",
                "the region is parsed out of the built-in EU base URL.",
            ),
            (
                "handles missing regions for explicit, scoped, and ambient profiles",
                "explicit, credential-scoped and ambient profiles all keep the derived endpoint and region.",
            ),
            (
                "still passes custom Bedrock endpoints through to the SDK client",
                "a custom baseUrl reaches the client config unchanged.",
            ),
            (
                "extracts region from inference profile ARN regardless of AWS_REGION",
                "the region comes from the ARN, not from AWS_REGION.",
            ),
            ("extracts region from GovCloud inference profile ARN", "a us-gov-* region is parsed out of the ARN."),
            (
                "preserves ambient AWS auth for custom model IDs through compat dispatch",
                "ambient AWS credentials survive the compat dispatch path for a custom model id.",
            ),
            (
                "uses the generic API key option as a Bedrock bearer token",
                "StreamOptions.apiKey becomes the Bedrock bearer token.",
            ),
        ],
    ),
    "test_bedrock_custom_headers.py": (
        REASON_MIDDLEWARE,
        [
            (
                "VC1: registers a build-step middleware that injects the caller header (happy path)",
                "a build-step middleware named piCustomHeaders adds the caller headers to the signed request.",
            ),
            (
                "VC2: skips reserved headers case-insensitively while applying allowed ones",
                "host/authorization/x-amz-* are skipped whatever their case; other headers are applied.",
            ),
            (
                "VC3: registers no middleware when headers is undefined",
                "middlewareStack carries no piCustomHeaders entry.",
            ),
            ("VC3: registers no middleware when headers is empty", "an empty headers object registers no middleware."),
            (
                "VC3 (structural guard): passes through unchanged when the request has no headers",
                "a Smithy request object without a headers property is returned untouched.",
            ),
            (
                "VC4: streamSimpleBedrock forwards headers end-to-end (regression guard)",
                "headers passed to streamSimple reach the signed request.",
            ),
        ],
    ),
    "test_bedrock_credentials.py": (
        REASON_SDK,
        [
            (
                "prefers explicit and scoped profiles over ambient AWS access keys",
                "credentials is fromNodeProviderChain({profile}) for both an explicit AWS_PROFILE option and a credential-scoped profile.",
            ),
            (
                "uses ambient AWS access keys when no profile is configured",
                "credentials is left undefined so the SDK's own chain picks up the ambient keys.",
            ),
            (
                "uses ambient AWS access keys when only an ambient profile is set",
                "an environment-only profile does not override the ambient access keys.",
            ),
        ],
    ),
    "test_bedrock_raw_stop_reason.py": (
        REASON_SDK,
        [
            (
                "preserves raw Bedrock stop reasons for successful stops",
                "rawStopReason is 'end_turn', stopReason is 'stop' and errorMessage is unset.",
            ),
            (
                "preserves raw Bedrock stop reasons for provider error stops",
                "an unmapped 'guardrail_intervened' keeps rawStopReason, maps stopReason to 'error' "
                "and sets errorMessage to 'Provider stopped with: guardrail_intervened'.",
            ),
        ],
    ),
}


def slug(name: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return out


for filename, (reason, cases) in CASES.items():
    path = pathlib.Path(filename)
    text = path.read_text().rstrip("\n")
    assert text.endswith("from __future__ import annotations"), filename
    parts = [text, "", "import pytest", "", f'_REASON = (\n    "{reason}"\n)', ""]
    seen: set[str] = set()
    for name, asserts in cases:
        fn = slug(name)
        if fn in seen:
            fn = f"{fn}_2"
        seen.add(fn)
        parts.append("")
        parts.append("@pytest.mark.skip(reason=_REASON)")
        parts.append(f"def test_{fn}() -> None:")
        parts.append(f'    """`it("{name}")` asserts that {asserts}"""')
    path.write_text("\n".join(parts) + "\n")
    print(filename, len(cases))
