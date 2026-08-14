# pi-ai

Unified LLM API with provider collections, automatic auth resolution, token and cost tracking, and serializable context hand-off across models.

**Note**: This library only includes chat models that support tool calling (function calling), as this is essential for agentic workflows. Some generated catalog entries are present for discovery but have unported streaming implementations; those are called out below.

## Table of Contents

- [Supported Providers](#supported-providers)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Providers and Models](#providers-and-models)
  - [Provider Factories](#provider-factories)
  - [All Built-in Providers](#all-built-in-providers)
  - [Querying Models](#querying-models)
  - [Static Catalog Reads](#static-catalog-reads)
  - [Dynamic Providers](#dynamic-providers)
- [Auth](#auth)
  - [How Auth Resolves](#how-auth-resolves)
  - [Transforming Request Headers](#transforming-request-headers)
  - [Credential Store](#credential-store)
  - [Environment Variables](#environment-variables)
- [Tools](#tools)
  - [Defining Tools](#defining-tools)
  - [Constrained Sampling for Tools](#constrained-sampling-for-tools)
  - [Handling Tool Calls](#handling-tool-calls)
  - [Streaming Tool Calls with Partial JSON](#streaming-tool-calls-with-partial-json)
  - [Validating Tool Arguments](#validating-tool-arguments)
  - [Complete Event Reference](#complete-event-reference)
- [Image Input](#image-input)
- [Image Generation](#image-generation)
  - [Basic Image Generation](#basic-image-generation)
  - [Notes and Limitations](#notes-and-limitations)
- [Thinking/Reasoning](#thinkingreasoning)
  - [Unified Interface](#unified-interface-streamsimplecompletesimple)
  - [Provider-Specific Options](#provider-specific-options-streamcomplete)
  - [Streaming Thinking Content](#streaming-thinking-content)
- [Stop Reasons](#stop-reasons)
- [Error Handling](#error-handling)
  - [Aborting Requests](#aborting-requests)
  - [Continuing After Abort](#continuing-after-abort)
  - [Debugging Provider Payloads](#debugging-provider-payloads)
- [Custom Providers](#custom-providers)
  - [createProvider()](#createprovider)
  - [Calling API Implementations Directly](#calling-api-implementations-directly)
  - [OpenAI Compatibility Settings](#openai-compatibility-settings)
- [Faux Provider for Tests](#faux-provider-for-tests)
- [Cross-Provider Handoffs](#cross-provider-handoffs)
- [Context Serialization](#context-serialization)
- [Browser Usage](#browser-usage)
- [Bundling and Tree Shaking](#bundling-and-tree-shaking)
- [OAuth Providers](#oauth-providers)
  - [Vertex AI](#vertex-ai)
  - [CLI Login](#cli-login)
  - [Programmatic OAuth](#programmatic-oauth)
- [Migrating from the Old Global API](#migrating-from-the-old-global-api)
- [Development](#development)
- [License](#license)

## Supported Providers

- **OpenAI**
- **Ant Ling**
- **Azure OpenAI (Responses)**
- **OpenAI Codex**: models are listed, but `openai-codex-responses` streaming is not ported and raises `NotImplementedError`.
- **DeepSeek**
- **NVIDIA NIM**
- **Anthropic**
- **Google**
- **Vertex AI** (Gemini via Vertex AI)
- **Mistral**
- **Groq**
- **Cerebras**
- **Cloudflare AI Gateway**
- **Cloudflare Workers AI**
- **xAI**
- **OpenRouter**
- **Vercel AI Gateway**
- **ZAI Coding Plan (Global)** (with separate China provider)
- **MiniMax** (with separate China provider)
- **Together AI**
- **Baseten**
- **Hugging Face**
- **Moonshot AI** (with separate China provider)
- **GitHub Copilot** (requires OAuth, see below)
- **Amazon Bedrock**: models are listed, but `bedrock-converse-stream` streaming is not ported and raises `NotImplementedError`.
- **OpenCode Zen**
- **OpenCode Go**
- **Fireworks** (uses OpenAI- and Anthropic-compatible APIs)
- **Kimi For Coding** (Moonshot AI subscription endpoint, uses Anthropic-compatible API)
- **Qwen Token Plan** (separate Individual and existing catalogs, with a separate China provider)
- **Xiaomi MiMo** (defaults to API billing endpoint, with separate Token Plan providers for `cn`/`ams`/`sgp` regions)
- **Radius**
- **Any OpenAI-compatible API**: Ollama, vLLM, LM Studio, etc.

## Installation

```bash
uv add pi-ai
# or
pip install pi-ai
```

The package installs a small auth CLI named `pp-ai`.

There is no TypeBox dependency in the Python port. Tool schemas are plain JSON Schema dictionaries. The `pi_ai.string_enum()` helper builds a Google-compatible string enum schema.

## Quick Start

You build a `Models` collection of providers and stream through it. `Models.stream()` and `Models.stream_simple()` are async in Python; `pi_ai.complete()` drains the returned stream and gives the final `AssistantMessage`.

```python
import asyncio
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from pi_ai import (
    Context,
    StreamOptions,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    complete,
    now_ms,
)
from pi_ai.providers.all import builtin_models


async def main() -> None:
    models = builtin_models()
    model = models.get_model("openai", "gpt-4o-mini")
    assert model is not None

    tools = [
        Tool(
            name="get_time",
            description="Get the current time",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Optional timezone, for example America/New_York",
                    }
                },
            },
        )
    ]

    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="What time is it?", timestamp=now_ms())],
        tools=tools,
    )

    stream = await models.stream(model, context, StreamOptions())

    async for event in stream:
        if event.type == "start":
            print(f"Starting with {event.partial.model}")
        elif event.type == "text_start":
            print("\n[Text started]")
        elif event.type == "text_delta":
            sys.stdout.write(event.delta)
        elif event.type == "text_end":
            print("\n[Text ended]")
        elif event.type == "thinking_start":
            print("[Model is thinking...]")
        elif event.type == "thinking_delta":
            sys.stdout.write(event.delta)
        elif event.type == "thinking_end":
            print("[Thinking complete]")
        elif event.type == "toolcall_start":
            print(f"\n[Tool call started: index {event.content_index}]")
        elif event.type == "toolcall_delta":
            block = event.partial.content[event.content_index]
            if isinstance(block, ToolCall):
                print(f"[Streaming args for {block.name}]")
        elif event.type == "toolcall_end":
            print(f"\nTool called: {event.tool_call.name}")
            print(f"Arguments: {json.dumps(event.tool_call.arguments)}")
        elif event.type == "done":
            print(f"\nFinished: {event.reason}")
        elif event.type == "error":
            print(f"Error: {event.error.error_message}")

    final_message = await stream.result()
    context.messages.append(final_message)

    tool_calls = [block for block in final_message.content if isinstance(block, ToolCall)]
    for call in tool_calls:
        if call.name == "get_time":
            timezone = str(call.arguments.get("timezone") or "UTC")
            result = datetime.now(ZoneInfo(timezone)).isoformat()
        else:
            result = "Unknown tool"

        context.messages.append(
            ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                content=[TextContent(text=result)],
                is_error=False,
                timestamp=now_ms(),
            )
        )

    if tool_calls:
        continuation = await complete(await models.stream(model, context))
        context.messages.append(continuation)
        print("After tool execution:", continuation.content)

    print(f"Total tokens: {final_message.usage.input} in, {final_message.usage.output} out")
    print(f"Cost: ${final_message.usage.cost.total:.4f}")

    response = await complete(await models.stream(model, context))
    for block in response.content:
        if block.type == "text":
            print(block.text)
        elif block.type == "toolCall":
            print(f"Tool: {block.name}({json.dumps(block.arguments)})")


if __name__ == "__main__":
    asyncio.run(main())
```

Snippets below assume a `models` collection set up like this, with the relevant providers registered.

## Providers and Models

A **provider** is the runtime unit: it owns its model catalog, its auth (API key resolution, OAuth flows), and its stream behavior. A `Models` collection holds providers and routes every request to the provider that owns the model.

Providers internally share **API implementations** (the wire protocols): Anthropic models use `anthropic-messages`, OpenAI uses `openai-responses`, while xAI, Groq, Cerebras, OpenRouter, and most others share `openai-completions`. Mixed-API providers (GitHub Copilot, OpenCode Zen, Fireworks) dispatch per model.

### Provider Factories

For apps that only need specific providers, there is one factory per built-in provider. Python modules use snake_case names:

```python
from pi_ai.registry import Models
from pi_ai.providers.amazon_bedrock import amazon_bedrock_provider
from pi_ai.providers.anthropic import anthropic_provider
from pi_ai.providers.openai import openai_provider
from pi_ai.providers.openrouter import openrouter_provider

models = Models()
models.add(anthropic_provider())
models.add(openai_provider())
models.add(openrouter_provider())
models.add(amazon_bedrock_provider())
```

Provider factories import their committed JSON model catalog and an HTTP API module. The Python port does not have TypeScript-style lazy `*.lazy.ts` wrappers because it does not load vendor SDKs for the implemented APIs.

### All Built-in Providers

For apps that want everything:

```python
from pi_ai.providers.all import builtin_models

models = builtin_models()
```

`builtin_models()` returns a `Models` collection with every built-in provider registered. `builtin_providers()` returns the provider list if you want to register them on your own collection.

### Querying Models

Reads are synchronous and return the last-known lists:

```python
from pi_ai.providers.all import builtin_models

models = builtin_models()
providers = models.get_providers()
provider = models.get_provider("anthropic")

all_models = models.get_models()
anthropic_models = models.get_models("anthropic")
model = models.get_model("anthropic", "claude-sonnet-4-5")

for item in anthropic_models:
    print(f"{item.id}: {item.name}")
    print(f"  API: {item.api}")
    print(f"  Context: {item.context_window} tokens")
    print(f"  Vision: {'image' in item.input}")
    print(f"  Reasoning: {item.reasoning}")
```

Use `has_api()` when dynamically selected models need API-specific option dataclasses:

```python
from pi_ai import Context, SimpleStreamOptions, UserMessage, complete, has_api, now_ms
from pi_ai.api.anthropic_messages import AnthropicOptions
from pi_ai.providers.all import builtin_models


async def example() -> None:
    models = builtin_models()
    model = models.get_model("anthropic", "claude-sonnet-4-5")
    context = Context(messages=[UserMessage(content="Hello", timestamp=now_ms())])
    if model is not None and has_api(model, "anthropic-messages"):
        stream = await models.stream(
            model,
            context,
            AnthropicOptions(thinking_enabled=True, thinking_budget_tokens=2048),
        )
        message = await complete(stream)
        print(message.stop_reason)
```

### Static Catalog Reads

For tooling that wants the generated built-in catalog independent of any collection:

```python
from pi_ai.providers.all import get_builtin_model, get_builtin_models, get_builtin_providers

model = get_builtin_model("openai", "gpt-4o-mini")
providers = get_builtin_providers()
anthropic = get_builtin_models("anthropic")

print(model.id if model else "missing")
print(providers[:3])
print(len(anthropic))
```

### Dynamic Providers

The TypeScript remote model catalog (`refreshModels`, `ModelsPublication`, `Models.refresh()`, and `ModelsStore`-driven dynamic publication) is not ported for chat providers. Generated built-in catalogs are committed JSON files loaded by `pi_ai.model_catalog`.

Radius is the exception: it has no generated catalog and can be refreshed explicitly with `refresh_radius_models()`.

```python
from pi_ai.auth.types import Credential
from pi_ai.providers.radius import radius_provider, refresh_radius_models
from pi_ai.registry import Models


async def example() -> None:
    provider = radius_provider(credential=Credential(type="api_key", key="radius-key"))
    await refresh_radius_models(provider, credential=Credential(type="api_key", key="radius-key"))

    models = Models()
    models.add(provider)
    print([model.id for model in models.get_models("radius")])
```

## Auth

Every provider owns its auth: how API keys resolve (stored credentials, environment variables, ambient sources like AWS profiles or gcloud ADC) and, where supported, OAuth login/refresh flows.

### How Auth Resolves

When you call `models.stream()`, the collection resolves auth through the owning provider and merges it into the request. Explicit per-request values win for the API key field:

```python
from pi_ai import Context, StreamOptions, UserMessage, complete, now_ms
from pi_ai.providers.all import builtin_models


async def example() -> None:
    models = builtin_models()
    model = models.get_model("openai", "gpt-4o-mini")
    assert model is not None
    context = Context(messages=[UserMessage(content="Hello", timestamp=now_ms())])

    response = await complete(await models.stream(model, context))
    explicit = await complete(await models.stream(model, context, StreamOptions(api_key="sk-explicit")))
    print(response.stop_reason, explicit.stop_reason)
```

You can inspect resolution without making a request. Pass a provider id for provider-scoped auth, or a model to include its static `model.headers`:

```python
from pi_ai.providers.all import builtin_models


async def example() -> None:
    models = builtin_models()
    model = models.get_model("anthropic", "claude-sonnet-4-5")
    assert model is not None

    provider_auth = await models.get_auth(model.provider)
    model_auth = await models.get_auth(model)

    print(provider_auth.source if provider_auth else "provider not configured")
    if model_auth is not None:
        print(f"configured via {model_auth.source}")
        print(model_auth.auth.headers)
    else:
        print("not configured")
```

`get_auth()`, `check_auth()`, `get_available()`, login, and logout resolve credentials, refresh expired OAuth where applicable, and raise `ModelsError` when something is broken. Request paths surface setup failures as stream error messages.

### Transforming Request Headers

The TypeScript `transformHeaders` option is not ported. In Python, pass explicit header overrides through `StreamOptions.headers`; pass `None` as a header value to suppress lower-level defaults that support deletion.

```python
from pi_ai import Context, StreamOptions, UserMessage, complete, now_ms
from pi_ai.providers.all import builtin_models


async def example() -> None:
    models = builtin_models()
    model = models.get_model("openai", "gpt-4o-mini")
    assert model is not None
    context = Context(messages=[UserMessage(content="Hello", timestamp=now_ms())])

    response = await complete(
        await models.stream(
            model,
            context,
            StreamOptions(headers={"X-Client": "my-app", "X-Request-ID": "request-123"}),
        )
    )
    print(response.stop_reason)
```

The merge order is:

```text
provider auth headers -> model.headers -> explicit options.headers -> Provider.stream()
```

### Credential Store

Stored credentials (API keys entered interactively, OAuth tokens) live in a `CredentialStore`. The default is in-memory; apps inject persistent storage by subclassing `CredentialStore`.

```python
from pi_ai.auth.types import Credential, CredentialInfo, CredentialStore
from pi_ai.providers.all import builtin_models
from pi_ai.registry import Models


class DictCredentialStore(CredentialStore):
    def __init__(self) -> None:
        self._credentials: dict[str, Credential] = {}

    async def get(self, provider_id: str) -> Credential | None:
        return self._credentials.get(provider_id)

    async def set(self, provider_id: str, credential: Credential) -> None:
        self._credentials[provider_id] = credential

    async def delete(self, provider_id: str) -> None:
        self._credentials.pop(provider_id, None)

    async def list(self) -> list[CredentialInfo]:
        return [CredentialInfo(provider_id=key, type=value.type) for key, value in self._credentials.items()]


models = Models(providers=builtin_models().get_providers(), credential_store=DictCredentialStore())
```

API-key credentials use the same discriminator as pi's `auth.json` and can carry provider-scoped env/config values:

```python
from pi_ai.auth.types import Credential

credential = Credential(
    type="api_key",
    key="...",
    env={
        "CLOUDFLARE_ACCOUNT_ID": "account-id",
        "CLOUDFLARE_GATEWAY_ID": "gateway-id",
    },
)
```

A stored credential owns its provider: environment variables are consulted only when no usable stored credential is present. OAuth token refresh updates the stored credential through the provider-owned flow.

### Environment Variables

Built-in providers resolve these env vars:

| Provider | Environment Variable(s) |
|----------|------------------------|
| OpenAI | `OPENAI_API_KEY` |
| Ant Ling | `ANT_LING_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_BASE_URL` or `AZURE_OPENAI_RESOURCE_NAME`. Optional: `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT_NAME_MAP`. |
| Anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_OAUTH_TOKEN`, or `ANTHROPIC_AUTH_TOKEN` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| NVIDIA NIM | `NVIDIA_API_KEY` |
| Google | `GEMINI_API_KEY` |
| Vertex AI | `GOOGLE_CLOUD_API_KEY` or `GOOGLE_CLOUD_PROJECT` (or `GCLOUD_PROJECT`) + `GOOGLE_CLOUD_LOCATION` + ADC/access token configuration |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Cerebras | `CEREBRAS_API_KEY` |
| Cloudflare AI Gateway | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_GATEWAY_ID` |
| Cloudflare Workers AI | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` |
| xAI | `XAI_API_KEY` |
| Radius | `RADIUS_API_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |
| Together AI | `TOGETHER_API_KEY` |
| Baseten | `BASETEN_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Vercel AI Gateway | `AI_GATEWAY_API_KEY` |
| ZAI Coding Plan (Global) | `ZAI_API_KEY` |
| ZAI Coding Plan (China) | `ZAI_CODING_CN_API_KEY` |
| MiniMax (Global) | `MINIMAX_API_KEY` |
| MiniMax (China) | `MINIMAX_CN_API_KEY` |
| Moonshot AI / Moonshot AI (China) | `MOONSHOT_API_KEY` |
| Hugging Face | `HF_TOKEN` |
| OpenCode Zen / OpenCode Go | `OPENCODE_API_KEY` |
| Kimi For Coding | `KIMI_API_KEY` |
| Qwen Token Plan (existing catalog) | `QWEN_TOKEN_PLAN_API_KEY` |
| Qwen Token Plan (Individual) | `QWEN_TOKEN_PLAN_API_KEY` |
| Qwen Token Plan (China) | `QWEN_TOKEN_PLAN_CN_API_KEY` |
| Xiaomi MiMo (API billing) | `XIAOMI_API_KEY` |
| Xiaomi MiMo Token Plan (China) | `XIAOMI_TOKEN_PLAN_CN_API_KEY` |
| Xiaomi MiMo Token Plan (Amsterdam) | `XIAOMI_TOKEN_PLAN_AMS_API_KEY` |
| Xiaomi MiMo Token Plan (Singapore) | `XIAOMI_TOKEN_PLAN_SGP_API_KEY` |
| GitHub Copilot | `COPILOT_GITHUB_TOKEN` |

`qwen-token-plan-individual` and `qwen-token-plan` share the international endpoint and `QWEN_TOKEN_PLAN_API_KEY`. Stored credentials remain provider-scoped, so save the key under the provider id you register.

Amazon Bedrock auth discovery is present, but the streaming API is not ported. Vertex AI auth supports API-key and access-token paths in the Python API modules; Application Default Credentials support is limited compared with the TypeScript SDK-backed implementation.

## Tools

Tools enable LLMs to interact with external systems. Python tool definitions use plain JSON Schema dictionaries and `jsonschema` validation.

### Defining Tools

```python
from pi_ai import Tool, string_enum

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name or coordinates"},
            "units": string_enum(["celsius", "fahrenheit"], default="celsius"),
        },
        "required": ["location"],
    },
)

book_meeting_tool = Tool(
    name="book_meeting",
    description="Schedule a meeting",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "start_time": {"type": "string", "format": "date-time"},
            "end_time": {"type": "string", "format": "date-time"},
            "attendees": {"type": "array", "items": {"type": "string", "format": "email"}, "minItems": 1},
        },
        "required": ["title", "start_time", "end_time", "attendees"],
    },
)
```

For Google API compatibility, use `string_enum()` instead of JSON Schema forms that produce `anyOf`/`const` enum variants.

### Constrained Sampling for Tools

Tools can opt in to provider-side constrained sampling. The Python dataclasses are `JsonSchemaConstrainedSampling` and `GrammarConstrainedSampling`.

```python
from pi_ai import Tool
from pi_ai.types import JsonSchemaConstrainedSampling

strict_tool = Tool(
    name="edit_file",
    description="Edit a file",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
)
```

Strict JSON-schema constrained sampling is implemented in the OpenAI, Anthropic, Mistral, Google Generative AI, Vertex, and Pi Messages adapters where the underlying API supports it. Bedrock is not runnable in this port, so its TypeScript strict-tool support is not available at runtime.

OpenAI-compatible grammar tools use `GrammarConstrainedSampling`. Native grammar tools must have an object parameter schema with exactly one required string property:

```python
from pi_ai import Tool
from pi_ai.types import GrammarConstrainedSampling

patch_tool = Tool(
    name="apply_patch",
    description="Apply a patch",
    parameters={
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
        "additionalProperties": False,
    },
    constrained_sampling=GrammarConstrainedSampling(variants={"openai_lark": "start: /.+/s"}),
)
```

### Handling Tool Calls

Tool results use content blocks and can include both text and images:

```python
import base64
import json
from pathlib import Path

from pi_ai import Context, ImageContent, TextContent, ToolCall, ToolResultMessage, UserMessage, complete, now_ms
from pi_ai.providers.all import builtin_models


async def execute_weather_api(arguments: dict[str, object]) -> dict[str, object]:
    return {"location": arguments.get("location"), "temperature": 18}


async def example() -> None:
    models = builtin_models()
    model = models.get_model("openai", "gpt-4o-mini")
    assert model is not None
    context = Context(messages=[UserMessage(content="What is the weather in London?", timestamp=now_ms())])

    response = await complete(await models.stream(model, context))
    for block in response.content:
        if isinstance(block, ToolCall):
            result = await execute_weather_api(block.arguments)
            context.messages.append(
                ToolResultMessage(
                    tool_call_id=block.id,
                    tool_name=block.name,
                    content=[TextContent(text=json.dumps(result))],
                    is_error=False,
                    timestamp=now_ms(),
                )
            )

    image_bytes = Path("chart.png").read_bytes()
    context.messages.append(
        ToolResultMessage(
            tool_call_id="tool_xyz",
            tool_name="generate_chart",
            content=[
                TextContent(text="Generated chart showing temperature trends"),
                ImageContent(data=base64.b64encode(image_bytes).decode("ascii"), mime_type="image/png"),
            ],
            is_error=False,
            timestamp=now_ms(),
        )
    )
```

`ImageContent.data` must be base64-encoded image bytes. The example uses local file access only to show where image bytes enter your program; encode real data with `base64.b64encode(...).decode("ascii")`.

### Streaming Tool Calls with Partial JSON

During streaming, tool call arguments are progressively parsed as they arrive. This enables real-time UI updates before the complete arguments are available:

```python
from pi_ai import Context, ToolCall, UserMessage, now_ms
from pi_ai.providers.all import builtin_models


async def example() -> None:
    models = builtin_models()
    model = models.get_model("openai", "gpt-4o-mini")
    assert model is not None
    context = Context(messages=[UserMessage(content="Write a file", timestamp=now_ms())])
    stream = await models.stream(model, context)

    async for event in stream:
        if event.type == "toolcall_delta":
            tool_call = event.partial.content[event.content_index]
            if isinstance(tool_call, ToolCall) and tool_call.arguments:
                if tool_call.name == "write_file" and tool_call.arguments.get("path"):
                    print(f"Writing to: {tool_call.arguments['path']}")
                    content = tool_call.arguments.get("content")
                    if isinstance(content, str):
                        print(f"Content preview: {content[:100]}...")

        if event.type == "toolcall_end":
            tool_call = event.tool_call
            print(f"Tool completed: {tool_call.name}", tool_call.arguments)
```

Important notes about partial tool arguments:

- During `toolcall_delta` events, `arguments` contains the best-effort parse of partial JSON.
- Fields may be missing or incomplete; always check before use.
- String values may be truncated mid-word.
- Arrays and nested objects may be partially populated.
- At minimum, `arguments` is an empty dictionary, not `None`.
- The Google provider does not support function call streaming. Instead, you receive a single `toolcall_delta` event with the full arguments.

### Validating Tool Arguments

When implementing your own tool execution loop, use `validate_tool_call()` to validate arguments before passing them to your tools:

```python
from pi_ai import Context, TextContent, Tool, ToolCall, ToolResultMessage, UserMessage, validate_tool_call, now_ms
from pi_ai.providers.all import builtin_models


def execute_my_tool(name: str, arguments: dict[str, object]) -> str:
    return f"{name}: {arguments}"


async def example(weather_tool: Tool, calculator_tool: Tool) -> None:
    tools = [weather_tool, calculator_tool]
    models = builtin_models()
    model = models.get_model("openai", "gpt-4o-mini")
    assert model is not None
    context = Context(messages=[UserMessage(content="Use a tool", timestamp=now_ms())], tools=tools)
    stream = await models.stream(model, context)

    async for event in stream:
        if event.type == "toolcall_end":
            tool_call = event.tool_call
            try:
                validated_args = validate_tool_call(tools, tool_call)
                result = execute_my_tool(tool_call.name, validated_args)
                context.messages.append(
                    ToolResultMessage(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content=[TextContent(text=result)],
                        is_error=False,
                        timestamp=now_ms(),
                    )
                )
            except Exception as error:
                context.messages.append(
                    ToolResultMessage(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content=[TextContent(text=str(error))],
                        is_error=True,
                        timestamp=now_ms(),
                    )
                )
```

### Complete Event Reference

All streaming events emitted during assistant message generation:

| Event Type | Description | Key Properties |
|------------|-------------|----------------|
| `start` | Stream begins | `partial`: initial assistant message structure |
| `text_start` | Text block starts | `content_index`: position in content array |
| `text_delta` | Text chunk received | `delta`, `content_index` |
| `text_end` | Text block complete | `content`, `content_index` |
| `thinking_start` | Thinking block starts | `content_index` |
| `thinking_delta` | Thinking chunk received | `delta`, `content_index` |
| `thinking_end` | Thinking block complete | `content`, `content_index` |
| `toolcall_start` | Tool call begins | `content_index` |
| `toolcall_delta` | Tool arguments streaming | `delta`, `partial.content[content_index].arguments` |
| `toolcall_end` | Tool call complete | `tool_call` with `id`, `name`, `arguments` |
| `done` | Stream complete | `reason`, `message` |
| `error` | Error occurred | `reason`, `error` assistant message |

Streaming events for different content blocks are not guaranteed to be contiguous. Consumers must use `content_index` to associate each delta/end event with its block and must not assume that a block's `*_start`/`*_delta`/`*_end` sequence is uninterrupted by events for other blocks.

## Image Input

Models with vision capabilities can process images. Check if a model supports images via the `input` property. If you pass images to a non-vision model, they are silently ignored.

```python
import base64
from pathlib import Path

from pi_ai import Context, ImageContent, TextContent, UserMessage, complete, now_ms
from pi_ai.providers.all import builtin_models


async def example() -> None:
    models = builtin_models()
    model = models.get_model("openai", "gpt-4o-mini")
    assert model is not None

    if "image" in model.input:
        print("Model supports vision")

    image_data = base64.b64encode(Path("image.png").read_bytes()).decode("ascii")
    response = await complete(
        await models.stream(
            model,
            Context(
                messages=[
                    UserMessage(
                        content=[
                            TextContent(text="What is in this image?"),
                            ImageContent(data=image_data, mime_type="image/png"),
                        ],
                        timestamp=now_ms(),
                    )
                ]
            ),
        )
    )

    for block in response.content:
        if block.type == "text":
            print(block.text)
```

## Image Generation

Image generation uses a separate API surface from text/chat generation. An `ImagesModels` collection holds `ImagesProvider` objects, reads are sync, and auth resolves through the owning provider. Image generation is one-shot: `generate_images()` waits for the provider response and returns the final `AssistantImages` result.

### Basic Image Generation

```python
from pi_ai import ImagesContext, TextContent
from pi_ai.providers.all import builtin_images_models


async def example() -> None:
    images_models = builtin_images_models()
    model = images_models.get_model("openrouter", "google/gemini-2.5-flash-image")
    assert model is not None

    result = await images_models.generate_images(
        model,
        ImagesContext(input=[TextContent(text="Generate a red circle on a plain white background.")]),
    )

    for block in result.output:
        if block.type == "text":
            print(block.text)
        elif block.type == "image":
            print(block.mime_type)
            print(block.data[:32])
```

Like the chat side, you can build the collection from parts: `create_images_models()`, the `openrouter_images_provider()` factory, and `create_images_provider()` for custom image providers. `ImagesModels.refresh(provider_id)` exists for image providers with a `refresh` function.

The old global image API remains available:

```python
from pi_ai import ImagesContext, ImagesOptions, TextContent
from pi_ai.image_models import get_image_model
from pi_ai.images import generate_images


async def example() -> None:
    model = get_image_model("openrouter", "google/gemini-2.5-flash-image")
    assert model is not None
    result = await generate_images(
        model,
        ImagesContext(input=[TextContent(text="Generate a red circle on a plain white background.")]),
        ImagesOptions(api_key="openrouter-key"),
    )
    print(result.stop_reason)
```

Some models also support image input:

```python
import base64
from pathlib import Path

from pi_ai import ImageContent, ImagesContext, TextContent
from pi_ai.providers.all import builtin_images_models


async def example() -> None:
    images_models = builtin_images_models()
    model = images_models.get_model("openrouter", "google/gemini-2.5-flash-image")
    assert model is not None
    image_data = base64.b64encode(Path("input.png").read_bytes()).decode("ascii")
    result = await images_models.generate_images(
        model,
        ImagesContext(
            input=[
                TextContent(text="Create a variation of this image with a blue background."),
                ImageContent(data=image_data, mime_type="image/png"),
            ]
        ),
    )
    print(result.stop_reason)
```

Check capabilities on the model metadata:

```python
from pi_ai.providers.all import builtin_images_models

images_models = builtin_images_models()
model = images_models.get_model("openrouter", "google/gemini-2.5-flash-image")
assert model is not None
print(model.input)
print(model.output)
```

### Notes and Limitations

- Image models live in `ImagesModels` collections, chat models in `Models` collections; the two are separate surfaces.
- Use `generate_images()`, not the chat/stream APIs.
- Image-generation models do not participate in tool calling.
- Outputs are returned in `AssistantImages.output` and can include both base64-encoded `ImageContent` blocks and `TextContent` blocks.
- Some models return only images, others return images plus text. Check `model.output`.
- Some models accept image input, others are text-to-image only. Check `model.input`.
- Image generation supports options such as `api_key`, `signal`, `headers`, `on_payload`, and `on_response`, and results may include `stop_reason`, `response_id`, and `usage`.
- If you want a model to analyze images in a conversation or call tools, use the regular chat APIs with a model that supports image input.
- At the moment, image generation is available through OpenRouter.

## Thinking/Reasoning

Many models support thinking/reasoning capabilities. Check if a model supports reasoning via the `reasoning` property. If you pass reasoning options to a non-reasoning model, they are silently ignored.

### Unified Interface (streamSimple/completeSimple)

```python
from pi_ai import Context, SimpleStreamOptions, UserMessage, complete, get_supported_thinking_levels, now_ms
from pi_ai.providers.all import builtin_models


async def example() -> None:
    models = builtin_models()
    model = models.get_model("anthropic", "claude-sonnet-4-5")
    assert model is not None

    if model.reasoning:
        print("Model supports reasoning/thinking")
    print(get_supported_thinking_levels(model))

    response = await complete(
        await models.stream_simple(
            model,
            Context(messages=[UserMessage(content="Solve: 2x + 5 = 13", timestamp=now_ms())]),
            SimpleStreamOptions(reasoning="medium"),
        )
    )

    for block in response.content:
        if block.type == "thinking":
            print("Thinking:", block.thinking)
        elif block.type == "text":
            print("Response:", block.text)
```

`xhigh` and `max` are model-specific, opt-in levels. Use `get_supported_thinking_levels(model)` to determine whether a concrete model exposes either level.

### Provider-Specific Options (stream/complete)

`models.stream()` accepts the owning API's option dataclass. Use `has_api()` to narrow a dynamically looked-up model before selecting the option type.

```python
from pi_ai import Context, UserMessage, complete, has_api, now_ms
from pi_ai.api.anthropic_messages import AnthropicOptions
from pi_ai.api.google_generative_ai import GoogleOptions, GoogleThinkingOptions
from pi_ai.api.openai_responses import OpenAIResponsesOptions
from pi_ai.providers.all import builtin_models


async def example() -> None:
    models = builtin_models()
    context = Context(messages=[UserMessage(content="Explain the problem", timestamp=now_ms())])

    openai_model = models.get_model("openai", "gpt-5-mini")
    if openai_model is not None and has_api(openai_model, "openai-responses"):
        await complete(
            await models.stream(
                openai_model,
                context,
                OpenAIResponsesOptions(reasoning_effort="medium", reasoning_summary="detailed"),
            )
        )

    anthropic_model = models.get_model("anthropic", "claude-sonnet-4-5")
    if anthropic_model is not None and has_api(anthropic_model, "anthropic-messages"):
        await complete(
            await models.stream(
                anthropic_model,
                context,
                AnthropicOptions(thinking_enabled=True, thinking_budget_tokens=8192),
            )
        )

    google_model = models.get_model("google", "gemini-2.5-flash")
    if google_model is not None and has_api(google_model, "google-generative-ai"):
        await complete(
            await models.stream(
                google_model,
                context,
                GoogleOptions(thinking=GoogleThinkingOptions(enabled=True, budget_tokens=8192)),
            )
        )
```

### Streaming Thinking Content

When streaming, thinking content is delivered through specific events:

```python
import sys

from pi_ai import Context, SimpleStreamOptions, UserMessage, now_ms
from pi_ai.providers.all import builtin_models


async def example() -> None:
    models = builtin_models()
    model = models.get_model("anthropic", "claude-sonnet-4-5")
    assert model is not None
    context = Context(messages=[UserMessage(content="Reason step by step", timestamp=now_ms())])
    stream = await models.stream_simple(model, context, SimpleStreamOptions(reasoning="high"))

    async for event in stream:
        if event.type == "thinking_start":
            print("[Model started thinking]")
        elif event.type == "thinking_delta":
            sys.stdout.write(event.delta)
        elif event.type == "thinking_end":
            print("\n[Thinking complete]")
```

## Stop Reasons

Every `AssistantMessage` includes a `stop_reason` field that indicates how the generation ended:

- `"pending"` - Only present in partial messages when the stop reason is not known yet.
- `"stop"` - This is the final message the model will produce this turn.
- `"length"` - Output hit the maximum token limit.
- `"toolUse"` - Model is calling tools and expects tool results.
- `"error"` - An error occurred during generation.
- `"aborted"` - Request was cancelled via abort signal.
- `"deferred"` - A provider returned a deferred response handle.

`AssistantMessage` may also include `response_id`, a provider-specific upstream response or message identifier when the underlying API exposes one.

## Error Handling

Request failures do not throw out of stream functions after a stream is created. When a request ends with an error, the streaming API emits an error event and the final message carries the details:

```python
from pi_ai import Context, UserMessage, now_ms
from pi_ai.providers.all import builtin_models


async def example() -> None:
    models = builtin_models()
    model = models.get_model("openai", "gpt-4o-mini")
    assert model is not None
    stream = await models.stream(model, Context(messages=[UserMessage(content="Hello", timestamp=now_ms())]))

    async for event in stream:
        if event.type == "error":
            print(f"Error ({event.reason}):", event.error.error_message)
            print("Partial content:", event.error.content)

    message = await stream.result()
    if message.stop_reason in ("error", "aborted"):
        print("Request failed:", message.error_message)
        print("Partial content received:", message.content)
        print("Tokens used:", message.usage)
```

Auth failures (no key configured, OAuth refresh failed, unknown provider) surface the same way: as a stream error with `stop_reason="error"`.

### Aborting Requests

The abort signal allows you to cancel in-progress requests. Aborted requests have `stop_reason == "aborted"`.

```python
import asyncio
import sys

from pi_ai import Context, StreamOptions, UserMessage, now_ms
from pi_ai.providers.all import builtin_models
from pi_ai.utils.abort import AbortController


async def example() -> None:
    models = builtin_models()
    model = models.get_model("openai", "gpt-4o-mini")
    assert model is not None
    controller = AbortController()

    async def abort_later() -> None:
        await asyncio.sleep(2)
        controller.abort()

    asyncio.create_task(abort_later())
    stream = await models.stream(
        model,
        Context(messages=[UserMessage(content="Write a long story", timestamp=now_ms())]),
        StreamOptions(signal=controller.signal),
    )

    async for event in stream:
        if event.type == "text_delta":
            sys.stdout.write(event.delta)
        elif event.type == "error":
            label = "Aborted" if event.reason == "aborted" else "Error"
            print(f"{label}:", event.error.error_message)

    response = await stream.result()
    if response.stop_reason == "aborted":
        print("Request was aborted:", response.error_message)
        print("Partial content received:", response.content)
        print("Tokens used:", response.usage)
```

### Continuing After Abort

Aborted messages can be added to the conversation context and continued in subsequent requests:

```python
import asyncio

from pi_ai import Context, StreamOptions, UserMessage, complete, now_ms
from pi_ai.providers.all import builtin_models
from pi_ai.utils.abort import AbortController


async def example() -> None:
    models = builtin_models()
    model = models.get_model("openai", "gpt-4o-mini")
    assert model is not None
    context = Context(messages=[UserMessage(content="Explain quantum computing in detail", timestamp=now_ms())])

    controller = AbortController()

    async def abort_later() -> None:
        await asyncio.sleep(2)
        controller.abort()

    asyncio.create_task(abort_later())
    partial = await complete(await models.stream(model, context, StreamOptions(signal=controller.signal)))

    context.messages.append(partial)
    context.messages.append(UserMessage(content="Please continue", timestamp=now_ms()))

    continuation = await complete(await models.stream(model, context))
    print(continuation.stop_reason)
```

### Debugging Provider Payloads

Use the `on_payload` callback to inspect or replace the request payload sent to the provider. This is useful for debugging request formatting issues or provider validation errors.

```python
import json
from typing import Any

from pi_ai import Context, StreamOptions, UserMessage, complete, now_ms
from pi_ai.providers.all import builtin_models


def on_payload(payload: Any, model: object) -> Any:
    print("Provider payload:", json.dumps(payload, indent=2))
    return payload


async def example() -> None:
    models = builtin_models()
    model = models.get_model("openai", "gpt-4o-mini")
    assert model is not None
    response = await complete(
        await models.stream(
            model,
            Context(messages=[UserMessage(content="Hello", timestamp=now_ms())]),
            StreamOptions(on_payload=on_payload),
        )
    )
    print(response.stop_reason)
```

The callback is supported by `stream`, `stream_simple`, and image generation options where the provider implementation uses HTTP request construction.

## Custom Providers

### createProvider()

`create_provider()` builds a provider from parts: identity, auth, a model list, and an API implementation. Use it for local inference servers, proxies, or any OpenAI/Anthropic-compatible endpoint.

```python
from pi_ai import Context, Model, ModelCost, UserMessage, complete, now_ms
from pi_ai.api import openai_completions
from pi_ai.auth.types import ApiKeyAuth, AuthResult, ProviderAuth, ResolvedAuth
from pi_ai.registry import Models, create_provider


async def resolve_keyless(credential: object = None, env: object = None) -> AuthResult:
    return AuthResult(auth=ResolvedAuth(), source="keyless")


async def example() -> None:
    ollama_model = Model(
        id="llama-3.1-8b",
        name="Llama 3.1 8B (Ollama)",
        api="openai-completions",
        provider="ollama",
        base_url="http://localhost:11434/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=32000,
    )

    ollama = create_provider(
        id="ollama",
        name="Ollama",
        base_url="http://localhost:11434/v1",
        auth=ProviderAuth(api_key=ApiKeyAuth(name="Ollama", resolve=resolve_keyless)),
        models=[ollama_model],
        api=openai_completions,
    )

    models = Models()
    models.add(ollama)
    model = models.get_model("ollama", "llama-3.1-8b")
    assert model is not None
    context = Context(messages=[UserMessage(content="Hello", timestamp=now_ms())])
    response = await complete(await models.stream(model, context))
    print(response.stop_reason)
```

For providers with real keys, `env_api_key_auth(display_name, env_vars)` gives the standard behavior: stored credential wins, then the first set environment variable.

```python
from pi_ai.api import openai_completions
from pi_ai.auth.helpers import env_api_key_auth
from pi_ai.auth.types import ProviderAuth
from pi_ai.registry import create_provider

proxy = create_provider(
    id="my-proxy",
    name="My Proxy",
    auth=ProviderAuth(api_key=env_api_key_auth("My proxy API key", ["MY_PROXY_API_KEY"])),
    models=[],
    api=openai_completions,
)
```

Mixed-API providers pass a map keyed by `model.api`; each model dispatches to its API implementation:

```python
from pi_ai.api import anthropic_messages, openai_responses
from pi_ai.auth.helpers import env_api_key_auth
from pi_ai.auth.types import ProviderAuth
from pi_ai.registry import create_provider


gateway = create_provider(
    id="my-gateway",
    name="My Gateway",
    auth=ProviderAuth(api_key=env_api_key_auth("Gateway key", ["GATEWAY_API_KEY"])),
    models=[],
    api={
        "anthropic-messages": anthropic_messages,
        "openai-responses": openai_responses,
    },
)
```

Provider-wide endpoint or request transformations belong in the provider's API implementation. Wrap the object you pass as `api` so every request goes through the transformation before dispatch.

```python
from dataclasses import replace
from typing import Any

from pi_ai import Context, Model, SimpleStreamOptions, StreamOptions
from pi_ai.api import openai_completions
from pi_ai.auth.helpers import env_api_key_auth
from pi_ai.auth.types import ProviderAuth
from pi_ai.registry import create_provider
from pi_ai.utils.event_stream import AssistantMessageEventStream


def tenant_streams(tenant_id: str) -> object:
    def with_tenant(model: Model) -> Model:
        return replace(model, base_url=model.base_url.replace("{tenant}", tenant_id))

    class TenantStreams:
        def stream(
            self,
            model: Model,
            context: Context,
            options: StreamOptions | None = None,
            **kwargs: Any,
        ) -> AssistantMessageEventStream:
            return openai_completions.stream(with_tenant(model), context, options, **kwargs)

        def stream_simple(
            self,
            model: Model,
            context: Context,
            options: SimpleStreamOptions | None = None,
            **kwargs: Any,
        ) -> AssistantMessageEventStream:
            return openai_completions.stream_simple(with_tenant(model), context, options, **kwargs)

    return TenantStreams()


tenant_gateway = create_provider(
    id="tenant-gateway",
    name="Tenant Gateway",
    auth=ProviderAuth(api_key=env_api_key_auth("Gateway key", ["GATEWAY_API_KEY"])),
    models=[],
    api=tenant_streams("tenant-1"),
)
```

Dynamic model publication through `fetchModels`, `Models.refresh()`, and `ModelsStore` is not ported for chat providers. Keep custom model lists in your application and replace `provider.models` when they change.

Custom models can carry `headers` and `compat` flags. `Models.get_auth(model)` includes model headers, and stream methods merge them before explicit request headers.

Some OpenAI-compatible servers do not understand the `developer` role used for reasoning-capable models. Set `compat["supportsDeveloperRole"]` or `compat["supports_developer_role"]` to `False` so the system prompt is sent as a `system` message instead. If the server also does not support `reasoning_effort`, set `supportsReasoningEffort` / `supports_reasoning_effort` to `False`.

Use model-level `thinking_level_map` to describe model-specific thinking controls. Keys are pi thinking levels (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`). Missing standard levels through `high` use provider defaults; `xhigh` and `max` are opt-in and require a non-null map entry.

```python
from pi_ai import Model, ModelCost

ollama_reasoning_model = Model(
    id="gpt-oss:20b",
    name="GPT-OSS 20B (Ollama)",
    api="openai-completions",
    provider="ollama",
    base_url="http://localhost:11434/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(),
    context_window=131072,
    max_tokens=32000,
    thinking_level_map={
        "minimal": None,
        "low": None,
        "medium": None,
        "high": "high",
        "xhigh": None,
    },
    compat={
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": False,
    },
)
```

### Calling API Implementations Directly

The API implementations are importable on their own. Each module exposes `stream()` and `stream_simple()`. Direct calls bypass provider auth; pass `api_key` explicitly.

```python
from pi_ai import Context, StreamOptions, UserMessage, complete, now_ms
from pi_ai.api.anthropic_messages import AnthropicOptions, stream
from pi_ai.providers.all import get_builtin_model


async def example() -> None:
    claude_model = get_builtin_model("anthropic", "claude-sonnet-4-5")
    assert claude_model is not None
    context = Context(messages=[UserMessage(content="Hello", timestamp=now_ms())])
    response = await complete(
        stream(
            claude_model,
            context,
            AnthropicOptions(
                api_key="anthropic-key",
                thinking_enabled=True,
                thinking_budget_tokens=2048,
            ),
        )
    )
    print(response.stop_reason)
```

Built-in API implementations live under `pi_ai.api.<api_module>`:

| API id | Python module | Options dataclass |
|--------|---------------|-------------------|
| `anthropic-messages` | `pi_ai.api.anthropic_messages` | `AnthropicOptions` |
| `openai-completions` | `pi_ai.api.openai_completions` | `OpenAICompletionsOptions` |
| `openai-responses` | `pi_ai.api.openai_responses` | `OpenAIResponsesOptions` |
| `openai-codex-responses` | `pi_ai.api.openai_codex_responses` | not ported; streaming raises `NotImplementedError` |
| `azure-openai-responses` | `pi_ai.api.azure_openai_responses` | `AzureOpenAIResponsesOptions` |
| `google-generative-ai` | `pi_ai.api.google_generative_ai` | `GoogleOptions` |
| `google-vertex` | `pi_ai.api.google_vertex` | `GoogleVertexOptions` |
| `mistral-conversations` | `pi_ai.api.mistral_conversations` | `MistralOptions` |
| `bedrock-converse-stream` | `pi_ai.api.bedrock_converse_stream` | not ported; streaming raises `NotImplementedError` |
| `pi-messages` | `pi_ai.api.pi_messages` | `PiMessagesOptions` |

There are no Python `*.lazy` modules. Importing an implementation module imports Python HTTP code, not a vendor SDK.

### OpenAI Compatibility Settings

The `openai-completions` API is implemented by many providers with minor differences. The Python port auto-detects compatibility settings by provider/base URL and then applies `model.compat` overrides. Both TypeScript camelCase keys and Python snake_case keys are accepted.

Common `openai-completions` compat keys include:

- `supportsStore` / `supports_store`
- `supportsDeveloperRole` / `supports_developer_role`
- `supportsReasoningEffort` / `supports_reasoning_effort`
- `supportsUsageInStreaming` / `supports_usage_in_streaming`
- `supportsStrictMode` / `supports_strict_mode`
- `supportsOpenAIGrammarTools` / `supports_openai_grammar_tools`
- `sendSessionAffinityHeaders` / `send_session_affinity_headers`
- `sessionAffinityFormat` / `session_affinity_format`
- `maxTokensField` / `max_tokens_field`
- `requiresToolResultName` / `requires_tool_result_name`
- `requiresAssistantAfterToolResult` / `requires_assistant_after_tool_result`
- `requiresThinkingAsText` / `requires_thinking_as_text`
- `requiresReasoningContentOnAssistantMessages` / `requires_reasoning_content_on_assistant_messages`
- `thinkingFormat` / `thinking_format`
- `chatTemplateKwargs` / `chat_template_kwargs`
- `chatTemplateArgs` / `chat_template_args`
- `cacheControlFormat` / `cache_control_format`
- `openRouterRouting` / `open_router_routing`
- `vercelGatewayRouting` / `vercel_gateway_routing`

For `openai-responses` models, compat keys include `supportsDeveloperRole`, `sessionAffinityFormat`, `supportsLongCacheRetention`, `supportsStrictMode`, `supportsOpenAIGrammarTools`, `supportsAdditionalTools`, `supportsToolSearch`, and `supportsExplicitPromptCacheMode` with the corresponding snake_case spellings.

If `compat` is not set, the library falls back to URL/provider detection. If `compat` is partially set, unspecified fields use detected defaults.

## Faux Provider for Tests

`faux_provider()` builds an in-memory provider with scripted responses for tests and demos:

```python
from pi_ai import Context, TextContent, ToolCall, ToolResultMessage, UserMessage, complete, now_ms
from pi_ai.providers.faux import (
    FauxModelDefinition,
    RegisterFauxProviderOptions,
    faux_assistant_message,
    faux_provider,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from pi_ai.registry import Models


async def example() -> None:
    faux = faux_provider(RegisterFauxProviderOptions(tokens_per_second=50))

    models = Models()
    models.add(faux.provider)

    model = faux.get_model()
    assert model is not None
    context = Context(messages=[UserMessage(content="Summarize package.json and then call echo", timestamp=now_ms())])

    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_thinking("Need to inspect package metadata first."),
                    faux_tool_call("echo", {"text": "package.json"}),
                ],
                stop_reason="toolUse",
            )
        ]
    )

    first = await complete(await models.stream(model, context))
    context.messages.append(first)

    tool_call = next(block for block in first.content if isinstance(block, ToolCall))
    context.messages.append(
        ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name="echo",
            content=[TextContent(text="package.json contents here")],
            is_error=False,
            timestamp=now_ms(),
        )
    )

    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_thinking("Now I can summarize the tool output."),
                    faux_text("Here is the summary."),
                ]
            )
        ]
    )

    stream = await models.stream(model, context)
    async for event in stream:
        print(event.type)

    multi_model = faux_provider(
        RegisterFauxProviderOptions(
            provider="faux-multi",
            models=[
                FauxModelDefinition(id="faux-fast", reasoning=False),
                FauxModelDefinition(id="faux-thinker", reasoning=True),
            ],
        )
    )
    models.add(multi_model.provider)
    thinker = multi_model.get_model("faux-thinker")

    print(thinker.reasoning if thinker else None)
    print(faux.get_pending_response_count())
    print(faux.state.call_count)
```

Notes:

- Responses are consumed from a queue in request start order.
- If the queue is empty, the faux provider returns an assistant error message with `error_message="No more faux responses queued"`.
- Use `faux.set_responses([...])` to replace the remaining queue and `faux.append_responses([...])` to add more responses.
- `faux.models` exposes all faux models. `faux.get_model()` returns the first one, and `faux.get_model(id)` returns a specific one.
- Use `faux_assistant_message(...)` for scripted assistant replies. Use `faux_text(...)`, `faux_thinking(...)`, and `faux_tool_call(...)` to build content blocks without filling in low-level fields manually.
- Usage is estimated at roughly 1 token per 4 characters. When `session_id` is present and `cache_retention` is not `"none"`, prompt cache reads and writes are simulated automatically.
- Tool call arguments stream incrementally via `toolcall_delta` chunks.
- Set `tokens_per_second` to pace chunk delivery in real time.
- The intended use is one deterministic scripted flow per handle. If you need independent concurrent flows, create separate faux providers with distinct `provider` ids.

## Cross-Provider Handoffs

The library supports handoffs between different LLM providers within the same conversation. This allows you to switch models mid-conversation while preserving context, including thinking blocks, tool calls, and tool results.

When messages from one provider are sent to a different provider, the library transforms them for compatibility:

- User and tool result messages are passed through unchanged.
- Assistant messages from the same provider/API are preserved as-is.
- Assistant messages from different providers have their thinking blocks converted to text.
- Tool calls and regular text are preserved unchanged.

```python
from pi_ai import Context, SimpleStreamOptions, UserMessage, complete, now_ms
from pi_ai.providers.anthropic import anthropic_provider
from pi_ai.providers.google import google_provider
from pi_ai.providers.openai import openai_provider
from pi_ai.registry import Models


async def example() -> None:
    models = Models()
    models.add(anthropic_provider())
    models.add(openai_provider())
    models.add(google_provider())

    context = Context(messages=[])

    claude = models.get_model("anthropic", "claude-sonnet-4-5")
    assert claude is not None
    context.messages.append(UserMessage(content="What is 25 * 18?", timestamp=now_ms()))
    context.messages.append(
        await complete(await models.stream_simple(claude, context, SimpleStreamOptions(reasoning="medium")))
    )

    gpt5 = models.get_model("openai", "gpt-5-mini")
    assert gpt5 is not None
    context.messages.append(UserMessage(content="Is that calculation correct?", timestamp=now_ms()))
    context.messages.append(await complete(await models.stream(gpt5, context)))

    gemini = models.get_model("google", "gemini-2.5-flash")
    assert gemini is not None
    context.messages.append(UserMessage(content="What was the original question?", timestamp=now_ms()))
    gemini_response = await complete(await models.stream(gemini, context))
    print(gemini_response.stop_reason)
```

All implemented providers can handle text, tool calls and results, image tool results for vision-capable models, thinking blocks transformed to text, and aborted messages with partial content.

## Context Serialization

The `Context` object is made of dataclasses and standard containers, so it can be serialized after converting dataclasses to dictionaries. Reconstructing dataclasses from JSON is application-specific because message/content variants are discriminated by `role` and `type`.

```python
import json
from dataclasses import asdict

from pi_ai import Context, UserMessage, complete, now_ms
from pi_ai.providers.all import builtin_models


async def example() -> None:
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="What is Python?", timestamp=now_ms())],
    )

    models = builtin_models()
    model = models.get_model("openai", "gpt-4o-mini")
    assert model is not None
    response = await complete(await models.stream(model, context))
    context.messages.append(response)

    serialized = json.dumps(asdict(context))
    print(serialized)

    restored_payload = json.loads(serialized)
    restored = Context(system_prompt=restored_payload.get("system_prompt"), messages=[])
    restored.messages.append(UserMessage(content="Tell me more about its type system", timestamp=now_ms()))

    new_model = models.get_model("anthropic", "claude-3-5-haiku-20241022")
    assert new_model is not None
    continuation = await complete(await models.stream(new_model, restored))
    print(continuation.stop_reason)
```

Models are plain serializable dataclasses too, so persisting which model was used is a JSON conversion away. If the context contains images, the base64 strings are serialized with the rest of the content.

## Browser Usage

Not applicable. This is a Python package for Python runtimes, not a browser bundle. Do not expose provider API keys in frontend code; use a backend service if a browser application needs model access.

## Bundling and Tree Shaking

Not applicable in the TypeScript bundler sense. Import only the provider factories you need when you want a smaller import surface or faster startup; `pi_ai.providers.all` imports every built-in provider factory and catalog.

### Provider-Scoped Environment Overrides

Pass `env` in stream options to scope provider configuration to a request. Values in `env` are passed to provider API code for configuration such as Cloudflare account IDs, Azure OpenAI settings, Vertex project/location, Bedrock settings, `PI_CACHE_RETENTION`, and `HTTP_PROXY`/`HTTPS_PROXY`. In the current Python port, `Models.stream()` passes `env` into auth resolution only when `StreamOptions.api_key` is also set; otherwise ambient auth resolution reads the collection environment or process environment.

```python
from pi_ai import Context, StreamOptions, UserMessage, complete, now_ms
from pi_ai.providers.all import builtin_models


async def example() -> None:
    models = builtin_models()
    model = models.get_model("cloudflare-ai-gateway", "workers-ai/@cf/moonshotai/kimi-k2.6")
    assert model is not None

    response = await complete(
        await models.stream(
            model,
            Context(messages=[UserMessage(content="Hello", timestamp=now_ms())]),
            StreamOptions(
                api_key="...",
                env={
                    "CLOUDFLARE_ACCOUNT_ID": "account-id",
                    "CLOUDFLARE_GATEWAY_ID": "gateway-id",
                },
            ),
        )
    )
    print(response.stop_reason)
```

Use this when one process needs different provider settings per request, or when ambient environment variables should not leak into a provider call.

## OAuth Providers

Several providers support OAuth authentication instead of static API keys:

- **Anthropic** (Claude Pro/Max subscription)
- **OpenAI Codex** (ChatGPT Plus/Pro subscription): login/catalog support is present, but the Codex streaming transport is not ported.
- **GitHub Copilot** (Copilot subscription)
- **OpenRouter** (OAuth PKCE that mints a user-controlled API key)
- **Kimi For Coding**
- **Radius**
- **xAI**

Each provider carries an `OAuthAuth` on `provider.auth.oauth` with three operations: `login(interaction)`, `refresh(credential, signal)`, and `to_auth(credential)`. Refresh is automatic in `models.get_auth(provider_id)` and request paths when a stored OAuth credential is expired.

```python
from pi_ai.auth.types import AuthEvent, AuthInteraction, AuthPrompt
from pi_ai.providers.anthropic import anthropic_provider
from pi_ai.registry import Models
from pi_ai.utils.abort import AbortSignal


class ConsoleInteraction(AuthInteraction):
    def __init__(self) -> None:
        self.signal = AbortSignal()

    async def prompt(self, prompt: AuthPrompt) -> str:
        return input(f"{prompt.message}: ")

    def notify(self, event: AuthEvent) -> None:
        if event.type == "info" and event.message:
            print(event.message)
            for link in event.links:
                print(f"{link.label or 'More information'}: {link.url}")
        if event.type == "auth_url":
            print(f"Open: {event.url}")
        if event.type == "device_code":
            print(f"Code: {event.user_code} at {event.verification_uri}")
        if event.type == "progress" and event.message:
            print(event.message)


async def example() -> None:
    models = Models()
    models.add(anthropic_provider())

    await models.login_oauth("anthropic", ConsoleInteraction())

    model = models.get_model("anthropic", "claude-sonnet-4-5")
    assert model is not None
    await models.logout("anthropic")
```

### Vertex AI

Vertex AI models support a Google Cloud API key, access-token configuration, and limited ADC-style environment discovery in the Python port.

- **API key**: Set `GOOGLE_CLOUD_API_KEY` or pass `api_key` in call options.
- **ADC environment**: Set `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT` (or `GCLOUD_PROJECT`), and `GOOGLE_CLOUD_LOCATION`.
- **Explicit access token**: Use `GoogleVertexOptions(access_token=..., project=..., location=...)` or the corresponding environment handled by the provider module.

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="my-project"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

Official docs: [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials)

### CLI Login

The quickest way to authenticate OAuth providers:

```bash
pp-ai login              # interactive provider selection
pp-ai login anthropic    # login to a specific provider
pp-ai list               # list OAuth-capable providers
```

Credentials are saved to `auth.json` in the current directory.

### Programmatic OAuth

Built-in login and refresh flows are provider implementations. Use provider-owned `OAuthAuth`, which composes with `CredentialStore` and gets auto-refresh through `Models`.

Provider notes:

**OpenAI Codex**: Requires a ChatGPT Plus or Pro subscription. The catalog and OAuth pieces are present, but `openai-codex-responses` streaming is not ported and raises `NotImplementedError`.

**Azure OpenAI (Responses)**: Uses the Responses API only. Set `AZURE_OPENAI_API_KEY` and either `AZURE_OPENAI_BASE_URL` or `AZURE_OPENAI_RESOURCE_NAME`. Use `AZURE_OPENAI_API_VERSION` to override the API version if needed. Deployment names are treated as model IDs by default; override with `AzureOpenAIResponsesOptions.azure_deployment_name` or `AZURE_OPENAI_DEPLOYMENT_NAME_MAP` using comma-separated `model-id=deployment` pairs.

**GitHub Copilot**: If you get "The requested model is not supported", enable the model manually in VS Code: open Copilot Chat, click the model selector, select the model, and click "Enable".

## Migrating from the Old Global API

Older TypeScript versions exposed a global API: `stream()`/`complete()` dispatching on `model.api` via a global registry, sync `getModel()`/`getModels()`/`getProviders()` catalog reads, `registerApiProvider()`, `getEnvApiKey()`, and per-API lazy stream functions. The Python port keeps a smaller compatibility surface in `pi_ai.compat`.

```python
from pi_ai import Context, UserMessage, now_ms
from pi_ai.compat import complete
from pi_ai.providers.all import get_builtin_model


async def example() -> None:
    model = get_builtin_model("openai", "gpt-4o-mini")
    assert model is not None
    response = await complete(model, Context(messages=[UserMessage(content="Hello", timestamp=now_ms())]))
    print(response.stop_reason)
```

Migration table:

| Old TypeScript | Python collection API |
|----------------|-----------------------|
| `getModel('openai', 'gpt-4o-mini')` | `models.get_model('openai', 'gpt-4o-mini')` or `get_builtin_model()` |
| `getModels('anthropic')` / `getProviders()` | `models.get_models('anthropic')` / `models.get_providers()` |
| `stream(model, ctx, opts)` | `await models.stream(model, ctx, opts)` |
| `complete(model, ctx, opts)` | `await complete(await models.stream(model, ctx, opts))` |
| `registerApiProvider({ api, stream, streamSimple })` | `create_provider(..., api=...)` + `models.add()`; global `pi_ai.compat.register_api_provider()` remains for compatibility tests |
| `getEnvApiKey('openai')` | `await models.get_auth(model.provider)` or `pi_ai.env_api_keys.get_env_api_key()` |
| `streamAnthropic(model, ctx, opts)` | `stream` from `pi_ai.api.anthropic_messages`, or a provider in a collection |
| `registerFauxProvider()` | `faux_provider()` + `models.add()`; `pi_ai.compat.register_faux_provider()` remains for global-registry tests |

Extension provider registration (`registerProvider`/`unregisterProvider`) is not ported.

## Development

### Adding a New Provider

Adding a new LLM provider requires changes across multiple files. The layered layout is: API implementations in `src/pi_ai/api/`, provider factories in `src/pi_ai/providers/`, committed generated catalog JSON under `src/pi_ai/providers/data/`, and provider exports in `src/pi_ai/providers/__init__.py` and `src/pi_ai/providers/all.py`.

#### 1. Core Types (`src/types.ts`)

Python path: `src/pi_ai/types.py`.

- Add the API identifier to `KnownApi` if it is a new API.
- Use or extend `Model`, `StreamOptions`, `SimpleStreamOptions`, and provider-specific options dataclasses.
- Keep wire string literals in TypeScript spelling; Python field names are snake_case.

#### 2. API Implementation (`src/api/<api-id>.ts`, only for a new API)

Python path: `src/pi_ai/api/<api_id>.py`.

Create a new API implementation module that exports `stream()` and `stream_simple()`, plus:

- An options dataclass extending `StreamOptions` when provider-specific fields are needed.
- Message conversion from `Context` to the provider payload.
- Tool conversion if the provider supports tools.
- Response parsing to emit standardized events (`text`, `tool_call`, `thinking`, `usage`, `stop`).

There is no Python lazy wrapper layer.

#### 3. Model Generation (`scripts/generate-models.ts`, `scripts/generate-image-models.ts`)

Python scripts live under `packages/pi-ai/scripts/`.

- Add logic to fetch and parse models from the provider's source.
- Map chat/tool-capable provider model data to `Model` JSON consumed by `pi_ai.model_catalog`.
- Map image-generation provider model data to `ImagesModel` JSON consumed by `pi_ai.image_models`.
- Handle provider-specific quirks: pricing format, capability flags, model ID transformations, and compatibility flags.

#### 4. Provider Factory (`src/providers/<id>.ts`)

Python path: `src/pi_ai/providers/<id>.py`.

- Wire `create_provider()` with catalog, auth, and API module.
- Auth: use `env_api_key_auth` for standard key providers, custom `ApiKeyAuth` for ambient auth, and `lazy_oauth` where an OAuth flow exists.
- Register the factory in `src/pi_ai/providers/all.py`.
- Re-export it from `src/pi_ai/providers/__init__.py`.
- If it is a new global compat API, register it in `src/pi_ai/compat.py`.

#### 5. Tests (`test/`)

Create or update targeted tests for streaming, tool use, token usage, aborts, empty messages, context limits, image handling, Unicode handling, orphaned tool calls, image tool results, total token accounting, cross-provider handoff, provider listing, and auth resolution.

For providers with non-standard auth (AWS, Google Vertex), add credential detection helpers next to the provider or tests.

#### 6. Coding Agent Integration (`../coding-agent/`)

For the Python port, update the corresponding `pi_coding_agent` model resolver and CLI help if the coding agent exposes the provider.

#### 7. Documentation

Update this README:

- Add the provider to Supported Providers.
- Document provider-specific options and authentication requirements.
- Add environment variables to the Environment Variables table.
- State plainly if a catalog exists before streaming support is ported.

#### 8. Changelog

Add an entry under `## [Unreleased]` in the package changelog, if the Python package has one.

### Added

- Python port documentation for `pi_ai` provider/model APIs.

## License

MIT

---

`pp-ai` is developed in [HSPK/pp_ai](https://github.com/HSPK/pp_ai). It was split out of the `pp` monorepo; sibling packages (`pp-ai`, `pp-agent-core`, `pp-tui`, `pp-coding-agent`, ...) each live in their own
repository and are consumed from PyPI.
