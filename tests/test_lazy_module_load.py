"""Python port of `packages/ai/test/lazy-module-load.test.ts`.

The TypeScript suite hooks Node's module resolver and asserts that importing the
package never resolves `@anthropic-ai/sdk`, `openai`, `@google/genai` or
`@aws-sdk/client-bedrock-runtime` until a request is actually streamed, and that
streaming Anthropic then pulls in exactly one of them. That machinery exists to
keep Node-only SDKs out of the browser/Bun bundles, through the `*.lazy.ts`
wrapper modules.

This port speaks every provider wire protocol itself over `httpx` and depends on
no vendor SDK, so "which SDK got loaded" has no Python analogue: the answer is
always "none". The probes below keep what does carry over:

* no vendor SDK is ever imported, on any of the three entrypoints or when
  streaming Anthropic (the TypeScript expectation `loadedSpecifiers == []`,
  which for the Anthropic case is `["@anthropic-ai/sdk"]` upstream only because
  TypeScript delegates to that SDK);
* importing the root barrel stays cheap — it pulls in neither the provider api
  layer nor the HTTP transport.

The `providers/all` and `compat` entrypoints deliberately are not asserted to be
transport-free: they resolve api modules through ordinary top-level imports,
because the `*.lazy.ts` indirection only exists to satisfy a bundler.
"""

from __future__ import annotations

import json
import subprocess
import sys

VENDOR_SDK_MODULES = ["anthropic", "openai", "google.genai", "boto3", "botocore"]


def run_probe(action: str) -> dict[str, list[str]]:
    script = f"""
import json, sys
import pi_ai
{action}
targets = {VENDOR_SDK_MODULES!r}
loaded = [name for name in targets if name in sys.modules]
api_modules = sorted(name for name in sys.modules if name.startswith("pi_ai.api."))
print(json.dumps({{"loadedSdks": loaded, "apiModules": api_modules, "httpx": "httpx" in sys.modules}}))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"Probe failed (exit {result.returncode})\n{result.stdout}\n{result.stderr}"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines, f"Probe produced no output\n{result.stderr}"
    return json.loads(lines[-1])


def test_does_not_load_provider_sdks_when_importing_the_root_barrel():
    result = run_probe("")
    assert result["loadedSdks"] == []


def test_the_root_barrel_pulls_in_neither_the_api_layer_nor_the_http_transport():
    result = run_probe("")
    assert result["apiModules"] == []
    assert result["httpx"] is False


def test_does_not_load_provider_sdks_when_building_all_builtin_providers():
    result = run_probe(
        "from pi_ai.providers import all as providers_all\nproviders_all.builtin_models().get_models()\n"
    )
    assert result["loadedSdks"] == []


def test_does_not_load_provider_sdks_when_importing_the_compat_entrypoint():
    result = run_probe("import pi_ai.compat\n")
    assert result["loadedSdks"] == []


def test_loads_no_vendor_sdk_when_streaming_through_the_anthropic_api_module_directly():
    """TS: "loads only the Anthropic SDK when streaming through the lazy API
    wrapper" calls `compat.anthropicMessagesApi().streamSimple(...)` directly,
    a distinct entry point from the generic `compat.streamSimple` dispatch
    covered below (TS: "loads only the Anthropic SDK when dispatching through
    streamSimple"). This probe mirrors that direct-module entry point by
    importing `pi_ai.api.anthropic_messages` and calling its `stream_simple`
    without going through `compat`.
    """
    result = run_probe(
        "\n".join(
            [
                "import asyncio",
                "from pi_ai.api import anthropic_messages",
                "from pi_ai.types import Context, Model, ModelCost, SimpleStreamOptions, UserMessage",
                # The payload hook aborts the request before any socket is opened,
                # so the probe stays offline while still exercising dispatch.
                "def on_payload(payload, request_model):",
                "    raise RuntimeError('payload captured')",
                "async def main():",
                "    model = Model(",
                "        id='claude-sonnet-4-6', name='Claude Sonnet 4', api='anthropic-messages',",
                "        provider='anthropic', base_url='https://api.anthropic.com', reasoning=True,",
                "        input=['text'], cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),",
                "        context_window=200000, max_tokens=8192,",
                "    )",
                "    context = Context(messages=[UserMessage(content='hi')])",
                "    await anthropic_messages.stream_simple(",
                "        model, context, SimpleStreamOptions(api_key='fake', on_payload=on_payload)",
                "    ).result()",
                "asyncio.run(main())",
            ]
        )
    )
    assert result["loadedSdks"] == []


def test_loads_no_vendor_sdk_when_dispatching_through_stream_simple():
    result = run_probe(
        "\n".join(
            [
                "import asyncio",
                "from pi_ai import compat",
                "from pi_ai.providers.all import get_builtin_model",
                "from pi_ai.types import Context, SimpleStreamOptions, UserMessage",
                # The payload hook aborts the request before any socket is opened,
                # so the probe stays offline while still exercising dispatch.
                "def on_payload(payload, request_model):",
                "    raise RuntimeError('payload captured')",
                "async def main():",
                "    model = get_builtin_model('anthropic', 'claude-sonnet-4-6')",
                "    context = Context(messages=[UserMessage(content='hi')])",
                "    await compat.stream_simple(",
                "        model, context, SimpleStreamOptions(api_key='fake', on_payload=on_payload)",
                "    ).result()",
                "asyncio.run(main())",
            ]
        )
    )
    assert result["loadedSdks"] == []
