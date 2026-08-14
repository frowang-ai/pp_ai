"""Provider API implementations.

Python port of `packages/ai/src/api/`. Each module speaks one provider wire
protocol and exposes the same `stream`/`stream_simple` pair.

**The lazy-loading layer has no Python module.** TypeScript wraps every API in
a `*.lazy.ts` shim (`api/anthropic-messages.lazy.ts`,
`api/azure-openai-responses.lazy.ts`, `api/google-generative-ai.lazy.ts`,
`api/google-vertex.lazy.ts`, `api/mistral-conversations.lazy.ts`,
`api/openai-codex-responses.lazy.ts`, `api/openai-completions.lazy.ts`,
`api/openai-responses.lazy.ts`, `api/openrouter-images.lazy.ts`,
`api/pi-messages.lazy.ts`) built on `api/lazy.ts`'s `lazyApi`/`lazyStream`.
Those exist so a provider's vendor SDK is only `import()`ed on first use,
keeping CLI startup and bundle size down: `lazyStream` returns a stream
*synchronously* while the module loads behind it, and converts a setup failure
into an `error` event on that stream.

Neither half applies here. This port has no vendor SDKs -- every API talks
HTTP through :mod:`pi_ai.utils.http` -- so there is nothing heavy to defer, and
`Provider.stream`/`stream_simple` are `async`, so a setup failure surfaces as a
raised exception at the `await` rather than as an error event. Callers that
want the error-event shape can catch and push one; nothing in this port needs
it.
"""

from __future__ import annotations
