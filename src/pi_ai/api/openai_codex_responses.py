"""OpenAI Codex Responses API — not ported.

Placeholder for `packages/ai/src/apis/openai-codex-responses.ts`, a documented
omission of this port: it drives the ChatGPT backend's private Codex protocol
(account-scoped session headers, rate-limit windows and response-item
replay) on top of the OAuth flow, and none of that surface is exercised by the
Python packages yet.

The module still exists so `openai-codex` models stay discoverable through the
generated catalog (`pi_ai.providers.openai_codex`); any attempt to actually
stream raises :class:`NotImplementedError`.
"""

from __future__ import annotations

from typing import Any, NoReturn

from ..types import Context, Model, SimpleStreamOptions, StreamOptions

_MESSAGE = (
    "The openai-codex-responses API is not ported to Python. "
    "openai-codex models are listed for discovery only; use the `openai` provider to run them."
)


def stream(model: Model, context: Context, options: StreamOptions | None = None, **kwargs: Any) -> NoReturn:
    raise NotImplementedError(_MESSAGE)


def stream_simple(
    model: Model, context: Context, options: SimpleStreamOptions | None = None, **kwargs: Any
) -> NoReturn:
    raise NotImplementedError(_MESSAGE)
