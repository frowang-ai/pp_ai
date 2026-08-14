"""Amazon Bedrock Converse streaming API — not ported.

Placeholder for `packages/ai/src/api/bedrock-converse-stream.ts`, which is a
documented omission of this port: it depends on the AWS SDK's SigV4 signer and
credential-provider chain, which has no equivalent here yet. The two thin
wrappers around it, `packages/ai/src/bedrock-provider.ts` and
`packages/ai/src/api/bedrock-converse-stream.lazy.ts`, exist only to keep the
Node-only AWS SDK out of TypeScript's browser and Bun bundles; a Python import
never pulls a bundler along, so they have nothing to port either.

The module still exists so `amazon-bedrock` models stay discoverable through
the generated catalog (`pi_ai.providers.amazon_bedrock`); any attempt to
actually stream raises :class:`NotImplementedError` instead of failing with an
attribute error deep inside the registry.
"""

from __future__ import annotations

from typing import Any, NoReturn

from ..types import Context, Model, SimpleStreamOptions, StreamOptions

_MESSAGE = (
    "The bedrock-converse-stream API is not ported to Python. "
    "amazon-bedrock models are listed for discovery only; use another provider to run them."
)


def stream(model: Model, context: Context, options: StreamOptions | None = None, **kwargs: Any) -> NoReturn:
    raise NotImplementedError(_MESSAGE)


def stream_simple(
    model: Model, context: Context, options: SimpleStreamOptions | None = None, **kwargs: Any
) -> NoReturn:
    raise NotImplementedError(_MESSAGE)
