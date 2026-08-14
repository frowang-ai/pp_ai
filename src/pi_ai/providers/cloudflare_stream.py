"""Cloudflare endpoint templating for Workers AI and AI Gateway.

Python port of `packages/ai/src/providers/cloudflare-stream.ts`.

Cloudflare base URLs in the generated catalog carry `{CLOUDFLARE_ACCOUNT_ID}`
and `{CLOUDFLARE_GATEWAY_ID}` placeholders (see
`pi_ai.api.cloudflare`), because the endpoint is account-specific and the
account id only becomes known once auth resolves. :func:`cloudflare_streams`
wraps an API module so those placeholders materialize from the resolved
provider env just before dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..types import Context, Model, SimpleStreamOptions, StreamOptions
from ..utils.event_stream import AssistantMessageEventStream

CLOUDFLARE_ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_GATEWAY_ID = "CLOUDFLARE_GATEWAY_ID"


def resolve_cloudflare_model(model: Model, env: dict[str, str] | None) -> Model:
    """Substitute account/gateway placeholders in ``model.base_url``."""
    if not env:
        return model
    base_url = model.base_url
    for name in (CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_GATEWAY_ID):
        base_url = base_url.replace(f"{{{name}}}", env.get(name, f"{{{name}}}"))
    return model if base_url == model.base_url else replace(model, base_url=base_url)


@dataclass
class CloudflareStreams:
    """An API module wrapper that resolves Cloudflare endpoint placeholders."""

    streams: Any

    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream:
        env = options.env if options is not None else None
        return self.streams.stream(resolve_cloudflare_model(model, env), context, options, **kwargs)

    def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream:
        env = options.env if options is not None else None
        return self.streams.stream_simple(resolve_cloudflare_model(model, env), context, options, **kwargs)


def cloudflare_streams(streams: Any) -> CloudflareStreams:
    """Wrap an API module so Cloudflare endpoint placeholders resolve per request."""
    return CloudflareStreams(streams=streams)
