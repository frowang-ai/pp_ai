"""OpenRouter image generation.

Python port of `packages/ai/src/api/openrouter-images.ts`.

Unlike the streaming chat providers, this is a single non-streaming JSON POST
to `{baseUrl}/chat/completions` with `modalities: ["image", "text"]`
(OpenRouter's image-generation extension to the OpenAI Chat Completions API)
-- no SSE, no vendor SDK. `generate_images` returns one `AssistantImages`,
not a stream, matching the TypeScript `ImagesFunction` contract.

The TypeScript source wraps its request in `retryProviderRequest`. No
provider in this Python port implements client-side retries yet (see
`utils/provider_retry.py`, unused elsewhere in the port), so this makes a
single request with no retry, consistent with every other ported provider.

`ImagesOptions` used to be declared here, because this was the only image
module in the port. It now lives in :mod:`pi_ai.types` next to
:class:`~pi_ai.types.StreamOptions` — where TypeScript declares it too — so the
image registry layer can share it, and is re-exported here for callers that
already import it from this module.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from ..types import (
    AssistantImages,
    Cost,
    ImageContent,
    ImagesContext,
    ImagesModel,
    ImagesOptions,
    ProviderResponse,
    TextContent,
    Usage,
    now_ms,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.headers import headers_to_record, provider_headers_to_record
from ..utils.http import ProviderHttpError, build_timeout
from ..utils.sanitize_unicode import sanitize_surrogates

_DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)

__all__ = ["ImagesOptions", "build_params", "generate_images", "parse_usage"]


def build_params(model: ImagesModel, context: ImagesContext) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for item in context.input:
        if isinstance(item, TextContent):
            content.append({"type": "text", "text": sanitize_surrogates(item.text)})
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"}})

    return {
        "model": model.id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "modalities": ["image", "text"] if "text" in model.output else ["image"],
    }


def parse_usage(raw_usage: dict[str, Any], model: ImagesModel) -> Usage:
    prompt_tokens = raw_usage.get("prompt_tokens") or 0
    details = raw_usage.get("prompt_tokens_details") or {}
    reported_cached_tokens = details.get("cached_tokens") or 0
    cache_write_tokens = details.get("cache_write_tokens") or 0
    cache_read_tokens = (
        max(0, reported_cached_tokens - cache_write_tokens) if cache_write_tokens > 0 else reported_cached_tokens
    )
    input_tokens = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
    output_tokens = raw_usage.get("completion_tokens") or 0

    cost_input = (model.cost.input / 1_000_000) * input_tokens
    cost_output = (model.cost.output / 1_000_000) * output_tokens
    cost_cache_read = (model.cost.cache_read / 1_000_000) * cache_read_tokens
    cost_cache_write = (model.cost.cache_write / 1_000_000) * cache_write_tokens

    return Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read_tokens,
        cache_write=cache_write_tokens,
        total_tokens=input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
        cost=Cost(
            input=cost_input,
            output=cost_output,
            cache_read=cost_cache_read,
            cache_write=cost_cache_write,
            total=cost_input + cost_output + cost_cache_read + cost_cache_write,
        ),
    )


async def generate_images(
    model: ImagesModel,
    context: ImagesContext,
    options: ImagesOptions | None = None,
    client: httpx.AsyncClient | None = None,
) -> AssistantImages:
    """Generate images from an OpenRouter chat-completions-with-images model.

    Failures are reported on the returned `AssistantImages` (`stop_reason`
    `"error"`/`"aborted"` plus `error_message`), never raised, matching the
    TypeScript `generateImages` contract.
    """
    options = options or ImagesOptions()
    output = AssistantImages(
        api=model.api,
        provider=model.provider,
        model=model.id,
        output=[],
        stop_reason="stop",
        timestamp=now_ms(),
    )

    try:
        # TypeScript hands `options.signal` to the openai SDK, which rejects
        # immediately when it is already aborted. This module posts with httpx,
        # so the check is explicit.
        if options.signal is not None and options.signal.aborted:
            raise RuntimeError("Request was aborted")

        api_key = options.api_key
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")

        params: Any = build_params(model, context)
        if options.on_payload is not None:
            replacement = options.on_payload(params, model)
            if hasattr(replacement, "__await__"):
                replacement = await replacement
            if replacement is not None:
                params = replacement

        headers: dict[str, str] = {"authorization": f"Bearer {api_key}", "content-type": "application/json"}
        headers.update(provider_headers_to_record({**model.headers, **(options.headers or {})}) or {})

        url = f"{model.base_url.rstrip('/')}/chat/completions"
        owns_client = client is None
        http_client = client or httpx.AsyncClient(timeout=build_timeout(options.timeout_ms))
        try:
            response = await http_client.post(
                url, headers=headers, json=params, timeout=build_timeout(options.timeout_ms)
            )
            if response.status_code >= 400:
                raise ProviderHttpError(response.status_code, response.text, dict(response.headers))

            if options.on_response is not None:
                result = options.on_response(
                    ProviderResponse(status=response.status_code, headers=headers_to_record(response.headers)),
                    model,
                )
                if hasattr(result, "__await__"):
                    await result

            data = response.json()
        finally:
            if owns_client:
                await http_client.aclose()

        output.response_id = data.get("id")
        raw_usage = data.get("usage")
        if raw_usage:
            output.usage = parse_usage(raw_usage, model)

        choices = data.get("choices") or []
        choice = choices[0] if choices else None
        if choice:
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content:
                output.output.append(TextContent(text=content))

            for image in message.get("images") or []:
                image_url_field = image.get("image_url")
                image_url = image_url_field if isinstance(image_url_field, str) else (image_url_field or {}).get("url")
                if not image_url or not image_url.startswith("data:"):
                    continue
                match = _DATA_URL_RE.match(image_url)
                if not match:
                    continue
                output.output.append(ImageContent(mime_type=match.group(1), data=match.group(2)))

        return output
    except BaseException as error:
        output.stop_reason = "aborted" if (options.signal is not None and options.signal.aborted) else "error"
        output.error_message = format_provider_error(normalize_provider_error(error))
        return output
