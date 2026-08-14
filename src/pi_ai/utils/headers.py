"""HTTP header conversion helpers.

Python port of `packages/ai/src/utils/headers.ts`.
"""

from __future__ import annotations

from collections.abc import Mapping


def headers_to_record(headers: Mapping[str, str]) -> dict[str, str]:
    """Convert any header mapping (e.g. ``httpx.Headers``) to a plain dict."""
    return dict(headers.items())


def provider_headers_to_record(headers: Mapping[str, str | None] | None) -> dict[str, str] | None:
    """Drop ``None`` values from a provider header mapping.

    Returns ``None`` when ``headers`` is ``None``/empty or every value is
    ``None``, matching the TypeScript ``undefined`` return.
    """
    if not headers:
        return None
    result = {key: value for key, value in headers.items() if value is not None}
    return result if result else None


def merge_provider_headers(
    base: Mapping[str, str | None] | None,
    override: Mapping[str, str | None] | None,
) -> dict[str, str | None]:
    """Merge two provider header maps, case-insensitively, override winning.

    Port of `mergeHeaders` in `packages/ai/src/models.ts`: an override replaces
    a differently-cased base entry rather than adding a second spelling, and a
    ``None`` value is preserved (it means "suppress this header downstream").
    """
    merged: dict[str, str | None] = dict(base or {})
    for name, value in (override or {}).items():
        lowered = name.lower()
        for existing in [key for key in merged if key.lower() == lowered]:
            del merged[existing]
        merged[name] = value
    return merged


def apply_header_overrides(headers: dict[str, str], overrides: Mapping[str, str | None] | None) -> dict[str, str]:
    """Apply caller-supplied header overrides case-insensitively, in place.

    A ``None`` value removes the header. HTTP header names are
    case-insensitive, but adapters build their defaults with lowercase keys
    (``authorization``) while callers and provider auth commonly use the
    canonical spelling (``Authorization``). TypeScript never hits this because
    it lets the vendor SDK set ``Authorization`` and then spreads the caller's
    headers over the same key; a plain case-sensitive ``dict`` update here
    would instead send *both* spellings, which httpx joins into a single
    comma-separated header value.
    """
    if not overrides:
        return headers
    for key, value in overrides.items():
        lowered = key.lower()
        for existing in [k for k in headers if k.lower() == lowered]:
            del headers[existing]
        if value is not None:
            headers[key] = value
    return headers
