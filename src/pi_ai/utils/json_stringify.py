"""`JSON.stringify`-compatible serialization.

Python's `json.dumps` defaults differ from JavaScript's `JSON.stringify` in two
ways that are visible on the wire: it inserts a space after `,` and `:`, and it
escapes non-ASCII characters as `\\uXXXX`. Provider payloads that carry a
JSON-encoded string (tool-call `arguments`, reasoning-item signatures) must match
the TypeScript original byte for byte, so they go through :func:`json_stringify`.
"""

from __future__ import annotations

import json
from typing import Any


def json_stringify(value: Any) -> str:
    """Serialize ``value`` exactly as `JSON.stringify(value)` would."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
