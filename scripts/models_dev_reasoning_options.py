"""models.dev reasoning-option conversion.

Python port of `packages/ai/scripts/models-dev-reasoning-options.ts`.
"""

from __future__ import annotations

from typing import Any

THINKING_LEVELS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")


def get_effort_thinking_level_map(options: list[dict[str, Any]] | None) -> dict[str, str | None] | None:
    """Convert models.dev verified effort values into pi's selectable thinking levels.

    Values without a pi equivalent (``default`` and JSON ``null``) are
    intentionally omitted.
    """
    effort_values: list[str | None] = []
    for option in options or []:
        if option.get("type") == "effort":
            effort_values.extend(option.get("values") or [])
    if not effort_values:
        return None

    supported = set(effort_values)
    if not any(level in supported for level in THINKING_LEVELS) and "none" not in supported:
        return None

    level_map: dict[str, str | None] = {"off": "none" if "none" in supported else None}
    for level in THINKING_LEVELS:
        level_map[level] = level if level in supported else None
    return level_map
