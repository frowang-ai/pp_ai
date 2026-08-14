"""pi's own User-Agent for provider requests.

Python port of `packages/ai/src/utils/pi-user-agent.ts`.

TypeScript loads `node:os` through `process.getBuiltinModule` so a browser or
Vite build does not break on a top-level Node import. Python has no such
constraint -- `platform` is in the standard library and always importable --
so the string is built directly.
"""

from __future__ import annotations

import platform


def get_pi_user_agent() -> str:
    """`pi (<system> <release>; <machine>)`, matching the TypeScript shape."""
    return f"pi ({platform.system().lower()} {platform.release()}; {platform.machine()})"


__all__ = ["get_pi_user_agent"]
