"""Provider environment variable resolution.

Python port of `packages/ai/src/utils/provider-env.ts`. Only `getProviderEnvValue`
is ported: the TypeScript file also has a Bun-specific fallback that reads
`/proc/self/environ` to work around https://github.com/oven-sh/bun/issues/27802
(a Bun compiled-binary bug where `process.env` can appear empty inside Linux
sandboxes). That workaround has no meaning for a Python process and is
intentionally dropped.
"""

from __future__ import annotations

import os
from collections.abc import Mapping


def get_provider_env_value(name: str, env: Mapping[str, str] | None = None) -> str | None:
    """Resolve a provider env value from scoped overrides, then `os.environ`."""
    if env is not None:
        value = env.get(name)
        if value:
            return value
    return os.environ.get(name) or None
