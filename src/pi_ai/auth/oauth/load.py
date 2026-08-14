"""OAuth flow loaders.

Python port of `packages/ai/src/auth/oauth/load.ts`.

The TypeScript version loads each flow through a dynamic `import()` behind a
variable specifier so bundlers cannot follow it into Node-only code
(`node:http` callback servers, `node:crypto` PKCE) when building a browser
bundle. This port has no browser bundle target, so the flow modules are
imported normally at module load time; the `load_*` functions still exist so
callers (and :func:`pi_ai.auth.helpers.lazy_oauth`) have the same "construct
on first use" entry points as the TypeScript loaders.
"""

from __future__ import annotations

from ..types import OAuthAuth
from .anthropic import build_anthropic_oauth
from .github_copilot import build_github_copilot_oauth
from .kimi_coding import build_kimi_coding_oauth
from .openrouter import build_openrouter_oauth
from .radius import create_radius_oauth
from .xai import build_xai_oauth


async def load_anthropic_oauth() -> OAuthAuth:
    return build_anthropic_oauth()


async def load_github_copilot_oauth() -> OAuthAuth:
    return build_github_copilot_oauth()


async def load_openrouter_oauth() -> OAuthAuth:
    return build_openrouter_oauth()


async def load_kimi_coding_oauth() -> OAuthAuth:
    return build_kimi_coding_oauth()


async def load_xai_oauth() -> OAuthAuth:
    return build_xai_oauth()


async def load_radius_oauth(name: str, gateway: str) -> OAuthAuth:
    return create_radius_oauth(name, gateway)
