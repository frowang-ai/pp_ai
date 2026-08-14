"""OAuth flow implementations.

Python port of `packages/ai/src/auth/oauth/`.
"""

from __future__ import annotations

from .device_code import (
    REAL_CLOCK,
    DeviceCodeClock,
    DeviceCodeError,
    DeviceCodePollResult,
    poll_oauth_device_code_flow,
)
from .load import (
    load_anthropic_oauth,
    load_github_copilot_oauth,
    load_kimi_coding_oauth,
    load_openrouter_oauth,
    load_radius_oauth,
    load_xai_oauth,
)
from .oauth_page import CallbackResult, OAuthCallbackServer, oauth_error_html, oauth_success_html
from .pkce import Pkce, generate_pkce

__all__ = [
    "REAL_CLOCK",
    "CallbackResult",
    "DeviceCodeClock",
    "DeviceCodeError",
    "DeviceCodePollResult",
    "OAuthCallbackServer",
    "Pkce",
    "generate_pkce",
    "load_anthropic_oauth",
    "load_github_copilot_oauth",
    "load_kimi_coding_oauth",
    "load_openrouter_oauth",
    "load_radius_oauth",
    "load_xai_oauth",
    "oauth_error_html",
    "oauth_success_html",
    "poll_oauth_device_code_flow",
]
