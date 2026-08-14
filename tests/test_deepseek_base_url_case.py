"""Port of upstream's DeepSeek base-URL case fix (`b647d1879`).

`detectCompat` branches on the host to decide `max_tokens` vs
`max_completion_tokens` and several other quirks. Matching the host
case-sensitively meant a provider configured with a mixed-case URL silently
missed every DeepSeek branch and got the wrong request shape.
"""

from __future__ import annotations

import pytest
from pi_ai.api.openai_completions import detect_compat
from pi_ai.types import Model


def _model(base_url: str, provider: str = "custom") -> Model:
    return Model(
        id="m",
        name="m",
        provider=provider,
        api="openai-completions",
        base_url=base_url,
        reasoning=False,
        input=["text"],
        cost=None,
    )


@pytest.mark.parametrize(
    "base_url",
    ["https://API.DeepSeek.COM/v1", "https://Api.Deepseek.Com/v1", "HTTPS://API.DEEPSEEK.COM/V1"],
)
def test_mixed_case_deepseek_hosts_resolve_like_the_lowercase_one(base_url):
    assert detect_compat(_model(base_url)) == detect_compat(_model("https://api.deepseek.com/v1"))


def test_the_provider_id_still_matches_on_its_own():
    """A provider named `deepseek` is DeepSeek regardless of its URL."""
    assert detect_compat(_model("https://proxy.internal/v1", provider="deepseek")) == detect_compat(
        _model("https://api.deepseek.com/v1")
    )


def test_an_unrelated_host_is_not_treated_as_deepseek():
    assert detect_compat(_model("https://api.openai.com/v1")) != detect_compat(_model("https://api.deepseek.com/v1"))
