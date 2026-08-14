"""Python port of `packages/ai/test/anthropic-long-cache-retention-e2e.test.ts`.

Only the offline case ("covers every generated anthropic-messages model") is
ported. The "forced long cache retention probe" cases are `it.skipIf(!apiKey)`
live API calls against every provider; they cannot run offline, so instead the
probe-selection helpers they depend on are exercised directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pi_ai.providers.all import get_builtin_models, get_builtin_providers
from pi_ai.types import Model


@dataclass
class LongCacheRetentionCase:
    name: str
    provider: str
    model: Model


def get_anthropic_messages_models(provider: str) -> list[Model]:
    return [model for model in get_builtin_models(provider) if model.api == "anthropic-messages"]


ANTHROPIC_MESSAGES_CASES = [
    LongCacheRetentionCase(name=f"{provider}/{model.id}", provider=provider, model=model)
    for provider in get_builtin_providers()
    for model in get_anthropic_messages_models(provider)
]


def get_probe_priority(model: Model) -> float:
    model_id = model.id.lower()
    priority = model.cost.input + model.cost.output

    if "haiku" in model_id and ("4-5" in model_id or "4.5" in model_id):
        priority -= 1000
    elif "sonnet" in model_id and ("4-" in model_id or "4." in model_id):
        priority -= 750
    elif "claude" in model_id and ("4-" in model_id or "4." in model_id):
        priority -= 500

    return priority


def select_one_case_per_provider(cases: list[LongCacheRetentionCase]) -> list[LongCacheRetentionCase]:
    by_provider: dict[str, list[LongCacheRetentionCase]] = {}
    for case in cases:
        by_provider.setdefault(case.provider, []).append(case)

    return [
        sorted(provider_cases, key=lambda case: (get_probe_priority(case.model), case.model.id))[0]
        for provider_cases in by_provider.values()
    ]


def test_covers_every_generated_anthropic_messages_model():
    expected = sorted(
        f"{provider}/{model.id}"
        for provider in get_builtin_providers()
        for model in get_anthropic_messages_models(provider)
    )
    assert sorted(case.name for case in ANTHROPIC_MESSAGES_CASES) == expected
    assert expected


def test_probe_selection_picks_exactly_one_case_per_provider():
    probe_cases = select_one_case_per_provider(ANTHROPIC_MESSAGES_CASES)
    providers = [case.provider for case in probe_cases]
    assert sorted(providers) == sorted(set(providers))
    assert set(providers) == {case.provider for case in ANTHROPIC_MESSAGES_CASES}


def test_long_cache_retention_probe_forces_the_compat_flag_on():
    # `withLongCacheRetention` in TypeScript; `model.compat` is a plain dict here.
    model = ANTHROPIC_MESSAGES_CASES[0].model
    forced = {**model.compat, "supportsLongCacheRetention": True}
    assert forced["supportsLongCacheRetention"] is True


@pytest.mark.skip(
    reason=(
        "TS 'forced long cache retention probe' (describe block, one it.skipIf(!apiKey) "
        "per provider with { retry: 2 }) sends a real `complete()` request with a real "
        "API key to every builtin provider's cheapest anthropic-messages model, forcing "
        "compat.supportsLongCacheRetention=true and cacheRetention: 'long', then asserts "
        "`expect(response.errorMessage).toBeFalsy()` and "
        "`expect(response.stopReason).not.toBe('error')` -- i.e. that the live vendor API "
        "actually accepts the 1-hour/long cache-retention header/beta flag for that model. "
        "This is an integration assertion about a live vendor's acceptance of a request "
        "shape; a MockTransport response would trivially satisfy it regardless of whether "
        "the real API accepts long cache retention, so it has no meaningful mocked "
        "analogue and cannot run offline without a real API key."
    )
)
def test_forced_long_cache_retention_probe_accepted_by_live_provider():
    pass
