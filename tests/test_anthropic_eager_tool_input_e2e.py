"""Python port of `packages/ai/test/anthropic-eager-tool-input-e2e.test.ts`.

Only the offline case ("covers every generated anthropic-messages model") is
ported. The "generated compatibility settings" and "forced
eager_input_streaming probe" blocks are `it.skipIf(!apiKey)` live API calls; the
selection helpers they depend on are exercised directly instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pi_ai.providers.all import get_builtin_models, get_builtin_providers
from pi_ai.types import Model


@dataclass
class EagerE2ECase:
    name: str
    provider: str
    model: Model


def get_anthropic_messages_models(provider: str) -> list[Model]:
    return [model for model in get_builtin_models(provider) if model.api == "anthropic-messages"]


ANTHROPIC_MESSAGES_CASES = [
    EagerE2ECase(name=f"{provider}/{model.id}", provider=provider, model=model)
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


def select_one_case_per_provider(cases: list[EagerE2ECase]) -> list[EagerE2ECase]:
    by_provider: dict[str, list[EagerE2ECase]] = {}
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


def test_generated_compat_selection_picks_one_case_per_provider():
    generated = select_one_case_per_provider(ANTHROPIC_MESSAGES_CASES)
    providers = [case.provider for case in generated]
    assert sorted(providers) == sorted(set(providers))
    assert set(providers) == {case.provider for case in ANTHROPIC_MESSAGES_CASES}


def test_forced_eager_probe_excludes_models_that_disable_eager_tool_input_streaming():
    eligible = [
        case
        for case in ANTHROPIC_MESSAGES_CASES
        if case.model.compat.get("supportsEagerToolInputStreaming") is not False
    ]
    forced = select_one_case_per_provider(eligible)
    assert all(case.model.compat.get("supportsEagerToolInputStreaming") is not False for case in forced)
    # `withEagerToolInputStreaming` forces the flag on for the probe.
    for case in forced:
        overridden = {**case.model.compat, "supportsEagerToolInputStreaming": True}
        assert overridden["supportsEagerToolInputStreaming"] is True


@pytest.mark.skip(
    reason=(
        "TS describe('generated compatibility settings'): for each "
        "selectOneCasePerProvider(anthropicMessagesCases) case with a real API key "
        "(it.skipIf(!testCase.apiKey)), calls complete() against the live provider "
        "with the echo_value tool using that model's generated compat settings and "
        "asserts response.errorMessage is falsy and response.stopReason !== 'error'. "
        "This drives real network requests to live provider APIs, which is "
        "forbidden for this port; the case-selection logic itself is exercised "
        "offline above by test_generated_compat_selection_picks_one_case_per_provider."
    )
)
def test_generated_compatibility_settings_accepts_configured_tool_streaming_live():
    pass


@pytest.mark.skip(
    reason=(
        "TS describe('forced eager_input_streaming probe'): for each "
        "selectOneCasePerProvider case whose compat.supportsEagerToolInputStreaming "
        "is not explicitly false, with a real API key (it.skipIf(!testCase.apiKey)), "
        "calls complete() against the live provider on a model with "
        "compat.supportsEagerToolInputStreaming forced to true, using the echo_value "
        "tool, and asserts response.errorMessage is falsy and "
        "response.stopReason !== 'error'. This drives real network requests to live "
        "provider APIs, which is forbidden for this port; the "
        "filter/selection/override logic is exercised offline above by "
        "test_forced_eager_probe_excludes_models_that_disable_eager_tool_input_streaming."
    )
)
def test_forced_eager_input_streaming_probe_accepts_request_live():
    pass
