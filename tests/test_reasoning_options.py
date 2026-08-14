"""Python port of `packages/ai/test/reasoning-options.test.ts`."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from models_dev_reasoning_options import get_effort_thinking_level_map  # noqa: E402


def test_exposes_only_verified_effort_values_and_none() -> None:
    assert get_effort_thinking_level_map(
        [{"type": "toggle"}, {"type": "effort", "values": ["none", "low", "high", "max"]}]
    ) == {
        "off": "none",
        "minimal": None,
        "low": "low",
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": "max",
    }


def test_does_not_infer_thinking_off_from_an_effort_list() -> None:
    assert get_effort_thinking_level_map([{"type": "effort", "values": ["low", "high", "max"]}]) == {
        "off": None,
        "minimal": None,
        "low": "low",
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": "max",
    }


def test_leaves_toggle_and_budget_controls_for_adapter_specific_implementations() -> None:
    assert get_effort_thinking_level_map([{"type": "toggle"}]) is None
    assert get_effort_thinking_level_map([{"type": "budget_tokens", "min": 1024, "max": 32000}]) is None
    assert get_effort_thinking_level_map([{"type": "effort", "values": [None, "default"]}]) is None
