"""Python port of `packages/ai/test/generate-models-strict.test.ts`.

TypeScript runs the generator in an isolated copy of the package with
`globalThis.fetch` preloaded to serve a fake `models.dev/api.json`. The Python
generator has no global fetch to override, but it does accept
`--models-dev-file` (a port addition for offline regeneration), which serves the
same purpose: the fixture catalog is read from disk and no network call is made.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

GENERATED_PATHS = [
    "src/pi_ai/providers/data/qwen-token-plan-individual.json",
    "src/pi_ai/providers/data/.manifest.json",
]

MODEL_IDS = [
    "deepseek-v4-flash-0731",
    "deepseek-v4-pro",
    "glm-5.2",
    "qwen3.6-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.8-max",
    "qwen3.8-max-preview",
]


def test_fails_before_mutating_generated_data_when_an_individual_model_loses_tool_support(tmp_path: Path):
    isolated_package_root = tmp_path / "package"
    isolated_package_root.mkdir()
    for entry in ("scripts", "src"):
        shutil.copytree(PACKAGE_ROOT / entry, isolated_package_root / entry)

    catalog = {
        "alibaba-token-plan": {
            "models": {
                model_id: {"id": model_id, "name": model_id, "tool_call": model_id != "deepseek-v4-flash-0731"}
                for model_id in MODEL_IDS
            }
        }
    }
    catalog_path = tmp_path / "models-dev.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    source_before = [(PACKAGE_ROOT / path).read_text(encoding="utf-8") for path in GENERATED_PATHS]
    isolated_before = [(isolated_package_root / path).read_text(encoding="utf-8") for path in GENERATED_PATHS]

    result = subprocess.run(
        [
            sys.executable,
            str(isolated_package_root / "scripts" / "generate_models.py"),
            "--strict",
            "--models-dev-file",
            str(catalog_path),
        ],
        cwd=isolated_package_root,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 1
    assert "qwen-token-plan-individual model IDs do not match (missing: deepseek-v4-flash-0731)" in (
        f"{result.stdout}\n{result.stderr}"
    )
    assert [(isolated_package_root / path).read_text(encoding="utf-8") for path in GENERATED_PATHS] == isolated_before
    assert [(PACKAGE_ROOT / path).read_text(encoding="utf-8") for path in GENERATED_PATHS] == source_before
