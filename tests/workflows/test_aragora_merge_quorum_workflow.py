from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/aragora-merge-quorum.yml"


def _workflow() -> dict[str, Any]:
    data = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(data, dict)
    return data


def _merge_quorum_steps() -> list[dict[str, Any]]:
    workflow = _workflow()
    steps = workflow["jobs"]["merge-quorum"]["steps"]
    assert isinstance(steps, list)
    return [step for step in steps if isinstance(step, dict)]


def test_merge_quorum_uses_current_default_branch_tooling() -> None:
    steps = _merge_quorum_steps()
    checkout_step = next(
        step for step in steps if str(step.get("uses", "")).startswith("actions/checkout@")
    )

    assert (
        checkout_step.get("with", {}).get("ref") == "${{ github.event.repository.default_branch }}"
    )


def test_merge_quorum_enables_advisory_dissent_settlement_gate() -> None:
    steps = _merge_quorum_steps()
    evaluate_step = next(step for step in steps if step.get("name") == "Evaluate merge quorum")
    env = evaluate_step.get("env", {})

    assert env["ARAGORA_ENABLE_SEVERITY_GATED_DISSENT"] == "1"
    assert env["ARAGORA_ENABLE_TIERED_MERGE_GATE"] == "1"
    assert env["ARAGORA_ENABLE_ADVISORY_DISSENT_SETTLE"] == "1"
