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


def test_merge_quorum_uses_current_default_branch_tooling() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["merge-quorum"]["steps"]
    checkout_step = next(
        step
        for step in steps
        if isinstance(step, dict) and str(step.get("uses", "")).startswith("actions/checkout@")
    )

    assert (
        checkout_step.get("with", {}).get("ref") == "${{ github.event.repository.default_branch }}"
    )
