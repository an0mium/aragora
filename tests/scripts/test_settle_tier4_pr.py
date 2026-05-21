"""Tests for ``scripts/settle_tier4_pr.py`` pure guard helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


settler = _load_module("settle_tier4_pr.py")


def test_missing_operator_comment_blocks_settlement() -> None:
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head="57c740022e3c432718462efa12ca79f1df4f674d",
        pr_view={
            "headRefOid": "57c740022e3c432718462efa12ca79f1df4f674d",
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "comments": [{"body": "looks good"}],
            "reviews": [],
        },
        merge_packet={"admin_squash_allowed": False, "not_ready": ["human_risk_settlement"]},
    )

    assert result["ok"] is False
    assert "missing repo-visible Tier 4 operator settlement comment" in result["blockers"]


def test_exact_head_operator_comment_allows_check_result() -> None:
    head = "57c740022e3c432718462efa12ca79f1df4f674d"
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head=head,
        pr_view={
            "headRefOid": head,
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "comments": [
                {
                    "body": (
                        "Tier-4 Human Settlement Authorization\n"
                        f"Authorized Head SHA: {head}\n"
                        "Authorized Action: admin_squash_merge on PR #7423\n"
                    )
                }
            ],
            "reviews": [],
        },
        merge_packet={"admin_squash_allowed": False, "not_ready": ["human_risk_settlement"]},
    )

    assert result["ok"] is True
    assert result["blockers"] == []


def test_head_mismatch_blocks_before_authorization() -> None:
    result = settler.evaluate_tier4_gate(
        pr=7423,
        expected_head="expected",
        pr_view={
            "headRefOid": "actual",
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "comments": [],
            "reviews": [],
        },
        merge_packet={},
    )

    assert result["ok"] is False
    assert "head mismatch: expected expected, got actual" in result["blockers"]
