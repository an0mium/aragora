"""Tests for ``scripts/build_next_prompt.py``."""

from __future__ import annotations

import importlib.util
import json
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


prompt_builder = _load_module("build_next_prompt.py")


def test_prompt_starts_with_mailbox_and_owner_verification(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "P106-merge-gate-settlement",
                    "owner_session": "droid-P106-merge-gate-settlement-20260521T2118Z",
                    "status": "working",
                    "pr_number": 7423,
                    "branch": "claude/recover-merge-gate-reconciliation",
                    "next_action": "settle exact-head governance gate",
                }
            ]
        ),
        encoding="utf-8",
    )

    prompt = prompt_builder.build_prompt(
        registry_path=registry,
        lane_id="P106-merge-gate-settlement",
        pr=7423,
    )

    assert prompt.startswith("Start from live repo truth")
    assert "Before lane work, check your Aragora operator-steering mailbox" in prompt
    assert (
        "python3 scripts/read_operator_steering.py --lane-id P106-merge-gate-settlement" in prompt
    )
    assert (
        "Continue only if you are owner_session droid-P106-merge-gate-settlement-20260521T2118Z"
        in prompt
    )
    assert (
        "If the prompt above accomplishes no incremental progress make the next prompt one that does"
        in prompt
    )


def test_prompt_for_non_owner_read_only_when_no_lane_match(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")

    prompt = prompt_builder.build_prompt(registry_path=registry, pr=7407)

    assert "If you cannot map yourself to a lane, run read-only only" in prompt
    assert "Do not paste raw transcripts" in prompt
