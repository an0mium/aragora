"""Tests for scripts/generate_contract_drift_issue_plan.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.generate_contract_drift_issue_plan as issue_plan


def test_generates_issue_plan_outputs(monkeypatch, tmp_path: Path):
    backlog = {
        "generated_on": "2026-02-13",
        "total_items": 25,
        "counts_by_source": {
            "verify_python_sdk_drift": 10,
            "routes_missing_in_spec": 8,
            "sdk_missing_from_both": 7,
        },
        "counts_by_owner": {
            "@team-platform": 15,
            "@team-core": 6,
            "@team-integrations": 4,
        },
        "tickets": [
            {
                "ticket": "CD-001",
                "source": "verify_python_sdk_drift",
                "domain": "auth",
                "owner": "@team-platform",
                "open_items": 10,
                "target_weekly_burn": 1,
            },
            {
                "ticket": "CD-002",
                "source": "routes_missing_in_spec",
                "domain": "debates",
                "owner": "@team-core",
                "open_items": 8,
                "target_weekly_burn": 1,
            },
            {
                "ticket": "CD-003",
                "source": "sdk_missing_from_both",
                "domain": "chat",
                "owner": "@team-integrations",
                "open_items": 7,
                "target_weekly_burn": 1,
            },
        ],
    }

    backlog_path = tmp_path / "backlog.json"
    md_out = tmp_path / "issue_plan.md"
    json_out = tmp_path / "issue_plan.json"
    backlog_path.write_text(json.dumps(backlog) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_contract_drift_issue_plan.py",
            "--backlog-json",
            str(backlog_path),
            "--markdown-out",
            str(md_out),
            "--json-out",
            str(json_out),
            "--max-tickets",
            "3",
        ],
    )

    assert issue_plan.main() == 0
    assert md_out.exists()
    assert json_out.exists()

    md_content = md_out.read_text(encoding="utf-8")
    assert "Contract Drift Issue Plan" in md_content
    assert "CD-001" in md_content

    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["program_total_items"] == 25
    assert plan["waves"]
    first_ticket = plan["waves"][0]["tickets"][0]
    assert "Contract drift:" in first_ticket["issue_title"]
