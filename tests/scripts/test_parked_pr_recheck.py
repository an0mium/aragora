"""Unit tests for scripts/parked_pr_recheck.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def test_recheck_records_marks_changed_heads() -> None:
    import parked_pr_recheck as mod

    records = [
        mod.ParkedPr(
            pr=8908,
            head_sha="old8908",
            parked_at="2026-07-08",
            blocker_class="current_head_p2",
            source="test",
        ),
        mod.ParkedPr(
            pr=8945,
            head_sha="same8945",
            parked_at="2026-07-08",
            blocker_class="tier4",
            source="test",
        ),
    ]

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        pr = int(cmd[3])
        head = {8908: "new8908", 8945: "same8945"}[pr]
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"headRefOid": head}), "")

    results, ok = mod.recheck_records(records, repo="synaptent/aragora", run=fake_run)

    assert ok is True
    assert results[0].changed is True
    assert results[0].recommendation == "requeue candidate (head changed)"
    assert results[1].changed is False
    assert results[1].recommendation == "skip parked head"


def test_render_table_includes_requeue_candidate() -> None:
    import parked_pr_recheck as mod

    table = mod.render_table(
        [
            mod.RecheckResult(
                pr=8908,
                parked_head="old8908abcdef",
                live_head="new8908abcdef",
                changed=True,
                recommendation="requeue candidate (head changed)",
                blocker_class="current_head_p2",
                source="test",
            )
        ]
    )

    assert "#8908" in table
    assert "requeue candidate (head changed)" in table


def test_load_ledger_accepts_records_object(tmp_path: Path) -> None:
    import parked_pr_recheck as mod

    ledger = tmp_path / "parked.json"
    ledger.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "pr": 8908,
                        "head_sha": "abc",
                        "parked_at": "2026-07-08",
                        "blocker_class": "current_head_p2",
                        "source": "test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = mod.load_ledger(ledger)

    assert records == [
        mod.ParkedPr(
            pr=8908,
            head_sha="abc",
            parked_at="2026-07-08",
            blocker_class="current_head_p2",
            source="test",
        )
    ]


def test_recheck_records_preserves_state_on_lookup_failure() -> None:
    import parked_pr_recheck as mod

    records = [
        mod.ParkedPr(
            pr=8908,
            head_sha="old8908",
            parked_at="2026-07-08",
            blocker_class="current_head_p2",
            source="test",
        )
    ]

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, "", "not found")

    results, ok = mod.recheck_records(records, repo="synaptent/aragora", run=fake_run)

    assert ok is False
    assert results[0].changed is None
    assert results[0].recommendation == "lookup failed; preserve parked state"
    assert results[0].error == "not found"
