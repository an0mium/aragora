"""Tests for scripts/reclassify_dogfood_report.py report-shape guards."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import reclassify_dogfood_report  # noqa: E402


def test_reclassify_report_rejects_empty_runs() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        reclassify_dogfood_report.reclassify_report({"runs": []})


def test_reclassify_report_rejects_non_object_run_before_mutation() -> None:
    report = {"runs": ["not-a-run"]}

    with pytest.raises(ValueError, match=r"runs\[0\] must be an object"):
        reclassify_dogfood_report.reclassify_report(report)

    assert report == {"runs": ["not-a-run"]}


def test_reclassify_report_counts_warning_only_and_blocker_runs() -> None:
    report = {
        "runs": [
            {
                "stderr_excerpt": (
                    "ResourceWarning: unclosed transport\n"
                    "ResourceWarning: Enable tracemalloc to get the object allocation traceback"
                )
            },
            {"stderr_excerpt": "Debate timed out after 120s"},
        ]
    }

    updated = reclassify_dogfood_report.reclassify_report(report)

    assert updated["runtime_blockers_zero"] is False
    assert updated["warning_only_runs"] == 1
    assert updated["blocker_runs"] == 1
    assert updated["runs"][0]["warning_only"] is True
    assert "debate_timeout" in updated["runs"][1]["runtime_blockers"]
