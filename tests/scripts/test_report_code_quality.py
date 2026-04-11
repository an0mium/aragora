"""Tests for scripts/report_code_quality.py."""

from __future__ import annotations

import sys
from pathlib import Path

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import report_code_quality  # noqa: E402


def test_default_file_loc_ratchet_is_tighter_than_legacy_ceiling() -> None:
    assert report_code_quality.RATCHET["max_file_loc"] <= 5400


def test_check_ratchet_flags_files_above_file_loc_ceiling() -> None:
    violations = report_code_quality.check_ratchet(
        {"except_exception": 0, "type_ignore": 0, "noqa": 0},
        [
            {
                "top5_largest": [
                    {
                        "file": "aragora/nomic/dev_coordination.py",
                        "loc": report_code_quality.RATCHET["max_file_loc"] + 1,
                    }
                ]
            }
        ],
    )

    assert violations == [
        "aragora/nomic/dev_coordination.py: "
        f"{report_code_quality.RATCHET['max_file_loc'] + 1} LOC > "
        f"{report_code_quality.RATCHET['max_file_loc']}"
    ]


def test_check_ratchet_allows_files_at_file_loc_ceiling() -> None:
    violations = report_code_quality.check_ratchet(
        {"except_exception": 0, "type_ignore": 0, "noqa": 0},
        [
            {
                "top5_largest": [
                    {
                        "file": "aragora/swarm/boss_loop.py",
                        "loc": report_code_quality.RATCHET["max_file_loc"],
                    }
                ]
            }
        ],
    )

    assert violations == []
