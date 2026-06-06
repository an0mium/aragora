"""Tests for scripts/benchmark_debate.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import benchmark_debate  # noqa: E402


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--agents", "0"),
        ("--rounds", "0"),
        ("--concurrent", "0"),
        ("--large-panel-agents", "0"),
        ("--large-panel-rounds", "0"),
        ("--agents", "-1"),
    ],
)
def test_parse_args_rejects_non_positive_benchmark_counts(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        benchmark_debate.parse_args([flag, value])


def test_parse_args_accepts_positive_benchmark_counts() -> None:
    args = benchmark_debate.parse_args(
        [
            "--agents",
            "2",
            "--rounds",
            "1",
            "--concurrent",
            "3",
            "--large-panel-agents",
            "4",
            "--large-panel-rounds",
            "2",
        ]
    )

    assert args.agents == 2
    assert args.rounds == 1
    assert args.concurrent == 3
    assert args.large_panel_agents == 4
    assert args.large_panel_rounds == 2
