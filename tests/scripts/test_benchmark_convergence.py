"""Tests for scripts/benchmark_convergence.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import benchmark_convergence  # noqa: E402


def test_run_single_debate_rejects_non_positive_round_count() -> None:
    with pytest.raises(ValueError, match="max_rounds must be a positive integer"):
        benchmark_convergence.run_single_debate(
            max_rounds=0,
            agents=["analyst"],
        )


def test_run_single_debate_rejects_empty_agent_list() -> None:
    with pytest.raises(ValueError, match="agents must include at least one persona"):
        benchmark_convergence.run_single_debate(
            max_rounds=2,
            agents=[],
        )


def test_run_single_debate_rejects_unknown_agent_names() -> None:
    with pytest.raises(
        ValueError,
        match="unknown convergence benchmark agent\\(s\\): unknown-agent",
    ):
        benchmark_convergence.run_single_debate(
            max_rounds=2,
            agents=["analyst", "unknown-agent"],
        )


def test_run_single_debate_records_valid_rounds() -> None:
    result = benchmark_convergence.run_single_debate(
        max_rounds=2,
        agents=["analyst", "critic"],
        convergence_threshold=0.99,
    )

    assert result.max_rounds == 2
    assert result.rounds_executed >= 1
    assert result.rounds_saved >= 0
    assert result.agents == ["analyst", "critic"]
    assert result.round_metrics
