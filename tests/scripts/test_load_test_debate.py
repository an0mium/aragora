"""Tests for scripts/load_test_debate.py."""

from __future__ import annotations

import scripts.load_test_debate as load_test_debate


def test_dispatch_concurrency_ratio_parallel_dispatch_is_near_one() -> None:
    ratio = load_test_debate._dispatch_concurrency_ratio(
        dispatch_duration_s=0.041,
        response_delays_s=[0.02, 0.03, 0.04],
    )

    assert 0.97 < ratio <= 1.0


def test_dispatch_concurrency_ratio_sequential_dispatch_is_penalized() -> None:
    ratio = load_test_debate._dispatch_concurrency_ratio(
        dispatch_duration_s=0.09,
        response_delays_s=[0.02, 0.03, 0.04],
    )

    assert 0.44 < ratio < 0.45


def test_dispatch_concurrency_ratio_rejects_invalid_inputs() -> None:
    assert load_test_debate._dispatch_concurrency_ratio(0.0, [0.01]) == 0.0
    assert load_test_debate._dispatch_concurrency_ratio(0.1, []) == 0.0
    assert load_test_debate._dispatch_concurrency_ratio(0.1, [0.0, -1.0]) == 0.0


def test_validate_against_slos_counts_parallel_dispatch_as_passing() -> None:
    metrics = load_test_debate.DebateLoadMetrics(
        total_debates=3,
        completed=3,
        failed=0,
        debate_latencies_ms=[1000.0, 1200.0, 1300.0],
        first_token_latencies_ms=[100.0, 120.0, 130.0],
        consensus_latencies_ms=[50.0, 55.0, 60.0],
        dispatch_concurrency_samples=[0.95, 0.98, 1.0],
    )

    result = load_test_debate.validate_against_slos(metrics)

    assert result["agent_dispatch_concurrency"]["actual"] == 0.977
    assert result["agent_dispatch_concurrency"]["passed"] is True
