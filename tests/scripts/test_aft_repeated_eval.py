"""Focused regression tests for AFT repeated-eval aggregation."""

from __future__ import annotations

import math

import pytest

from scripts.aft_repeated_eval import aggregate


def test_aggregate_rejects_non_finite_condition_metric() -> None:
    summaries = [
        {
            "conditions": {
                "local_advocate": {
                    "accuracy": math.nan,
                    "brier": 0.25,
                    "cost_usd_total": 0.0,
                    "latency_ms_mean": 10.0,
                }
            },
            "pairwise_significance": {},
        }
    ]

    with pytest.raises(ValueError, match=r"conditions\.local_advocate\.accuracy"):
        aggregate(summaries)


def test_aggregate_rejects_non_finite_pairwise_p_value() -> None:
    summaries = [
        {
            "conditions": {},
            "pairwise_significance": {"baseline_vs_local": {"p_value_bonferroni": float("inf")}},
        }
    ]

    with pytest.raises(
        ValueError,
        match=r"pairwise_significance\.baseline_vs_local\.p_value_bonferroni",
    ):
        aggregate(summaries)


def test_aggregate_rejects_non_numeric_pairwise_p_value() -> None:
    summaries = [
        {
            "conditions": {},
            "pairwise_significance": {"baseline_vs_local": {"p_value_bonferroni": "not-a-number"}},
        }
    ]

    with pytest.raises(ValueError, match="non-numeric p_value_bonferroni"):
        aggregate(summaries)
