"""Tests for TaskRoutingContext (pure routing-input dataclass)."""

from __future__ import annotations

import pytest

from aragora.routing.config import TaskRoutingContext


def test_minimal_construction_and_defaults() -> None:
    ctx = TaskRoutingContext(task_id="t1", domain="software")
    assert ctx.task_id == "t1"
    assert ctx.domain == "software"
    assert ctx.complexity_score == 0.0
    assert ctx.budget_usd is None
    assert ctx.latency_ms_sla is None
    assert ctx.quality_floor == 0.0
    assert ctx.diversity_preference == 0.0


def test_frozen_is_immutable() -> None:
    ctx = TaskRoutingContext(task_id="t1", domain="software")
    with pytest.raises(Exception):
        ctx.task_id = "other"  # type: ignore[misc]


def test_to_dict_roundtrip_shape() -> None:
    ctx = TaskRoutingContext(
        task_id="t1",
        domain="legal",
        complexity_score=6.5,
        budget_usd=1.25,
        latency_ms_sla=2000.0,
        quality_floor=0.7,
        diversity_preference=0.4,
    )
    d = ctx.to_dict()
    assert d == {
        "task_id": "t1",
        "domain": "legal",
        "complexity_score": 6.5,
        "budget_usd": 1.25,
        "latency_ms_sla": 2000.0,
        "quality_floor": 0.7,
        "diversity_preference": 0.4,
    }


def test_empty_task_id_rejected() -> None:
    with pytest.raises(ValueError, match="task_id"):
        TaskRoutingContext(task_id="", domain="software")


@pytest.mark.parametrize("bad", [-0.1, 10.1, 99.0])
def test_complexity_out_of_range_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="complexity_score"):
        TaskRoutingContext(task_id="t1", domain="d", complexity_score=bad)


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_quality_floor_out_of_range_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="quality_floor"):
        TaskRoutingContext(task_id="t1", domain="d", quality_floor=bad)


def test_diversity_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="diversity_preference"):
        TaskRoutingContext(task_id="t1", domain="d", diversity_preference=2.0)


def test_negative_budget_rejected_but_none_ok() -> None:
    with pytest.raises(ValueError, match="budget_usd"):
        TaskRoutingContext(task_id="t1", domain="d", budget_usd=-5.0)
    assert TaskRoutingContext(task_id="t1", domain="d", budget_usd=None).budget_usd is None
    assert TaskRoutingContext(task_id="t1", domain="d", budget_usd=0.0).budget_usd == 0.0


def test_nonpositive_latency_rejected_but_none_ok() -> None:
    with pytest.raises(ValueError, match="latency_ms_sla"):
        TaskRoutingContext(task_id="t1", domain="d", latency_ms_sla=0.0)
    assert TaskRoutingContext(task_id="t1", domain="d", latency_ms_sla=None).latency_ms_sla is None
