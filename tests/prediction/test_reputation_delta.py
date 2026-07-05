"""Tests for aragora.prediction.reputation_delta (AGT-05 sub-deliverable 5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aragora.prediction.reputation_delta import ReputationDelta, compute_reputation_deltas
from aragora.prediction.stakeable_claim import QuestionType, ResolutionStatus, StakeableClaim


def _claim(
    *,
    claim_id: str = "c-1",
    status: ResolutionStatus = ResolutionStatus.RESOLVED_YES,
    resolution_value: bool | None = True,
    positions: dict[str, float] | None = None,
    days_before_cutoff: int = 10,
    cutoff: datetime | None = None,
) -> StakeableClaim:
    ref = cutoff or datetime.now(UTC)
    expiry = (ref - timedelta(days=days_before_cutoff)).isoformat()
    return StakeableClaim(
        claim_id=claim_id,
        question="Will PR #1 merge?",
        question_type=QuestionType.PR_MERGE,
        target_ref="owner/repo#1",
        expiry=expiry,
        resolution_status=status,
        resolution_value=resolution_value,
        positions=positions or {},
    )


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    def test_flag_off_raises(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_PREDICTION_MARKETS_ENABLED", raising=False)
        with pytest.raises(RuntimeError, match="disabled"):
            compute_reputation_deltas([_claim(positions={"a": 0.9})])

    def test_require_enabled_false_bypasses(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_PREDICTION_MARKETS_ENABLED", raising=False)
        result = compute_reputation_deltas([_claim(positions={"a": 0.9})], require_enabled=False)
        assert len(result) == 1

    def test_flag_on_succeeds(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_PREDICTION_MARKETS_ENABLED", "1")
        result = compute_reputation_deltas([_claim(positions={"a": 0.9})])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Claim filtering
# ---------------------------------------------------------------------------


class TestClaimFiltering:
    def test_open_excluded(self):
        c = _claim(status=ResolutionStatus.OPEN, resolution_value=None, positions={"a": 0.8})
        assert compute_reputation_deltas([c], require_enabled=False) == []

    def test_expired_excluded(self):
        c = _claim(status=ResolutionStatus.EXPIRED, resolution_value=None, positions={"a": 0.8})
        assert compute_reputation_deltas([c], require_enabled=False) == []

    def test_inconclusive_excluded(self):
        c = _claim(status=ResolutionStatus.INCONCLUSIVE, resolution_value=None, positions={"a": 0.8})
        assert compute_reputation_deltas([c], require_enabled=False) == []

    def test_no_positions_skipped(self):
        assert compute_reputation_deltas([_claim(positions={})], require_enabled=False) == []

    def test_old_claim_outside_window_excluded(self):
        cutoff = datetime.now(UTC)
        c = _claim(days_before_cutoff=200, positions={"a": 0.9}, cutoff=cutoff)
        assert compute_reputation_deltas([c], window_days=90, cutoff_dt=cutoff, require_enabled=False) == []

    def test_recent_claim_included(self):
        cutoff = datetime.now(UTC)
        c = _claim(days_before_cutoff=10, positions={"a": 0.9}, cutoff=cutoff)
        result = compute_reputation_deltas([c], window_days=90, cutoff_dt=cutoff, require_enabled=False)
        assert len(result) == 1

    def test_future_expiry_excluded(self):
        cutoff = datetime.now(UTC)
        future = (cutoff + timedelta(days=10)).isoformat()
        c = StakeableClaim(
            claim_id="c-f", question="Q", question_type=QuestionType.PR_MERGE,
            target_ref="r#1", expiry=future,
            resolution_status=ResolutionStatus.RESOLVED_YES, resolution_value=True,
            positions={"a": 0.9},
        )
        assert compute_reputation_deltas([c], cutoff_dt=cutoff, require_enabled=False) == []


# ---------------------------------------------------------------------------
# Delta arithmetic
# ---------------------------------------------------------------------------


class TestDeltaArithmetic:
    def test_perfect_yes_prediction(self):
        d = compute_reputation_deltas([_claim(positions={"a": 1.0})], require_enabled=False)[0]
        assert d.brier_score == pytest.approx(0.0, abs=1e-6)
        assert d.delta == pytest.approx(0.25, abs=1e-6)

    def test_perfect_no_prediction(self):
        c = _claim(status=ResolutionStatus.RESOLVED_NO, resolution_value=False, positions={"a": 0.0})
        d = compute_reputation_deltas([c], require_enabled=False)[0]
        assert d.brier_score == pytest.approx(0.0, abs=1e-6)
        assert d.delta == pytest.approx(0.25, abs=1e-6)

    def test_no_skill_baseline_zero_delta(self):
        d = compute_reputation_deltas([_claim(positions={"a": 0.5})], require_enabled=False)[0]
        assert d.brier_score == pytest.approx(0.25, abs=1e-6)
        assert d.delta == pytest.approx(0.0, abs=1e-6)

    def test_maximally_wrong_most_negative_delta(self):
        c = _claim(status=ResolutionStatus.RESOLVED_NO, resolution_value=False, positions={"a": 1.0})
        d = compute_reputation_deltas([c], require_enabled=False)[0]
        assert d.brier_score == pytest.approx(1.0, abs=1e-6)
        assert d.delta == pytest.approx(-0.75, abs=1e-6)

    def test_multiple_agents_scored_independently(self):
        c = _claim(
            status=ResolutionStatus.RESOLVED_YES, resolution_value=True,
            positions={"skilled": 1.0, "unskilled": 0.0},
        )
        by_agent = {d.agent_id: d for d in compute_reputation_deltas([c], require_enabled=False)}
        assert by_agent["skilled"].delta == pytest.approx(0.25, abs=1e-6)
        assert by_agent["unskilled"].delta == pytest.approx(-0.75, abs=1e-6)

    def test_multiple_claims_produce_multiple_deltas(self):
        cutoff = datetime.now(UTC)
        claims = [
            _claim(claim_id=f"c-{i}", positions={"a": 0.8}, cutoff=cutoff) for i in range(3)
        ]
        deltas = compute_reputation_deltas(claims, cutoff_dt=cutoff, require_enabled=False)
        assert len(deltas) == 3
        assert {d.claim_id for d in deltas} == {"c-0", "c-1", "c-2"}


# ---------------------------------------------------------------------------
# ReputationDelta dataclass
# ---------------------------------------------------------------------------


class TestReputationDelta:
    def test_to_dict_has_all_keys(self):
        d = ReputationDelta(
            agent_id="a", claim_id="c", delta=0.25, brier_score=0.0,
            resolved_yes=True, agent_probability=1.0, computed_at="2026-07-05T00:00:00+00:00",
        )
        result = d.to_dict()
        for key in ("agent_id", "claim_id", "delta", "brier_score",
                    "resolved_yes", "agent_probability", "computed_at"):
            assert key in result

    def test_resolved_yes_field_matches_status(self):
        yes_d = compute_reputation_deltas([_claim(positions={"a": 0.7})], require_enabled=False)[0]
        no_d = compute_reputation_deltas(
            [_claim(claim_id="c-no", status=ResolutionStatus.RESOLVED_NO,
                    resolution_value=False, positions={"a": 0.3})],
            require_enabled=False,
        )[0]
        assert yes_d.resolved_yes is True
        assert no_d.resolved_yes is False
