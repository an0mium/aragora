"""Tests for stale-policy integration in settle_claim (AGT-05 SD-4).

Verifies that:
- ``stale_policy=None`` (the default) is backward-compatible; existing
  callers that omit the kwarg are completely unaffected.
- Fresh claims (<fresh_days old) settle without a stale annotation.
- Stale claims (>=stale_days but <hard_limit_days) settle normally but
  carry ``reason[\"stale_warning\"]`` and ``reason[\"staleness_age_days\"]``.
- Expired claims (>=hard_limit_days) raise SettlementError before any
  scoring runs, and the error message names the claim and explains why.
- Custom StalePolicy bounds are respected.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aragora.reputation.settlement import SettlementError, settle_claim
from aragora.reputation.stale_policy import StalePolicy
from aragora.reputation.types import (
    DOMAIN_PREDICTION_MARKET,
    ResolvedClaim,
    StakeableClaim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _claim(created_at: str, *, probability: float = 0.8) -> StakeableClaim:
    return StakeableClaim.create(
        agent_id="alice",
        domain=DOMAIN_PREDICTION_MARKET,
        statement="Event will occur",
        position="yes",
        stake_units=10,
        resolution_source="synthetic_github",
        resolution_id="mkt_abc",
        predicted_probability=probability,
        created_at=created_at,
    )


def _resolved(claim: StakeableClaim, outcome: str = "yes") -> ResolvedClaim:
    return ResolvedClaim(
        claim_id=claim.claim_id,
        outcome=outcome,  # type: ignore[arg-type]
        resolved_at=_iso(datetime.now(UTC)),
        resolution_source="synthetic_github",
    )


@pytest.fixture()
def default_policy() -> StalePolicy:
    return StalePolicy()


# ---------------------------------------------------------------------------
# Backward compatibility — stale_policy=None
# ---------------------------------------------------------------------------


class TestNoStalePolicyBackwardCompat:
    def test_omitting_stale_policy_settles_any_age(self) -> None:
        # A 200-day-old claim must still settle when stale_policy is not passed.
        claim = _claim(_iso(datetime.now(UTC) - timedelta(days=200)))
        delta = settle_claim(claim, _resolved(claim))  # no stale_policy keyword
        assert isinstance(delta.delta, float)
        assert "stale_warning" not in delta.reason

    def test_explicit_none_is_neutral(self) -> None:
        claim = _claim(_iso(datetime.now(UTC) - timedelta(days=100)))
        delta = settle_claim(claim, _resolved(claim), stale_policy=None)
        assert "stale_warning" not in delta.reason
        assert "staleness_age_days" not in delta.reason


# ---------------------------------------------------------------------------
# Fresh claims
# ---------------------------------------------------------------------------


class TestFreshClaim:
    def test_fresh_claim_no_stale_annotation(self, default_policy: StalePolicy) -> None:
        claim = _claim(_iso(datetime.now(UTC) - timedelta(hours=1)))
        delta = settle_claim(claim, _resolved(claim), stale_policy=default_policy)
        assert "stale_warning" not in delta.reason
        assert "staleness_age_days" not in delta.reason

    def test_fresh_claim_delta_is_computed_correctly(self, default_policy: StalePolicy) -> None:
        # probability=0.8, outcome=yes → brier=(0.8-1)²=0.04 → fraction=0.92 → delta=9.2
        claim = _claim(_iso(datetime.now(UTC) - timedelta(hours=1)), probability=0.8)
        delta = settle_claim(
            claim, _resolved(claim), stale_policy=default_policy, scoring_rule="brier_proper"
        )
        assert delta.delta == pytest.approx(9.2, abs=1e-6)

    def test_just_under_stale_threshold_is_fresh(self, default_policy: StalePolicy) -> None:
        # Default stale_days=30; 29.5 days old should be fresh.
        claim = _claim(_iso(datetime.now(UTC) - timedelta(days=29, hours=12)))
        delta = settle_claim(claim, _resolved(claim), stale_policy=default_policy)
        assert "stale_warning" not in delta.reason


# ---------------------------------------------------------------------------
# Stale claims
# ---------------------------------------------------------------------------


class TestStaleClaim:
    def test_stale_claim_carries_warning(self, default_policy: StalePolicy) -> None:
        # 40 days old — past stale_days (30) but under hard_limit_days (180).
        claim = _claim(_iso(datetime.now(UTC) - timedelta(days=40)))
        delta = settle_claim(claim, _resolved(claim), stale_policy=default_policy)
        assert delta.reason.get("stale_warning") is True

    def test_stale_claim_includes_age_days_in_reason(self, default_policy: StalePolicy) -> None:
        claim = _claim(_iso(datetime.now(UTC) - timedelta(days=40)))
        delta = settle_claim(claim, _resolved(claim), stale_policy=default_policy)
        age = delta.reason.get("staleness_age_days")
        assert age is not None
        assert 39.0 < float(age) < 41.0

    def test_stale_claim_still_produces_delta(self, default_policy: StalePolicy) -> None:
        # Settlement must complete; the stale annotation is advisory, not blocking.
        claim = _claim(_iso(datetime.now(UTC) - timedelta(days=40)))
        delta = settle_claim(claim, _resolved(claim), stale_policy=default_policy)
        assert isinstance(delta.delta, float)

    def test_just_under_hard_limit_is_stale_not_expired(self) -> None:
        # 179 days old with default hard_limit=180: stale but not expired.
        claim = _claim(_iso(datetime.now(UTC) - timedelta(days=179)))
        delta = settle_claim(claim, _resolved(claim), stale_policy=StalePolicy())
        assert delta.reason.get("stale_warning") is True
        assert "staleness_age_days" in delta.reason


# ---------------------------------------------------------------------------
# Expired claims
# ---------------------------------------------------------------------------


class TestExpiredClaim:
    def test_expired_raises_settlement_error(self, default_policy: StalePolicy) -> None:
        claim = _claim(_iso(datetime.now(UTC) - timedelta(days=200)))
        with pytest.raises(SettlementError):
            settle_claim(claim, _resolved(claim), stale_policy=default_policy)

    def test_expired_error_names_claim_id(self, default_policy: StalePolicy) -> None:
        claim = _claim(_iso(datetime.now(UTC) - timedelta(days=200)))
        with pytest.raises(SettlementError, match=claim.claim_id):
            settle_claim(claim, _resolved(claim), stale_policy=default_policy)

    def test_expired_error_says_settlement_refused(self, default_policy: StalePolicy) -> None:
        claim = _claim(_iso(datetime.now(UTC) - timedelta(days=200)))
        with pytest.raises(SettlementError, match="settlement refused"):
            settle_claim(claim, _resolved(claim), stale_policy=default_policy)

    def test_no_delta_produced_for_expired_claim(self, default_policy: StalePolicy) -> None:
        # Verify no ReputationDelta is returned — only SettlementError.
        claim = _claim(_iso(datetime.now(UTC) - timedelta(days=200)))
        result = None
        try:
            result = settle_claim(claim, _resolved(claim), stale_policy=default_policy)
        except SettlementError:
            pass
        assert result is None

    def test_custom_hard_limit_respected(self) -> None:
        # Custom policy: hard_limit=90d; 100-day-old claim is expired.
        strict = StalePolicy(fresh_days=1, stale_days=7, hard_limit_days=90)
        claim = _claim(_iso(datetime.now(UTC) - timedelta(days=100)))
        with pytest.raises(SettlementError):
            settle_claim(claim, _resolved(claim), stale_policy=strict)

    def test_same_claim_settles_without_policy(self) -> None:
        # The claim is expired under the policy, but a caller without the
        # policy must still see a successful settlement.
        claim = _claim(_iso(datetime.now(UTC) - timedelta(days=200)))
        delta = settle_claim(claim, _resolved(claim))  # no stale_policy
        assert isinstance(delta.delta, float)
