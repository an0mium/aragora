"""Unit tests for the AGT-05 stale-policy seed.

These tests cover:

- The fresh / stale / expired bucketing.
- Bound validation in ``StalePolicy.__post_init__``.
- Future-dated claims being treated as fresh.
- ``policy_fingerprint`` stability and sensitivity to parameter changes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aragora.reputation.stale_policy import (
    DEFAULT_FRESH_DAYS,
    DEFAULT_STALE_DAYS,
    DEFAULT_HARD_LIMIT_DAYS,
    StaleDecision,
    StalePolicy,
    is_stale,
)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@pytest.fixture()
def now() -> datetime:
    # Pin the evaluation moment so tests are deterministic.
    return datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)


class TestBucketing:
    def test_fresh_just_made(self, now: datetime) -> None:
        decision = is_stale(claim_iso=_iso(now - timedelta(hours=1)), now_iso=_iso(now))
        assert decision.bucket == "fresh"
        assert decision.is_stale is False
        assert decision.is_expired is False

    def test_fresh_under_default_fresh_days(self, now: datetime) -> None:
        decision = is_stale(
            claim_iso=_iso(now - timedelta(days=DEFAULT_FRESH_DAYS - 0.5)),
            now_iso=_iso(now),
        )
        assert decision.bucket == "fresh"

    def test_stale_at_default_stale_days(self, now: datetime) -> None:
        # Age == stale_days hits the >= boundary.
        decision = is_stale(
            claim_iso=_iso(now - timedelta(days=DEFAULT_STALE_DAYS)),
            now_iso=_iso(now),
        )
        assert decision.bucket == "stale"
        assert decision.is_stale is True
        assert decision.is_expired is False

    def test_stale_between_stale_and_hard_limit(self, now: datetime) -> None:
        decision = is_stale(
            claim_iso=_iso(now - timedelta(days=90)),
            now_iso=_iso(now),
        )
        assert decision.bucket == "stale"
        assert decision.is_stale is True
        assert decision.is_expired is False

    def test_expired_past_hard_limit(self, now: datetime) -> None:
        decision = is_stale(
            claim_iso=_iso(now - timedelta(days=DEFAULT_HARD_LIMIT_DAYS + 1)),
            now_iso=_iso(now),
        )
        assert decision.bucket == "expired"
        assert decision.is_stale is True
        assert decision.is_expired is True


class TestFutureDated:
    def test_future_claim_treated_as_fresh(self, now: datetime) -> None:
        decision = is_stale(
            claim_iso=_iso(now + timedelta(days=5)),
            now_iso=_iso(now),
        )
        # Negative age is clamped to zero so the predicate is total.
        assert decision.bucket == "fresh"
        assert decision.age_days == 0.0


class TestPolicyValidation:
    def test_invalid_bounds_zero_fresh(self) -> None:
        with pytest.raises(ValueError):
            StalePolicy(fresh_days=0, stale_days=10, hard_limit_days=20)

    def test_invalid_bounds_unordered(self) -> None:
        with pytest.raises(ValueError):
            StalePolicy(fresh_days=10, stale_days=5, hard_limit_days=20)

    def test_invalid_bounds_hard_below_stale(self) -> None:
        with pytest.raises(ValueError):
            StalePolicy(fresh_days=1, stale_days=10, hard_limit_days=5)

    def test_valid_bounds_equal_allowed(self) -> None:
        # fresh == stale is allowed (immediate stale-on-creation policy)
        p = StalePolicy(fresh_days=1.0, stale_days=1.0, hard_limit_days=10.0)
        assert p.fresh_days == 1.0
        assert p.stale_days == 1.0


class TestFingerprint:
    def test_fingerprint_stable(self) -> None:
        p1 = StalePolicy()
        p2 = StalePolicy()
        assert p1.fingerprint() == p2.fingerprint()

    def test_fingerprint_changes_with_param(self) -> None:
        a = StalePolicy().fingerprint()
        b = StalePolicy(stale_days=DEFAULT_STALE_DAYS + 1).fingerprint()
        assert a != b

    def test_fingerprint_format(self) -> None:
        fp = StalePolicy().fingerprint()
        assert fp.startswith("sp_")
        assert len(fp) == 3 + 12  # "sp_" + 12 hex chars


class TestCustomPolicy:
    def test_custom_policy_applied(self, now: datetime) -> None:
        custom = StalePolicy(fresh_days=1.0, stale_days=2.0, hard_limit_days=3.0)
        decision = is_stale(
            claim_iso=_iso(now - timedelta(days=2.5)),
            now_iso=_iso(now),
            policy=custom,
        )
        assert decision.bucket == "stale"
        assert decision.policy_fingerprint == custom.fingerprint()


class TestInvalidInput:
    def test_empty_claim_iso_raises(self, now: datetime) -> None:
        with pytest.raises(ValueError):
            is_stale(claim_iso="", now_iso=_iso(now))

    def test_now_iso_default_used(self) -> None:
        # Just verify the function handles None now_iso without crashing
        decision = is_stale(claim_iso="2026-04-29T00:00:00Z")
        assert isinstance(decision, StaleDecision)
        assert decision.age_days >= 0.0


class TestReturnShape:
    def test_decision_is_dataclass_with_expected_fields(self, now: datetime) -> None:
        d = is_stale(claim_iso=_iso(now), now_iso=_iso(now))
        assert hasattr(d, "is_stale")
        assert hasattr(d, "is_expired")
        assert hasattr(d, "age_days")
        assert hasattr(d, "bucket")
        assert hasattr(d, "policy_fingerprint")
