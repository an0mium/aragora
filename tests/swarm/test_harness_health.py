"""Tests for aragora.swarm.harness_health (Round 30f phase 2)."""

from __future__ import annotations

import time

import pytest

from aragora.swarm.harness_health import (
    AUTH_FAILURE_CODES,
    PERMANENT_FAILURE_CODES,
    QUOTA_FAILURE_CODES,
    TRANSIENT_FAILURE_CODES,
    FailureCategory,
    HarnessHealthRegistry,
    classify_failure,
    get_harness_health_registry,
    reset_harness_health_registry,
)


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    reset_harness_health_registry()
    yield
    reset_harness_health_registry()


# --- classify_failure --------------------------------------------------


class TestClassifyFailure:
    def test_auth_status_codes(self) -> None:
        for code in AUTH_FAILURE_CODES:
            assert classify_failure(status_code=code, reason="") is FailureCategory.AUTH

    def test_quota_status_code(self) -> None:
        for code in QUOTA_FAILURE_CODES:
            assert classify_failure(status_code=code, reason="") is FailureCategory.QUOTA

    def test_transient_default(self) -> None:
        assert classify_failure(status_code=500, reason="server error") is FailureCategory.TRANSIENT

    def test_no_status_unauthorized_phrase(self) -> None:
        assert (
            classify_failure(status_code=None, reason="Error: Unauthorized API key")
            is FailureCategory.AUTH
        )

    def test_no_status_quota_phrase(self) -> None:
        assert (
            classify_failure(status_code=None, reason="Rate limit exceeded")
            is FailureCategory.QUOTA
        )

    def test_no_status_unknown_is_transient(self) -> None:
        assert (
            classify_failure(status_code=None, reason="connection reset")
            is FailureCategory.TRANSIENT
        )


class TestPermanentCodesUnion:
    def test_union_is_correct(self) -> None:
        assert PERMANENT_FAILURE_CODES == AUTH_FAILURE_CODES | QUOTA_FAILURE_CODES

    def test_transient_codes_disjoint_from_permanent(self) -> None:
        assert PERMANENT_FAILURE_CODES.isdisjoint(TRANSIENT_FAILURE_CODES)


# --- HarnessHealthRegistry --------------------------------------------


class TestRegistryInit:
    def test_default_construction(self) -> None:
        reg = HarnessHealthRegistry()
        assert reg.is_available("any-harness") is True

    def test_rejects_invalid_threshold(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            HarnessHealthRegistry(transient_threshold=0)

    def test_rejects_invalid_window(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            HarnessHealthRegistry(transient_window_seconds=0)


class TestRecordSuccessFailure:
    def test_success_recorded(self) -> None:
        reg = HarnessHealthRegistry()
        reg.record_success("claude-code", detail="ok")
        snap = reg.snapshot("claude-code")
        assert snap.last_outcome == "success"
        assert snap.available is True

    def test_failure_recorded_transient_below_threshold(self) -> None:
        reg = HarnessHealthRegistry()
        cat = reg.record_failure("claude-code", reason="timeout")
        assert cat is FailureCategory.TRANSIENT
        assert reg.is_available("claude-code") is True
        assert reg.transient_failures_in_window("claude-code") == 1

    def test_transient_threshold_pins_harness(self) -> None:
        reg = HarnessHealthRegistry(transient_threshold=3)
        for _ in range(3):
            reg.record_failure("claude-code", reason="timeout")
        assert reg.is_available("claude-code") is False
        reason = reg.permanent_pin_reason("claude-code")
        assert reason is not None
        assert "transient" in reason

    def test_auth_failure_pins_immediately(self) -> None:
        reg = HarnessHealthRegistry()
        cat = reg.record_failure("claude-code", reason="missing credential", status_code=401)
        assert cat is FailureCategory.AUTH
        assert reg.is_available("claude-code") is False
        assert "auth" in reg.permanent_pin_reason("claude-code")

    def test_quota_failure_pins_immediately(self) -> None:
        reg = HarnessHealthRegistry()
        cat = reg.record_failure("claude-code", reason="rate limit", status_code=429)
        assert cat is FailureCategory.QUOTA
        assert reg.is_available("claude-code") is False

    def test_success_resets_transient_window(self) -> None:
        reg = HarnessHealthRegistry(transient_threshold=3)
        reg.record_failure("claude-code", reason="timeout")
        reg.record_failure("claude-code", reason="timeout")
        reg.record_success("claude-code")
        assert reg.transient_failures_in_window("claude-code") == 0
        # And accumulating again starts from zero
        reg.record_failure("claude-code", reason="timeout")
        assert reg.is_available("claude-code") is True

    def test_success_does_not_lift_permanent_pin(self) -> None:
        reg = HarnessHealthRegistry()
        reg.record_failure("claude-code", reason="api key invalid", status_code=401)
        reg.record_success("claude-code")  # bizarre but valid call
        # Permanent pin survives — auth pin is for the session.
        assert reg.is_available("claude-code") is False

    def test_other_harnesses_unaffected_by_pin(self) -> None:
        reg = HarnessHealthRegistry()
        reg.record_failure("claude-code", reason="api key", status_code=401)
        assert reg.is_available("codex") is True
        assert reg.is_available("aider") is True


class TestSnapshot:
    def test_snapshot_for_unknown_harness_is_available(self) -> None:
        reg = HarnessHealthRegistry()
        snap = reg.snapshot("never-touched")
        assert snap.available is True
        assert snap.permanent_pin_reason is None
        assert snap.last_outcome is None
        assert snap.transient_failure_count_in_window == 0

    def test_snapshot_after_failure(self) -> None:
        reg = HarnessHealthRegistry()
        reg.record_failure("claude-code", reason="boom", status_code=500)
        snap = reg.snapshot("claude-code")
        assert snap.last_outcome == "failure"
        assert snap.last_failure_reason == "boom"
        assert snap.last_failure_category == "transient"

    def test_snapshot_to_dict_round_trip(self) -> None:
        reg = HarnessHealthRegistry()
        reg.record_success("claude-code")
        snap = reg.snapshot("claude-code")
        d = snap.to_dict()
        assert d["harness"] == "claude-code"
        assert d["available"] is True
        assert d["last_outcome"] == "success"

    def test_snapshot_all_includes_explicitly_requested(self) -> None:
        reg = HarnessHealthRegistry()
        reg.record_success("claude-code")
        snaps = reg.snapshot_all(harnesses=["codex", "aider"])
        names = {snap.harness for snap in snaps}
        assert {"claude-code", "codex", "aider"}.issubset(names)


class TestSingleton:
    def test_get_returns_same_instance(self) -> None:
        a = get_harness_health_registry()
        b = get_harness_health_registry()
        assert a is b

    def test_reset_creates_new_instance(self) -> None:
        a = get_harness_health_registry()
        reset_harness_health_registry()
        b = get_harness_health_registry()
        assert a is not b


class TestSlidingWindow:
    def test_old_failures_pruned_from_count(self) -> None:
        reg = HarnessHealthRegistry(transient_threshold=3, transient_window_seconds=0.1)
        reg.record_failure("claude-code", reason="t1")
        reg.record_failure("claude-code", reason="t2")
        time.sleep(0.15)
        # Window expired; count should be 0 even though 2 failures occurred
        assert reg.transient_failures_in_window("claude-code") == 0
        # And new failure must restart accumulation, not pin yet
        reg.record_failure("claude-code", reason="t3")
        assert reg.is_available("claude-code") is True
