import math

import pytest

from scripts.dr_metrics import SLA_TARGETS, calculate_compliance


def _assert_all_tiers_false(compliance: dict[str, bool]) -> None:
    assert set(compliance) == set(SLA_TARGETS)
    assert not any(compliance.values())


def test_calculate_compliance_preserves_valid_tier_results() -> None:
    rto_compliance, rpo_compliance = calculate_compliance(
        restore_time=2 * 3600,
        backup_age=30 * 60,
    )

    assert rto_compliance == {
        "free": True,
        "pro": True,
        "enterprise": False,
    }
    assert rpo_compliance == {
        "free": True,
        "pro": True,
        "enterprise": False,
    }


@pytest.mark.parametrize("restore_time", [-1.0, math.nan, math.inf, -math.inf, True])
def test_calculate_compliance_fails_closed_for_invalid_restore_time(
    restore_time: float,
) -> None:
    rto_compliance, rpo_compliance = calculate_compliance(
        restore_time=restore_time,
        backup_age=0.0,
    )

    _assert_all_tiers_false(rto_compliance)
    assert all(rpo_compliance.values())


@pytest.mark.parametrize("backup_age", [-1.0, math.nan, math.inf, -math.inf, False])
def test_calculate_compliance_fails_closed_for_invalid_backup_age(backup_age: float) -> None:
    rto_compliance, rpo_compliance = calculate_compliance(
        restore_time=0.0,
        backup_age=backup_age,
    )

    assert all(rto_compliance.values())
    _assert_all_tiers_false(rpo_compliance)
