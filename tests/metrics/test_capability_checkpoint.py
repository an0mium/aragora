"""Tests for aragora.metrics.capability_checkpoint — AGT-06 CP-* registry."""

from __future__ import annotations

import pytest

from aragora.metrics.capability_checkpoint import (
    CheckpointCode,
    CheckpointRecord,
    CheckpointRegistry,
    CheckpointRegistryError,
    CheckpointStatus,
    build_default_registry,
    capability_checkpoints_enabled,
)


def _rec(
    code: CheckpointCode, status: CheckpointStatus = CheckpointStatus.PASS
) -> CheckpointRecord:
    return CheckpointRecord.create(checkpoint_code=code, status=status, evaluator="t")


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


def test_flag_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARAGORA_CAPABILITY_CHECKPOINTS_ENABLED", raising=False)
    assert not capability_checkpoints_enabled()


@pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
def test_flag_enabled_truthy(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("ARAGORA_CAPABILITY_CHECKPOINTS_ENABLED", val)
    assert capability_checkpoints_enabled()


# ---------------------------------------------------------------------------
# Default registry structure (mirrors AGENT_CIVILIZATION_SUBSTRATE.md §5)
# ---------------------------------------------------------------------------


def test_default_registry_has_five_checkpoints() -> None:
    assert len(build_default_registry().all_checkpoints()) == 5


def test_all_checkpoints_start_pending() -> None:
    reg = build_default_registry()
    assert all(reg.status_of(cp.code) == CheckpointStatus.PENDING for cp in reg.all_checkpoints())


def test_dependency_chain() -> None:
    reg = build_default_registry()
    assert reg.checkpoint(CheckpointCode.CP1).depends_on is None
    for code, dep in [
        (CheckpointCode.CP2, CheckpointCode.CP1),
        (CheckpointCode.CP3, CheckpointCode.CP2),
        (CheckpointCode.CP4, CheckpointCode.CP3),
        (CheckpointCode.CP5, CheckpointCode.CP4),
    ]:
        assert reg.checkpoint(code).depends_on == dep


def test_all_window_weeks_are_4() -> None:
    assert all(cp.window_weeks == 4 for cp in build_default_registry().all_checkpoints())


def test_cp5_pass_condition_mentions_viah() -> None:
    assert "VIAH" in build_default_registry().checkpoint(CheckpointCode.CP5).pass_condition


def test_checkpoints_sorted_by_code() -> None:
    codes = [cp.code.value for cp in build_default_registry().all_checkpoints()]
    assert codes == sorted(codes)


def test_unknown_code_raises() -> None:
    small = CheckpointRegistry([build_default_registry().checkpoint(CheckpointCode.CP1)])
    with pytest.raises(CheckpointRegistryError):
        small.checkpoint(CheckpointCode.CP2)


# ---------------------------------------------------------------------------
# CheckpointRecord
# ---------------------------------------------------------------------------


def test_record_create_auto_timestamp() -> None:
    assert _rec(CheckpointCode.CP1).evaluated_at.endswith("Z")


def test_record_to_dict_shape() -> None:
    d = CheckpointRecord.create(
        checkpoint_code=CheckpointCode.CP3,
        status=CheckpointStatus.SKIPPED,
        evaluator="ci",
        notes="skip",
    ).to_dict()
    assert d["checkpoint_code"] == "CP-3" and d["status"] == "skipped" and d["notes"] == "skip"


def test_record_evidence_preserved() -> None:
    ev = {"shift_entry": "abc"}
    rec = CheckpointRecord.create(
        checkpoint_code=CheckpointCode.CP1,
        status=CheckpointStatus.PASS,
        evaluator="op",
        evidence=ev,
    )
    assert rec.evidence == ev


# ---------------------------------------------------------------------------
# Registry write path (flag-gated)
# ---------------------------------------------------------------------------


def test_record_blocked_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARAGORA_CAPABILITY_CHECKPOINTS_ENABLED", raising=False)
    with pytest.raises(CheckpointRegistryError, match="disabled"):
        build_default_registry().record(_rec(CheckpointCode.CP1))


def test_record_succeeds_and_updates_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_CAPABILITY_CHECKPOINTS_ENABLED", "1")
    reg = build_default_registry()
    r1 = _rec(CheckpointCode.CP2, CheckpointStatus.FAIL)
    r2 = _rec(CheckpointCode.CP2, CheckpointStatus.PASS)
    assert reg.record(r1) is r1
    reg.record(r2)
    assert reg.status_of(CheckpointCode.CP2) == CheckpointStatus.PASS
    assert reg.latest_record(CheckpointCode.CP2) is r2


def test_record_unknown_code_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_CAPABILITY_CHECKPOINTS_ENABLED", "1")
    small = CheckpointRegistry([build_default_registry().checkpoint(CheckpointCode.CP1)])
    with pytest.raises(CheckpointRegistryError, match="unknown checkpoint code"):
        small.record(_rec(CheckpointCode.CP2))


def test_latest_record_none_before_writes() -> None:
    assert build_default_registry().latest_record(CheckpointCode.CP5) is None
