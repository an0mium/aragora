"""Loop Control Plane v2 - per-loop budget policy, spend ledger, resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aragora.swarm.loop_budget import (
    DEFAULT_SPEND_FRESH_SECONDS,
    BudgetPolicy,
    read_loop_spend,
    record_loop_spend,
    resolve_loop_budget,
    spend_path,
)
from aragora.swarm.loop_control import LOOP_SPECS, LoopKind, LoopState, NextAction, classify_loop
from aragora.swarm.loop_control_io import collect_all


def _write_policy(repo_root: Path, payload: dict) -> Path:
    path = repo_root / ".aragora" / "loop_budgets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


# --- policy -----------------------------------------------------------------


class TestBudgetPolicy:
    def test_missing_file_yields_empty_policy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ARAGORA_LOOP_BUDGET_USD", raising=False)
        policy = BudgetPolicy.load(tmp_path)
        assert policy.ceiling_for("merge_arbiter") == (None, "none")
        assert policy.source == "none"

    def test_env_fleet_default_is_v1_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ARAGORA_LOOP_BUDGET_USD", "12.5")
        ceiling, source = BudgetPolicy.load(tmp_path).ceiling_for("boss_loop")
        assert ceiling == 12.5
        assert "env:ARAGORA_LOOP_BUDGET_USD" in source

    def test_per_loop_overrides_default_and_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ARAGORA_LOOP_BUDGET_USD", "99")
        _write_policy(
            tmp_path,
            {"default_ceiling_usd": 25.0, "loops": {"merge_arbiter": {"ceiling_usd": 5.0}}},
        )
        policy = BudgetPolicy.load(tmp_path)
        arbiter_ceiling, arbiter_source = policy.ceiling_for("merge_arbiter")
        assert arbiter_ceiling == 5.0
        assert arbiter_source.endswith("#loops.merge_arbiter")
        other_ceiling, other_source = policy.ceiling_for("boss_loop")
        assert other_ceiling == 25.0
        assert other_source.endswith("#default")

    def test_invalid_file_degrades_to_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ARAGORA_LOOP_BUDGET_USD", "7")
        path = tmp_path / ".aragora" / "loop_budgets.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")
        assert BudgetPolicy.load(tmp_path).ceiling_for("nomic")[0] == 7.0

    def test_malformed_env_default_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ARAGORA_LOOP_BUDGET_USD", "not-a-number")
        assert BudgetPolicy.load(tmp_path).ceiling_for("boss_loop") == (None, "none")

    def test_non_finite_ceilings_are_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # NaN would fail open downstream (NaN <= 0 is False), so it must never
        # become a ceiling.
        monkeypatch.setenv("ARAGORA_LOOP_BUDGET_USD", "nan")
        assert BudgetPolicy.load(tmp_path).ceiling_for("boss_loop") == (None, "none")
        monkeypatch.delenv("ARAGORA_LOOP_BUDGET_USD")
        path = _write_policy(
            tmp_path,
            {"default_ceiling_usd": float("inf"), "loops": {"nomic": {"ceiling_usd": None}}},
        )
        assert "Infinity" in path.read_text()  # json.dump emits non-finite literals
        policy = BudgetPolicy.load(tmp_path)
        assert policy.ceiling_for("nomic") == (None, "none")
        assert policy.ceiling_for("boss_loop") == (None, "none")

    def test_negative_and_malformed_loop_ceilings_are_ignored(self, tmp_path: Path) -> None:
        _write_policy(
            tmp_path,
            {"loops": {"boss_loop": {"ceiling_usd": -1}, "nomic": "not-a-dict"}},
        )
        policy = BudgetPolicy.load(tmp_path)
        assert policy.ceiling_for("boss_loop") == (None, "none")
        assert policy.ceiling_for("nomic") == (None, "none")

    def test_spend_fresh_seconds_knob(self, tmp_path: Path) -> None:
        _write_policy(tmp_path, {"spend_fresh_seconds": 60})
        assert BudgetPolicy.load(tmp_path).spend_fresh_seconds == 60.0
        _write_policy(tmp_path, {"spend_fresh_seconds": -5})
        assert BudgetPolicy.load(tmp_path).spend_fresh_seconds == DEFAULT_SPEND_FRESH_SECONDS


# --- spend ledger -----------------------------------------------------------


class TestSpendLedger:
    def test_record_then_read_round_trip(self, tmp_path: Path) -> None:
        path = record_loop_spend(
            tmp_path, "boss_loop", 3.21, source="gate-debates", window_start="2026-06-11T00:00:00Z"
        )
        assert path == spend_path(tmp_path, "boss_loop")
        record = read_loop_spend(tmp_path, "boss_loop")
        assert record is not None
        assert record["spend_usd"] == 3.21
        assert record["window_start"] == "2026-06-11T00:00:00Z"
        assert record["source"] == "gate-debates"
        assert record["age_s"] < 60

    def test_negative_and_non_finite_spend_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            record_loop_spend(tmp_path, "boss_loop", -0.01, source="x")
        with pytest.raises(ValueError):
            record_loop_spend(tmp_path, "boss_loop", float("nan"), source="x")
        with pytest.raises(ValueError):
            record_loop_spend(tmp_path, "boss_loop", float("inf"), source="x")

    def test_path_escaping_loop_ids_are_rejected(self, tmp_path: Path) -> None:
        # The loop id is a filename component; it must never traverse out of
        # the spend directory or resolve to an absolute path.
        for hostile in ("/etc/passwd", "../escape", "a/b", "..", "", "Boss Loop", "no.dots"):
            with pytest.raises(ValueError):
                spend_path(tmp_path, hostile)
            with pytest.raises(ValueError):
                record_loop_spend(tmp_path, hostile, 1.0, source="x")
        assert spend_path(tmp_path, "boss_loop").name == "boss_loop.json"

    def test_absent_and_corrupt_snapshots_read_as_none(self, tmp_path: Path) -> None:
        assert read_loop_spend(tmp_path, "publisher") is None
        path = spend_path(tmp_path, "publisher")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{broken")
        assert read_loop_spend(tmp_path, "publisher") is None
        path.write_text(json.dumps({"spend_usd": -4}))
        assert read_loop_spend(tmp_path, "publisher") is None
        # Python's json module parses bare NaN; a NaN spend would fail open
        # downstream, so the reader must reject it.
        path.write_text('{"spend_usd": NaN}')
        assert read_loop_spend(tmp_path, "publisher") is None

    def test_snapshot_is_world_readable(self, tmp_path: Path) -> None:
        path = record_loop_spend(tmp_path, "boss_loop", 1.0, source="x")
        assert path.stat().st_mode & 0o077 == 0o044

    def test_staleness_uses_injected_now(self, tmp_path: Path) -> None:
        record_loop_spend(tmp_path, "nomic", 1.0, source="x")
        mtime = spend_path(tmp_path, "nomic").stat().st_mtime
        record = read_loop_spend(tmp_path, "nomic", now=mtime + 100.0)
        assert record is not None
        assert record["age_s"] == pytest.approx(100.0)


# --- resolution -------------------------------------------------------------


class TestResolveLoopBudget:
    def test_no_ceiling_no_spend_is_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ARAGORA_LOOP_BUDGET_USD", raising=False)
        budget = resolve_loop_budget(tmp_path, "merge_arbiter")
        assert budget["source_status"] == "unavailable"
        assert budget["remaining_usd"] is None

    def test_ceiling_with_fresh_spend_is_ok_with_remaining(self, tmp_path: Path) -> None:
        _write_policy(tmp_path, {"loops": {"boss_loop": {"ceiling_usd": 10.0}}})
        record_loop_spend(tmp_path, "boss_loop", 4.0, source="gate-debates")
        budget = resolve_loop_budget(tmp_path, "boss_loop")
        assert budget["source_status"] == "ok"
        assert budget["remaining_usd"] == pytest.approx(6.0)
        assert "#loops.boss_loop" in budget["source"]
        assert "ledger:" in budget["source"]

    def test_ceiling_without_spend_is_degraded_and_never_computes_remaining(
        self, tmp_path: Path
    ) -> None:
        _write_policy(tmp_path, {"loops": {"publisher": {"ceiling_usd": 2.0}}})
        budget = resolve_loop_budget(tmp_path, "publisher")
        assert budget["source_status"] == "degraded"
        assert budget["ceiling_usd"] == 2.0
        assert budget["spend_usd"] is None
        assert budget["remaining_usd"] is None

    def test_stale_spend_is_visible_but_never_yields_remaining(self, tmp_path: Path) -> None:
        _write_policy(tmp_path, {"loops": {"nomic": {"ceiling_usd": 5.0}}})
        record_loop_spend(tmp_path, "nomic", 9.0, source="x")
        mtime = spend_path(tmp_path, "nomic").stat().st_mtime
        budget = resolve_loop_budget(tmp_path, "nomic", now=mtime + DEFAULT_SPEND_FRESH_SECONDS + 1)
        assert budget["source_status"] == "degraded"
        assert budget["spend_usd"] == 9.0
        assert budget["remaining_usd"] is None
        assert "(stale)" in budget["source"]

    def test_spend_without_ceiling_is_degraded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ARAGORA_LOOP_BUDGET_USD", raising=False)
        record_loop_spend(tmp_path, "docs_sync_drift", 0.0, source="no-model-calls")
        budget = resolve_loop_budget(tmp_path, "docs_sync_drift")
        assert budget["source_status"] == "degraded"
        assert budget["spend_usd"] == 0.0
        assert budget["ceiling_usd"] is None


# --- classification + fleet integration -------------------------------------


class TestClassification:
    def test_exhausted_budget_halts_the_loop(self, tmp_path: Path) -> None:
        _write_policy(tmp_path, {"loops": {"boss_loop": {"ceiling_usd": 5.0}}})
        record_loop_spend(tmp_path, "boss_loop", 5.0, source="gate-debates")
        budget = resolve_loop_budget(tmp_path, "boss_loop")
        record = classify_loop(
            LOOP_SPECS[LoopKind.BOSS_LOOP],
            {"source_status": "ok", "alive": True, "budget": budget},
        )
        assert record.state == LoopState.BUDGET_EXHAUSTED.value
        assert record.next_action == NextAction.HALT.value
        assert record.blocker == "budget exhausted"

    def test_overspend_yields_negative_remaining_and_halts(self, tmp_path: Path) -> None:
        # remaining_usd is the honest remainder, not capped at zero.
        _write_policy(tmp_path, {"loops": {"boss_loop": {"ceiling_usd": 5.0}}})
        record_loop_spend(tmp_path, "boss_loop", 9.0, source="gate-debates")
        budget = resolve_loop_budget(tmp_path, "boss_loop")
        assert budget["remaining_usd"] == pytest.approx(-4.0)
        record = classify_loop(
            LOOP_SPECS[LoopKind.BOSS_LOOP],
            {"source_status": "ok", "alive": True, "budget": budget},
        )
        assert record.state == LoopState.BUDGET_EXHAUSTED.value
        assert record.next_action == NextAction.HALT.value

    def test_degraded_budget_never_halts(self, tmp_path: Path) -> None:
        _write_policy(tmp_path, {"loops": {"boss_loop": {"ceiling_usd": 5.0}}})
        budget = resolve_loop_budget(tmp_path, "boss_loop")
        record = classify_loop(
            LOOP_SPECS[LoopKind.BOSS_LOOP],
            {"source_status": "ok", "alive": True, "budget": budget},
        )
        assert record.state == LoopState.RUNNING.value
        assert record.next_action == NextAction.CONTINUE.value

    def test_collect_all_degrades_on_corrupt_policy_and_ledger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Loops write the ledger concurrently with collection; a torn or
        # corrupt file must degrade the budget, never raise into the fleet.
        monkeypatch.delenv("ARAGORA_LOOP_BUDGET_USD", raising=False)
        policy_file = tmp_path / ".aragora" / "loop_budgets.json"
        policy_file.parent.mkdir(parents=True, exist_ok=True)
        policy_file.write_text('{"default_ceiling_usd": 9.0, "loops": {tor')
        ledger = spend_path(tmp_path, "merge_arbiter")
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text('{"spend_usd": 1.')
        raw = collect_all(
            tmp_path, timeout=0.5, allow_network=False, kinds=[LoopKind.MERGE_ARBITER]
        )
        budget = raw[LoopKind.MERGE_ARBITER]["budget"]
        assert budget["source_status"] == "unavailable"  # corrupt policy + corrupt ledger
        assert budget["remaining_usd"] is None

    def test_collect_all_attaches_per_loop_budgets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ARAGORA_LOOP_BUDGET_USD", raising=False)
        _write_policy(
            tmp_path,
            {"default_ceiling_usd": 20.0, "loops": {"merge_arbiter": {"ceiling_usd": 5.0}}},
        )
        record_loop_spend(tmp_path, "merge_arbiter", 1.0, source="x")
        raw = collect_all(
            tmp_path,
            timeout=0.5,
            allow_network=False,
            kinds=[LoopKind.MERGE_ARBITER, LoopKind.BOSS_LOOP, LoopKind.NOMIC],
        )
        arbiter = raw[LoopKind.MERGE_ARBITER]["budget"]
        assert arbiter["ceiling_usd"] == 5.0
        assert arbiter["remaining_usd"] == pytest.approx(4.0)
        assert arbiter["source_status"] == "ok"
        boss = raw[LoopKind.BOSS_LOOP]["budget"]
        assert boss["ceiling_usd"] == 20.0
        assert boss["source_status"] == "degraded"  # default ceiling, no spend written
        nomic = raw[LoopKind.NOMIC]["budget"]
        assert nomic["ceiling_usd"] == 20.0
        assert nomic["remaining_usd"] is None
