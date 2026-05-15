"""Tests for the proof-loop alerter (aragora.review.alert)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aragora.review import alert as alert_module
from aragora.review.alert import (
    ALERTING_STATUSES,
    EVENT_KIND_CHANGED,
    DEFAULT_MAX_EVENTS,
    EVENT_FILENAME_PREFIX,
    EVENT_FILENAME_SUFFIX,
    EVENT_KIND_HEARTBEAT,
    EVENT_KIND_OPENED,
    EVENT_KIND_RECOVERED,
    EVENTS_SUBDIR,
    STATE_FILENAME,
    AlertEvent,
    AlertState,
    alerting_surface_names,
    determine_event_kind,
    evaluate,
    load_state,
    prune_event_files,
    save_state,
    write_event,
)
from aragora.review.health import (
    STATUS_AGING,
    STATUS_EMPTY,
    STATUS_FRESH,
    STATUS_MISSING,
    STATUS_STALE,
    HealthReport,
    SurfaceCheck,
)

UTC = timezone.utc


def _now() -> datetime:
    return datetime(2026, 5, 14, 17, 0, tzinfo=UTC)


def _report(
    surfaces: list[SurfaceCheck], *, overall: str = STATUS_FRESH, at: datetime | None = None
) -> HealthReport:
    return HealthReport(
        generated_at=at if at is not None else _now(),
        overall_status=overall,
        surfaces=surfaces,
    )


def _surf(name: str, status: str = STATUS_FRESH, **kw: object) -> SurfaceCheck:
    return SurfaceCheck(name=name, status=status, **kw)


class TestAlertingStatuses:
    def test_only_stale_and_missing_alert(self) -> None:
        assert ALERTING_STATUSES == {STATUS_STALE, STATUS_MISSING}

    def test_aging_is_not_alerting(self) -> None:
        report = _report([_surf("briefs", STATUS_AGING)])
        assert alerting_surface_names(report) == []

    def test_empty_is_not_alerting(self) -> None:
        report = _report([_surf("briefs", STATUS_EMPTY)])
        assert alerting_surface_names(report) == []

    def test_stale_alerts(self) -> None:
        report = _report([_surf("briefs", STATUS_STALE)])
        assert alerting_surface_names(report) == ["briefs"]

    def test_missing_alerts(self) -> None:
        report = _report([_surf("settlement_receipts", STATUS_MISSING)])
        assert alerting_surface_names(report) == ["settlement_receipts"]

    def test_alerting_names_sorted(self) -> None:
        report = _report(
            [
                _surf("z", STATUS_STALE),
                _surf("a", STATUS_MISSING),
                _surf("m", STATUS_FRESH),
            ]
        )
        assert alerting_surface_names(report) == ["a", "z"]


class TestDetermineEventKind:
    def test_idle_to_idle_no_event(self) -> None:
        assert determine_event_kind([], []) is None

    def test_idle_to_idle_heartbeat(self) -> None:
        assert determine_event_kind([], [], emit_heartbeat=True) == EVENT_KIND_HEARTBEAT

    def test_idle_to_alerting_opens(self) -> None:
        assert determine_event_kind([], ["briefs"]) == EVENT_KIND_OPENED

    def test_alerting_to_idle_recovers(self) -> None:
        assert determine_event_kind(["briefs"], []) == EVENT_KIND_RECOVERED

    def test_alerting_to_alerting_same_set_no_event(self) -> None:
        assert determine_event_kind(["briefs"], ["briefs"]) is None

    def test_alerting_to_alerting_same_set_heartbeat(self) -> None:
        assert (
            determine_event_kind(["briefs"], ["briefs"], emit_heartbeat=True)
            == EVENT_KIND_HEARTBEAT
        )

    def test_alerting_to_alerting_changed(self) -> None:
        assert determine_event_kind(["briefs"], ["briefs", "b0"]) == EVENT_KIND_CHANGED
        assert determine_event_kind(["briefs", "b0"], ["briefs"]) == EVENT_KIND_CHANGED
        assert determine_event_kind(["briefs"], ["b0"]) == EVENT_KIND_CHANGED


class TestStatePersistence:
    def test_load_state_missing_returns_empty(self, tmp_path: Path) -> None:
        state = load_state(tmp_path / "nope.json")
        assert state == AlertState()

    def test_load_state_corrupt_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        p.write_text("{not valid json", encoding="utf-8")
        state = load_state(p)
        assert state == AlertState()

    def test_load_state_non_dict_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        p.write_text("[]", encoding="utf-8")
        state = load_state(p)
        assert state == AlertState()

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        original = AlertState(
            alerting_surfaces=["briefs", "b0_publication"],
            last_event_at=datetime(2026, 5, 14, 12, 0, tzinfo=UTC),
            last_run_at=datetime(2026, 5, 14, 17, 0, tzinfo=UTC),
            last_event_kind=EVENT_KIND_OPENED,
        )
        path = tmp_path / STATE_FILENAME
        save_state(original, path)
        loaded = load_state(path)
        assert loaded == original

    def test_save_state_atomic_creates_parent(self, tmp_path: Path) -> None:
        state = AlertState(alerting_surfaces=["briefs"])
        path = tmp_path / "nested" / "dir" / STATE_FILENAME
        save_state(state, path)
        assert path.exists()

    def test_save_state_no_tempfile_remnants(self, tmp_path: Path) -> None:
        save_state(AlertState(), tmp_path / STATE_FILENAME)
        leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".state-")]
        assert leftover == []


class TestEvaluateTransitions:
    def test_first_run_no_alerts_no_event(self, tmp_path: Path) -> None:
        report = _report([_surf("briefs", STATUS_FRESH)])
        decision = evaluate(report, state_dir=tmp_path)
        assert decision.event is None
        assert decision.event_path is None
        assert decision.state.alerting_surfaces == []
        assert decision.state.last_run_at is not None

    def test_first_run_with_alerts_opens(self, tmp_path: Path) -> None:
        report = _report(
            [
                _surf("settlement_receipts", STATUS_MISSING),
                _surf("briefs", STATUS_FRESH),
            ]
        )
        decision = evaluate(report, state_dir=tmp_path)
        assert decision.event is not None
        assert decision.event_path is None
        assert decision.event.kind == EVENT_KIND_OPENED
        assert decision.event.previous_alerting == []
        assert decision.event.current_alerting == ["settlement_receipts"]
        assert decision.event_path is None
        assert decision.state.alerting_surfaces == ["settlement_receipts"]
        assert decision.state.last_event_kind == EVENT_KIND_OPENED

    def test_alerting_to_alerting_same_set_no_event(self, tmp_path: Path) -> None:
        save_state(
            AlertState(alerting_surfaces=["briefs"], last_event_kind=EVENT_KIND_OPENED),
            tmp_path / STATE_FILENAME,
        )
        report = _report([_surf("briefs", STATUS_STALE)])
        decision = evaluate(report, state_dir=tmp_path)
        assert decision.event is None
        assert decision.state.alerting_surfaces == ["briefs"]
        # Heartbeat fields update even when no event fires
        assert decision.state.last_run_at is not None

    def test_alerting_set_grows(self, tmp_path: Path) -> None:
        save_state(
            AlertState(alerting_surfaces=["briefs"], last_event_kind=EVENT_KIND_OPENED),
            tmp_path / STATE_FILENAME,
        )
        report = _report(
            [
                _surf("briefs", STATUS_STALE),
                _surf("b0_publication", STATUS_MISSING),
            ]
        )
        decision = evaluate(report, state_dir=tmp_path)
        assert decision.event is not None
        assert decision.event.kind == EVENT_KIND_CHANGED
        assert decision.event.previous_alerting == ["briefs"]
        assert decision.event.current_alerting == ["b0_publication", "briefs"]

    def test_alerting_set_shrinks_but_not_recovered(self, tmp_path: Path) -> None:
        save_state(
            AlertState(
                alerting_surfaces=["briefs", "b0_publication"],
                last_event_kind=EVENT_KIND_OPENED,
            ),
            tmp_path / STATE_FILENAME,
        )
        report = _report([_surf("briefs", STATUS_STALE)])
        decision = evaluate(report, state_dir=tmp_path)
        assert decision.event is not None
        assert decision.event.kind == EVENT_KIND_CHANGED

    def test_alerting_recovers(self, tmp_path: Path) -> None:
        save_state(
            AlertState(alerting_surfaces=["briefs"], last_event_kind=EVENT_KIND_OPENED),
            tmp_path / STATE_FILENAME,
        )
        report = _report([_surf("briefs", STATUS_FRESH)])
        decision = evaluate(report, state_dir=tmp_path)
        assert decision.event is not None
        assert decision.event.kind == EVENT_KIND_RECOVERED
        assert decision.state.alerting_surfaces == []

    def test_heartbeat_emitted_when_requested_and_no_change(self, tmp_path: Path) -> None:
        save_state(AlertState(alerting_surfaces=["briefs"]), tmp_path / STATE_FILENAME)
        report = _report([_surf("briefs", STATUS_STALE)])
        decision = evaluate(report, state_dir=tmp_path, emit_heartbeat=True)
        assert decision.event is not None
        assert decision.event.kind == EVENT_KIND_HEARTBEAT


class TestEventPayload:
    def test_event_includes_overall_status(self, tmp_path: Path) -> None:
        report = _report(
            [_surf("briefs", STATUS_STALE)],
            overall=STATUS_STALE,
        )
        decision = evaluate(report, state_dir=tmp_path)
        assert decision.event is not None
        assert decision.event.overall_status == STATUS_STALE

    def test_event_surfaces_include_relevant_only(self, tmp_path: Path) -> None:
        report = _report(
            [
                _surf("briefs", STATUS_STALE, path="/x"),
                _surf("settlement_receipts", STATUS_FRESH, path="/y"),
            ]
        )
        decision = evaluate(report, state_dir=tmp_path)
        assert decision.event is not None
        names = {s["name"] for s in decision.event.surfaces}
        assert names == {"briefs"}

    def test_event_to_dict_roundtrips(self, tmp_path: Path) -> None:
        event = AlertEvent(
            kind=EVENT_KIND_OPENED,
            generated_at=_now(),
            previous_alerting=[],
            current_alerting=["briefs"],
            surfaces=[{"name": "briefs", "status": STATUS_STALE}],
            overall_status=STATUS_STALE,
        )
        data = event.to_dict()
        assert data["kind"] == EVENT_KIND_OPENED
        assert data["current_alerting"] == ["briefs"]
        # serializable
        json.dumps(data)


class TestWriteEvent:
    def test_creates_events_dir(self, tmp_path: Path) -> None:
        event = AlertEvent(
            kind=EVENT_KIND_OPENED,
            generated_at=_now(),
            previous_alerting=[],
            current_alerting=["briefs"],
            surfaces=[],
            overall_status=STATUS_STALE,
        )
        events_dir = tmp_path / EVENTS_SUBDIR
        path = write_event(event, events_dir)
        assert path.exists()
        assert path.parent == events_dir

    def test_filename_encodes_kind_and_ts(self, tmp_path: Path) -> None:
        event = AlertEvent(
            kind=EVENT_KIND_RECOVERED,
            generated_at=_now(),
            previous_alerting=["briefs"],
            current_alerting=[],
            surfaces=[],
            overall_status=STATUS_FRESH,
        )
        path = write_event(event, tmp_path)
        assert path.name.startswith("event-")
        assert EVENT_KIND_RECOVERED in path.name
        assert path.name.endswith(".json")

    def test_collision_suffixes(self, tmp_path: Path) -> None:
        event = AlertEvent(
            kind=EVENT_KIND_OPENED,
            generated_at=_now(),
            previous_alerting=[],
            current_alerting=["briefs"],
            surfaces=[],
            overall_status=STATUS_STALE,
        )
        p1 = write_event(event, tmp_path)
        p2 = write_event(event, tmp_path)
        assert p1 != p2
        assert p1.exists() and p2.exists()

    def test_complete_payload_written(self, tmp_path: Path) -> None:
        """A successful write produces the complete, parseable event JSON
        at the final path — never a partial file."""
        event = AlertEvent(
            kind=EVENT_KIND_OPENED,
            generated_at=_now(),
            previous_alerting=[],
            current_alerting=["briefs", "settlement_receipts"],
            surfaces=[
                {"name": "briefs", "status": STATUS_STALE, "age_hours": 25.0},
                {"name": "settlement_receipts", "status": STATUS_MISSING, "age_hours": None},
            ],
            overall_status=STATUS_MISSING,
        )
        path = write_event(event, tmp_path)
        # File exists and parses to the exact expected payload.
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["kind"] == EVENT_KIND_OPENED
        assert loaded["current_alerting"] == ["briefs", "settlement_receipts"]
        assert loaded["overall_status"] == STATUS_MISSING
        assert len(loaded["surfaces"]) == 2

    def test_no_temp_remnants_on_success(self, tmp_path: Path) -> None:
        """The atomic-write tempfile (``.event-*`` prefix) must be cleaned up
        on success — leaving it behind would clutter the events directory."""
        event = AlertEvent(
            kind=EVENT_KIND_OPENED,
            generated_at=_now(),
            previous_alerting=[],
            current_alerting=["briefs"],
            surfaces=[],
            overall_status=STATUS_STALE,
        )
        write_event(event, tmp_path)
        leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".event-")]
        assert leftover == []

    def test_failed_replace_cleans_temp_and_leaves_no_partial(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate a failure between writing the tempfile and renaming
        it into place. The final path must NOT exist (no partial file is
        observable), AND the tempfile must be cleaned up so the events
        directory does not accumulate .event-* fragments."""
        event = AlertEvent(
            kind=EVENT_KIND_OPENED,
            generated_at=_now(),
            previous_alerting=[],
            current_alerting=["briefs"],
            surfaces=[],
            overall_status=STATUS_STALE,
        )

        # Inject a failure at the os.replace call site (atomic rename).
        # write_event imports os at module scope, so patch
        # alert_module.os.replace specifically.
        def boom(*_args, **_kwargs):
            raise OSError("simulated rename failure")

        monkeypatch.setattr(alert_module.os, "replace", boom)

        with pytest.raises(OSError, match="simulated rename failure"):
            write_event(event, tmp_path)

        # The final ``event-*.json`` path must not exist — no partial file.
        final_files = [
            p
            for p in tmp_path.iterdir()
            if p.name.startswith("event-") and p.name.endswith(".json")
        ]
        assert final_files == [], (
            f"partial event file should not be observable after failed rename, "
            f"but found: {[p.name for p in final_files]}"
        )

        # And the temp file must be cleaned up.
        temp_files = [p for p in tmp_path.iterdir() if p.name.startswith(".event-")]
        assert temp_files == [], (
            f"temp file should be cleaned up after failed rename, "
            f"but found: {[p.name for p in temp_files]}"
        )


class TestRunAlertPersistenceOrder:
    def test_writes_event_before_saving_advanced_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = _report([_surf("briefs", STATUS_STALE)], overall=STATUS_STALE)
        calls: list[str] = []
        written_path = tmp_path / EVENTS_SUBDIR / "event.json"

        def fake_write_event(event: AlertEvent, events_dir: Path) -> Path:
            calls.append("write_event")
            assert event.kind == EVENT_KIND_OPENED
            assert events_dir == tmp_path / EVENTS_SUBDIR
            return written_path

        def fake_save_state(state: AlertState, path: Path) -> None:
            calls.append("save_state")
            assert state.alerting_surfaces == ["briefs"]
            assert path == tmp_path / STATE_FILENAME

        monkeypatch.setattr(alert_module, "gather_health", lambda **_: report)
        monkeypatch.setattr(alert_module, "write_event", fake_write_event)
        monkeypatch.setattr(alert_module, "save_state", fake_save_state)

        result = alert_module.run_alert(state_dir=tmp_path)

        assert calls == ["write_event", "save_state"]
        assert result.event_path == written_path


class TestEdgeTriggeredSemantics:
    """End-to-end: simulate launchd ticks and verify only state transitions fire events."""

    def test_steady_alerting_no_repeat_events(self, tmp_path: Path) -> None:
        from aragora.review.alert import EVENTS_SUBDIR as ES

        report = _report([_surf("briefs", STATUS_STALE)])
        # tick 1: opens
        decision1 = evaluate(report, state_dir=tmp_path)
        assert decision1.event is not None and decision1.event.kind == EVENT_KIND_OPENED
        save_state(decision1.state, tmp_path / STATE_FILENAME)
        # tick 2: same state, no event
        decision2 = evaluate(report, state_dir=tmp_path)
        assert decision2.event is None
        save_state(decision2.state, tmp_path / STATE_FILENAME)
        # tick 3: still same state, still no event
        decision3 = evaluate(report, state_dir=tmp_path)
        assert decision3.event is None

    def test_full_lifecycle(self, tmp_path: Path) -> None:
        # tick 1: idle
        d1 = evaluate(_report([_surf("briefs", STATUS_FRESH)]), state_dir=tmp_path)
        save_state(d1.state, tmp_path / STATE_FILENAME)
        assert d1.event is None
        # tick 2: opens
        d2 = evaluate(_report([_surf("briefs", STATUS_STALE)]), state_dir=tmp_path)
        save_state(d2.state, tmp_path / STATE_FILENAME)
        assert d2.event is not None and d2.event.kind == EVENT_KIND_OPENED
        # tick 3: grows
        d3 = evaluate(
            _report(
                [
                    _surf("briefs", STATUS_STALE),
                    _surf("b0_publication", STATUS_MISSING),
                ]
            ),
            state_dir=tmp_path,
        )
        save_state(d3.state, tmp_path / STATE_FILENAME)
        assert d3.event is not None and d3.event.kind == EVENT_KIND_CHANGED
        # tick 4: shrinks (still alerting on one surface)
        d4 = evaluate(_report([_surf("briefs", STATUS_STALE)]), state_dir=tmp_path)
        save_state(d4.state, tmp_path / STATE_FILENAME)
        assert d4.event is not None and d4.event.kind == EVENT_KIND_CHANGED
        # tick 5: recovers
        d5 = evaluate(_report([_surf("briefs", STATUS_FRESH)]), state_dir=tmp_path)
        save_state(d5.state, tmp_path / STATE_FILENAME)
        assert d5.event is not None and d5.event.kind == EVENT_KIND_RECOVERED


class TestStateDictRoundTrip:
    def test_state_to_dict_then_from_dict(self) -> None:
        original = AlertState(
            alerting_surfaces=["a", "b"],
            last_event_at=_now(),
            last_run_at=_now(),
            last_event_kind=EVENT_KIND_OPENED,
        )
        restored = AlertState.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_handles_bad_input(self) -> None:
        bad: dict = {"alerting_surfaces": "not a list", "last_event_at": "garbage"}
        state = AlertState.from_dict(bad)
        assert state.alerting_surfaces == []
        assert state.last_event_at is None

    def test_from_dict_drops_non_string_surfaces(self) -> None:
        bad: dict = {"alerting_surfaces": ["a", 1, None, "b"]}
        state = AlertState.from_dict(bad)
        assert state.alerting_surfaces == ["a", "b"]


class TestStaleAndMissingTogether:
    def test_mixed_stale_and_missing(self, tmp_path: Path) -> None:
        report = _report(
            [
                _surf("settlement_receipts", STATUS_MISSING),
                _surf("briefs", STATUS_STALE),
                _surf("boss_metrics", STATUS_AGING),
                _surf("automation_receipts", STATUS_FRESH),
            ]
        )
        decision = evaluate(report, state_dir=tmp_path)
        assert decision.event is not None
        assert decision.event.kind == EVENT_KIND_OPENED
        assert set(decision.event.current_alerting) == {"settlement_receipts", "briefs"}


@pytest.mark.parametrize(
    "status,expect_alerting",
    [
        (STATUS_FRESH, False),
        (STATUS_EMPTY, False),
        (STATUS_AGING, False),
        (STATUS_STALE, True),
        (STATUS_MISSING, True),
    ],
)
def test_status_alerting_membership(status: str, expect_alerting: bool) -> None:
    is_alerting = status in ALERTING_STATUSES
    assert is_alerting == expect_alerting


class TestPruneEventFiles:
    """Bounded retention for the events subdirectory.

    Pruning is conservative — only ``event-*.json`` files are eligible, and
    pruning happens by mtime (oldest first) up to the ``max_count`` cap.
    """

    def _make_event_file(self, events_dir: Path, name: str, mtime: float) -> Path:
        path = events_dir / name
        path.write_text("{}\n", encoding="utf-8")
        os.utime(path, (mtime, mtime))
        return path

    def test_below_cap_does_nothing(self, tmp_path: Path) -> None:
        events_dir = tmp_path / EVENTS_SUBDIR
        events_dir.mkdir()
        for i in range(5):
            self._make_event_file(
                events_dir, f"event-2026051{i}T120000Z-alert_opened.json", 1000.0 + i
            )
        removed = prune_event_files(events_dir, max_count=10)
        assert removed == []
        # All files survive.
        assert len(list(events_dir.iterdir())) == 5

    def test_above_cap_prunes_oldest_first(self, tmp_path: Path) -> None:
        events_dir = tmp_path / EVENTS_SUBDIR
        events_dir.mkdir()
        # 12 files with strictly increasing mtimes; only the newest 5 should survive.
        created = []
        for i in range(12):
            created.append(
                self._make_event_file(events_dir, f"event-{i:02d}-alert_opened.json", 1000.0 + i)
            )
        removed = prune_event_files(events_dir, max_count=5)
        # 7 oldest files removed.
        assert len(removed) == 7
        # Surviving files are the 5 newest by mtime.
        remaining = sorted(events_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        assert [p.name for p in remaining] == [
            f"event-{i:02d}-alert_opened.json" for i in range(7, 12)
        ]
        # And removed files are exactly the 7 oldest.
        assert {p.name for p in removed} == {f"event-{i:02d}-alert_opened.json" for i in range(7)}

    def test_unrelated_files_are_not_touched(self, tmp_path: Path) -> None:
        events_dir = tmp_path / EVENTS_SUBDIR
        events_dir.mkdir()
        # 10 real event files...
        for i in range(10):
            self._make_event_file(events_dir, f"event-{i:02d}-alert_opened.json", 1000.0 + i)
        # ...plus assorted unrelated files an operator might place there.
        unrelated_paths = {
            events_dir / "README.md": "operator notes\n",
            events_dir / "snapshot.tar.gz": "binary blob",
            events_dir / "event-broken.txt": "wrong suffix",  # doesn't end in .json
            events_dir / "log-20260514.json": "wrong prefix",  # doesn't start with event-
            events_dir / ".hidden_event.json": "dotfile",  # also no event- prefix
        }
        for path, body in unrelated_paths.items():
            path.write_text(body, encoding="utf-8")
            os.utime(path, (500.0, 500.0))  # OLDER than all event files
        removed = prune_event_files(events_dir, max_count=3)
        # 7 of the 10 event files should be pruned.
        assert len(removed) == 7
        # All unrelated files must survive untouched.
        for path, body in unrelated_paths.items():
            assert path.exists(), f"unrelated file removed: {path.name}"
            assert path.read_text(encoding="utf-8") == body, f"unrelated file modified: {path.name}"
        # Verify only event-*.json files are affected.
        for path in removed:
            assert path.name.startswith(EVENT_FILENAME_PREFIX)
            assert path.name.endswith(EVENT_FILENAME_SUFFIX)

    def test_missing_directory_is_no_op(self, tmp_path: Path) -> None:
        events_dir = tmp_path / "nonexistent_events"
        assert not events_dir.exists()
        removed = prune_event_files(events_dir, max_count=10)
        assert removed == []
        # Function must not create the directory.
        assert not events_dir.exists()

    def test_nondirectory_path_is_no_op(self, tmp_path: Path) -> None:
        # If someone passes a file path by mistake, prune should bail safely.
        not_a_dir = tmp_path / "not_a_dir"
        not_a_dir.write_text("oops", encoding="utf-8")
        removed = prune_event_files(not_a_dir, max_count=10)
        assert removed == []
        # File should still be there, untouched.
        assert not_a_dir.read_text(encoding="utf-8") == "oops"

    def test_zero_or_negative_max_disables_pruning(self, tmp_path: Path) -> None:
        events_dir = tmp_path / EVENTS_SUBDIR
        events_dir.mkdir()
        for i in range(20):
            self._make_event_file(events_dir, f"event-{i:02d}-alert_opened.json", 1000.0 + i)
        # max_count = 0 → no pruning (caller opt-out).
        assert prune_event_files(events_dir, max_count=0) == []
        # max_count < 0 → no pruning (defensive).
        assert prune_event_files(events_dir, max_count=-1) == []
        # All 20 files survive.
        assert len(list(events_dir.iterdir())) == 20

    def test_default_max_events_constant(self) -> None:
        # The default is documented in the module; this test pins the value
        # so a casual edit cannot silently relax the retention policy.
        assert DEFAULT_MAX_EVENTS == 200

    def test_run_alert_invokes_prune_after_write(self, tmp_path: Path, monkeypatch) -> None:
        """End-to-end: a run that writes an event should also prune below the cap."""
        import aragora.review.alert as alert_module

        events_dir = tmp_path / EVENTS_SUBDIR
        events_dir.mkdir()
        # Pre-populate with 50 stale event files (older than what run_alert will write).
        for i in range(50):
            path = events_dir / f"event-{i:02d}-alert_opened.json"
            path.write_text("{}\n", encoding="utf-8")
            os.utime(path, (1000.0 + i, 1000.0 + i))

        # Stub gather_health to return a stale-fired report so an event is written.
        from aragora.review.health import HealthReport, SurfaceCheck

        def fake_gather_health(**_kwargs):
            return HealthReport(
                generated_at=_now(),
                overall_status=STATUS_STALE,
                surfaces=[SurfaceCheck(name="briefs", status=STATUS_STALE)],
            )

        monkeypatch.setattr(alert_module, "gather_health", fake_gather_health)

        # Trigger one alert run with cap=10.
        result = alert_module.run_alert(state_dir=tmp_path, max_events=10)

        # Event should have been written.
        assert result.event is not None
        assert result.event_path is not None

        # After prune, at most 10 event files should remain — pre-existing
        # older ones got pruned to make room for the new one.
        surviving = [
            p
            for p in events_dir.iterdir()
            if p.name.startswith(EVENT_FILENAME_PREFIX) and p.name.endswith(EVENT_FILENAME_SUFFIX)
        ]
        assert len(surviving) <= 10
        # The newly-written event must be among the survivors (it is the
        # newest by mtime).
        assert result.event_path in surviving


class TestCmdHealthAlertExitCode:
    """Regression for the mixed-state CLI exit-code bug.

    ``aragora/review/health.py`` ranks statuses ``fresh < aging < stale < empty < missing``.
    A surface set like ``[empty, stale]`` produces ``overall_status == "empty"``
    even though a stale surface is firing. ``_cmd_health_alert`` must drive
    its exit code from ``state.alerting_surfaces`` (the actual set of firing
    surfaces), NOT from ``report.overall_status``.
    """

    def test_exit_1_when_stale_present_even_if_overall_is_empty(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        import argparse
        import aragora.cli.commands.review_queue as rq

        # Build a stub AlertResult representing the regression scenario:
        # one empty surface (NOT alerting) + one stale surface (alerting).
        # alert.alerting_surface_names() correctly returns only ['briefs'],
        # so state.alerting_surfaces is non-empty even though
        # report.overall_status is "empty" (3 > stale=2 in severity rank).
        from aragora.review.alert import AlertResult, AlertState
        from aragora.review.health import HealthReport, SurfaceCheck

        report = HealthReport(
            generated_at=_now(),
            overall_status=STATUS_EMPTY,  # the regression: overall is NOT stale/missing
            surfaces=[
                SurfaceCheck(name="settlement_receipts", status=STATUS_EMPTY),
                SurfaceCheck(name="briefs", status=STATUS_STALE),
            ],
        )
        state = AlertState(
            alerting_surfaces=["briefs"],  # actual firing surface
            last_event_at=_now(),
            last_run_at=_now(),
            last_event_kind=EVENT_KIND_OPENED,
        )
        stub_result = AlertResult(
            state=state,
            event=None,
            report=report,
            state_path=tmp_path / STATE_FILENAME,
            event_path=None,
        )

        # Patch run_alert to return our stub.
        monkeypatch.setattr(rq, "_resolve_repo_root", lambda _x: tmp_path, raising=False)

        def fake_run_alert(**_kwargs):
            return stub_result

        # _cmd_health_alert imports run_alert + _resolve_repo_root lazily inside
        # the function body, so we patch their source modules directly.
        import aragora.review.alert as alert_module
        import aragora.review.health as health_module

        monkeypatch.setattr(alert_module, "run_alert", fake_run_alert)
        monkeypatch.setattr(health_module, "_resolve_repo_root", lambda _x: tmp_path, raising=False)

        # Minimal namespace matching the CLI parser surface.
        args = argparse.Namespace(
            repo_root=str(tmp_path),
            review_queue_root=None,
            overnight_root=None,
            automation_receipts_root=None,
            state_dir=str(tmp_path),
            heartbeat=False,
            json_output=False,
            json=False,
        )

        exit_code = rq._cmd_health_alert(args)
        assert exit_code == 1, (
            "exit code must be 1 because state.alerting_surfaces is non-empty, "
            "even though report.overall_status is 'empty' (not in {stale, missing})"
        )

    def test_exit_0_when_no_alerting_surfaces(self, tmp_path: Path, monkeypatch) -> None:
        """Symmetric check: no alerting surfaces → exit 0 regardless of overall status."""
        import argparse
        import aragora.cli.commands.review_queue as rq
        from aragora.review.alert import AlertResult, AlertState
        from aragora.review.health import HealthReport, SurfaceCheck
        import aragora.review.alert as alert_module
        import aragora.review.health as health_module

        report = HealthReport(
            generated_at=_now(),
            overall_status=STATUS_AGING,  # aging is also not stale/missing
            surfaces=[SurfaceCheck(name="briefs", status=STATUS_AGING)],
        )
        state = AlertState(
            alerting_surfaces=[],  # nothing firing
            last_event_at=None,
            last_run_at=_now(),
            last_event_kind=None,
        )
        stub_result = AlertResult(
            state=state,
            event=None,
            report=report,
            state_path=tmp_path / STATE_FILENAME,
            event_path=None,
        )

        monkeypatch.setattr(alert_module, "run_alert", lambda **_kw: stub_result)
        monkeypatch.setattr(health_module, "_resolve_repo_root", lambda _x: tmp_path, raising=False)

        args = argparse.Namespace(
            repo_root=str(tmp_path),
            review_queue_root=None,
            overnight_root=None,
            automation_receipts_root=None,
            state_dir=str(tmp_path),
            heartbeat=False,
            json_output=False,
            json=False,
        )

        exit_code = rq._cmd_health_alert(args)
        assert exit_code == 0
