"""Tests for the proof-loop alerter (aragora.review.alert)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aragora.review.alert import (
    ALERTING_STATUSES,
    EVENT_KIND_CHANGED,
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
        assert decision.event.kind == EVENT_KIND_OPENED
        assert decision.event.previous_alerting == []
        assert decision.event.current_alerting == ["settlement_receipts"]
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
