"""Tests for :mod:`aragora.pdb.brief_state`.

These tests pin the lifecycle transition table. Every legal transition is
verified; every illegal transition (including self-transitions, which are
explicitly rejected) raises :class:`StateTransitionError`.
"""

from __future__ import annotations

import itertools

import pytest

from aragora.pdb.brief_state import (
    LEGAL_TRANSITIONS,
    BriefLifecycleState,
    StateTransitionError,
    validate_transition,
)


# ---------------------------------------------------------------------------
# Enum surface
# ---------------------------------------------------------------------------


class TestEnumSurface:
    def test_six_states_present(self):
        assert {s.value for s in BriefLifecycleState} == {
            "absent",
            "queued",
            "running",
            "ready",
            "failed",
            "stale",
        }

    def test_states_are_str_subclass(self):
        """String-mixin lets ``state == "ready"`` work at API boundary."""
        assert BriefLifecycleState.READY == "ready"
        assert BriefLifecycleState.QUEUED == "queued"

    def test_stringification_returns_value(self):
        assert str(BriefLifecycleState.RUNNING) == "running"


# ---------------------------------------------------------------------------
# Legal transitions (enumerated — each must pass)
# ---------------------------------------------------------------------------


LEGAL_PAIRS = [
    (BriefLifecycleState.ABSENT, BriefLifecycleState.QUEUED),
    (BriefLifecycleState.QUEUED, BriefLifecycleState.RUNNING),
    (BriefLifecycleState.QUEUED, BriefLifecycleState.FAILED),
    (BriefLifecycleState.QUEUED, BriefLifecycleState.ABSENT),
    (BriefLifecycleState.RUNNING, BriefLifecycleState.READY),
    (BriefLifecycleState.RUNNING, BriefLifecycleState.FAILED),
    (BriefLifecycleState.RUNNING, BriefLifecycleState.ABSENT),
    (BriefLifecycleState.READY, BriefLifecycleState.STALE),
    (BriefLifecycleState.FAILED, BriefLifecycleState.QUEUED),
    (BriefLifecycleState.STALE, BriefLifecycleState.QUEUED),
]


class TestLegalTransitions:
    @pytest.mark.parametrize("source,destination", LEGAL_PAIRS)
    def test_each_legal_transition_passes(
        self, source: BriefLifecycleState, destination: BriefLifecycleState
    ):
        # Should not raise.
        validate_transition(source, destination)

    def test_legal_transitions_table_matches_documented_pairs(self):
        documented = set(LEGAL_PAIRS)
        actual = {(src, dst) for src, dests in LEGAL_TRANSITIONS.items() for dst in dests}
        assert documented == actual, "LEGAL_TRANSITIONS table drifted from the documented pairs."


# ---------------------------------------------------------------------------
# Illegal transitions (exhaustive — every non-legal pair raises)
# ---------------------------------------------------------------------------


ALL_STATES = list(BriefLifecycleState)


def _illegal_pairs() -> list[tuple[BriefLifecycleState, BriefLifecycleState]]:
    """Enumerate every ordered pair not in the legal table.

    Self-transitions are included because they are also rejected.
    """
    legal = set(LEGAL_PAIRS)
    return [(a, b) for a, b in itertools.product(ALL_STATES, ALL_STATES) if (a, b) not in legal]


class TestIllegalTransitions:
    @pytest.mark.parametrize("source,destination", _illegal_pairs())
    def test_each_illegal_transition_raises(
        self, source: BriefLifecycleState, destination: BriefLifecycleState
    ):
        with pytest.raises(StateTransitionError) as exc_info:
            validate_transition(source, destination)
        err = exc_info.value
        assert err.source == source
        assert err.destination == destination
        assert source.value in str(err)
        assert destination.value in str(err)

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_self_transition_rejected(self, state: BriefLifecycleState):
        with pytest.raises(StateTransitionError) as exc_info:
            validate_transition(state, state)
        assert "self-transition" in str(exc_info.value)

    def test_state_transition_error_is_value_error(self):
        """Callers can catch as ValueError without importing this package."""
        with pytest.raises(ValueError):
            validate_transition(BriefLifecycleState.READY, BriefLifecycleState.QUEUED)
