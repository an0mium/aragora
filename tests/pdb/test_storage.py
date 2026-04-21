"""Tests for :mod:`aragora.pdb.storage`.

Covers:

- ``get_state`` inference across all 6 on-disk layouts
- Each lifecycle transition (``queue_generation``, ``mark_running``,
  ``mark_ready``, ``mark_failed``, ``cancel_generation``,
  ``invalidate_if_head_changed``)
- Atomicity of writes under simulated interrupt (no partial ``.json``
  file surfaces)
- ``index.jsonl`` event append behavior
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.pdb import storage
from aragora.pdb.brief_state import BriefLifecycleState, StateTransitionError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def briefs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the briefs root under a tmp dir and return the briefs dir."""
    monkeypatch.setenv("ARAGORA_REVIEW_QUEUE_ROOT", str(tmp_path))
    briefs = tmp_path / "briefs"
    briefs.mkdir(parents=True, exist_ok=True)
    return briefs


PR = 1234
SHA = "a" * 40
SHA_SHORT = SHA[:12]
FILENAME = f"pr-{PR}-{SHA_SHORT}.json"


def _index_events(briefs: Path) -> list[dict]:
    index = briefs / storage.INDEX_FILENAME
    if not index.exists():
        return []
    return [
        json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# ---------------------------------------------------------------------------
# get_state: 6 on-disk layouts
# ---------------------------------------------------------------------------


class TestGetState:
    def test_absent_returns_absent(self, briefs_dir):
        assert storage.get_state(PR, SHA) == BriefLifecycleState.ABSENT

    def test_queued_layout(self, briefs_dir):
        (briefs_dir / storage.QUEUED_SUBDIR).mkdir()
        (briefs_dir / storage.QUEUED_SUBDIR / FILENAME).write_text("{}", encoding="utf-8")
        assert storage.get_state(PR, SHA) == BriefLifecycleState.QUEUED

    def test_running_layout(self, briefs_dir):
        (briefs_dir / storage.RUNNING_SUBDIR).mkdir()
        (briefs_dir / storage.RUNNING_SUBDIR / FILENAME).write_text("{}", encoding="utf-8")
        assert storage.get_state(PR, SHA) == BriefLifecycleState.RUNNING

    def test_ready_layout(self, briefs_dir):
        (briefs_dir / FILENAME).write_text("{}", encoding="utf-8")
        assert storage.get_state(PR, SHA) == BriefLifecycleState.READY

    def test_failed_layout(self, briefs_dir):
        (briefs_dir / storage.FAILED_SUBDIR).mkdir()
        (briefs_dir / storage.FAILED_SUBDIR / FILENAME).write_text("{}", encoding="utf-8")
        assert storage.get_state(PR, SHA) == BriefLifecycleState.FAILED

    def test_stale_layout(self, briefs_dir):
        (briefs_dir / storage.INVALIDATED_SUBDIR).mkdir()
        (briefs_dir / storage.INVALIDATED_SUBDIR / FILENAME).write_text("{}", encoding="utf-8")
        assert storage.get_state(PR, SHA) == BriefLifecycleState.STALE

    def test_terminal_state_wins_over_in_progress(self, briefs_dir):
        """If a crash leaves both running/ and the final artifact, READY wins."""
        (briefs_dir / storage.RUNNING_SUBDIR).mkdir()
        (briefs_dir / storage.RUNNING_SUBDIR / FILENAME).write_text("{}", encoding="utf-8")
        (briefs_dir / FILENAME).write_text("{}", encoding="utf-8")
        assert storage.get_state(PR, SHA) == BriefLifecycleState.READY

    def test_different_sha_is_independent(self, briefs_dir):
        (briefs_dir / FILENAME).write_text("{}", encoding="utf-8")
        other_sha = "b" * 40
        assert storage.get_state(PR, other_sha) == BriefLifecycleState.ABSENT


# ---------------------------------------------------------------------------
# queue_generation
# ---------------------------------------------------------------------------


class TestQueueGeneration:
    def test_absent_to_queued(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude", "pdb.core.gpt"])
        assert storage.get_state(PR, SHA) == BriefLifecycleState.QUEUED
        queued_file = briefs_dir / storage.QUEUED_SUBDIR / FILENAME
        assert queued_file.exists()
        record = json.loads(queued_file.read_text())
        assert record["pr_number"] == PR
        assert record["head_sha"] == SHA
        assert record["panel_models"] == ["pdb.core.claude", "pdb.core.gpt"]
        assert "requested_at" in record

    def test_index_event_recorded(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        events = _index_events(briefs_dir)
        assert len(events) == 1
        assert events[0]["event"] == "queued"
        assert events[0]["pr_number"] == PR
        assert events[0]["head_sha"] == SHA
        assert events[0]["panel_models"] == ["pdb.core.claude"]

    def test_queued_to_queued_raises(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        with pytest.raises(StateTransitionError):
            storage.queue_generation(PR, SHA, ["pdb.core.claude"])

    def test_running_to_queued_raises(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        storage.mark_running(PR, SHA, phase="findings_round")
        with pytest.raises(StateTransitionError):
            storage.queue_generation(PR, SHA, ["pdb.core.claude"])

    def test_failed_to_queued_clears_failed_record(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        storage.mark_running(PR, SHA, phase="findings_round")
        storage.mark_failed(PR, SHA, "boom", "findings_round", cost_usd_so_far=1.5)
        assert storage.get_state(PR, SHA) == BriefLifecycleState.FAILED

        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        assert storage.get_state(PR, SHA) == BriefLifecycleState.QUEUED
        assert not (briefs_dir / storage.FAILED_SUBDIR / FILENAME).exists()


# ---------------------------------------------------------------------------
# mark_running
# ---------------------------------------------------------------------------


class TestMarkRunning:
    def test_queued_to_running_moves_record(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        storage.mark_running(PR, SHA, phase="findings_round")

        assert storage.get_state(PR, SHA) == BriefLifecycleState.RUNNING
        assert not (briefs_dir / storage.QUEUED_SUBDIR / FILENAME).exists()
        running_file = briefs_dir / storage.RUNNING_SUBDIR / FILENAME
        assert running_file.exists()
        record = json.loads(running_file.read_text())
        assert record["current_phase"] == "findings_round"
        assert record["panel_models"] == ["pdb.core.claude"]
        assert "started_at" in record

    def test_absent_to_running_raises(self, briefs_dir):
        with pytest.raises(StateTransitionError):
            storage.mark_running(PR, SHA, phase="findings_round")


# ---------------------------------------------------------------------------
# write_running_phase
# ---------------------------------------------------------------------------


class TestWriteRunningPhase:
    def test_updates_phase_and_cost_without_state_change(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        storage.mark_running(PR, SHA, phase="findings_round")

        storage.write_running_phase(PR, SHA, phase="critique_round", cost_usd_so_far=2.41)

        assert storage.get_state(PR, SHA) == BriefLifecycleState.RUNNING
        record = json.loads((briefs_dir / storage.RUNNING_SUBDIR / FILENAME).read_text())
        assert record["current_phase"] == "critique_round"
        assert record["cost_usd_so_far"] == pytest.approx(2.41)

        events = [e for e in _index_events(briefs_dir) if e["event"] == "running_phase"]
        assert len(events) == 1
        assert events[0]["current_phase"] == "critique_round"


# ---------------------------------------------------------------------------
# mark_ready
# ---------------------------------------------------------------------------


class TestMarkReady:
    def test_running_to_ready(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        storage.mark_running(PR, SHA, phase="synthesis")

        brief = {
            "pr_number": PR,
            "head_sha": SHA,
            "verdict": "approve_candidate",
            "confidence": 4,
            "logic": "ok",
            "security": "ok",
            "maintainability": "ok",
            "skeptic": "ok",
        }
        storage.mark_ready(PR, SHA, brief, signature="ed25519:sig-abc123")

        assert storage.get_state(PR, SHA) == BriefLifecycleState.READY
        ready_path = briefs_dir / FILENAME
        assert ready_path.exists()
        loaded = json.loads(ready_path.read_text())
        assert loaded["signature"] == "ed25519:sig-abc123"
        assert loaded["verdict"] == "approve_candidate"
        # Running record cleared
        assert not (briefs_dir / storage.RUNNING_SUBDIR / FILENAME).exists()

    def test_mark_ready_appends_index_event(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        storage.mark_running(PR, SHA, phase="synthesis")
        storage.mark_ready(PR, SHA, {"verdict": "approve_candidate"}, signature=None)

        events = [e for e in _index_events(briefs_dir) if e["event"] == "ready"]
        assert len(events) == 1
        assert events[0]["signature_present"] is False

    def test_mark_ready_without_running_raises(self, briefs_dir):
        with pytest.raises(StateTransitionError):
            storage.mark_ready(PR, SHA, {"verdict": "approve_candidate"})

    def test_load_ready_brief_roundtrip(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        storage.mark_running(PR, SHA, phase="synthesis")
        storage.mark_ready(PR, SHA, {"verdict": "approve_candidate", "confidence": 5})

        loaded = storage.load_ready_brief(PR, SHA)
        assert loaded is not None
        assert loaded["verdict"] == "approve_candidate"
        assert loaded["confidence"] == 5


# ---------------------------------------------------------------------------
# mark_failed
# ---------------------------------------------------------------------------


class TestMarkFailed:
    def test_running_to_failed(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        storage.mark_running(PR, SHA, phase="findings_round")
        storage.mark_failed(PR, SHA, "rate limit exhausted", "critique_round", cost_usd_so_far=3.20)

        assert storage.get_state(PR, SHA) == BriefLifecycleState.FAILED
        failed_file = briefs_dir / storage.FAILED_SUBDIR / FILENAME
        record = json.loads(failed_file.read_text())
        assert record["error_message"] == "rate limit exhausted"
        assert record["failed_phase"] == "critique_round"
        assert record["cost_usd_so_far"] == pytest.approx(3.20)
        assert not (briefs_dir / storage.RUNNING_SUBDIR / FILENAME).exists()

    def test_queued_to_failed(self, briefs_dir):
        """Cancel-before-pickup may still choose to record a failed entry."""
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        storage.mark_failed(PR, SHA, "cancelled by user", "queued", cost_usd_so_far=0.0)
        assert storage.get_state(PR, SHA) == BriefLifecycleState.FAILED
        assert not (briefs_dir / storage.QUEUED_SUBDIR / FILENAME).exists()


# ---------------------------------------------------------------------------
# invalidate_if_head_changed
# ---------------------------------------------------------------------------


class TestInvalidateIfHeadChanged:
    def test_moves_ready_with_different_sha(self, briefs_dir):
        old_sha = "oldsha1234567890" + "0" * 22
        new_sha = "newsha9876543210" + "0" * 22

        # Seed a ready brief under the old SHA via full lifecycle.
        storage.queue_generation(PR, old_sha, ["pdb.core.claude"])
        storage.mark_running(PR, old_sha, phase="synthesis")
        storage.mark_ready(PR, old_sha, {"verdict": "approve_candidate"})

        moved = storage.invalidate_if_head_changed(PR, new_sha)
        assert moved is True

        old_short = old_sha[:12]
        old_name = f"pr-{PR}-{old_short}.json"
        assert not (briefs_dir / old_name).exists()
        assert (briefs_dir / storage.INVALIDATED_SUBDIR / old_name).exists()

        events = [e for e in _index_events(briefs_dir) if e["event"] == "stale"]
        assert len(events) == 1
        assert events[0]["reason"] == "head_advanced"
        assert events[0]["head_sha"] == old_sha
        assert events[0]["new_head_sha_short"] == new_sha[:12]

    def test_no_move_when_sha_unchanged(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        storage.mark_running(PR, SHA, phase="synthesis")
        storage.mark_ready(PR, SHA, {"verdict": "approve_candidate"})

        moved = storage.invalidate_if_head_changed(PR, SHA)
        assert moved is False
        assert (briefs_dir / FILENAME).exists()
        assert not (briefs_dir / storage.INVALIDATED_SUBDIR).exists() or not list(
            (briefs_dir / storage.INVALIDATED_SUBDIR).glob("*.json")
        )

    def test_returns_false_when_no_ready_exists(self, briefs_dir):
        moved = storage.invalidate_if_head_changed(PR, SHA)
        assert moved is False


# ---------------------------------------------------------------------------
# cancel_generation
# ---------------------------------------------------------------------------


class TestCancelGeneration:
    def test_cancel_queued_returns_absent(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        final = storage.cancel_generation(PR, SHA)
        assert final == BriefLifecycleState.ABSENT
        assert storage.get_state(PR, SHA) == BriefLifecycleState.ABSENT
        assert not (briefs_dir / storage.QUEUED_SUBDIR / FILENAME).exists()
        events = [e for e in _index_events(briefs_dir) if e["event"] == "cancelled"]
        assert len(events) == 1
        assert events[0]["previous_state"] == "queued"

    def test_cancel_running_returns_absent(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        storage.mark_running(PR, SHA, phase="findings_round")
        final = storage.cancel_generation(PR, SHA)
        assert final == BriefLifecycleState.ABSENT
        assert not (briefs_dir / storage.RUNNING_SUBDIR / FILENAME).exists()

    def test_cancel_ready_is_noop(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        storage.mark_running(PR, SHA, phase="synthesis")
        storage.mark_ready(PR, SHA, {"verdict": "approve_candidate"})
        final = storage.cancel_generation(PR, SHA)
        assert final == BriefLifecycleState.READY
        assert storage.get_state(PR, SHA) == BriefLifecycleState.READY

    def test_cancel_absent_is_noop(self, briefs_dir):
        final = storage.cancel_generation(PR, SHA)
        assert final == BriefLifecycleState.ABSENT

    def test_cancel_failed_is_noop(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        storage.mark_failed(PR, SHA, "oops", "queued")
        final = storage.cancel_generation(PR, SHA)
        assert final == BriefLifecycleState.FAILED


# ---------------------------------------------------------------------------
# load_latest_ready_brief (compat helper)
# ---------------------------------------------------------------------------


class TestLoadLatestReadyBrief:
    def test_returns_none_when_absent(self, briefs_dir):
        assert storage.load_latest_ready_brief(PR) is None

    def test_returns_most_recent_mtime(self, briefs_dir):
        sha_a = "a" * 40
        sha_b = "b" * 40

        storage.queue_generation(PR, sha_a, ["pdb.core.claude"])
        storage.mark_running(PR, sha_a, phase="synthesis")
        storage.mark_ready(PR, sha_a, {"verdict": "approve_candidate", "tag": "older"})

        storage.queue_generation(PR, sha_b, ["pdb.core.claude"])
        storage.mark_running(PR, sha_b, phase="synthesis")
        storage.mark_ready(PR, sha_b, {"verdict": "approve_candidate", "tag": "newer"})

        # Nudge mtimes to guarantee ordering on fast-disk filesystems.
        import os as _os
        import time as _time

        older = briefs_dir / f"pr-{PR}-{sha_a[:12]}.json"
        newer = briefs_dir / f"pr-{PR}-{sha_b[:12]}.json"
        _os.utime(older, (1_000_000.0, 1_000_000.0))
        _os.utime(newer, (_time.time(), _time.time()))

        loaded = storage.load_latest_ready_brief(PR)
        assert loaded is not None
        assert loaded["tag"] == "newer"


# ---------------------------------------------------------------------------
# Atomicity: an interrupted write leaves no partial .json
# ---------------------------------------------------------------------------


class TestAtomicity:
    def test_interrupted_atomic_write_leaves_tmp_only(self, briefs_dir):
        """Simulate a KeyboardInterrupt mid-write; final file must not appear.

        We patch ``os.replace`` (inside the storage module) to raise
        KeyboardInterrupt. The tmp file may remain on disk; the final
        target must NOT.
        """
        target = briefs_dir / storage.QUEUED_SUBDIR / FILENAME

        with patch.object(storage.os, "replace", side_effect=KeyboardInterrupt("simulated")):
            with pytest.raises(KeyboardInterrupt):
                storage.queue_generation(PR, SHA, ["pdb.core.claude"])

        assert not target.exists(), "final .json must not be visible after interrupt"
        # The tmp sibling may remain; that's expected and recoverable.
        tmp_candidates = list((briefs_dir / storage.QUEUED_SUBDIR).glob("*.tmp"))
        # Zero or one tmp files are acceptable; what matters is the
        # final target was not created.
        assert len(tmp_candidates) <= 1


# ---------------------------------------------------------------------------
# index.jsonl event log
# ---------------------------------------------------------------------------


class TestIndexEventLog:
    def test_full_lifecycle_logged(self, briefs_dir):
        storage.queue_generation(PR, SHA, ["pdb.core.claude"])
        storage.mark_running(PR, SHA, phase="findings_round")
        storage.write_running_phase(PR, SHA, "critique_round", cost_usd_so_far=1.1)
        storage.mark_ready(PR, SHA, {"verdict": "approve_candidate"})

        events = _index_events(briefs_dir)
        event_types = [e["event"] for e in events]
        assert event_types == ["queued", "running", "running_phase", "ready"]
        for event in events:
            assert "timestamp" in event
            assert event["pr_number"] == PR

    def test_append_only_across_calls(self, briefs_dir):
        storage.append_index_event(PR, SHA, "custom_event", {"note": "one"})
        storage.append_index_event(PR, SHA, "custom_event", {"note": "two"})
        events = _index_events(briefs_dir)
        assert [e["note"] for e in events] == ["one", "two"]
