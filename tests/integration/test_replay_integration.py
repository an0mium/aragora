#!/usr/bin/env python3
"""Replay integration checks (originally repo-root ``test_replay_integration.py``).

Offline tests exercising the replay recorder/storage and the API handler's
default replay-storage wiring. No live agents or network access required.
"""

import json
import tempfile
from pathlib import Path

from aragora.replay.recorder import ReplayRecorder
from aragora.replay.storage import ReplayStorage
from aragora.server.api import DebateAPIHandler


def test_recorder():
    """ReplayRecorder records a session and writes finalized metadata to disk."""
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = Path(temp_dir) / "replays"

        recorder = ReplayRecorder(
            debate_id="test_debate_001",
            topic="Test debate topic",
            proposal="Test proposal",
            agents=[
                {"name": "Agent1", "role": "proposer"},
                {"name": "Agent2", "role": "critic"},
            ],
            storage_dir=str(replay_dir),
        )

        recorder.start()
        recorder.record_phase_change("round_1_start")
        recorder.record_turn("Agent1", "This is a test proposal", 1)
        recorder.record_turn("Agent2", "This is a critique", 1)
        recorder.record_vote("Agent1", "Option A", "Because it's better")
        recorder.record_phase_change("consensus_reached: Option A")
        session_path = recorder.finalize("Option A", {"Option A": 2, "Option B": 1})

        session_dir = replay_dir / "test_debate_001"
        assert Path(session_path) == session_dir
        assert session_dir.is_dir()

        meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["status"] == "completed"
        assert meta["final_verdict"] == "Option A"
        assert meta["vote_tally"] == {"Option A": 2, "Option B": 1}
        assert (session_dir / "events.jsonl").exists()


def test_storage():
    """ReplayStorage lists no recordings for an empty storage directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = ReplayStorage(str(temp_dir))

        recordings = storage.list_recordings()
        assert len(recordings) == 0


def test_api():
    """DebateAPIHandler has no replay storage configured by default."""
    assert DebateAPIHandler.replay_storage is None
