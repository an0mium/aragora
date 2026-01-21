#!/usr/bin/env python3
"""
Test script for replay integration.
"""

import tempfile
from pathlib import Path
import json

from aragora.replay.recorder import ReplayRecorder
from aragora.replay.storage import ReplayStorage
from aragora.server.api import DebateAPIHandler


def test_recorder():
    """Test that ReplayRecorder can be created and used."""

    # Create temporary directory for replays
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = Path(temp_dir) / "replays"

        # Create recorder
        recorder = ReplayRecorder(
            debate_id="test_debate_001",
            topic="Test debate topic",
            proposal="Test proposal",
            agents=[{"name": "Agent1", "role": "proposer"}, {"name": "Agent2", "role": "critic"}],
            storage_dir=str(replay_dir),
        )

        # Start recording
        recorder.start()

        # Record some events
        recorder.record_phase_change("round_1_start")
        recorder.record_turn("Agent1", "This is a test proposal", 1)
        recorder.record_turn("Agent2", "This is a critique", 1)
        recorder.record_vote("Agent1", "Option A", "Because it's better")
        recorder.record_phase_change("consensus_reached: Option A")

        # Wait a bit for async writing
        import time

        time.sleep(0.1)

        # Finalize
        recorder.finalize("Option A", {"Option A": 2, "Option B": 1})

        # Check if files were created
        session_dir = replay_dir / "test_debate_001"
        meta_file = session_dir / "meta.json"
        events_file = session_dir / "events.jsonl"

        if meta_file.exists():
            with open(meta_file, "r") as f:
                json.load(f)
        else:
            pass

        if events_file.exists():
            # Count events
            with open(events_file, "r") as f:
                events = [json.loads(line) for line in f]
            for event in events:
                pass
        else:
            pass


def test_storage():
    """Test ReplayStorage."""

    with tempfile.TemporaryDirectory() as temp_dir:
        storage = ReplayStorage(str(temp_dir))

        # Should list empty
        recordings = storage.list_recordings()
        assert len(recordings) == 0, f"Expected 0 recordings, got {len(recordings)}"


def test_api():
    """Test API endpoints (basic import test)."""

    # Test that the class attributes work
    assert DebateAPIHandler.replay_storage is None, "Replay storage should be None initially"


if __name__ == "__main__":
    test_recorder()
    test_storage()
    test_api()
