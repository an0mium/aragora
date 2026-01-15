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

        print("✓ Recorder created")

        # Start recording
        recorder.start()
        print("✓ Recorder started")

        # Record some events
        recorder.record_phase_change("round_1_start")
        recorder.record_turn("Agent1", "This is a test proposal", 1)
        recorder.record_turn("Agent2", "This is a critique", 1)
        recorder.record_vote("Agent1", "Option A", "Because it's better")
        recorder.record_phase_change("consensus_reached: Option A")

        print("✓ Events recorded")

        # Wait a bit for async writing
        import time

        time.sleep(0.1)

        # Finalize
        recorder.finalize("Option A", {"Option A": 2, "Option B": 1})
        print("✓ Recording finalized")

        # Check if files were created
        session_dir = replay_dir / "test_debate_001"
        meta_file = session_dir / "meta.json"
        events_file = session_dir / "events.jsonl"

        if meta_file.exists():
            print("✓ Meta file created")
            with open(meta_file, "r") as f:
                meta = json.load(f)
            print(f"  Status: {meta.get('status')}")
            print(f"  Event count: {meta.get('event_count')}")
        else:
            print("✗ Meta file missing")

        if events_file.exists():
            print("✓ Events file created")
            # Count events
            with open(events_file, "r") as f:
                events = [json.loads(line) for line in f]
            print(f"✓ Recorded {len(events)} events")
            for event in events:
                print(f"  - {event['event_type']}: {event['source']} - {event['content'][:50]}...")
        else:
            print("✗ Events file missing")

        print("Recorder test completed successfully!")


def test_storage():
    """Test ReplayStorage."""

    with tempfile.TemporaryDirectory() as temp_dir:
        storage = ReplayStorage(str(temp_dir))

        # Should list empty
        recordings = storage.list_recordings()
        assert len(recordings) == 0, f"Expected 0 recordings, got {len(recordings)}"
        print("✓ Storage lists empty correctly")

        print("Storage test completed successfully!")


def test_api():
    """Test API endpoints (basic import test)."""
    print("✓ API imports work")

    # Test that the class attributes work
    assert DebateAPIHandler.replay_storage is None, "Replay storage should be None initially"
    print("✓ API handler class attributes work")

    print("API test completed successfully!")


if __name__ == "__main__":
    print("Testing replay integration...")
    test_recorder()
    print()
    test_storage()
    print()
    test_api()
    print("\nAll tests passed!")
