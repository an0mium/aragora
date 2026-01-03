import pytest
import time
from aragora.replay.reader import ReplayReader
from aragora.replay.recorder import ReplayRecorder

def test_reader_bundle(tmp_path):
    # Create test data using the recorder
    session_dir = tmp_path / "test"
    recorder = ReplayRecorder("test", "Test Topic", "Test Proposal", [], str(tmp_path))
    recorder.start()
    recorder.record_turn("agent1", "Hello world", 1)
    time.sleep(0.1)  # Let background writer flush
    recorder.finalize("approved", {"yes": 1})
    time.sleep(0.1)  # Let finalize complete

    # Now read back the data
    reader = ReplayReader(str(session_dir))
    bundle = reader.to_bundle()
    assert "meta" in bundle
    assert "events" in bundle
    assert len(bundle["events"]) >= 1