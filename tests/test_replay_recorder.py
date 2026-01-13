import pytest
import tempfile
import time
from pathlib import Path
from aragora.replay.recorder import ReplayRecorder


def test_recorder_files(tmp_path):
    recorder = ReplayRecorder("test", "topic", "prop", [], str(tmp_path))
    recorder.start()
    recorder.record_turn("a1", "hi", 1)
    time.sleep(0.1)
    recorder.finalize("yes", {})
    assert (tmp_path / "test" / "meta.json").exists()
    assert (tmp_path / "test" / "events.jsonl").exists()
