import pytest
from aragora.replay.schema import ReplayEvent, ReplayMeta


def test_event_jsonl():
    event = ReplayEvent("id", 0, 0, "turn", "a1", "hi", {"round": 1})
    jsonl = event.to_jsonl()
    restored = ReplayEvent.from_jsonl(jsonl)
    assert restored.content == "hi"


def test_meta_json():
    meta = ReplayMeta(debate_id="test")
    json_str = meta.to_json()
    restored = ReplayMeta.from_json(json_str)
    assert restored.debate_id == "test"
