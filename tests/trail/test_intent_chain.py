"""Tests for the TET intent hash-chain (``aragora/trail/intent_chain.py``).

Spec: docs/specs/TAMPER_EVIDENT_TRAIL.md Component 2 (phase T1).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from aragora.trail.intent_chain import (
    ACTOR_CLASSES,
    GENESIS_PREV_HASH,
    INTENT_TYPES,
    ChainError,
    append_intent,
    chain_head_hash,
    compute_record_hash,
    default_chain_path,
    read_records,
    record_intent,
    verify_chain,
)

TARGET = {"repo": "synaptent/aragora", "pr": 1234}


def _ts() -> str:
    return "2026-06-11T22:00:00+00:00"


def _append(path: Path, n: int = 1) -> list[dict]:
    return [
        append_intent(
            path,
            actor_class="agent-claude",
            intent_type="publish_pr",
            target=TARGET,
            payload={"n": i},
            now=_ts,
        )
        for i in range(n)
    ]


class TestAppendAndVerify:
    def test_append_builds_valid_chain(self, tmp_path: Path) -> None:
        chain = tmp_path / "chain.jsonl"
        records = _append(chain, 3)
        ok, broken = verify_chain(chain)
        assert ok and broken is None
        assert [r["seq"] for r in records] == [0, 1, 2]
        assert chain_head_hash(chain) == records[-1]["record_hash"]

    def test_genesis_prev_hash_is_64_zeros(self, tmp_path: Path) -> None:
        chain = tmp_path / "chain.jsonl"
        (record,) = _append(chain, 1)
        assert record["prev_hash"] == GENESIS_PREV_HASH == "0" * 64

    def test_each_record_links_to_previous(self, tmp_path: Path) -> None:
        chain = tmp_path / "chain.jsonl"
        records = _append(chain, 3)
        assert records[1]["prev_hash"] == records[0]["record_hash"]
        assert records[2]["prev_hash"] == records[1]["record_hash"]

    def test_record_carries_injected_ts_and_intent_id(self, tmp_path: Path) -> None:
        chain = tmp_path / "chain.jsonl"
        record = append_intent(
            chain,
            actor_class="human",
            intent_type="merge_pr",
            target=TARGET,
            intent_id="fixed-id",
            now=lambda: "2026-01-01T00:00:00+00:00",
        )
        assert record["ts"] == "2026-01-01T00:00:00+00:00"
        assert record["intent_id"] == "fixed-id"

    def test_empty_and_missing_chain(self, tmp_path: Path) -> None:
        chain = tmp_path / "missing.jsonl"
        assert read_records(chain) == []
        assert chain_head_hash(chain) is None
        ok, broken = verify_chain(chain)
        assert ok and broken is None


class TestTamperDetection:
    def test_edited_middle_record_breaks_at_that_seq(self, tmp_path: Path) -> None:
        chain = tmp_path / "chain.jsonl"
        _append(chain, 4)
        lines = chain.read_text().splitlines()
        middle = json.loads(lines[1])
        middle["payload"]["n"] = 999  # silent rewrite attempt
        lines[1] = json.dumps(middle)
        chain.write_text("\n".join(lines) + "\n")
        ok, broken = verify_chain(chain)
        assert not ok
        assert broken == 1

    def test_rehashed_middle_record_still_breaks_chain(self, tmp_path: Path) -> None:
        # Adversary edits a record AND recomputes its hash: the next record's
        # prev_hash no longer matches, so the break surfaces at seq 2.
        chain = tmp_path / "chain.jsonl"
        _append(chain, 4)
        lines = chain.read_text().splitlines()
        middle = json.loads(lines[1])
        middle["payload"]["n"] = 999
        middle.pop("record_hash")
        middle["record_hash"] = compute_record_hash(middle)
        lines[1] = json.dumps(middle)
        chain.write_text("\n".join(lines) + "\n")
        ok, broken = verify_chain(chain)
        assert not ok
        assert broken == 2

    def test_truncated_then_extended_tail_breaks(self, tmp_path: Path) -> None:
        chain = tmp_path / "chain.jsonl"
        records = _append(chain, 3)
        lines = chain.read_text().splitlines()
        chain.write_text("\n".join(lines[:2]) + "\n")
        # The shortened chain is internally consistent (detection of pure
        # truncation needs the external anchor)…
        ok, _ = verify_chain(chain)
        assert ok
        # …but the head no longer matches what was (or would have been) anchored.
        assert chain_head_hash(chain) != records[-1]["record_hash"]

    def test_duplicate_seq_rejected(self, tmp_path: Path) -> None:
        chain = tmp_path / "chain.jsonl"
        _append(chain, 2)
        lines = chain.read_text().splitlines()
        chain.write_text("\n".join([*lines, lines[1]]) + "\n")
        ok, broken = verify_chain(chain)
        assert not ok
        assert broken == 1  # claimed seq of the out-of-place record

    def test_out_of_order_seq_rejected(self, tmp_path: Path) -> None:
        chain = tmp_path / "chain.jsonl"
        _append(chain, 3)
        lines = chain.read_text().splitlines()
        chain.write_text("\n".join([lines[0], lines[2], lines[1]]) + "\n")
        ok, broken = verify_chain(chain)
        assert not ok
        assert broken == 2

    def test_corrupt_json_line_fails_verification(self, tmp_path: Path) -> None:
        chain = tmp_path / "chain.jsonl"
        _append(chain, 2)
        with chain.open("a") as fh:
            fh.write("{not json\n")
        with pytest.raises(ChainError):
            read_records(chain)
        ok, _ = verify_chain(chain)
        assert not ok


class TestCanonicalization:
    def test_hash_is_key_order_independent(self) -> None:
        base = {
            "seq": 0,
            "ts": _ts(),
            "actor_class": "agent-claude",
            "intent_type": "publish_pr",
            "target": {"repo": "synaptent/aragora", "ref": "main"},
            "intent_id": "abc",
            "payload": {"b": 2, "a": 1},
            "prev_hash": GENESIS_PREV_HASH,
        }
        shuffled = dict(reversed(list(base.items())))
        shuffled["target"] = {"ref": "main", "repo": "synaptent/aragora"}
        shuffled["payload"] = {"a": 1, "b": 2}
        assert compute_record_hash(base) == compute_record_hash(shuffled)

    def test_unicode_payload_hashes_stably(self, tmp_path: Path) -> None:
        chain = tmp_path / "chain.jsonl"
        record = append_intent(
            chain,
            actor_class="agent-codex",
            intent_type="issue_create",
            target=TARGET,
            payload={"title": "café ☃ ✓"},
            now=_ts,
        )
        # Round-tripping through JSON file storage must reproduce the hash.
        stored = read_records(chain)[0]
        assert compute_record_hash(stored) == record["record_hash"]
        ok, _ = verify_chain(chain)
        assert ok

    def test_extra_unhashed_field_rejected(self) -> None:
        with pytest.raises(ChainError, match="extra fields"):
            compute_record_hash({"seq": 0, "smuggled": "content"})


class TestValidation:
    def test_unknown_actor_class_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ChainError, match="actor_class"):
            append_intent(
                tmp_path / "c.jsonl",
                actor_class="agent-rogue",
                intent_type="publish_pr",
                target=TARGET,
            )

    def test_unknown_intent_type_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ChainError, match="intent_type"):
            append_intent(
                tmp_path / "c.jsonl",
                actor_class="human",
                intent_type="rm_rf",
                target=TARGET,
            )

    def test_target_requires_repo(self, tmp_path: Path) -> None:
        with pytest.raises(ChainError, match="repo"):
            append_intent(
                tmp_path / "c.jsonl",
                actor_class="human",
                intent_type="merge_pr",
                target={"pr": 1},
            )

    def test_enums_match_spec(self) -> None:
        assert ACTOR_CLASSES == {
            "human",
            "agent-claude",
            "agent-codex",
            "agent-app",
            "daemon-publisher",
            "daemon-arbiter",
            "daemon-boss",
        }
        assert INTENT_TYPES == {
            "publish_pr",
            "merge_pr",
            "settle_pr",
            "close_pr",
            "branch_delete",
            "issue_create",
        }


class TestConcurrency:
    def test_concurrent_appends_do_not_corrupt(self, tmp_path: Path) -> None:
        chain = tmp_path / "chain.jsonl"
        threads = [
            threading.Thread(
                target=lambda: [
                    append_intent(
                        chain,
                        actor_class="daemon-publisher",
                        intent_type="publish_pr",
                        target=TARGET,
                        now=_ts,
                    )
                    for _ in range(10)
                ]
            )
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        records = read_records(chain)
        assert len(records) == 40
        assert [r["seq"] for r in records] == list(range(40))
        ok, broken = verify_chain(chain)
        assert ok and broken is None


class TestRecordIntentHelper:
    def test_off_by_default(self, tmp_path: Path) -> None:
        chain = tmp_path / "chain.jsonl"
        result = record_intent(
            actor_class="agent-app",
            intent_type="settle_pr",
            target=TARGET,
            path=chain,
            env={},
        )
        assert result is None
        assert not chain.exists()

    def test_appends_when_enabled(self, tmp_path: Path) -> None:
        chain = tmp_path / "chain.jsonl"
        result = record_intent(
            actor_class="agent-app",
            intent_type="settle_pr",
            target=TARGET,
            payload={"action": "post_quorum_evidence"},
            path=chain,
            env={"ARAGORA_TRAIL": "1"},
        )
        assert result is not None and result["seq"] == 0
        ok, _ = verify_chain(chain)
        assert ok

    def test_never_raises_on_bad_input(self, tmp_path: Path) -> None:
        result = record_intent(
            actor_class="not-a-class",
            intent_type="settle_pr",
            target=TARGET,
            path=tmp_path / "chain.jsonl",
            env={"ARAGORA_TRAIL": "1"},
        )
        assert result is None

    def test_default_path_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_TRAIL_CHAIN", "/tmp/custom-chain.jsonl")
        assert default_chain_path() == Path("/tmp/custom-chain.jsonl")
        monkeypatch.delenv("ARAGORA_TRAIL_CHAIN")
        assert default_chain_path() == Path(".aragora/trail/intent-chain.jsonl")
