"""Tests for ``scripts/send_operator_steering.py`` — Phase B mailbox writer.

Fixture-driven; never touches the real
``.aragora/operator-steering/`` directory. All writes go to
``tmp_path`` via the ``--steering-inbox-root`` override.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest


def _load_module() -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "send_operator_steering.py"
    spec = importlib.util.spec_from_file_location("send_operator_steering_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sos = _load_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inbox_dir(tmp_path: Path, recipient: str) -> Path:
    return tmp_path / recipient


def _list_messages(tmp_path: Path, recipient: str) -> list[Path]:
    inbox = _inbox_dir(tmp_path, recipient)
    if not inbox.is_dir():
        return []
    return sorted(inbox.glob("*.json"))


# ---------------------------------------------------------------------------
# Schema + sha256 round-trip
# ---------------------------------------------------------------------------


class TestSchemaShape:
    def test_happy_path_writes_v1_schema_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = sos.main(
            [
                "--to",
                "fixture-session-1",
                "--body",
                "first steering message",
                "--steering-inbox-root",
                str(tmp_path),
            ]
        )
        assert rc == 0
        msgs = _list_messages(tmp_path, "fixture-session-1")
        assert len(msgs) == 1
        payload = json.loads(msgs[0].read_text(encoding="utf-8"))
        assert payload["schema_version"] == "aragora-operator-steering/1.0"
        assert payload["to_session"] == "fixture-session-1"
        assert payload["from"] == "operator"
        assert payload["priority"] == "normal"
        assert payload["subject"] == "first steering message"
        assert payload["body"] == "first steering message"
        assert payload["lane_id_hint"] is None
        assert payload["pr_hint"] is None
        assert payload["message_sha256"]
        assert len(payload["message_sha256"]) == 64

    def test_message_sha256_round_trips(self, tmp_path: Path) -> None:
        sos.main(
            [
                "--to",
                "fixture-roundtrip",
                "--body",
                "verify the binding sha",
                "--from",
                "operator-test",
                "--priority",
                "high",
                "--lane-id",
                "P29-fixture",
                "--pr",
                "9999",
                "--steering-inbox-root",
                str(tmp_path),
            ]
        )
        msgs = _list_messages(tmp_path, "fixture-roundtrip")
        assert len(msgs) == 1
        payload = json.loads(msgs[0].read_text(encoding="utf-8"))
        claimed, recomputed, matches = sos.verify_message_sha256(payload)
        assert claimed == payload["message_sha256"]
        assert recomputed == claimed
        assert matches is True


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestArgValidation:
    def test_missing_to_exits_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = sos.main(["--body", "no recipient", "--steering-inbox-root", str(tmp_path)])
        assert rc == 2

    def test_missing_body_exits_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = sos.main(["--to", "x", "--steering-inbox-root", str(tmp_path)])
        assert rc == 2

    def test_both_body_and_body_file_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        body_file = tmp_path / "body.md"
        body_file.write_text("from file")
        rc = sos.main(
            [
                "--to",
                "x",
                "--body",
                "from inline",
                "--body-file",
                str(body_file),
                "--steering-inbox-root",
                str(tmp_path),
            ]
        )
        assert rc == 2

    def test_missing_body_file_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = sos.main(
            [
                "--to",
                "x",
                "--body-file",
                str(tmp_path / "absent.md"),
                "--steering-inbox-root",
                str(tmp_path),
            ]
        )
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_empty_body_exits_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = sos.main(
            [
                "--to",
                "x",
                "--body",
                "   \n   ",  # whitespace-only
                "--steering-inbox-root",
                str(tmp_path),
            ]
        )
        assert rc == 2
        assert "empty" in capsys.readouterr().err

    def test_invalid_priority_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = sos.main(
            [
                "--to",
                "x",
                "--body",
                "hi",
                "--priority",
                "URGENT",  # not in PRIORITY_CHOICES
                "--steering-inbox-root",
                str(tmp_path),
            ]
        )
        assert rc == 2


class TestRecipientPathSafety:
    def test_empty_recipient_fails_before_writing(self, tmp_path: Path) -> None:
        rc = sos.main(
            [
                "--to",
                "",
                "--body",
                "must not write",
                "--steering-inbox-root",
                str(tmp_path / "inbox"),
            ]
        )
        assert rc == 2
        assert not (tmp_path / "inbox").exists()
        assert list(tmp_path.rglob("*.json")) == []

    @pytest.mark.parametrize(
        "recipient",
        [
            "../escape",
            "nested/session",
            "nested\\session",
            "/tmp/escape",
            ".",
            "..",
            " session-with-space-prefix",
            "session-with-space-suffix ",
        ],
    )
    def test_unsafe_recipients_fail_before_writing(self, tmp_path: Path, recipient: str) -> None:
        rc = sos.main(
            [
                "--to",
                recipient,
                "--body",
                "must not escape",
                "--steering-inbox-root",
                str(tmp_path / "inbox"),
            ]
        )
        assert rc == 2
        assert not (tmp_path / "escape").exists()
        assert not (tmp_path / "inbox").exists()
        assert list(tmp_path.rglob("*.json")) == []

    def test_validator_rejects_resolving_outside_root(self, tmp_path: Path) -> None:
        root = tmp_path / "inbox"
        with pytest.raises(ValueError, match="path separators|plain session identifier|outside"):
            sos.validate_to_session("../escape", steering_inbox_root=root)
        assert not root.exists()
        assert not (tmp_path / "escape").exists()


# ---------------------------------------------------------------------------
# Subject derivation + body-file path
# ---------------------------------------------------------------------------


class TestSubjectAndBodyFile:
    def test_subject_is_first_80_chars(self, tmp_path: Path) -> None:
        long_body = "x" * 200
        sos.main(
            [
                "--to",
                "subj-fixture",
                "--body",
                long_body,
                "--steering-inbox-root",
                str(tmp_path),
            ]
        )
        msg = json.loads(_list_messages(tmp_path, "subj-fixture")[0].read_text())
        assert len(msg["subject"]) == 80
        assert msg["subject"] == "x" * 80
        assert msg["body"] == "x" * 200

    def test_subject_is_first_line_only(self, tmp_path: Path) -> None:
        body = "single-line subject\nbody line 2\nbody line 3"
        sos.main(
            [
                "--to",
                "subj-multi",
                "--body",
                body,
                "--steering-inbox-root",
                str(tmp_path),
            ]
        )
        msg = json.loads(_list_messages(tmp_path, "subj-multi")[0].read_text())
        assert msg["subject"] == "single-line subject"
        assert msg["body"] == body  # full body preserved

    def test_body_file_is_read(self, tmp_path: Path) -> None:
        body_file = tmp_path / "msg.md"
        body_file.write_text("# Heading\n\nfile body content\n", encoding="utf-8")
        sos.main(
            [
                "--to",
                "body-file-fixture",
                "--body-file",
                str(body_file),
                "--steering-inbox-root",
                str(tmp_path),
            ]
        )
        msg = json.loads(_list_messages(tmp_path, "body-file-fixture")[0].read_text())
        assert msg["body"] == "# Heading\n\nfile body content\n"
        assert msg["subject"] == "# Heading"


# ---------------------------------------------------------------------------
# Ordering + directory creation
# ---------------------------------------------------------------------------


class TestOrderingAndDir:
    def test_dir_auto_created_for_new_recipient(self, tmp_path: Path) -> None:
        # Recipient dir does NOT exist beforehand.
        assert not (tmp_path / "new-recipient").exists()
        sos.main(
            [
                "--to",
                "new-recipient",
                "--body",
                "first message ever",
                "--steering-inbox-root",
                str(tmp_path),
            ]
        )
        assert (tmp_path / "new-recipient").is_dir()
        assert len(_list_messages(tmp_path, "new-recipient")) == 1

    def test_idempotent_on_rerun_same_recipient(self, tmp_path: Path) -> None:
        # Two writes in succession both succeed; both messages persist.
        for body in ("msg-A", "msg-B"):
            sos.main(
                [
                    "--to",
                    "rerun-fixture",
                    "--body",
                    body,
                    "--steering-inbox-root",
                    str(tmp_path),
                ]
            )
        msgs = _list_messages(tmp_path, "rerun-fixture")
        assert len(msgs) == 2
        bodies = sorted(json.loads(m.read_text())["body"] for m in msgs)
        assert bodies == ["msg-A", "msg-B"]

    def test_multi_message_ordering_filenames_strictly_increasing(self, tmp_path: Path) -> None:
        # The filename starts with the message's sent_at_utc; messages
        # written later should sort after messages written earlier.
        for i in range(3):
            sos.main(
                [
                    "--to",
                    "ordering-fixture",
                    "--body",
                    f"message-{i}",
                    "--steering-inbox-root",
                    str(tmp_path),
                ]
            )
            time.sleep(0.005)  # ensure distinct sent_at_utc timestamps
        msgs = _list_messages(tmp_path, "ordering-fixture")
        assert len(msgs) == 3
        # Sorted by filename should match sorted by sent_at_utc inside
        # each message. zip without strict because adjacent-pair count
        # is naturally len(msgs)-1.
        for prev, curr in zip(msgs, msgs[1:]):
            prev_payload = json.loads(prev.read_text())
            curr_payload = json.loads(curr.read_text())
            assert prev_payload["sent_at_utc"] <= curr_payload["sent_at_utc"]

    def test_tmp_files_do_not_appear_in_glob(self, tmp_path: Path) -> None:
        sos.main(
            [
                "--to",
                "tmp-leak-fixture",
                "--body",
                "atomic-write check",
                "--steering-inbox-root",
                str(tmp_path),
            ]
        )
        inbox = _inbox_dir(tmp_path, "tmp-leak-fixture")
        # Real consumers do `glob("*.json")` — only completed messages
        # should match. The atomic-write tmp prefix starts with '.tmp-'
        # which neither matches *.json nor pollutes a glob.
        completed = list(inbox.glob("*.json"))
        assert len(completed) == 1
        leftover_tmp = list(inbox.glob(".tmp-*"))
        assert leftover_tmp == []


# ---------------------------------------------------------------------------
# --json output
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_json_includes_written_path_and_full_record(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = sos.main(
            [
                "--to",
                "json-fixture",
                "--body",
                "json mode body",
                "--from",
                "operator-test",
                "--lane-id",
                "P29-fixture",
                "--pr",
                "9000",
                "--priority",
                "high",
                "--json",
                "--steering-inbox-root",
                str(tmp_path),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["schema_version"] == "aragora-operator-steering/1.0"
        assert parsed["to_session"] == "json-fixture"
        assert parsed["from"] == "operator-test"
        assert parsed["lane_id_hint"] == "P29-fixture"
        assert parsed["pr_hint"] == 9000
        assert parsed["priority"] == "high"
        assert parsed["_written_path"].endswith(".json")
        # The written file should match the JSON-mode output (sans the
        # injected _written_path field).
        written_file = Path(parsed["_written_path"])
        assert written_file.is_file()
        on_disk = json.loads(written_file.read_text())
        for k in (
            "schema_version",
            "to_session",
            "from",
            "lane_id_hint",
            "pr_hint",
            "priority",
            "body",
            "subject",
            "message_sha256",
        ):
            assert on_disk[k] == parsed[k]


# ---------------------------------------------------------------------------
# Active-owner routing
# ---------------------------------------------------------------------------


class TestActiveOwnerRouting:
    def _write_lanes(self, tmp_path: Path, lanes: list[dict[str, Any]]) -> Path:
        registry = tmp_path / "lanes.json"
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(json.dumps(lanes), encoding="utf-8")
        return registry

    def test_to_owner_pr_routes_to_single_active_owner(self, tmp_path: Path) -> None:
        registry = self._write_lanes(
            tmp_path,
            [
                {
                    "lane_id": "old-7292",
                    "owner_session": "codex-old",
                    "status": "completed",
                    "pr_number": 7292,
                    "updated_at": "2026-05-18T01:00:00Z",
                },
                {
                    "lane_id": "q23-repair-7292",
                    "owner_session": "codex-q23",
                    "status": "active",
                    "pr_number": 7292,
                    "branch": "droid/P16-stage2",
                    "updated_at": "2026-05-19T16:05:53Z",
                },
            ],
        )

        rc = sos.main(
            [
                "--to-owner-pr",
                "7292",
                "--body",
                "continue only if you own Q23",
                "--json",
                "--lane-registry-path",
                str(registry),
                "--steering-inbox-root",
                str(tmp_path / "inbox"),
            ]
        )

        assert rc == 0
        msgs = _list_messages(tmp_path / "inbox", "codex-q23")
        assert len(msgs) == 1
        payload = json.loads(msgs[0].read_text(encoding="utf-8"))
        assert payload["to_session"] == "codex-q23"

    def test_default_owner_resolution_reads_user_and_repo_registries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_registry = self._write_lanes(tmp_path / "repo", [])
        user_registry = self._write_lanes(
            tmp_path / "user",
            [
                {
                    "lane_id": "q23-repair-7292",
                    "owner_session": "codex-q23",
                    "status": "active",
                    "pr_number": 7292,
                    "updated_at": "2026-05-19T16:05:53Z",
                }
            ],
        )
        monkeypatch.setattr(sos, "LANE_REGISTRY_DEFAULT", repo_registry)
        monkeypatch.setattr(sos, "USER_LANE_REGISTRY_DEFAULT", user_registry)

        resolved = sos.resolve_active_owner(pr=7292)

        assert resolved["lane_id"] == "q23-repair-7292"
        assert resolved["owner_session"] == "codex-q23"

    def test_to_owner_pr_json_includes_route(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        registry = self._write_lanes(
            tmp_path,
            [
                {
                    "lane_id": "q23-repair-7292",
                    "owner_session": "codex-q23",
                    "status": "active",
                    "pr_number": 7292,
                    "updated_at": "2026-05-19T16:05:53Z",
                }
            ],
        )

        rc = sos.main(
            [
                "--to-owner-pr",
                "7292",
                "--body",
                "route metadata please",
                "--json",
                "--lane-registry-path",
                str(registry),
                "--steering-inbox-root",
                str(tmp_path / "inbox"),
            ]
        )

        assert rc == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["_route"]["resolved_via"] == "pr"
        assert parsed["_route"]["lane_id"] == "q23-repair-7292"
        assert parsed["_route"]["owner_session"] == "codex-q23"

    def test_to_owner_pr_rejects_completed_only_owner(self, tmp_path: Path) -> None:
        registry = self._write_lanes(
            tmp_path,
            [
                {
                    "lane_id": "q23-repair-7292",
                    "owner_session": "codex-q23",
                    "status": "completed",
                    "pr_number": 7292,
                    "updated_at": "2026-05-19T16:16:29Z",
                }
            ],
        )

        rc = sos.main(
            [
                "--to-owner-pr",
                "7292",
                "--body",
                "must not route to completed lane",
                "--lane-registry-path",
                str(registry),
                "--steering-inbox-root",
                str(tmp_path / "inbox"),
            ]
        )

        assert rc == 2
        assert list((tmp_path / "inbox").rglob("*.json")) == []

    def test_to_owner_pr_rejects_duplicate_active_owners(self, tmp_path: Path) -> None:
        registry = self._write_lanes(
            tmp_path,
            [
                {
                    "lane_id": "lane-a",
                    "owner_session": "codex-a",
                    "status": "active",
                    "pr_number": 7292,
                    "updated_at": "2026-05-19T16:00:00Z",
                },
                {
                    "lane_id": "lane-b",
                    "owner_session": "codex-b",
                    "status": "active",
                    "pr_number": 7292,
                    "updated_at": "2026-05-19T16:01:00Z",
                },
            ],
        )

        rc = sos.main(
            [
                "--to-owner-pr",
                "7292",
                "--body",
                "must not route to ambiguous owners",
                "--lane-registry-path",
                str(registry),
                "--steering-inbox-root",
                str(tmp_path / "inbox"),
            ]
        )

        assert rc == 2
        assert list((tmp_path / "inbox").rglob("*.json")) == []


# ---------------------------------------------------------------------------
# Build / verify helpers exposed for downstream callers
# ---------------------------------------------------------------------------


class TestBuildVerifyHelpers:
    def test_build_message_stamps_canonical_sha(self) -> None:
        msg = sos.build_message(
            to_session="target",
            body="hello canonical",
            sent_at_utc="2026-05-18T05:00:00.000Z",
        )
        verify_copy = {k: v for k, v in msg.items() if k != "message_sha256"}
        expected = hashlib.sha256(sos.canonical_json(verify_copy).encode("utf-8")).hexdigest()
        assert msg["message_sha256"] == expected

    def test_verify_detects_tampered_message(self) -> None:
        msg = sos.build_message(to_session="t", body="original body")
        msg["body"] = "tampered body"
        claimed, recomputed, matches = sos.verify_message_sha256(msg)
        assert matches is False
        assert claimed != recomputed
