"""Tests for scripts/agent_session_digest.py (cross-agent session digest)."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

_SPEC = importlib.util.spec_from_file_location(
    "agent_session_digest",
    Path(__file__).resolve().parents[2] / "scripts" / "agent_session_digest.py",
)
assert _SPEC and _SPEC.loader
digest = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(digest)


def _write_rollout(tmp_path: Path) -> Path:
    rows = [
        {"type": "session_meta"},
        {
            "type": "response_item",
            "payload": {"role": "user", "content": "Repair PR #8718 dissent"},
        },
        {
            "type": "response_item",
            "payload": {
                "role": "assistant",
                "content": [{"text": "I'm editing the #8718 branch conservatively."}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"python3 -m pytest tests/swarm/test_queue_disposition.py","pr":"8718"}',
            },
        },
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "done"}},
    ]
    p = tmp_path / "rollout-test.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_extract_turns_pulls_prompts_decisions_commands_and_prs(tmp_path: Path) -> None:
    turns = digest.extract_turns(_write_rollout(tmp_path))
    assert turns["counts"]["prompts"] == 1
    assert turns["counts"]["decisions"] == 2
    assert turns["counts"]["commands"] == 1
    assert "8718" in turns["prs_referenced"]
    assert "pytest" in turns["commands"][0]
    assert "done" in turns["decisions"]


def test_extract_turns_streams_rollout_without_read_text(tmp_path: Path, monkeypatch) -> None:
    rollout = _write_rollout(tmp_path)

    def fail_read_text(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError(f"read_text should not be used for {self}")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    turns = digest.extract_turns(rollout)

    assert turns["counts"]["commands"] == 1
    assert "8718" in turns["prs_referenced"]


def test_extract_turns_parses_structured_function_call_arguments(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout-2026-06-30T12-00-00-019f197d.jsonl"
    rows = [
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "arguments": {
                    "cmd": "python3 -m pytest tests/scripts/test_agent_session_digest.py",
                    "pr": "8730",
                },
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "arguments": json.dumps(
                    {
                        "cmd": "gh pr view 8731 --json number",
                        "pr_number": 8731,
                    }
                ),
            },
        },
    ]
    rollout.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    turns = digest.extract_turns(rollout)

    assert turns["counts"]["commands"] == 2
    assert "pytest" in turns["commands"][0]
    assert "8730" in turns["prs_referenced"]
    assert "8731" in turns["prs_referenced"]


def test_rollout_session_id_preserves_full_uuid(tmp_path: Path) -> None:
    rollout = tmp_path / ("rollout-2026-06-15T15-40-37-019ecd03-c60c-7bc1-8b9c-6477893310f6.jsonl")
    rollout.write_text("", encoding="utf-8")

    turns = digest.extract_turns(rollout)

    assert turns["session_id"] == "019ecd03-c60c-7bc1-8b9c-6477893310f6"


def test_default_sessions_root_honors_aragora_codex_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARAGORA_CODEX_HOME", str(tmp_path))

    assert digest.default_sessions_root() == tmp_path / "sessions"


def test_find_rollout_latest_and_session_and_missing(tmp_path: Path) -> None:
    p = _write_rollout(tmp_path)
    # rename to a session-id-bearing name
    sess = tmp_path / "rollout-2026-06-30T12-00-00-019f197d.jsonl"
    p.rename(sess)
    assert digest.find_rollout(path=None, session="019f197d", latest=False, root=tmp_path) == sess
    assert digest.find_rollout(path=None, session=None, latest=True, root=tmp_path) == sess
    assert digest.find_rollout(path=None, session="nope", latest=False, root=tmp_path) is None


def test_coordinator_view_windows_by_mtime(tmp_path: Path) -> None:
    import os

    sess = tmp_path / "rollout-2026-06-30T12-00-00-019f197d.jsonl"
    _write_rollout(tmp_path).rename(sess)
    rows = digest.coordinator_view(root=tmp_path, since_hours=24.0)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "019f197d"
    assert "8718" in rows[0]["prs_referenced"]
    future = os.path.getmtime(sess) + 10 * 3600
    assert digest.coordinator_view(root=tmp_path, since_hours=1.0, now=future) == []


def test_rlm_summary_cleans_temporary_files(tmp_path: Path, monkeypatch) -> None:
    created: list[Path] = []

    class FakeTemporaryDirectory:
        def __init__(self, prefix: str) -> None:
            self.path = tmp_path / prefix.rstrip("-")

        def __enter__(self) -> str:
            self.path.mkdir()
            created.append(self.path)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            for child in self.path.iterdir():
                child.unlink()
            self.path.rmdir()

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        if "compress" in cmd:
            Path(cmd[cmd.index("-o") + 1]).write_text("{}", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="")
        if "query" in cmd:
            return SimpleNamespace(returncode=0, stdout="summary")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setattr(digest.subprocess, "run", fake_run)

    summary = digest.rlm_summary({"decisions": ["done"], "commands": ["pytest"]}, "what")

    assert summary == "summary"
    assert created
    assert all(not path.exists() for path in created)


def test_rlm_summary_children_inherit_repo_pythonpath(tmp_path: Path, monkeypatch) -> None:
    pythonpaths: list[str] = []

    class FakeTemporaryDirectory:
        def __init__(self, prefix: str) -> None:
            self.path = tmp_path / prefix.rstrip("-")

        def __enter__(self) -> str:
            self.path.mkdir()
            return str(self.path)

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            for child in self.path.iterdir():
                child.unlink()
            self.path.rmdir()

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        env = kwargs.get("env") or {}
        pythonpaths.append(env.get("PYTHONPATH", ""))
        if "compress" in cmd:
            Path(cmd[cmd.index("-o") + 1]).write_text("{}", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="")
        if "query" in cmd:
            return SimpleNamespace(returncode=0, stdout="summary")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setattr(digest.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "/existing/path")

    summary = digest.rlm_summary({"decisions": ["done"], "commands": ["pytest"]}, "what")

    assert summary == "summary"
    assert len(pythonpaths) == 2
    for value in pythonpaths:
        paths = value.split(os.pathsep)
        assert paths[0] == str(digest.REPO_ROOT)
        assert "/existing/path" in paths


def test_extract_rlm_answer_strips_cli_chrome() -> None:
    stdout = "\n".join(
        [
            "Loaded context with 3 nodes",
            "",
            "Query: what happened",
            "Strategy: auto",
            "",
            "-" * 60,
            "",
            "=" * 60,
            "ANSWER",
            "=" * 60,
            "The agent fixed PR #8730 and ran the tests.",
            "",
            "-" * 60,
            "Ready: True",
            "Confidence: 91.0%",
            "Tokens processed: 1,234",
        ]
    )
    assert digest._extract_rlm_answer(stdout) == "The agent fixed PR #8730 and ran the tests."


def test_extract_rlm_answer_falls_back_without_marker() -> None:
    assert digest._extract_rlm_answer("plain text, no banner") == "plain text, no banner"
    assert digest._extract_rlm_answer("   ") is None


def test_pr_ids_ignore_unicode_digits() -> None:
    # "²" passes str.isdigit() but crashes int(); must be rejected, not collected.
    assert digest._pr_ids_from_call_args({"pr": "²"}) == set()
    assert digest._pr_ids_from_call_args({"pr": "8730"}) == {"8730"}


def test_main_all_mode(tmp_path: Path, capsys) -> None:
    sess = tmp_path / "rollout-2026-06-30T12-00-00-019f197d.jsonl"
    _write_rollout(tmp_path).rename(sess)
    rc = digest.main(["--all", "--sessions-root", str(tmp_path), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert any(r["session_id"] == "019f197d" for r in out)


def test_extract_turns_reads_user_message_events(tmp_path: Path) -> None:
    """Codex stores human prompts as event_msg/user_message — must be captured."""
    rollout = tmp_path / "rollout-2026-06-30T12-00-00-usermsg.jsonl"
    rows = [
        {"type": "event_msg", "payload": {"type": "user_message", "message": "fix PR #8730"}},
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "on it"}},
    ]
    rollout.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    turns = digest.extract_turns(rollout)
    assert turns["counts"]["prompts"] == 1
    assert "fix PR #8730" in turns["prompts"]
    assert "8730" in turns["prs_referenced"]


def test_extract_turns_dedupes_message_in_both_representations(tmp_path: Path) -> None:
    """A turn present as BOTH event_msg and response_item is counted once."""
    rollout = tmp_path / "rollout-2026-06-30T12-00-00-dedupe.jsonl"
    rows = [
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "same decision"}},
        {
            "type": "response_item",
            "payload": {"role": "assistant", "content": [{"text": "same decision"}]},
        },
    ]
    rollout.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    turns = digest.extract_turns(rollout)
    assert turns["counts"]["decisions"] == 1


def test_extract_turns_reads_session_id_from_payload(tmp_path: Path) -> None:
    """Canonical rollouts nest the id under payload — the meta-id path must work."""
    rollout = tmp_path / "rollout-2026-06-30T12-00-00-metaid.jsonl"
    rows = [{"type": "session_meta", "payload": {"id": "canonical-sid"}}]
    rollout.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    turns = digest.extract_turns(rollout)
    assert turns["session_id"] == "canonical-sid"


def test_iter_rollouts_limit_is_respected(tmp_path: Path) -> None:
    """The scan cap is injectable; limit=None is exhaustive (session lookups need it)."""
    import os

    for i in range(3):
        f = tmp_path / f"rollout-2026-06-30T12-00-0{i}-sess{i}.jsonl"
        f.write_text("", encoding="utf-8")
        os.utime(f, (1000 + i, 1000 + i))
    assert len(list(digest._iter_rollouts(tmp_path, limit=2))) == 2
    assert len(list(digest._iter_rollouts(tmp_path, limit=None))) == 3
    # An older session (not among the newest) is still resolvable by --session.
    assert digest.find_rollout(path=None, session="sess0", latest=False, root=tmp_path) is not None


def test_main_json_output(tmp_path: Path, capsys) -> None:
    sess = tmp_path / "rollout-2026-06-30T12-00-00-019f197d.jsonl"
    _write_rollout(tmp_path).rename(sess)
    rc = digest.main(["--session", "019f197d", "--sessions-root", str(tmp_path), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["counts"]["commands"] == 1
    assert out["rlm_summary"] is None  # --rlm not passed
