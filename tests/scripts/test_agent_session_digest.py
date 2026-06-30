"""Tests for scripts/agent_session_digest.py (cross-agent session digest)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "agent_session_digest",
    Path(__file__).resolve().parents[2] / "scripts" / "agent_session_digest.py",
)
assert _SPEC and _SPEC.loader
digest = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(digest)


def _write_rollout(tmp_path: Path) -> Path:
    rows = [
        {"type": "session_meta", "id": "test"},
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
    assert turns["counts"]["decisions"] == 1
    assert turns["counts"]["commands"] == 1
    assert "8718" in turns["prs_referenced"]
    assert "pytest" in turns["commands"][0]


def test_find_rollout_latest_and_session_and_missing(tmp_path: Path) -> None:
    p = _write_rollout(tmp_path)
    # rename to a session-id-bearing name
    sess = tmp_path / "rollout-2026-06-30T12-00-00-019f197d.jsonl"
    p.rename(sess)
    assert digest.find_rollout(path=None, session="019f197d", latest=False, root=tmp_path) == sess
    assert digest.find_rollout(path=None, session=None, latest=True, root=tmp_path) == sess
    assert digest.find_rollout(path=None, session="nope", latest=False, root=tmp_path) is None


def test_main_json_output(tmp_path: Path, capsys) -> None:
    sess = tmp_path / "rollout-2026-06-30T12-00-00-019f197d.jsonl"
    _write_rollout(tmp_path).rename(sess)
    rc = digest.main(["--session", "019f197d", "--sessions-root", str(tmp_path), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["counts"]["commands"] == 1
    assert out["rlm_summary"] is None  # --rlm not passed
