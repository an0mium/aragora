"""Tests for claude_profile.sh subscription-pool routing of the debate Claude agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aragora.agents.claude_profile_pool import (
    build_claude_command,
    select_profile,
    strip_profile_preamble,
)

BASE_CMD = ["claude", "--print", "-p", "-"]


@pytest.fixture(autouse=True)
def _hermetic_pool_env(monkeypatch):
    """Isolate tests from the host's real Claude-pool configuration."""
    for var in (
        "ARAGORA_CLAUDE_REVIEW_PROFILES",
        "ARAGORA_CLAUDE_PROFILE",
        "ARAGORA_CLAUDE_DISABLE_PROFILE_POOL",
        "ARAGORA_CLAUDE_POOL_HEALTH_FILE",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_repo(tmp_path: Path, *, with_script: bool = True, health: dict | None = None) -> Path:
    if with_script:
        scripts = tmp_path / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / "claude_profile.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    if health is not None:
        hp = tmp_path / ".aragora" / "claude_pool_health.json"
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text(json.dumps(health), encoding="utf-8")
    return tmp_path


def test_build_command_wraps_in_profile_when_pool_available(tmp_path, monkeypatch):
    monkeypatch.delenv("ARAGORA_CLAUDE_DISABLE_PROFILE_POOL", raising=False)
    monkeypatch.delenv("ARAGORA_CLAUDE_PROFILE", raising=False)
    repo = _make_repo(tmp_path)

    command, used_profile = build_claude_command(BASE_CMD, repo_root=repo, index=0)

    assert used_profile is True
    assert command[0] == str(repo / "scripts" / "claude_profile.sh")
    assert command[1] == "exec"
    assert command[2].startswith("max-")
    assert command[3] == "--"
    assert command[4:] == BASE_CMD


def test_build_command_is_bare_when_script_absent(tmp_path):
    repo = _make_repo(tmp_path, with_script=False)

    command, used_profile = build_claude_command(BASE_CMD, repo_root=repo, index=0)

    assert used_profile is False
    assert command == BASE_CMD


def test_disable_env_forces_bare_command(tmp_path, monkeypatch):
    monkeypatch.setenv("ARAGORA_CLAUDE_DISABLE_PROFILE_POOL", "1")
    repo = _make_repo(tmp_path)

    command, used_profile = build_claude_command(BASE_CMD, repo_root=repo, index=0)

    assert used_profile is False
    assert command == BASE_CMD


def test_explicit_profile_override_is_honored(tmp_path, monkeypatch):
    monkeypatch.setenv("ARAGORA_CLAUDE_PROFILE", "max-07")
    repo = _make_repo(tmp_path)

    command, used_profile = build_claude_command(BASE_CMD, repo_root=repo, index=0)

    assert used_profile is True
    assert command[2] == "max-07"


def test_unhealthy_profiles_are_dropped(tmp_path, monkeypatch):
    monkeypatch.delenv("ARAGORA_CLAUDE_PROFILE", raising=False)
    # Only max-13 healthy; the rest expired/logged out.
    profiles = [{"profile": f"max-{i:02d}", "state": "expired"} for i in range(1, 13)]
    profiles.append({"profile": "max-13", "state": "ok"})
    repo = _make_repo(
        tmp_path, health={"generated_at": "2026-06-04T00:00:00Z", "profiles": profiles}
    )

    # Any rotation index must resolve to the single healthy profile.
    for idx in (0, 1, 5, 12, 99):
        assert select_profile(repo_root=repo, index=idx) == "max-13"


def test_flat_health_mapping_shape_is_supported(tmp_path, monkeypatch):
    monkeypatch.delenv("ARAGORA_CLAUDE_PROFILE", raising=False)
    health = {"generated_at": "x", "max-01": "expired", "max-02": "ok", "max-03": "ok"}
    monkeypatch.setenv("ARAGORA_CLAUDE_REVIEW_PROFILES", "max-01,max-02,max-03")
    repo = _make_repo(tmp_path, health=health)

    chosen = {select_profile(repo_root=repo, index=i) for i in range(6)}

    assert "max-01" not in chosen
    assert chosen <= {"max-02", "max-03"}


def test_rotation_spreads_across_healthy_profiles(tmp_path, monkeypatch):
    monkeypatch.delenv("ARAGORA_CLAUDE_PROFILE", raising=False)
    monkeypatch.setenv("ARAGORA_CLAUDE_REVIEW_PROFILES", "max-01,max-02")
    repo = _make_repo(tmp_path)  # no health file -> all configured profiles usable

    assert select_profile(repo_root=repo, index=0) == "max-01"
    assert select_profile(repo_root=repo, index=1) == "max-02"
    assert select_profile(repo_root=repo, index=2) == "max-01"


def test_strip_profile_preamble_removes_wrapper_lines():
    raw = "Using profile home: /home/x/.aragora-claude/max-01\nCommand: claude --print\nactual answer\nline two"
    assert strip_profile_preamble(raw) == "actual answer\nline two"


def test_strip_profile_preamble_noop_on_plain_text():
    assert strip_profile_preamble("just a response\nsecond line") == "just a response\nsecond line"


def test_all_unhealthy_snapshot_yields_no_profile(tmp_path, monkeypatch):
    """Regression: when the verify snapshot marks every configured profile
    unhealthy, selection must return None (=> bare claude => OpenRouter fallback)
    rather than handing back a known-bad profile and defeating the safety gate.
    """
    monkeypatch.setenv("ARAGORA_CLAUDE_REVIEW_PROFILES", "max-01,max-02,max-03")
    profiles = [
        {"profile": "max-01", "state": "expired"},
        {"profile": "max-02", "state": "logged_out"},
        {"profile": "max-03", "state": "unauthenticated"},
    ]
    repo = _make_repo(
        tmp_path, health={"generated_at": "2026-06-04T00:00:00Z", "profiles": profiles}
    )

    assert select_profile(repo_root=repo, index=0) is None
    command, used_profile = build_claude_command(BASE_CMD, repo_root=repo, index=0)
    assert used_profile is False
    assert command == BASE_CMD


def test_profiles_absent_from_snapshot_are_treated_as_usable(tmp_path, monkeypatch):
    """A profile not listed in the snapshot (unknown state) stays selectable."""
    monkeypatch.setenv("ARAGORA_CLAUDE_REVIEW_PROFILES", "max-01,max-09")
    # Snapshot only knows max-01 (expired); max-09 is unlisted => unknown => usable.
    repo = _make_repo(
        tmp_path,
        health={"generated_at": "x", "profiles": [{"profile": "max-01", "state": "expired"}]},
    )

    assert select_profile(repo_root=repo, index=0) == "max-09"


def test_strip_preamble_preserves_body_lines_matching_prefixes():
    """Only the leading wrapper block is stripped; a model answer that itself
    contains a 'Command:' line (past the preamble) must be preserved."""
    raw = (
        "Using profile home: /home/x/.aragora-claude/max-01\n"
        "Command: claude --print -p -\n"
        "To run the migration:\n"
        "Command: python manage.py migrate\n"
        "Using profile home: is also a valid sentence here."
    )
    assert strip_profile_preamble(raw) == (
        "To run the migration:\n"
        "Command: python manage.py migrate\n"
        "Using profile home: is also a valid sentence here."
    )


def test_strip_preamble_without_leading_wrapper_is_untouched():
    """If output does not start with the wrapper block, nothing is stripped."""
    raw = "Command: this is the model's first line\nsecond line"
    assert strip_profile_preamble(raw) == raw
