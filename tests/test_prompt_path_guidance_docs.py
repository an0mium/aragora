"""Regression tests for repo-root guidance in reusable agent prompts."""

from pathlib import Path


def test_recursive_prompts_do_not_assume_home_aragora_symlink():
    repo_root = Path(__file__).resolve().parents[1]
    prompt_paths = [
        repo_root / "docs/prompts/MASTER_FANOUT_PROMPT.md",
        repo_root / "docs/missions/H1_DISCIPLINED_MISSION_PROMPT.md",
    ]

    for path in prompt_paths:
        text = path.read_text(encoding="utf-8")
        assert "`~/aragora`" not in text
        assert "~/aragora" not in text
        assert "repo root" in text
        assert "worktree" in text


def test_stage_gate_conductor_prompt_uses_canonical_log_resolver():
    repo_root = Path(__file__).resolve().parents[1]
    prompt = (repo_root / "docs/prompts/STAGE_GATE_CONDUCTOR_LOG_PROMPT.md").read_text(
        encoding="utf-8"
    )

    assert "resolve_stage_gate_conductor_log_issue" in prompt
    assert "#8671" in prompt
    assert "#7162" not in prompt
