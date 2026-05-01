from pathlib import Path

import scripts.harvest_salvage_branches as mod


def test_process_branch_accepts_audit_name_key(monkeypatch):
    def fake_diff_stat(root: Path, base: str, branch: str) -> dict[str, object]:
        assert branch == "codex/example"
        return {
            "added": 12,
            "removed": 1,
            "files": [{"path": "docs/example.md", "added": 12, "removed": 1}],
        }

    def fake_commit_log(root: Path, base: str, branch: str) -> list[dict[str, str]]:
        assert branch == "codex/example"
        return [{"sha": "abc123", "subject": "docs: add example", "body": ""}]

    monkeypatch.setattr(mod, "_diff_stat", fake_diff_stat)
    monkeypatch.setattr(mod, "_commit_log", fake_commit_log)

    decision = mod._process_branch(
        record={
            "name": "codex/example",
            "head_sha": "abc123",
            "category": "salvage_diverged_local",
        },
        root=Path("/repo"),
        base="origin/main",
    )

    assert decision["branch"] == "codex/example"
    assert decision["decision"] == "auto_pr"
    assert decision["reason"] == "matches archetype: docs-only update"
