"""Regression tests for copy-paste automation merge contract snippets."""

from pathlib import Path


def test_boss_loop_snippet_uses_repo_root_coordination_db():
    repo_root = Path(__file__).resolve().parents[1]
    contract = (repo_root / "docs/briefs/automation-merge-contract.md").read_text(encoding="utf-8")

    assert 'export ARAGORA_REPO_ROOT="${ARAGORA_REPO_ROOT:-$PWD}"' in contract
    assert (
        'export ARAGORA_DEV_COORDINATION_DB="${ARAGORA_REPO_ROOT}/.aragora/dev_coordination.sqlite3"'
        in contract
    )
    assert "ARAGORA_POST_LOOP_ISSUE_REFILL=0 ./scripts/run_boss_cycle.sh \\" in contract
    assert "ARAGORA_DEV_COORDINATION_DB=~/aragora/.aragora/dev_coordination.sqlite3" not in contract
    assert "python3.11 -u -m aragora.cli.main swarm boss-loop" not in contract
