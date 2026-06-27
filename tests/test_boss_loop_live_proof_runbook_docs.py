"""Regression tests for boss-loop live proof runbook command snippets."""

from pathlib import Path


def test_live_proof_launch_uses_runtime_resolving_wrapper():
    repo_root = Path(__file__).resolve().parents[1]
    runbook = (repo_root / "docs/briefs/boss-loop-live-proof-runbook.md").read_text(
        encoding="utf-8"
    )

    assert (
        "ARAGORA_USER_ID=an0mium ARAGORA_POST_LOOP_ISSUE_REFILL=0 ./scripts/run_boss_cycle.sh \\"
    ) in runbook
    assert "python -m aragora.cli.main swarm boss-loop \\" not in runbook
    assert "python3 -m aragora.cli.main swarm boss-loop \\" not in runbook
