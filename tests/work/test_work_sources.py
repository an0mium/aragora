from __future__ import annotations

import json
from pathlib import Path

from aragora.work.scoring import build_recommendations
from aragora.work.sources import collect_automation_outbox


def test_collect_automation_outbox_preserves_rich_publication_metadata(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    payload = {
        "idempotency_key": "open-pr-codex-value-drain",
        "task": "Open a draft PR for authenticated outbox value drain.",
        "status": "pending",
        "branch": "codex/authenticated-outbox-value-drain-20260605",
        "owner": "primary-writer",
        "objective": "Publish one clean PR for useful automation output.",
        "context": "GitHub publication was unavailable during the writer pass.",
        "acceptance_criteria": [
            "publisher keeps rich handoff evidence",
            "work robot classifies the handoff as ready",
        ],
        "mutation_boundary": "Limited to value-drain publication tooling.",
        "validation": {
            "pytest": "tests/scripts/test_drain_codex_automation_value.py passed",
            "ruff": "all checks passed",
        },
        "dependencies_declared": True,
        "labels": ["codex", "codex-automation"],
        "requested_action": {
            "action": "open_pr",
            "branch": "codex/authenticated-outbox-value-drain-20260605",
            "desired_head_sha": "abc123",
            "draft": True,
            "labels": ["value-drain"],
        },
        "local_evidence": {
            "branch_state": "committed",
            "changed_files": [
                "scripts/drain_codex_automation_value.py",
                "tests/scripts/test_drain_codex_automation_value.py",
            ],
        },
        "requires_github": True,
        "publication_blocker": {
            "mode": "connectivity_failed",
            "summary": "gh unavailable during automation run",
        },
        "validation_summary": "focused value-drain tests passed",
    }
    (outbox / "open-pr-codex-value-drain.json").write_text(json.dumps(payload))

    items, health = collect_automation_outbox(tmp_path)

    assert health["status"] == "ok"
    assert len(items) == 1
    item = items[0]
    assert item.tags == ["codex", "codex-automation", "value-drain"]
    assert item.metadata["requested_action"] == payload["requested_action"]
    assert item.metadata["local_evidence"] == payload["local_evidence"]
    assert item.metadata["labels"] == payload["labels"]
    assert item.metadata["requires_github"] is True
    assert item.metadata["publication_blocker"] == payload["publication_blocker"]
    assert item.metadata["validation_summary"] == payload["validation_summary"]

    recommendation = build_recommendations(items)[0]
    assert recommendation.classification == "ready"
    assert recommendation.action == "publish_or_reconcile_handoff"
    assert recommendation.priority == "high"
    assert recommendation.blockers == []
