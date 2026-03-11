"""Repair task generation for known deterministic blockers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aragora.ralph.classifier import BlockerKind


@dataclass(slots=True)
class RepairTask:
    """A bounded repair task generated from a classified blocker."""

    title: str
    problem_statement: str
    allowed_paths: list[str]
    required_tests: list[str]
    done_condition: str
    blocker_kind: str
    affected_project_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "problem_statement": self.problem_statement,
            "allowed_paths": list(self.allowed_paths),
            "required_tests": list(self.required_tests),
            "done_condition": self.done_condition,
            "blocker_kind": self.blocker_kind,
            "affected_project_ids": list(self.affected_project_ids),
        }


_TEMPLATES: dict[BlockerKind, dict[str, Any]] = {
    BlockerKind.REVIEWER_MISSING_DIFF: {
        "title": "Fix campaign reviewer to include actual diff content",
        "problem_statement": (
            "CampaignReviewer._build_prompt() only passes metadata (branch/commit) "
            "to the review model. The reviewer cannot verify acceptance criteria "
            "because it never sees the actual file changes. Add _fetch_diff_content() "
            "to retrieve the git diff and include it in the review prompt."
        ),
        "allowed_paths": [
            "aragora/swarm/campaign.py",
            "tests/swarm/test_campaign_reviewer_diff.py",
        ],
        "required_tests": [
            "tests/swarm/test_campaign_reviewer_diff.py",
            "tests/swarm/test_campaign.py",
        ],
        "done_condition": (
            "_build_prompt includes actual diff content from git diff; "
            "all existing campaign tests pass"
        ),
    },
    BlockerKind.SCOPE_FALSE_POSITIVE: {
        "title": "Fix scope enforcement false positive in worker deliverable check",
        "problem_statement": (
            "Worker deliverable is being rejected by scope enforcement even though "
            "the changed files are within the declared file_scope_hints. The scope "
            "check path comparison may be incorrect."
        ),
        "allowed_paths": [
            "aragora/swarm/worker_launcher.py",
            "aragora/swarm/campaign.py",
            "tests/swarm/test_exit141_deliverable.py",
        ],
        "required_tests": [
            "tests/swarm/test_exit141_deliverable.py",
        ],
        "done_condition": (
            "Worker deliverables within declared file_scope_hints are accepted; "
            "scope enforcement still rejects out-of-scope changes"
        ),
    },
    BlockerKind.WORKER_CLEAN_EXIT_NO_EFFECT: {
        "title": "Fix worker producing no deliverable on docs-only tasks",
        "problem_statement": (
            "Worker exits cleanly (exit 0) but produces no committed files. "
            "The auto-commit gate may not detect untracked files, or the worker "
            "prompt may lack sufficient context to produce output."
        ),
        "allowed_paths": [
            "aragora/swarm/worker_launcher.py",
            "aragora/swarm/boss_loop.py",
            "tests/swarm/test_exit141_deliverable.py",
        ],
        "required_tests": [
            "tests/swarm/test_exit141_deliverable.py",
        ],
        "done_condition": (
            "Worker auto-commit detects untracked files; "
            "docs-only tasks produce at least one committed file"
        ),
    },
    BlockerKind.MANIFEST_IDENTIFIER_COLLISION: {
        "title": "Fix identifier collision in campaign manifest",
        "problem_statement": (
            "Campaign manifest contains duplicate file_scope_hints or project IDs "
            "that collide with existing artifacts (e.g., ADR numbers already occupied). "
            "Renumber or deduplicate the colliding entries."
        ),
        "allowed_paths": [
            "docs/plans/",
            ".aragora/campaign_manifest.yaml",
        ],
        "required_tests": [],
        "done_condition": (
            "No duplicate file_scope_hints across projects; "
            "no collisions with existing files on main"
        ),
    },
    BlockerKind.RUNTIME_TIMEOUT_CONFIG: {
        "title": "Increase campaign runtime time limit",
        "problem_statement": (
            "Campaign timed out but was making progress (some projects completed). "
            "The runtime time_limit_hours is too low for the campaign scope."
        ),
        "allowed_paths": [
            ".aragora/campaign_manifest.yaml",
        ],
        "required_tests": [],
        "done_condition": "time_limit_hours increased to accommodate remaining projects",
    },
    BlockerKind.RECEIPT_EMISSION_GAP: {
        "title": "Fix missing receipt emission on terminal project transition",
        "problem_statement": (
            "A project reached a terminal status (completed/failed/blocked/skipped) "
            "without a receipt being written to docs/receipts/. The _emit_receipt() "
            "call may be missing from a code path."
        ),
        "allowed_paths": [
            "aragora/swarm/campaign.py",
            "tests/swarm/test_campaign_receipt.py",
        ],
        "required_tests": [
            "tests/swarm/test_campaign_receipt.py",
        ],
        "done_condition": (
            "All terminal project transitions emit a receipt; "
            "receipt file exists for every terminal project"
        ),
    },
}


def generate_repair_task(
    blocker_kind: BlockerKind,
    *,
    affected_project_ids: list[str] | None = None,
) -> RepairTask | None:
    """Generate a bounded repair task for a deterministic blocker.

    Returns ``None`` if the blocker kind has no known repair template
    (i.e., it should be escalated instead).
    """
    template = _TEMPLATES.get(blocker_kind)
    if template is None:
        return None
    return RepairTask(
        title=template["title"],
        problem_statement=template["problem_statement"],
        allowed_paths=list(template["allowed_paths"]),
        required_tests=list(template["required_tests"]),
        done_condition=template["done_condition"],
        blocker_kind=blocker_kind.value,
        affected_project_ids=list(affected_project_ids or []),
    )
