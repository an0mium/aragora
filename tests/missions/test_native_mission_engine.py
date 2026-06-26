"""Tests for the native mission engine layers beyond the Phase-A spine."""

from __future__ import annotations

import json
from pathlib import Path

from aragora.missions import Feature, MissionState, Status
from aragora.missions.live_gate import LiveBossLoopGate
from aragora.missions.reconcile import (
    AdmissionPolicy,
    ArtifactCategory,
    ReconcileMode,
    WorkArtifact,
    apply_validation_result,
    classify_artifact,
    inject_validation_features,
    write_operator_receipt,
)
from aragora.missions.runtime import MissionRuntimeConfig


def test_reconcile_classifies_preserve_first_categories() -> None:
    cases = {
        "merged": WorkArtifact("wt-merged", kind="worktree", clean=True, already_merged=True),
        "open": WorkArtifact("pr-1", kind="pr", clean=True, open_pr=True),
        "valuable": WorkArtifact("branch-1", kind="branch", clean=True, unique_commits=True),
        "duplicate": WorkArtifact(
            "branch-dup", kind="branch", clean=True, represented_elsewhere=True
        ),
        "dirty": WorkArtifact("wt-dirty", kind="worktree", clean=False),
        "human": WorkArtifact("wt-owned", kind="worktree", clean=True, owner_active=True),
        "unknown": WorkArtifact("mystery", kind="branch", clean=True),
    }

    assert classify_artifact(cases["merged"]).category == ArtifactCategory.MERGED
    assert classify_artifact(cases["open"]).category == ArtifactCategory.OPEN_PR
    assert classify_artifact(cases["valuable"]).category == ArtifactCategory.VALUABLE_UNMERGED
    assert classify_artifact(cases["duplicate"]).category == ArtifactCategory.DUPLICATE
    assert classify_artifact(cases["dirty"]).category == ArtifactCategory.UNSAFE_DIRTY
    assert classify_artifact(cases["human"]).category == ArtifactCategory.NEEDS_HUMAN
    assert classify_artifact(cases["unknown"]).category == ArtifactCategory.UNKNOWN


def test_safe_clean_only_authorizes_freshly_safe_merged_artifacts() -> None:
    artifacts = [
        WorkArtifact("wt-merged", kind="worktree", clean=True, already_merged=True),
        WorkArtifact("wt-dirty", kind="worktree", clean=False, already_merged=True),
        WorkArtifact("pr-open", kind="pr", clean=True, open_pr=True),
        WorkArtifact("unknown", kind="branch", clean=True),
    ]

    report = ReconcileMode.SAFE_CLEAN.run(artifacts)

    assert [item.artifact_id for item in report.authorized_cleanup] == ["wt-merged"]
    assert {item.artifact_id for item in report.parked} == {"wt-dirty", "pr-open", "unknown"}


def test_auto_drain_authorizes_only_tier_0_to_2_exact_head_candidates() -> None:
    artifacts = [
        WorkArtifact(
            "pr-1",
            kind="pr",
            clean=True,
            open_pr=True,
            tier=2,
            head_sha="abc123",
            checks_green=True,
            quorum_satisfied=True,
        ),
        WorkArtifact(
            "pr-2",
            kind="pr",
            clean=True,
            open_pr=True,
            tier=3,
            head_sha="def456",
            checks_green=True,
            quorum_satisfied=True,
        ),
        WorkArtifact(
            "pr-3",
            kind="pr",
            clean=True,
            open_pr=True,
            tier=1,
            head_sha="fed999",
            checks_green=False,
            quorum_satisfied=True,
        ),
    ]

    report = ReconcileMode.AUTO_DRAIN.run(artifacts)

    assert [item.artifact_id for item in report.authorized_auto_drain] == ["pr-1"]
    assert {item.artifact_id for item in report.parked} == {"pr-2", "pr-3"}


def test_admission_policy_blocks_new_producer_work_under_backlog_pressure() -> None:
    policy = AdmissionPolicy(max_unresolved=1)
    report = ReconcileMode.REPORT.run(
        [
            WorkArtifact("valuable", kind="branch", clean=True, unique_commits=True),
            WorkArtifact("unknown", kind="branch", clean=True),
        ]
    )

    decision = policy.evaluate("Build a speculative new product surface", report)

    assert not decision.allowed
    assert "unresolved backlog" in decision.reason


def test_admission_policy_allows_cleanup_missions_under_backlog_pressure() -> None:
    policy = AdmissionPolicy(max_unresolved=1)
    report = ReconcileMode.REPORT.run(
        [
            WorkArtifact("valuable", kind="branch", clean=True, unique_commits=True),
            WorkArtifact("unknown", kind="branch", clean=True),
        ]
    )

    decision = policy.evaluate("Reconcile and cleanup queued work", report)

    assert decision.allowed


def test_validation_injection_adds_gated_validator_features() -> None:
    state = MissionState(
        mission_id="m",
        goal="ship carefully",
        milestones=["m1"],
        features=[
            Feature(
                id="impl",
                description="implement feature",
                milestone="m1",
                status=Status.COMPLETED,
                fulfills=["VAL-1"],
            )
        ],
    )

    injected = inject_validation_features(state, milestone="m1")

    assert [f.id for f in injected] == ["validate-m1-tests", "validate-m1-scrutiny"]
    assert {f.id for f in state.features} == {
        "impl",
        "validate-m1-tests",
        "validate-m1-scrutiny",
    }
    assert state.get("validate-m1-tests").preconditions == ["feature:impl"]
    assert state.get("validate-m1-tests").metadata["validation_for"] == "m1"
    assert state.get("validate-m1-tests").fulfills == ["VAL-1"]


def test_failed_validation_reopens_parent_features() -> None:
    state = MissionState(
        mission_id="m",
        goal="ship carefully",
        milestones=["m1"],
        features=[
            Feature(
                id="impl", description="implement feature", milestone="m1", status=Status.COMPLETED
            ),
            Feature(
                id="validate-m1-tests",
                description="Validate milestone m1 with tests",
                milestone="m1",
                status=Status.COMPLETED,
                metadata={"validation_for": "m1", "validates": ["impl"]},
            ),
        ],
    )

    apply_validation_result(state, "validate-m1-tests", passed=False, reason="regression failed")

    assert state.get("impl").status == Status.PENDING
    assert "validation validate-m1-tests failed: regression failed" in state.get("impl").notes
    assert state.get("validate-m1-tests").status == Status.BLOCKED


def test_passed_validation_completes_validator_without_reopening_parent() -> None:
    state = MissionState(
        mission_id="m",
        goal="ship carefully",
        milestones=["m1"],
        features=[
            Feature(
                id="impl", description="implement feature", milestone="m1", status=Status.COMPLETED
            ),
            Feature(
                id="validate-m1-tests",
                description="Validate milestone m1 with tests",
                milestone="m1",
                metadata={"validation_for": "m1", "validates": ["impl"]},
            ),
        ],
    )

    apply_validation_result(state, "validate-m1-tests", passed=True, reason="")

    assert state.get("impl").status == Status.COMPLETED
    assert state.get("validate-m1-tests").status == Status.COMPLETED


def test_operator_receipt_is_structured_and_append_only(tmp_path: Path) -> None:
    receipt_dir = tmp_path / "receipts"
    path = write_operator_receipt(
        receipt_dir,
        feature_id="f-tier3",
        blocker="tier-3 surface requires operator settlement",
        evidence=["head abc123", "merge-packet requires human risk settlement"],
        next_action="Ask operator for exact-head settlement approval",
        human_required=True,
    )

    payload = json.loads(path.read_text())
    assert payload["feature_id"] == "f-tier3"
    assert payload["human_required"] is True
    assert payload["next_action"] == "Ask operator for exact-head settlement approval"
    assert payload["evidence_checked"] == [
        "head abc123",
        "merge-packet requires human risk settlement",
    ]

    second = write_operator_receipt(
        receipt_dir,
        feature_id="f-tier3",
        blocker="still blocked",
        evidence=["head abc123"],
        next_action="keep parked",
        human_required=True,
    )
    assert second != path
    assert len(list(receipt_dir.glob("*.json"))) == 2


def test_live_gate_reads_feature_metadata_and_never_uses_admin_merge() -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], cwd: Path) -> str:
        calls.append(cmd)
        if cmd[:3] == ["git", "rev-parse", "codex/native-mission-engine"]:
            return "abc123\n"
        if cmd[:2] == ["git", "branch"]:
            return ""
        if cmd[:2] == ["git", "log"]:
            return "abc123\tmission: implement native engine\n"
        if cmd[:4] == ["python", "-m", "aragora.cli.main", "review-queue"]:
            return json.dumps(
                {
                    "ready": [
                        {
                            "pr_number": 8625,
                            "head_sha": "abc123",
                            "tier": 2,
                            "status": "satisfied",
                            "verdict": "admin_squash_allowed",
                            "admin_squash_allowed": True,
                            "requires_human_risk_settlement": False,
                            "requires_human_preapproval": False,
                            "unresolved_dissent": False,
                        }
                    ]
                }
            )
        if cmd[:3] == ["gh", "pr", "merge"]:
            return ""
        raise AssertionError(f"unexpected command: {cmd}")

    gate = LiveBossLoopGate(repo_root=Path("/repo"), repo_slug="synaptent/aragora", runner=runner)
    feature = Feature(
        id="engine",
        description="ship engine",
        milestone="m",
        metadata={"branch": "codex/native-mission-engine", "pr": 8625, "tier": 2},
    )

    branch = gate.branch_for(feature)
    head = gate.head_of(branch)
    verdict = gate.collect_evidence(branch, head)

    assert branch == "codex/native-mission-engine"
    assert head == "abc123"
    assert verdict.satisfied
    assert verdict.tier == 2
    assert gate.merge_head_bound(branch, head)
    merge_call = [cmd for cmd in calls if cmd[:3] == ["gh", "pr", "merge"]][0]
    assert "--match-head-commit" in merge_call
    assert "abc123" in merge_call
    assert "--admin" not in merge_call


def test_live_gate_tier_falls_back_to_merge_packet_when_metadata_omits_tier() -> None:
    def runner(cmd: list[str], cwd: Path) -> str:
        if cmd[:4] == ["python", "-m", "aragora.cli.main", "review-queue"]:
            return json.dumps({"entries": [{"pr_number": 8625, "head_sha": "abc123", "tier": 2}]})
        raise AssertionError(f"unexpected command: {cmd}")

    gate = LiveBossLoopGate(repo_root=Path("/repo"), repo_slug="synaptent/aragora", runner=runner)

    assert (
        gate.tier_of(
            Feature(
                id="engine",
                description="ship engine",
                milestone="m",
                metadata={"branch": "codex/native-mission-engine", "pr": 8625},
            )
        )
        == 2
    )


def test_live_gate_foreign_commits_inspects_subjects_on_mission_branches() -> None:
    def runner(cmd: list[str], cwd: Path) -> str:
        if cmd[:2] == ["git", "log"]:
            return (
                "bad123\tfix(memory): unrelated cache work\n"
                "ok456\tmission: implement native engine\n"
            )
        raise AssertionError(f"unexpected command: {cmd}")

    gate = LiveBossLoopGate(repo_root=Path("/repo"), runner=runner)

    assert gate.foreign_commits("mission/engine", "origin/main", ("mission/", "structex/")) == [
        "bad123 fix(memory): unrelated cache work"
    ]


def test_live_gate_requires_exact_head_packet_entry() -> None:
    def runner(cmd: list[str], cwd: Path) -> str:
        if cmd[:4] == ["python", "-m", "aragora.cli.main", "review-queue"]:
            return json.dumps(
                {
                    "ready": [
                        {
                            "pr_number": 8625,
                            "model_review_quorum": {"verdict": "satisfied", "tier": 2},
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected command: {cmd}")

    gate = LiveBossLoopGate(repo_root=Path("/repo"), repo_slug="synaptent/aragora", runner=runner)
    branch = gate.branch_for(
        Feature(
            id="engine",
            description="ship engine",
            milestone="m",
            metadata={"branch": "codex/native-mission-engine", "pr": 8625, "tier": 2},
        )
    )

    verdict = gate.collect_evidence(branch, "abc123")

    assert not verdict.satisfied
    assert verdict.tier == 3
    assert verdict.dissent == ["merge-packet had no exact-head entry for PR 8625 at abc123"]


def test_live_gate_uses_canonical_admin_squash_authorization() -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], cwd: Path) -> str:
        calls.append(cmd)
        if cmd[:4] == ["python", "-m", "aragora.cli.main", "review-queue"]:
            return json.dumps(
                {
                    "entries": [
                        {
                            "pr_number": 8625,
                            "head_sha": "abc123",
                            "tier": 2,
                            "status": "satisfied",
                            "verdict": "admin_squash_allowed",
                            "admin_squash_allowed": True,
                            "requires_human_risk_settlement": False,
                            "requires_human_preapproval": False,
                            "unresolved_dissent": False,
                        }
                    ]
                }
            )
        if cmd[:3] == ["gh", "pr", "merge"]:
            return ""
        raise AssertionError(f"unexpected command: {cmd}")

    gate = LiveBossLoopGate(repo_root=Path("/repo"), repo_slug="synaptent/aragora", runner=runner)
    branch = gate.branch_for(
        Feature(
            id="engine",
            description="ship engine",
            milestone="m",
            metadata={"branch": "codex/native-mission-engine", "pr": 8625, "tier": 2},
        )
    )

    verdict = gate.collect_evidence(branch, "abc123")

    assert verdict.satisfied
    assert verdict.tier == 2
    assert gate.merge_head_bound(branch, "abc123")
    assert any(cmd[:3] == ["gh", "pr", "merge"] for cmd in calls)


def test_live_gate_refuses_admin_squash_when_packet_has_human_blocker() -> None:
    def runner(cmd: list[str], cwd: Path) -> str:
        if cmd[:4] == ["python", "-m", "aragora.cli.main", "review-queue"]:
            return json.dumps(
                {
                    "entries": [
                        {
                            "pr_number": 8625,
                            "head_sha": "abc123",
                            "tier": 2,
                            "status": "satisfied",
                            "verdict": "admin_squash_allowed",
                            "admin_squash_allowed": True,
                            "requires_human_risk_settlement": True,
                            "requires_human_preapproval": False,
                            "unresolved_dissent": False,
                        }
                    ]
                }
            )
        if cmd[:3] == ["gh", "pr", "merge"]:
            raise AssertionError("merge must not be attempted")
        raise AssertionError(f"unexpected command: {cmd}")

    gate = LiveBossLoopGate(repo_root=Path("/repo"), repo_slug="synaptent/aragora", runner=runner)
    branch = gate.branch_for(
        Feature(
            id="engine",
            description="ship engine",
            milestone="m",
            metadata={"branch": "codex/native-mission-engine", "pr": 8625, "tier": 2},
        )
    )

    assert not gate.collect_evidence(branch, "abc123").satisfied
    assert not gate.merge_head_bound(branch, "abc123")


def test_headless_runtime_config_uses_provider_env_without_enabling_flag(monkeypatch) -> None:
    monkeypatch.setenv("ARAGORA_MISSION_RUNTIME", "headless-api")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.delenv("ARAGORA_ENABLE_NATIVE_MISSION", raising=False)

    config = MissionRuntimeConfig.from_env()

    assert config.mode == "headless-api"
    assert config.available_provider_env_vars == {"openai": "OPENAI_API_KEY"}
    assert not config.enables_native_mission_flag
