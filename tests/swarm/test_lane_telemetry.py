"""Regression coverage for swarm lane telemetry."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from aragora.nomic.dev_coordination import DevCoordinationStore
from aragora.swarm.boss_loop import BossLoop, BossLoopConfig
from aragora.swarm.campaign import (
    CampaignExecutor,
    CampaignManifest,
    CampaignProject,
    CampaignProjectStatus,
    CampaignReviewGate,
    CampaignReviewStatus,
    save_campaign_manifest,
)
from aragora.swarm.lane_telemetry import LaneTelemetryCollector, LaneTelemetryRecord
from aragora.swarm.supervisor import SwarmSupervisor

UTC = timezone.utc


class TestLaneTelemetryCollector:
    def test_default_db_path_survives_later_cwd_changes(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ARAGORA_DATA_DIR", "runtime")

        collector = LaneTelemetryCollector()
        assert Path(collector.db_path).is_absolute()

        other_dir = tmp_path / "other"
        other_dir.mkdir()
        monkeypatch.chdir(other_dir)

        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="boss_dispatch",
                lane_id="boss-1",
                terminal_outcome="deliverable_created",
                deliverable_type="branch",
                timestamp=datetime.now(UTC).timestamp(),
            )
        )

        assert collector.get_throughput(window_days=7) == 1

    def test_queries_cover_success_false_success_and_merge_metrics(self) -> None:
        collector = LaneTelemetryCollector(db_path=":memory:")
        now = datetime.now(UTC).timestamp()

        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="boss_dispatch",
                lane_id="boss-1",
                terminal_outcome="deliverable_created",
                deliverable_type="branch",
                receipt_id="lane-1",
                duration_seconds=12.0,
                time_to_pr_seconds=30.0,
                false_success_candidate=False,
                timestamp=now,
            )
        )
        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="campaign_project",
                lane_id="proj-1",
                terminal_outcome="pr_adopted",
                deliverable_type="adopted_pr",
                receipt_id="receipt-1",
                human_intervention_required=False,
                merge_ref="main@abc123",
                merged_at="2026-03-29T12:00:00+00:00",
                time_to_merge_seconds=120.0,
                false_success_candidate=False,
                timestamp=now,
            )
        )
        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="supervisor_work_order",
                lane_id="wo-1",
                terminal_outcome="deliverable_created",
                deliverable_type="",
                human_intervention_required=False,
                false_success_candidate=True,
                timestamp=now,
            )
        )

        assert collector.get_throughput(window_days=7) == 3
        assert collector.get_success_rate(window_days=7) == 2 / 3
        assert collector.get_false_success_candidate_count(window_days=7) == 1
        assert collector.get_human_intervention_rate(window_days=7) == 0.0
        assert collector.get_merge_yield(window_days=7) == 0.5
        assert collector.get_avg_time_to_pr(window_days=7) == 30.0
        assert collector.get_avg_time_to_merge(window_days=7) == 120.0

    def test_rates_ignore_unclassified_terminal_rows(self) -> None:
        collector = LaneTelemetryCollector(db_path=":memory:")
        now = datetime.now(UTC).timestamp()

        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="boss_dispatch",
                lane_id="boss-1",
                terminal_outcome="deliverable_created",
                deliverable_type="branch",
                false_success_candidate=False,
                timestamp=now,
            )
        )
        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="boss_dispatch",
                lane_id="boss-2",
                terminal_outcome="needs_human",
                human_intervention_required=True,
                timestamp=now,
            )
        )
        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="boss_dispatch",
                lane_id="boss-3",
                terminal_outcome="preview_only",
                human_intervention_required=False,
                timestamp=now,
            )
        )
        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="boss_dispatch",
                lane_id="boss-4",
                terminal_outcome="",
                human_intervention_required=False,
                timestamp=now,
            )
        )

        assert collector.get_throughput(window_days=7) == 4
        assert collector.get_success_rate(window_days=7) == 0.5
        assert collector.get_human_intervention_rate(window_days=7) == 0.5

    def test_rates_ignore_legacy_noncanonical_terminal_outcomes(self) -> None:
        collector = LaneTelemetryCollector(db_path=":memory:")
        now = datetime.now(UTC).timestamp()

        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="boss_dispatch",
                lane_id="boss-success",
                terminal_outcome="deliverable_created",
                deliverable_type="branch",
                false_success_candidate=False,
                timestamp=now,
            )
        )
        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="boss_dispatch",
                lane_id="boss-human",
                terminal_outcome="needs_human",
                human_intervention_required=True,
                timestamp=now,
            )
        )
        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="boss_dispatch",
                lane_id="boss-legacy-completed",
                terminal_outcome="completed",
                human_intervention_required=True,
                timestamp=now,
            )
        )
        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="boss_dispatch",
                lane_id="boss-legacy-failed",
                terminal_outcome="failed",
                human_intervention_required=True,
                timestamp=now,
            )
        )

        assert collector.get_throughput(window_days=7) == 4
        assert collector.get_success_rate(window_days=7) == 0.5
        assert collector.get_human_intervention_rate(window_days=7) == 0.5


class TestBossDispatchTelemetry:
    def test_emit_lane_receipt_records_boss_dispatch_terminal_event(self) -> None:
        collector = LaneTelemetryCollector(db_path=":memory:")
        loop = BossLoop(config=BossLoopConfig(max_iterations=1, iteration_interval_seconds=0.0))
        worker_result = {
            "run_id": "run-123",
            "lease_id": "lease-123",
            "agent_id": "boss-loop",
            "outcome": "needs_human",
            "deliverable": {
                "type": "branch",
                "branch": "codex/recovered",
                "commit_shas": ["abc123"],
            },
            "reasons": ["Recovered deliverable requires review."],
        }

        with (
            patch("aragora.swarm.boss_loop._LANE_TELEMETRY", collector),
            patch("aragora.receipts.lane.emit_lane_receipt", return_value="lane-receipt-1"),
        ):
            receipt_id = loop._emit_lane_receipt(
                worker_result, {"number": 42, "title": "Fix lane"}, 3.5
            )

        assert receipt_id == "lane-receipt-1"
        records = collector.get_recent_lanes()
        assert len(records) == 1
        assert records[0].lane_kind == "boss_dispatch"
        assert records[0].lane_id == "run-123"
        assert records[0].terminal_outcome == "needs_human"
        assert records[0].deliverable_type == "branch"
        assert records[0].receipt_id == "lane-receipt-1"
        assert records[0].human_intervention_required is True

    def test_missing_outcome_falls_back_to_deliverable_created(self) -> None:
        collector = LaneTelemetryCollector(db_path=":memory:")
        loop = BossLoop(config=BossLoopConfig(max_iterations=1, iteration_interval_seconds=0.0))
        worker_result = {
            "run_id": "run-124",
            "status": "completed",
            "deliverable": {
                "type": "branch",
                "branch": "codex/recovered",
                "commit_shas": ["abc123"],
            },
        }

        with patch("aragora.swarm.boss_loop._LANE_TELEMETRY", collector):
            loop._record_lane_telemetry(
                worker_result, {"number": 43, "title": "Fix lane"}, 2.0, None
            )

        records = collector.get_recent_lanes()
        assert len(records) == 1
        assert records[0].terminal_outcome == "deliverable_created"
        assert records[0].deliverable_type == "branch"

    def test_missing_outcome_completed_without_deliverable_becomes_clean_exit_no_deliverable(
        self,
    ) -> None:
        collector = LaneTelemetryCollector(db_path=":memory:")
        loop = BossLoop(config=BossLoopConfig(max_iterations=1, iteration_interval_seconds=0.0))
        worker_result = {
            "run_id": "run-125",
            "status": "completed",
        }

        with patch("aragora.swarm.boss_loop._LANE_TELEMETRY", collector):
            loop._record_lane_telemetry(
                worker_result, {"number": 44, "title": "No deliverable lane"}, 2.0, None
            )

        records = collector.get_recent_lanes()
        assert len(records) == 1
        assert records[0].terminal_outcome == "clean_exit_no_deliverable"
        assert records[0].deliverable_type == ""
        assert records[0].human_intervention_required is True

    def test_preview_only_outcome_is_excluded_from_human_intervention(self) -> None:
        collector = LaneTelemetryCollector(db_path=":memory:")
        loop = BossLoop(config=BossLoopConfig(max_iterations=1, iteration_interval_seconds=0.0))
        worker_result = {
            "run_id": "run-preview",
            "status": "needs_human",
            "outcome": "preview_only",
        }

        with patch("aragora.swarm.boss_loop._LANE_TELEMETRY", collector):
            loop._record_lane_telemetry(
                worker_result, {"number": 99, "title": "Preview lane"}, 1.0, None
            )

        records = collector.get_recent_lanes()
        assert len(records) == 1
        assert records[0].terminal_outcome == "preview_only"
        assert records[0].human_intervention_required is False


class TestSupervisorTelemetry:
    def test_terminal_work_order_records_supervisor_event(self, tmp_path: Path) -> None:
        collector = LaneTelemetryCollector(db_path=":memory:")
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
        supervisor = SwarmSupervisor(repo_root=repo, store=DevCoordinationStore(repo_root=repo))
        item = {
            "task_key": "run-1:wo-1",
            "work_order_id": "wo-1",
            "status": "needs_human",
            "worker_outcome": "crash_with_salvage",
            "branch": "codex/recovered",
            "commit_shas": ["abc123"],
            "failure_reason": "worker_crash_with_salvage",
            "blocking_question": "Should the recovered deliverable be adopted?",
            "dispatched_at": "2026-03-29T12:00:00+00:00",
            "completed_at": "2026-03-29T12:01:00+00:00",
        }

        with patch("aragora.swarm.supervisor._LANE_TELEMETRY", collector):
            supervisor._record_terminal_work_order_telemetry("run-1", [item])

        records = collector.get_recent_lanes()
        assert len(records) == 1
        assert records[0].lane_kind == "supervisor_work_order"
        assert records[0].lane_id == "run-1:wo-1"
        assert records[0].terminal_outcome == "crash"
        assert records[0].worker_outcome == "crash_with_salvage"
        assert records[0].deliverable_type == "branch"
        assert records[0].human_intervention_required is True
        assert records[0].duration_seconds == 60.0


class TestCampaignTelemetry:
    def test_emit_receipt_records_campaign_project_terminal_event(self, tmp_path: Path) -> None:
        collector = LaneTelemetryCollector(db_path=":memory:")
        repo = tmp_path / "repo"
        repo.mkdir()
        manifest_path = repo / ".aragora" / "campaign_manifest.yaml"
        manifest_path.parent.mkdir(parents=True)

        project = CampaignProject(
            project_id="proj-1",
            title="ADR lane",
            status=CampaignProjectStatus.COMPLETED.value,
            last_run_outcome="deliverable_created",
            run_id="run-1",
            branch="codex/adr",
            commit_shas=["abc123"],
            worker_receipt_id="worker-receipt-1",
            review=CampaignReviewGate(
                required=True,
                review_model="claude",
                status=CampaignReviewStatus.PASSED.value,
                findings=[],
            ),
        )
        manifest = CampaignManifest(
            campaign_id="campaign-1",
            created_at="2026-03-29T12:00:00+00:00",
            source_kind="manual",
            source_ref="spec",
            projects=[project],
        )
        save_campaign_manifest(manifest_path, manifest)

        executor = CampaignExecutor(
            manifest_path=manifest_path,
            repo_root=repo,
        )
        run_dict = {
            "run_id": "run-1",
            "status": "completed",
            "work_orders": [
                {
                    "work_order_id": "wo-1",
                    "status": "completed",
                    "branch": "codex/adr",
                    "commit_shas": ["abc123"],
                    "worker_outcome": "completed",
                    "receipt_id": "worker-receipt-1",
                }
            ],
        }

        with patch("aragora.swarm.campaign._LANE_TELEMETRY", collector):
            receipt_path = executor._emit_receipt(manifest, project, run_dict)

        assert receipt_path.exists()
        records = collector.get_recent_lanes()
        assert len(records) == 1
        assert records[0].lane_kind == "campaign_project"
        assert records[0].lane_id == "campaign-1:proj-1"
        assert records[0].run_id == "run-1"
        assert records[0].terminal_outcome == "deliverable_created"
        assert records[0].deliverable_type == "branch"
        assert records[0].receipt_id == project.receipt_id
        assert records[0].human_intervention_required is False

    def test_emit_receipt_falls_back_to_qualified_outcome_when_project_outcome_missing(
        self, tmp_path: Path
    ) -> None:
        collector = LaneTelemetryCollector(db_path=":memory:")
        repo = tmp_path / "repo"
        repo.mkdir()
        manifest_path = repo / ".aragora" / "campaign_manifest.yaml"
        manifest_path.parent.mkdir(parents=True)

        project = CampaignProject(
            project_id="proj-2",
            title="ADR lane",
            status=CampaignProjectStatus.COMPLETED.value,
            last_run_outcome=None,
            run_id="run-2",
            branch="codex/adr",
            commit_shas=["abc123"],
            review=CampaignReviewGate(
                required=True,
                review_model="claude",
                status=CampaignReviewStatus.PENDING.value,
                findings=[],
            ),
        )
        manifest = CampaignManifest(
            campaign_id="campaign-2",
            created_at="2026-03-29T12:00:00+00:00",
            source_kind="manual",
            source_ref="spec",
            projects=[project],
        )
        save_campaign_manifest(manifest_path, manifest)

        executor = CampaignExecutor(
            manifest_path=manifest_path,
            repo_root=repo,
        )
        run_dict = {
            "run_id": "run-2",
            "status": "completed",
            "work_orders": [
                {
                    "work_order_id": "wo-2",
                    "status": "completed",
                    "branch": "codex/adr",
                    "commit_shas": ["abc123"],
                    "worker_outcome": "completed",
                }
            ],
        }

        with patch("aragora.swarm.campaign._LANE_TELEMETRY", collector):
            executor._emit_receipt(manifest, project, run_dict)

        records = collector.get_recent_lanes()
        assert len(records) == 1
        assert records[0].terminal_outcome == "deliverable_created"
        assert records[0].deliverable_type == "branch"
