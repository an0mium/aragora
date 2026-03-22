from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aragora.swarm.tranche_queue import (
    QUEUE_ITEM_STATUS_COMPLETED,
    QUEUE_ITEM_STATUS_NEEDS_HUMAN,
    QUEUE_ITEM_STATUS_PENDING,
    QUEUE_ITEM_STATUS_RUNNING,
    QUEUE_STATUS_COMPLETED,
    QUEUE_STATUS_RUNNING,
    QUEUE_STATUS_STOPPED,
    TrancheQueueExecutor,
    TrancheQueueItem,
    TrancheQueueItemRunState,
    TrancheQueueManifest,
    TrancheQueueRunState,
    compile_tranche_queue,
    _queue_merge_policy,
    _resolve_queue_autonomy_mode,
    queue_state_path_for_queue,
    reconcile_tranche_queue,
)
from aragora.swarm.tranche import TrancheLaneArtifact
from aragora.swarm.tranche_state import (
    LANE_STATUS_NEEDS_HUMAN,
    TRANCHE_STATUS_NEEDS_HUMAN,
    LaneRunState,
    TrancheRunState,
)


def _write_queue(queue_path: Path) -> TrancheQueueManifest:
    manifest = TrancheQueueManifest.from_dict(
        {
            "queue_id": "overnight",
            "items": [
                {"id": "issue-1046", "kind": "issue", "source": "1046", "merge_class": "manual"},
                {
                    "id": "intake-docs",
                    "kind": "intake",
                    "source": "bundles/docs.yaml",
                    "merge_class": "manual",
                },
                {"id": "issue-1047", "kind": "issue", "source": "1047", "merge_class": "manual"},
            ],
        }
    )
    queue_path.write_text(manifest.to_yaml(), encoding="utf-8")
    return manifest


def test_compile_queue_writes_execute_doc_source_as_intake_bundle(tmp_path: Path) -> None:
    sources_path = tmp_path / "sources.yaml"
    doc_path = tmp_path / "roadmap.md"
    output_path = tmp_path / "overnight.yaml"
    doc_path.write_text("# Roadmap\n\nShip tranche queue.\n", encoding="utf-8")
    sources_path.write_text(
        """
queue_id: overnight
sources:
  - id: roadmap-doc
    kind: doc
    mode: execute
    path: roadmap.md
    objective: Ship the roadmap tranche work.
    merge_class: manual
""".strip(),
        encoding="utf-8",
    )

    payload = compile_tranche_queue(
        sources_path=sources_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    assert payload["item_count"] == 1
    assert payload["proposal_count"] == 0
    manifest = TrancheQueueManifest.load(output_path)
    assert manifest.items[0].kind == "intake"
    assert manifest.items[0].max_lanes == 1
    bundle_path = (output_path.parent / manifest.items[0].source).resolve()
    assert bundle_path.exists()
    bundle = bundle_path.read_text(encoding="utf-8")
    assert "Ship the roadmap tranche work." in bundle
    assert "merge_policy: manual" in bundle


def test_compile_queue_records_proposal_for_synthesize_doc_source(tmp_path: Path) -> None:
    sources_path = tmp_path / "sources.yaml"
    output_path = tmp_path / "overnight.yaml"
    sources_path.write_text(
        """
queue_id: overnight
sources:
  - id: goals-doc
    kind: doc
    mode: synthesize
    path: goals.md
    merge_class: manual
""".strip(),
        encoding="utf-8",
    )

    payload = compile_tranche_queue(
        sources_path=sources_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    assert payload["item_count"] == 0
    assert payload["proposal_count"] == 1
    assert payload["status"] == "needs_human"
    assert payload["wrote_queue"] is False
    assert payload["proposals"][0]["status"] == "needs_human"
    assert "not executable yet" in payload["proposals"][0]["reason"]
    assert payload["detail"] == "All sources produced proposals. Review proposals before running."
    assert output_path.exists() is False


def test_compile_queue_records_proposal_for_missing_execute_doc_source(tmp_path: Path) -> None:
    sources_path = tmp_path / "sources.yaml"
    output_path = tmp_path / "overnight.yaml"
    sources_path.write_text(
        """
queue_id: overnight
sources:
  - id: missing-doc
    kind: doc
    mode: execute
    path: does-not-exist.md
    objective: Ship the missing doc work.
    merge_class: manual
""".strip(),
        encoding="utf-8",
    )

    payload = compile_tranche_queue(
        sources_path=sources_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    assert payload["item_count"] == 0
    assert payload["proposal_count"] == 1
    assert payload["status"] == "needs_human"
    assert payload["wrote_queue"] is False
    assert payload["proposals"][0]["reason"] == "Execute doc source path was not found."
    assert payload["detail"] == "All sources produced proposals. Review proposals before running."
    assert output_path.exists() is False


def test_compile_queue_propagates_verification_commands_to_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verification commands from sources must flow through to compiled queue items."""
    sources_path = tmp_path / "sources.yaml"
    output_path = tmp_path / "overnight.yaml"
    sources_path.write_text(
        """
queue_id: overnight
sources:
  - id: test-issue
    kind: issue
    mode: execute
    url: https://github.com/org/repo/issues/42
    verification_commands:
      - python3 -m pytest tests/cli/test_setup.py -q
    merge_class: manual
""".strip(),
        encoding="utf-8",
    )

    payload = compile_tranche_queue(
        sources_path=sources_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    assert payload["item_count"] == 1
    manifest = TrancheQueueManifest.load(output_path)
    item = manifest.items[0]
    assert item.verification_commands == ["python3 -m pytest tests/cli/test_setup.py -q"]


def test_compile_queue_expands_issue_query_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources_path = tmp_path / "sources.yaml"
    output_path = tmp_path / "overnight.yaml"
    sources_path.write_text(
        """
queue_id: overnight
sources:
  - id: autonomy-query
    kind: issue_query
    mode: execute
    query: "label:autonomy sort:created-asc"
    repo: synaptent/aragora
    limit: 2
""".strip(),
        encoding="utf-8",
    )

    def fake_run(cmd, text, capture_output, timeout, check):
        assert cmd[:3] == ["gh", "issue", "list"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "number": 1046,
                        "title": "A",
                        "url": "https://github.com/org/repo/issues/1046",
                    },
                    {
                        "number": 1047,
                        "title": "B",
                        "url": "https://github.com/org/repo/issues/1047",
                    },
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr("aragora.swarm.tranche_queue.subprocess.run", fake_run)

    payload = compile_tranche_queue(
        sources_path=sources_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    assert payload["item_count"] == 2
    manifest = TrancheQueueManifest.load(output_path)
    assert [item.kind for item in manifest.items] == ["issue", "issue"]
    assert all(item.max_lanes == 1 for item in manifest.items)
    assert manifest.items[0].source == "https://github.com/org/repo/issues/1046"
    assert manifest.items[1].source == "https://github.com/org/repo/issues/1047"


def test_queue_item_defaults_to_single_lane() -> None:
    item = TrancheQueueItem.from_dict(
        {
            "id": "issue-1046",
            "kind": "issue",
            "source": "https://github.com/org/repo/issues/1046",
            "merge_class": "manual",
        }
    )

    assert item.max_lanes == 1


def test_low_risk_queue_item_keeps_fire_and_forget_and_sets_auto_merge_policy(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "overnight.yaml"
    queue_path.write_text(
        "queue_id: overnight\nitems:\n- id: one\n  kind: intake\n  source: bundle.yaml\n"
    )
    bundle_path = tmp_path / "bundle.yaml"
    bundle_path.write_text(
        "objective: test\ncandidate_lanes:\n- lane_id: lane-a\n  owner_role: critical_path_engineer\n",
        encoding="utf-8",
    )
    executor = TrancheQueueExecutor(queue_path=queue_path, repo_root=tmp_path)
    item = TrancheQueueItem(
        item_id="one",
        kind="intake",
        source="bundle.yaml",
        merge_class="low_risk",
        autonomy_mode="fire_and_forget",
    )

    effective_autonomy, downgraded = _resolve_queue_autonomy_mode(
        item.autonomy_mode,
        merge_class=item.merge_class,
    )
    bundle = executor._bundle_from_intake_item(item, effective_autonomy_mode=effective_autonomy)

    assert effective_autonomy == "fire_and_forget"
    assert downgraded is False
    assert bundle["autonomy_mode"] == "fire_and_forget"
    assert bundle["candidate_lanes"][0]["merge_policy"] == "auto"


def test_manual_queue_item_downgrades_fire_and_forget_and_keeps_manual_policy(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "overnight.yaml"
    queue_path.write_text(
        "queue_id: overnight\nitems:\n- id: one\n  kind: intake\n  source: bundle.yaml\n"
    )
    bundle_path = tmp_path / "bundle.yaml"
    bundle_path.write_text(
        "objective: test\ncandidate_lanes:\n- lane_id: lane-a\n  owner_role: critical_path_engineer\n",
        encoding="utf-8",
    )
    executor = TrancheQueueExecutor(queue_path=queue_path, repo_root=tmp_path)
    item = TrancheQueueItem(
        item_id="one",
        kind="intake",
        source="bundle.yaml",
        merge_class="manual",
        autonomy_mode="fire_and_forget",
    )

    effective_autonomy, downgraded = _resolve_queue_autonomy_mode(
        item.autonomy_mode,
        merge_class=item.merge_class,
    )
    bundle = executor._bundle_from_intake_item(item, effective_autonomy_mode=effective_autonomy)

    assert effective_autonomy == "adaptive"
    assert downgraded is True
    assert bundle["autonomy_mode"] == "adaptive"
    assert bundle["candidate_lanes"][0]["merge_policy"] == _queue_merge_policy(merge_class="manual")


def test_issue_queue_item_collapses_multi_lane_planner_output_to_single_fallback_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "overnight.yaml"
    queue_path.write_text(
        "queue_id: overnight\nitems:\n- id: issue-one\n  kind: issue\n  source: https://github.com/org/repo/issues/1046\n",
        encoding="utf-8",
    )
    executor = TrancheQueueExecutor(queue_path=queue_path, repo_root=tmp_path)
    item = TrancheQueueItem(
        item_id="issue-one",
        kind="issue",
        source="https://github.com/org/repo/issues/1046",
        merge_class="manual",
        max_lanes=1,
        objective_override="Implement one CLI-only slice.",
        allowed_write_scope=["aragora/cli/**", "tests/cli/**"],
        autonomy_mode="adaptive",
    )

    class _FakePlanner:
        def plan_from_items(self, items, source_kind, source_ref):
            assert source_kind == "queue_issue"
            assert source_ref == "https://github.com/org/repo/issues/1046"
            return SimpleNamespace(projects=["proj-1", "proj-2"])

    monkeypatch.setattr(executor, "_planner", lambda: _FakePlanner())
    monkeypatch.setattr(
        "aragora.swarm.tranche_queue._fetch_issue_payload",
        lambda source: {
            "number": 1046,
            "title": "Close the product loop",
            "body": "CLI-first bounded issue body.",
            "url": "https://github.com/org/repo/issues/1046",
        },
    )
    monkeypatch.setattr(
        "aragora.swarm.tranche_queue.campaign_projects_to_candidate_lanes",
        lambda projects, planner: [
            {
                "lane_id": "proj-001",
                "title": "First planner lane",
                "prompt": "planner lane 1",
                "owner_role": "implementation_engineer",
                "allowed_write_scope": ["aragora/cli/**"],
                "dependencies": [],
            },
            {
                "lane_id": "proj-002",
                "title": "Second planner lane",
                "prompt": "planner lane 2",
                "owner_role": "implementation_engineer",
                "allowed_write_scope": ["aragora/cli/**"],
                "dependencies": [],
            },
        ],
    )

    bundle = executor._bundle_from_issue_item(item, effective_autonomy_mode="adaptive")

    assert bundle["autonomy_mode"] == "adaptive"
    assert len(bundle["candidate_lanes"]) == 1
    lane = bundle["candidate_lanes"][0]
    assert lane["lane_id"] == "issue-one"
    assert lane["merge_policy"] == "manual"
    assert lane["queue_item_id"] == "issue-one"
    assert lane["allowed_write_scope"] == ["aragora/cli/**", "tests/cli/**"]
    assert "Source issue context" in lane["prompt"]


def test_intake_queue_item_limits_candidate_lanes_to_max_lanes(tmp_path: Path) -> None:
    queue_path = tmp_path / "overnight.yaml"
    queue_path.write_text(
        "queue_id: overnight\nitems:\n- id: intake\n  kind: intake\n  source: bundle.yaml\n",
        encoding="utf-8",
    )
    bundle_path = tmp_path / "bundle.yaml"
    bundle_path.write_text(
        """
objective: test
candidate_lanes:
  - lane_id: lane-a
    title: Lane A
    prompt: Implement lane A
    owner_role: implementation_engineer
  - lane_id: lane-b
    title: Lane B
    prompt: Implement lane B
    owner_role: implementation_engineer
    dependencies:
      - lane-a
""".strip(),
        encoding="utf-8",
    )
    executor = TrancheQueueExecutor(queue_path=queue_path, repo_root=tmp_path)
    item = TrancheQueueItem(
        item_id="intake",
        kind="intake",
        source="bundle.yaml",
        merge_class="manual",
        max_lanes=1,
    )

    bundle = executor._bundle_from_intake_item(item, effective_autonomy_mode="adaptive")

    assert bundle["max_lanes"] == 1
    assert len(bundle["candidate_lanes"]) == 1
    assert bundle["candidate_lanes"][0]["lane_id"] == "lane-a"


@pytest.mark.asyncio
async def test_run_queue_processes_items_sequentially_and_continues_after_needs_human(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "overnight.yaml"
    manifest = _write_queue(queue_path)
    executor = TrancheQueueExecutor(queue_path=queue_path, repo_root=tmp_path)

    seen: list[str] = []

    async def fake_process_item(*, manifest, item, item_state, deadline):
        seen.append(item.item_id)
        if item.item_id == "issue-1046":
            item_state.status = QUEUE_ITEM_STATUS_COMPLETED
        elif item.item_id == "intake-docs":
            item_state.status = QUEUE_ITEM_STATUS_NEEDS_HUMAN
            item_state.findings.append("Missing source material.")
        else:
            item_state.status = QUEUE_ITEM_STATUS_COMPLETED
        item_state.finished_at = item_state.updated_at = item_state.started_at
        return None

    monkeypatch.setattr(executor, "_process_item", fake_process_item)

    result = await executor.run()

    assert seen == ["issue-1046", "intake-docs", "issue-1047"]
    assert result["status"] == QUEUE_STATUS_COMPLETED
    assert result["counts"] == {"completed": 2, "needs_human": 1}

    state = TrancheQueueRunState.load(queue_state_path_for_queue(queue_path), manifest=manifest)
    assert state.item_states["issue-1046"].status == QUEUE_ITEM_STATUS_COMPLETED
    assert state.item_states["intake-docs"].status == QUEUE_ITEM_STATUS_NEEDS_HUMAN
    assert state.item_states["issue-1047"].status == QUEUE_ITEM_STATUS_COMPLETED


@pytest.mark.asyncio
async def test_run_queue_process_item_exception_marks_item_needs_human_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "overnight.yaml"
    manifest = _write_queue(queue_path)
    executor = TrancheQueueExecutor(queue_path=queue_path, repo_root=tmp_path)

    seen: list[str] = []

    async def fake_process_item(*, manifest, item, item_state, deadline):
        seen.append(item.item_id)
        if item.item_id == "issue-1046":
            raise FileNotFoundError("missing tranche dir")
        item_state.status = QUEUE_ITEM_STATUS_COMPLETED
        item_state.finished_at = item_state.updated_at = item_state.started_at
        return None

    monkeypatch.setattr(executor, "_process_item", fake_process_item)

    result = await executor.run()

    assert seen == ["issue-1046", "intake-docs", "issue-1047"]
    assert result["status"] == QUEUE_STATUS_COMPLETED
    assert result["counts"] == {"completed": 2, "needs_human": 1}

    state = TrancheQueueRunState.load(queue_state_path_for_queue(queue_path), manifest=manifest)
    crashed = state.item_states["issue-1046"]
    assert crashed.status == QUEUE_ITEM_STATUS_NEEDS_HUMAN
    assert crashed.stop_reason == "processing_error"
    assert crashed.findings == ["Queue item processing failed. Check logs for detail."]
    assert crashed.result["processing_error"] == {"error_type": "FileNotFoundError"}


@pytest.mark.asyncio
async def test_run_queue_resumes_from_existing_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "overnight.yaml"
    manifest = _write_queue(queue_path)
    state_path = queue_state_path_for_queue(queue_path)
    state = TrancheQueueRunState(queue_id=manifest.queue_id)
    state.ensure_manifest(manifest)
    state.item_states["issue-1046"] = TrancheQueueItemRunState(
        item_id="issue-1046",
        status=QUEUE_ITEM_STATUS_COMPLETED,
    )
    state.item_states["intake-docs"] = TrancheQueueItemRunState(
        item_id="intake-docs",
        status=QUEUE_ITEM_STATUS_PENDING,
    )
    state.item_states["issue-1047"] = TrancheQueueItemRunState(
        item_id="issue-1047",
        status=QUEUE_ITEM_STATUS_PENDING,
    )
    state.save(state_path)

    executor = TrancheQueueExecutor(queue_path=queue_path, repo_root=tmp_path)
    seen: list[str] = []

    async def fake_process_item(*, manifest, item, item_state, deadline):
        seen.append(item.item_id)
        item_state.status = QUEUE_ITEM_STATUS_COMPLETED
        item_state.finished_at = item_state.updated_at = item_state.started_at
        return None

    monkeypatch.setattr(executor, "_process_item", fake_process_item)

    result = await executor.run()

    assert seen == ["intake-docs", "issue-1047"]
    assert result["status"] == QUEUE_STATUS_COMPLETED


@pytest.mark.asyncio
async def test_process_item_persists_manifest_path_before_drive_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "overnight.yaml"
    manifest = _write_queue(queue_path)
    executor = TrancheQueueExecutor(queue_path=queue_path, repo_root=tmp_path)
    item = manifest.items[0]
    item_state = TrancheQueueItemRunState(item_id=item.item_id)

    queue_state = TrancheQueueRunState(queue_id=manifest.queue_id, status="running")
    queue_state.ensure_manifest(manifest)
    queue_state.current_item_id = item.item_id
    queue_state.save(queue_state_path_for_queue(queue_path))

    tranche_dir = tmp_path / "tranches" / "tranche-a"
    tranche_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tranche_dir / "tranche.yaml"
    manifest_path.write_text("manifest_id: tranche-a\n", encoding="utf-8")
    normalized_path = tranche_dir / "normalized_bundle.yaml"
    normalized_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        executor,
        "_bundle_for_item",
        lambda item, effective_autonomy_mode: {"objective": "test"},
    )
    monkeypatch.setattr(
        "aragora.swarm.tranche_queue.submit_intake_bundle",
        lambda *args, **kwargs: {
            "manifest_id": "tranche-a",
            "manifest_path": str(manifest_path),
            "tranche_dir": str(tranche_dir),
            "submission_status": "submitted",
            "inspection_status": "ready",
            "recommended_action": "run",
        },
    )
    monkeypatch.setattr(
        "aragora.swarm.tranche_queue.load_tranche_manifest",
        lambda path: SimpleNamespace(manifest_id="tranche-a"),
    )
    monkeypatch.setattr(
        "aragora.swarm.tranche_queue.TrancheInspector",
        lambda repo_root: SimpleNamespace(
            inspect=lambda tranche_manifest: {"preflight_status": "ready"}
        ),
    )

    async def fake_design_review(*, manifest, normalized_bundle, inspection):
        return {"recommendation": "approved", "record": {"recommendation": "approved"}}

    monkeypatch.setattr("aragora.swarm.tranche_queue.run_design_review", fake_design_review)
    monkeypatch.setattr(
        "aragora.swarm.tranche_queue.save_design_review", lambda *args, **kwargs: None
    )

    async def fake_drive_manifest(*, item, item_state, manifest_path, tranche_manifest, deadline):
        state = TrancheQueueRunState.load(queue_state_path_for_queue(queue_path), manifest=manifest)
        persisted = state.item_states[item.item_id]
        assert persisted.status == "running"
        assert persisted.manifest_id == "tranche-a"
        assert persisted.manifest_path == str(manifest_path)
        assert state.current_item_id == item.item_id
        item_state.status = QUEUE_ITEM_STATUS_COMPLETED
        item_state.finished_at = item_state.updated_at = item_state.started_at
        return None

    monkeypatch.setattr(executor, "_drive_manifest", fake_drive_manifest)

    result = await executor._process_item(
        manifest=manifest,
        item=item,
        item_state=item_state,
        deadline=datetime.now(timezone.utc) + timedelta(minutes=10),
    )

    assert result is None
    assert item_state.status == QUEUE_ITEM_STATUS_COMPLETED


@pytest.mark.asyncio
async def test_run_queue_needs_human_does_not_count_toward_consecutive_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "overnight.yaml"
    _write_queue(queue_path)
    executor = TrancheQueueExecutor(
        queue_path=queue_path,
        repo_root=tmp_path,
        max_consecutive_failures=2,
    )
    seen: list[str] = []

    async def fake_process_item(*, manifest, item, item_state, deadline):
        seen.append(item.item_id)
        item_state.status = QUEUE_ITEM_STATUS_NEEDS_HUMAN
        item_state.finished_at = item_state.updated_at = item_state.started_at
        return None

    monkeypatch.setattr(executor, "_process_item", fake_process_item)

    result = await executor.run()

    assert seen == ["issue-1046", "intake-docs", "issue-1047"]
    assert result["status"] == QUEUE_STATUS_COMPLETED
    assert result["counts"]["needs_human"] == 3
    assert result["consecutive_failures"] == 0


@pytest.mark.asyncio
async def test_run_queue_stops_after_configured_consecutive_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "overnight.yaml"
    _write_queue(queue_path)
    executor = TrancheQueueExecutor(
        queue_path=queue_path,
        repo_root=tmp_path,
        max_consecutive_failures=2,
    )
    seen: list[str] = []

    async def fake_process_item(*, manifest, item, item_state, deadline):
        seen.append(item.item_id)
        item_state.status = "stopped"
        item_state.finished_at = item_state.updated_at = item_state.started_at
        return None

    monkeypatch.setattr(executor, "_process_item", fake_process_item)

    result = await executor.run()

    assert seen == ["issue-1046", "intake-docs"]
    assert result["status"] == QUEUE_STATUS_STOPPED
    assert result["stop_reason"] == "max_consecutive_failures"
    assert result["counts"]["stopped"] == 2
    assert result["counts"]["pending"] == 1


def test_reconcile_queue_reports_truthful_counts(tmp_path: Path) -> None:
    queue_path = tmp_path / "overnight.yaml"
    manifest = _write_queue(queue_path)
    state = TrancheQueueRunState(queue_id=manifest.queue_id, status="running")
    state.ensure_manifest(manifest)
    state.item_states["issue-1046"] = TrancheQueueItemRunState(
        item_id="issue-1046",
        status=QUEUE_ITEM_STATUS_COMPLETED,
    )
    state.item_states["intake-docs"] = TrancheQueueItemRunState(
        item_id="intake-docs",
        status=QUEUE_ITEM_STATUS_NEEDS_HUMAN,
        findings=["Missing source material."],
    )
    state.item_states["issue-1047"] = TrancheQueueItemRunState(
        item_id="issue-1047",
        status=QUEUE_ITEM_STATUS_PENDING,
    )
    state.save(queue_state_path_for_queue(queue_path))

    payload = reconcile_tranche_queue(queue_path=queue_path, repo_root=tmp_path)

    assert payload["status"] == QUEUE_STATUS_STOPPED
    assert payload["stop_reason"] == "driver_stopped"
    assert payload["current_item_id"] is None
    assert payload["counts"] == {"completed": 1, "needs_human": 1, "pending": 1}
    assert payload["items"][1]["findings"] == ["Missing source material."]


def test_reconcile_queue_terminalizes_stale_running_item_and_stops_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_path = tmp_path / "overnight.yaml"
    manifest = _write_queue(queue_path)
    manifest_path = tmp_path / "tranche.yaml"
    manifest_path.write_text("manifest_id: tranche-test\n", encoding="utf-8")
    state = TrancheQueueRunState(
        queue_id=manifest.queue_id,
        status=QUEUE_STATUS_RUNNING,
        current_item_id="intake-docs",
    )
    state.ensure_manifest(manifest)
    state.item_states["issue-1046"] = TrancheQueueItemRunState(
        item_id="issue-1046",
        status=QUEUE_ITEM_STATUS_COMPLETED,
    )
    state.item_states["intake-docs"] = TrancheQueueItemRunState(
        item_id="intake-docs",
        status=QUEUE_ITEM_STATUS_RUNNING,
        manifest_path=str(manifest_path),
        tranche_status="running",
    )
    state.item_states["issue-1047"] = TrancheQueueItemRunState(
        item_id="issue-1047",
        status=QUEUE_ITEM_STATUS_PENDING,
    )
    state.save(queue_state_path_for_queue(queue_path))

    class _FakeTrancheState:
        def __init__(self, status: str, *, pr_url: str | None = None) -> None:
            self.status = status
            lane = SimpleNamespace(pr_url=pr_url)
            self.lane_states = {"lane-a": lane}

        def to_dict(self) -> dict[str, object]:
            return {
                "status": self.status,
                "lane_states": {
                    lane_id: {"pr_url": lane_state.pr_url}
                    for lane_id, lane_state in self.lane_states.items()
                },
            }

        def save(self, path: str | Path) -> None:
            Path(path).write_text("status: completed\n", encoding="utf-8")

    monkeypatch.setattr(
        "aragora.swarm.tranche_queue.load_tranche_run_state",
        lambda _: _FakeTrancheState("running"),
    )
    monkeypatch.setattr(
        "aragora.swarm.tranche_queue.refresh_tranche_state",
        lambda current, **kwargs: _FakeTrancheState(
            "completed", pr_url="https://example.test/pr/1"
        ),
    )

    payload = reconcile_tranche_queue(queue_path=queue_path, repo_root=tmp_path)

    assert payload["status"] == QUEUE_STATUS_STOPPED
    assert payload["stop_reason"] == "driver_stopped"
    assert payload["current_item_id"] is None
    assert payload["counts"] == {"completed": 2, "pending": 1}
    intake = next(item for item in payload["items"] if item["item_id"] == "intake-docs")
    assert intake["status"] == QUEUE_ITEM_STATUS_COMPLETED
    assert intake["tranche_status"] == "completed"
    assert intake["pr_urls"] == ["https://example.test/pr/1"]

    repaired = TrancheQueueRunState.load(queue_state_path_for_queue(queue_path), manifest=manifest)
    assert repaired.status == QUEUE_STATUS_STOPPED
    assert repaired.stop_reason == "driver_stopped"
    assert repaired.current_item_id is None
    assert repaired.item_states["intake-docs"].status == QUEUE_ITEM_STATUS_COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("systemic_reason", "expected_stop_reason"),
    [
        ("reviewer_routing_unavailable", "reviewer_routing_unavailable"),
        ("controller_publication_unavailable", "controller_publication_unavailable"),
        ("time_limit_exceeded", "time_limit_exceeded"),
    ],
)
async def test_run_queue_stops_on_systemic_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    systemic_reason: str,
    expected_stop_reason: str,
) -> None:
    queue_path = tmp_path / "overnight.yaml"
    _write_queue(queue_path)
    executor = TrancheQueueExecutor(queue_path=queue_path, repo_root=tmp_path)

    async def fake_process_item(*, manifest, item, item_state, deadline):
        item_state.status = "stopped"
        item_state.finished_at = item_state.updated_at = item_state.started_at
        item_state.stop_reason = systemic_reason
        return systemic_reason

    monkeypatch.setattr(executor, "_process_item", fake_process_item)

    result = await executor.run()

    assert result["status"] == QUEUE_STATUS_STOPPED
    assert result["stop_reason"] == expected_stop_reason
    assert result["counts"] == {"stopped": 1, "pending": 2}


@pytest.mark.asyncio
async def test_drive_manifest_publishes_terminal_deliverable_without_recorded_pr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / "overnight.yaml"
    _write_queue(queue_path)
    executor = TrancheQueueExecutor(queue_path=queue_path, repo_root=tmp_path)
    manifest_path = tmp_path / "tranche.yaml"
    manifest_path.write_text("manifest_id: tranche-test\n", encoding="utf-8")
    item = TrancheQueueItem(
        item_id="issue-1046",
        kind="issue",
        source="1046",
        merge_class="manual",
    )
    item_state = TrancheQueueItemRunState(
        item_id=item.item_id,
        status=QUEUE_ITEM_STATUS_RUNNING,
        effective_autonomy_mode="adaptive",
    )
    current = TrancheRunState(
        manifest_id="tranche-test",
        status=TRANCHE_STATUS_NEEDS_HUMAN,
        autonomy_mode="adaptive",
        lane_states={
            "lane-a": LaneRunState(
                lane_id="lane-a",
                status=LANE_STATUS_NEEDS_HUMAN,
                run_id="run-123",
            )
        },
    )
    artifact = TrancheLaneArtifact(
        lane_id="lane-a",
        source_ref="issue-1046",
        status="completed",
        run_id="run-123",
        metadata={
            "deliverable": {
                "branch": "codex/swarm-lane-a",
                "commit_shas": ["abc123"],
            }
        },
    )

    monkeypatch.setattr(
        "aragora.swarm.tranche_queue.load_tranche_run_state",
        lambda _: current,
    )
    monkeypatch.setattr(
        "aragora.swarm.tranche_queue.claim_driver",
        lambda state, **kwargs: state,
    )
    monkeypatch.setattr(
        "aragora.swarm.tranche_queue.release_driver",
        lambda state, **kwargs: state,
    )

    async def fake_watch_loop(state, **kwargs):
        return state

    monkeypatch.setattr("aragora.swarm.tranche_queue.watch_loop", fake_watch_loop)
    monkeypatch.setattr(
        TrancheQueueExecutor,
        "_load_artifact_for_lane",
        staticmethod(lambda *args, **kwargs: artifact),
    )
    monkeypatch.setattr(
        "aragora.swarm.tranche_queue.DevCoordinationStore",
        lambda repo_root: object(),
    )

    async def fake_integrate_lane(**kwargs):
        return {
            "pr_url": "https://example.test/pr/42",
            "publish_result": {"action": "pr_created"},
            "recommendation": "needs_human",
            "executed": False,
        }

    monkeypatch.setattr("aragora.swarm.tranche_queue.integrate_lane", fake_integrate_lane)

    tranche_manifest = SimpleNamespace(manifest_id="tranche-test")
    deadline = datetime.now(timezone.utc) + timedelta(minutes=5)

    result = await executor._drive_manifest(
        item=item,
        item_state=item_state,
        manifest_path=manifest_path,
        tranche_manifest=tranche_manifest,
        deadline=deadline,
    )

    assert result is None
    assert item_state.status == QUEUE_ITEM_STATUS_NEEDS_HUMAN
    assert item_state.pr_urls == ["https://example.test/pr/42"]
    assert current.lane_states["lane-a"].pr_url == "https://example.test/pr/42"
    assert any(
        event.get("type") == "integrate" and event.get("pr_url") == "https://example.test/pr/42"
        for event in item_state.events
    )
