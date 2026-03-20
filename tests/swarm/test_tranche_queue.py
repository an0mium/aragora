from __future__ import annotations

from pathlib import Path

import pytest

from aragora.swarm.tranche_queue import (
    QUEUE_ITEM_STATUS_COMPLETED,
    QUEUE_ITEM_STATUS_NEEDS_HUMAN,
    QUEUE_ITEM_STATUS_PENDING,
    QUEUE_STATUS_COMPLETED,
    QUEUE_STATUS_STOPPED,
    TrancheQueueExecutor,
    TrancheQueueItemRunState,
    TrancheQueueManifest,
    TrancheQueueRunState,
    queue_state_path_for_queue,
    reconcile_tranche_queue,
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

    assert payload["status"] == "running"
    assert payload["counts"] == {"completed": 1, "needs_human": 1, "pending": 1}
    assert payload["items"][1]["findings"] == ["Missing source material."]


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
