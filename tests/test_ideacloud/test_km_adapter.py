"""Behavior tests for the Idea Cloud Knowledge Mound adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aragora.ideacloud.adapters.km_adapter import IdeaCloudAdapter
from aragora.ideacloud.graph.node import IdeaNode
from aragora.ideacloud.graph import node as node_module


def _make_cloud(*nodes: IdeaNode) -> SimpleNamespace:
    return SimpleNamespace(
        graph=SimpleNamespace(nodes={node.id: node for node in nodes}),
        save=Mock(),
        stats={},
    )


def _make_adapter(cloud: SimpleNamespace) -> IdeaCloudAdapter:
    adapter = IdeaCloudAdapter(idea_cloud=cloud, enable_tracing=False)
    adapter._record_metric = Mock()
    adapter._emit_event = Mock()
    return adapter


@pytest.mark.asyncio
async def test_sync_to_km_continues_after_node_failure() -> None:
    failed = IdeaNode(id="ic_fail", title="fail", body="body", km_synced=False)
    synced = IdeaNode(id="ic_ok", title="ok", body="body", km_synced=False)
    cloud = _make_cloud(failed, synced)
    adapter = _make_adapter(cloud)

    async def ingest(payload: dict[str, object]) -> None:
        if payload["source_id"] == "ic_fail":
            raise RuntimeError("boom")

    adapter.km = SimpleNamespace(ingest=AsyncMock(side_effect=ingest))

    result = await adapter.sync_to_km()

    assert result == {
        "records_synced": 1,
        "records_skipped": 0,
        "records_failed": 1,
    }
    assert failed.km_synced is False
    assert synced.km_synced is True
    cloud.save.assert_called_once()


@pytest.mark.asyncio
async def test_sync_from_km_continues_after_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = IdeaNode(id="ic_fail", title="fail", body="body", km_synced=True)
    updated = IdeaNode(id="ic_ok", title="ok", body="body", km_synced=True)
    cloud = _make_cloud(failed, updated)
    adapter = _make_adapter(cloud)

    counter = {"calls": 0}

    def fake_now_iso() -> str:
        counter["calls"] += 1
        if counter["calls"] == 1:
            raise RuntimeError("clock failed")
        return "2026-04-06T00:00:00+00:00"

    monkeypatch.setattr(node_module, "_now_iso", fake_now_iso)

    result = await adapter.sync_from_km(
        [
            {"source_id": "ic_fail", "confidence": 0.2, "validation_status": "disputed"},
            {"source_id": "ic_ok", "confidence": 0.9, "validation_status": "confirmed"},
        ]
    )

    assert result == {
        "records_analyzed": 2,
        "records_updated": 1,
    }
    assert failed.confidence == 0.0
    assert failed.pipeline_status == "inbox"
    assert updated.confidence == 0.9
    assert updated.pipeline_status == "candidate"
    cloud.save.assert_called_once()
