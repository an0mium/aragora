"""Tests for scripts/route_burndown_batches.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import route_burndown_batches as batches


def _write_inventory(path: Path, items: list[dict[str, str]]) -> None:
    path.write_text(json.dumps({"items": items}))


def _item(
    item_id: str, *, status: str = "open", source: str = "python_sdk_drift"
) -> dict[str, str]:
    return {
        "id": item_id,
        "source": source,
        "status": status,
        "class": "start_cohort",
        "discovered_on": "2026-04-17",
    }


def test_load_open_items_sorts_and_excludes_resolved(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.json"
    _write_inventory(
        inventory,
        [
            _item("python_sdk_drift:Z"),
            _item("python_sdk_drift:B", status="resolved"),
            _item("python_sdk_drift:A"),
        ],
    )

    result = batches.load_open_items(inventory)

    assert [item["id"] for item in result] == [
        "python_sdk_drift:A",
        "python_sdk_drift:Z",
    ]


def test_load_open_items_rejects_duplicate_ids(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.json"
    _write_inventory(inventory, [_item("python_sdk_drift:A"), _item("python_sdk_drift:A")])

    with pytest.raises(batches.InventoryError, match="duplicate inventory id"):
        batches.load_open_items(inventory)


def test_partition_and_render_are_stable() -> None:
    items = [_item(f"python_sdk_drift:{value}") for value in ("E", "A", "D", "B", "C")]

    first = batches.build_outputs(
        items,
        batch_size=2,
        inventory_label="inventory.json",
        playbook_label="playbook.md",
    )
    second = batches.build_outputs(
        list(reversed(items)),
        batch_size=2,
        inventory_label="inventory.json",
        playbook_label="playbook.md",
    )

    assert first == second
    assert sorted(first) == ["batch-001.md", "batch-002.md", "batch-003.md", "index.md"]
    assert "`python_sdk_drift:A`" in first["batch-001.md"]
    assert "`python_sdk_drift:C`" in first["batch-002.md"]
    assert "Open entries: `5`" in first["index.md"]


def test_packet_digest_changes_with_membership() -> None:
    one = [_item("python_sdk_drift:A")]
    two = [*one, _item("python_sdk_drift:B")]

    assert batches.batch_digest(one) != batches.batch_digest(two)
    assert batches.batch_digest([_item("A\nB")]) != batches.batch_digest([_item("A"), _item("B")])


def test_write_outputs_refuses_stale_packet(tmp_path: Path) -> None:
    (tmp_path / "batch-999.md").write_text("obsolete")

    with pytest.raises(batches.InventoryError, match="stale generated packets"):
        batches.write_outputs(tmp_path, {"index.md": "index\n"})


def test_write_outputs_round_trip(tmp_path: Path) -> None:
    outputs = {"index.md": "index\n", "batch-001.md": "packet\n"}

    batches.write_outputs(tmp_path, outputs)
    batches.write_outputs(tmp_path, outputs)

    assert {path.name: path.read_text() for path in tmp_path.iterdir()} == outputs
