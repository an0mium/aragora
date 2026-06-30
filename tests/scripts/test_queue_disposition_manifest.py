"""Tests for ``scripts/queue_disposition_manifest.py``.

All GitHub and inventory boundaries are injected; no test touches the network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


manifest_cli = _load_module("queue_disposition_manifest.py")
NOW = datetime(2026, 6, 30, 12, 0, 0, tzinfo=timezone.utc)


def test_run_manifest_writes_atomic_payload_and_summary(tmp_path: Path) -> None:
    out = tmp_path / "manifest.json"
    lines: list[str] = []

    rc = manifest_cli.run_manifest(
        list_prs=lambda: [
            {
                "number": 8389,
                "title": "feat(odr): in-package ODR verification engine",
                "isDraft": False,
                "mergeable": "MERGEABLE",
                "headRefName": "claude/odr3-verify-core",
                "headRefOid": "abc",
                "createdAt": "2026-06-20T12:00:00Z",
                "additions": 734,
                "deletions": 0,
                "changedFiles": 2,
                "labels": [],
            }
        ],
        merge_packet=lambda pr: {"tier": 1, "unresolved_dissent": False},
        inventory_candidates=lambda: [
            {
                "candidate_id": "wt",
                "classification": "patch_equivalent_or_merged",
                "decision": "cleanup_candidate",
                "git": {"branch": "codex/old", "head": "def"},
                "proof": ["patch-equivalent to base"],
            }
        ],
        out_file=str(out),
        summary=True,
        now=NOW,
        log=lines.append,
    )

    assert rc == manifest_cli.EXIT_OK
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["generated_at"] == "2026-06-30T12:00:00Z"
    assert payload["summary"]["total_items"] == 2
    assert payload["items"][0]["disposition"] == "harvest_now"
    assert payload["items"][1]["disposition"] == "close_or_delete_after_manifest"
    assert lines == [
        "queue disposition: total=2 harvest=1 human=0 park=0 close_delete=1 operator_required=1"
    ]


def test_run_manifest_records_merge_packet_failures() -> None:
    lines: list[str] = []

    def boom(pr: int) -> dict[str, Any]:
        raise RuntimeError("transport down")

    rc = manifest_cli.run_manifest(
        list_prs=lambda: [
            {
                "number": 1,
                "title": "feat(odr): verify",
                "isDraft": False,
                "mergeable": "MERGEABLE",
                "headRefName": "codex/x",
                "headRefOid": "abc",
                "createdAt": "2026-06-20T12:00:00Z",
                "additions": 10,
                "deletions": 0,
                "changedFiles": 1,
                "labels": [],
            }
        ],
        merge_packet=boom,
        now=NOW,
        log=lines.append,
    )

    assert rc == manifest_cli.EXIT_OK
    payload = json.loads(lines[0])
    assert payload["annotations"] == ["merge_packet_failed:#1:transport down"]
    assert payload["items"][0]["disposition"] == "harvest_now"


def test_collect_merge_packets_uses_bounded_parallel_workers() -> None:
    annotations: list[str] = []

    entries = manifest_cli.collect_merge_packets(
        [{"number": 1}, {"number": 2}],
        lambda pr: {"tier": pr},
        workers=2,
        annotations=annotations,
    )

    assert entries == {1: {"tier": 1}, 2: {"tier": 2}}
    assert annotations == []
