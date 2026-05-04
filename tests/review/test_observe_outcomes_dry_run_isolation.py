"""Round 31a Phase 3: dry-run isolation contract.

The insufficiency receipt produced by ``run_observe_outcomes`` in
dry-run mode (write=False) must:

1. Land under ``.aragora/evolve-round/observe-outcomes/<utc-iso>/`` and
   never under a round-id directory.
2. Never write under ``docs/`` or ``tests/`` even transiently.
3. Not actually create the file when write=False (path is proposed
   only; the payload is returned in the response so callers can inspect
   it without filesystem side effects).

These tests pin the contract so future refactors cannot quietly drift
back to writing dry-run artifacts under round-receipt directories.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from aragora.review.observe_outcomes_cli import (
    DEFAULT_PER_RECEIPT_EVENT_CAP,
    DEFAULT_WINDOW_DAYS,
    run_observe_outcomes,
)

UTC = timezone.utc


def _silent_provider(pr_number, head_sha, event_cap):
    return [], None


def _list_files(root: Path, prefix: str) -> list[Path]:
    target = root / prefix
    if not target.exists():
        return []
    return sorted(p for p in target.rglob("*") if p.is_file())


def test_dry_run_proposes_observe_outcomes_path(tmp_path: Path) -> None:
    """Dry-run must propose a path under
    .aragora/evolve-round/observe-outcomes/<utc-iso>/ — not under any
    round-id directory.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store_root = tmp_path / "store"
    store_root.mkdir()
    (store_root / "receipts").mkdir()

    summary = run_observe_outcomes(
        store_root=str(store_root),
        repo_root=repo_root,
        window_end=datetime(2026, 5, 1, tzinfo=UTC),
        window_days=DEFAULT_WINDOW_DAYS,
        max_receipts=10,
        per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
        write=False,
        timeline_provider=_silent_provider,
        round_id="2026-05-01a",
    )

    proposed = summary["insufficiency_receipt_path"]
    assert proposed is not None, "dry-run with no receipts must still propose a path"
    p = Path(proposed)
    parts = p.parts
    assert ".aragora" in parts
    assert "evolve-round" in parts
    assert "observe-outcomes" in parts, (
        f"dry-run insufficiency-receipt path must be under "
        f".aragora/evolve-round/observe-outcomes/, got: {p}"
    )
    assert p.name == "insufficiency-receipt.json"
    # The <utc-iso> directory must look like ISO-compact (YYYYMMDDTHHMMSSZ)
    iso_dir = p.parent.name
    assert len(iso_dir) == 16
    assert iso_dir.endswith("Z")
    assert iso_dir[:8].isdigit()


def test_dry_run_does_not_write_under_docs_or_tests(tmp_path: Path) -> None:
    """No artifact may appear under docs/ or tests/ during dry-run."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "docs").mkdir()
    (repo_root / "tests").mkdir()
    store_root = tmp_path / "store"
    store_root.mkdir()
    (store_root / "receipts").mkdir()

    run_observe_outcomes(
        store_root=str(store_root),
        repo_root=repo_root,
        window_end=datetime(2026, 5, 1, tzinfo=UTC),
        window_days=DEFAULT_WINDOW_DAYS,
        max_receipts=10,
        per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
        write=False,
        timeline_provider=_silent_provider,
        round_id="2026-05-01a",
    )

    docs_files = _list_files(repo_root, "docs")
    tests_files = _list_files(repo_root, "tests")
    assert docs_files == [], f"dry-run wrote under docs/: {docs_files}"
    assert tests_files == [], f"dry-run wrote under tests/: {tests_files}"


def test_dry_run_does_not_actually_write_file(tmp_path: Path) -> None:
    """write=False must NOT create the proposed file on disk."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store_root = tmp_path / "store"
    store_root.mkdir()
    (store_root / "receipts").mkdir()

    summary = run_observe_outcomes(
        store_root=str(store_root),
        repo_root=repo_root,
        window_end=datetime(2026, 5, 1, tzinfo=UTC),
        window_days=DEFAULT_WINDOW_DAYS,
        max_receipts=10,
        per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
        write=False,
        timeline_provider=_silent_provider,
        round_id="2026-05-01a",
    )

    proposed = Path(summary["insufficiency_receipt_path"])
    assert not proposed.exists(), f"dry-run actually wrote a file: {proposed}"
    # Payload still returned in response for caller inspection.
    assert summary["insufficiency_receipt"] is not None


def test_write_mode_routes_to_round_id_directory(tmp_path: Path) -> None:
    """write=True (non-dry-run) MUST route to the original
    .aragora/evolve-round/<round_id>/ directory, not the
    observe-outcomes/<utc-iso>/ dry-run isolation tree.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store_root = tmp_path / "store"
    store_root.mkdir()
    (store_root / "receipts").mkdir()

    summary = run_observe_outcomes(
        store_root=str(store_root),
        repo_root=repo_root,
        window_end=datetime(2026, 5, 1, tzinfo=UTC),
        window_days=DEFAULT_WINDOW_DAYS,
        max_receipts=10,
        per_receipt_event_cap=DEFAULT_PER_RECEIPT_EVENT_CAP,
        write=True,
        timeline_provider=_silent_provider,
        round_id="2026-05-01a",
    )

    proposed = Path(summary["insufficiency_receipt_path"])
    parts = proposed.parts
    assert "observe-outcomes" not in parts, (
        f"write=True must NOT route through observe-outcomes/<utc-iso>/, got: {proposed}"
    )
    assert "2026-05-01a" in parts, (
        f"write=True must route to the round_id directory, got: {proposed}"
    )
    assert proposed.name == "phase-a-observe-outcomes-insufficiency-receipt.json"
    assert proposed.exists()
    payload = json.loads(proposed.read_text(encoding="utf-8"))
    assert payload.get("kind") == "phase-a-observe-outcomes-insufficiency-receipt"
