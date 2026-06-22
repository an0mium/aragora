"""Productize ``blocked_auth_failure`` — the largest single failure class on
the rev-4 B0 corpus (7/28 ticks = 25%).

This test locks the productization contract for the failure class:

1. The terminal-truth fixture at
   ``benchmarks/fixtures/swarm/terminal_truth/blocked_auth_failure.json``
   carries five canonical shapes, including the dominant production-observed
   shape (``worker_outcome="blocked_auth_failure"``, ``status="needs_human"``).
2. Every shape classifies to ``TerminalClass.BLOCKED_AUTH_FAILURE`` via
   :func:`aragora.swarm.terminal_truth.classify_from_metrics`.
3. ``docs/benchmarks/rescue_productization.json`` records a durable ledger
   entry for the class with a link to A2 admission-class productization
   (issue #7209), satisfying Operating Law: "if humans intervene twice for
   the same class of failure, the next system change should absorb that
   rescue as product behavior".
4. The classifier still treats the ``auth`` substring in ``worker_outcome``
   as the canonical trigger; this guards the productization invariant
   that *any* future auth-failure outcome shape will be classified
   correctly without the fixture needing to enumerate it.
5. The preflight-auth hint tuple still contains ``"auth"`` so preflight
   failures with ``auth`` in their failed-check text are also classified
   correctly.

No agent or worker process is spawned. The test is fixture-driven and
finishes in under a second.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aragora.swarm.terminal_truth import (
    TerminalClass,
    _PREFLIGHT_AUTH_HINTS,
    classify_from_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    REPO_ROOT / "benchmarks" / "fixtures" / "swarm" / "terminal_truth" / "blocked_auth_failure.json"
)
RESCUE_PRODUCTIZATION_PATH = REPO_ROOT / "docs" / "benchmarks" / "rescue_productization.json"


def _load_fixture_rows() -> list[dict[str, Any]]:
    rows = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    return rows


def _load_productization() -> dict[str, Any]:
    payload = json.loads(RESCUE_PRODUCTIZATION_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_fixture_has_five_canonical_shapes() -> None:
    rows = _load_fixture_rows()
    assert len(rows) == 5, (
        "blocked_auth_failure fixture must carry the 5 canonical productization "
        "shapes (3 synthetic + 1 dominant production + 1 preflight)."
    )


def test_every_fixture_row_classifies_to_blocked_auth_failure() -> None:
    rows = _load_fixture_rows()
    for idx, row in enumerate(rows):
        result = classify_from_metrics(row)
        assert result is TerminalClass.BLOCKED_AUTH_FAILURE, (
            f"row[{idx}] outcome={row.get('worker_outcome')!r} classified as "
            f"{result.value!r}; expected blocked_auth_failure"
        )


def test_fixture_records_dominant_production_shape() -> None:
    rows = _load_fixture_rows()
    matching = [
        row
        for row in rows
        if row.get("worker_status") == "needs_human"
        and row.get("worker_outcome") == "blocked_auth_failure"
    ]
    assert matching, (
        "The dominant production-observed shape (worker_status=needs_human + "
        "worker_outcome=blocked_auth_failure) must be in the fixture so any "
        "future classifier refactor that drops it will fail this test."
    )


def test_fixture_records_preflight_subprocess_shape() -> None:
    """The A2 plan (#7209) notes that 7 of 7 rev-4 ticks come from preflight
    subprocess auth probes. This shape (``provider_auth_required``) must be
    in the fixture so the productization (wiring CredentialEnvelope into
    _run_preflight_worker) keeps a regression anchor."""
    rows = _load_fixture_rows()
    matching = [row for row in rows if row.get("worker_outcome") == "provider_auth_required"]
    assert matching, (
        "The preflight subprocess auth shape must be in the fixture; the A2 "
        "plan (#7209) productizes this exact shape by wiring CredentialEnvelope "
        "into _run_preflight_worker."
    )


def test_fixture_rows_satisfy_required_schema() -> None:
    required_keys = {
        "worker_status",
        "worker_outcome",
        "elapsed_seconds",
        "files_changed",
        "has_deliverable",
        "publish_action",
        "expected_class",
    }
    rows = _load_fixture_rows()
    for idx, row in enumerate(rows):
        missing = required_keys - set(row.keys())
        assert not missing, f"row[{idx}] missing keys {missing}"
        assert row["expected_class"] == "blocked_auth_failure"


def test_classifier_treats_auth_substring_as_trigger() -> None:
    """Productization invariant: the classifier must keep the ``"auth"``
    substring as the canonical trigger for ``BLOCKED_AUTH_FAILURE``, so the
    productization is robust against future auth-related outcome names
    without the fixture having to enumerate them ahead of time."""
    synthetic_row = {
        "worker_status": "failed",
        "worker_outcome": "auth_provider_quota_exceeded",
        "elapsed_seconds": 1.0,
        "files_changed": 0,
        "has_deliverable": False,
        "publish_action": "",
    }
    assert classify_from_metrics(synthetic_row) is TerminalClass.BLOCKED_AUTH_FAILURE


def test_preflight_hint_tuple_includes_auth() -> None:
    """Productization invariant: the preflight classifier hint tuple must
    contain ``"auth"`` so any preflight failure text containing that hint
    short-circuits to ``BLOCKED_AUTH_FAILURE`` rather than being mis-classified
    as a generic blocker."""
    assert "auth" in _PREFLIGHT_AUTH_HINTS


def test_rescue_productization_records_blocked_auth_failure() -> None:
    payload = _load_productization()
    entries = payload["entries"]
    assert isinstance(entries, list)

    classes = [entry["class"] for entry in entries]
    assert "blocked_auth_failure" in classes, (
        "rescue_productization.json must record a durable ledger entry for "
        "the blocked_auth_failure rescue class, per Operating Law."
    )

    entry = next(item for item in entries if item["class"] == "blocked_auth_failure")
    assert entry["target"] == "#7209"
    assert entry["target_kind"] == "issue"
    notes = entry["notes"]
    assert "blocked_auth_failure" in notes
    assert "CredentialEnvelope" in notes
    assert "preflight" in notes.lower()
    assert "rev-4" in notes


@pytest.mark.parametrize(
    "extra_synthetic_shape",
    [
        {"worker_status": "failed", "worker_outcome": "auth_expired"},
        {"worker_status": "error", "worker_outcome": "auth_disabled"},
        {"worker_status": "blocked", "worker_outcome": "auth_revoked"},
    ],
)
def test_classifier_extensibility_to_new_auth_shapes(
    extra_synthetic_shape: dict[str, Any],
) -> None:
    """Confirms new auth-failure outcomes appearing in production will be
    classified correctly without a fixture update being required first."""
    row = {
        **extra_synthetic_shape,
        "elapsed_seconds": 2.0,
        "files_changed": 0,
        "has_deliverable": False,
        "publish_action": "",
    }
    assert classify_from_metrics(row) is TerminalClass.BLOCKED_AUTH_FAILURE


def test_fixture_path_is_canonical() -> None:
    """The fixture must live at the canonical terminal-truth path so the
    existing parametrized test in tests/swarm/test_terminal_truth_benchmark.py
    keeps validating it alongside the other 13 fixtures."""
    assert FIXTURE_PATH.is_file()
    assert FIXTURE_PATH.parent.name == "terminal_truth"
    assert FIXTURE_PATH.parent.parent.name == "swarm"
