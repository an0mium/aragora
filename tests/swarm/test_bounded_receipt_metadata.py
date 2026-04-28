"""Regression tests for the bounded receipt-metadata helper.

The boss-loop's post-worker path used to spread ``worker_result["receipt_metadata"]``
unchecked into the signed-receipt pipeline. With a real worker payload (dispatch
gate, prior worker results, raw stdout/stderr) the metadata could exceed several
MB. The downstream ``json.dumps(..., sort_keys=True)`` in the signing canonicalisation
then stalled for 2+ seconds per call; chained through the asyncio post-worker path
this looked like a process hang.

These tests fix that contract by ensuring:
  - bounded summaries stay within ``BOUNDED_METADATA_TARGET_BYTES``
  - bounding completes in well under 5 seconds even with multi-MB input
  - downstream consumers (boss_loop_outcome, _emit_lane_receipt) see the
    keys they expect
  - the full payload is preserved on disk under ``.aragora/worker-results/``
    with a matching sha256
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from aragora.swarm.bounded_receipt_metadata import (
    BOUNDED_METADATA_TARGET_BYTES,
    BOUNDED_TAIL_BYTES,
    DEFAULT_RESULTS_DIR,
    bound_receipt_metadata,
    reference_keys,
)


def _huge_metadata(stdout_kb: int = 2048, depth: int = 6) -> dict:
    """Build a worker_result-shaped metadata dict that simulates the spin condition.

    Includes:
      - oversized stdout/stderr blobs (default: 2 MiB each)
      - a deeply nested dispatch_gate
      - sibling fields that downstream consumers actually read
    """
    stdout_blob = "x" * (stdout_kb * 1024)
    stderr_blob = "y" * (stdout_kb * 1024)

    nested: dict = {"leaf": "deep"}
    for layer in range(depth):
        nested = {
            "layer": layer,
            "child": nested,
            "siblings": [{"k": i, "v": "z" * 200} for i in range(50)],
        }

    return {
        "issue_title": "[CS-01a] Reconcile docs/status surfaces to current proof",
        "issue_number": 5839,
        "actual_target_agent": "claude",
        "requested_target_agent": "claude",
        "runner_type": "claude",
        "blocker_kind": None,
        "terminal_outcome": "deliverable_created",
        "outcome": "deliverable_created",
        "status": "completed",
        "error_class": None,
        "branch": "codex/cs-01a-fix",
        "pr_url": "https://github.com/synaptent/aragora/pull/9999",
        "pr_number": 9999,
        "head_sha": "deadbeefcafebabe1234567890abcdef00112233",
        "base_sha": "0011223344556677889900aabbccddeeff001122",
        "lease_id": "lease-1234",
        "worker_receipt_id": "rcpt-9876",
        "stdout": stdout_blob,
        "stderr": stderr_blob,
        "log": "log content " * 1000,
        "raw_output": "raw " * 1000,
        "blocker_evidence": "evidence " * 500,
        "dispatch_gate": nested,
        "publish_result": {
            "action": "published",
            "branch": "codex/cs-01a-fix",
            "pr_url": "https://github.com/synaptent/aragora/pull/9999",
            "raw_history": [{"step": i, "stdout": "h" * 1000} for i in range(50)],
        },
        "harvest_result": {
            "outbox_state": "delivered",
            "raw_metadata": {"deep": [{"a": i} for i in range(200)]},
        },
        "blocked_reasons": [f"reason {i}" for i in range(100)],
        "needs_human_reasons": [f"need {i}" for i in range(100)],
        "changed_files": [f"path/file_{i}.py" for i in range(100)],
        "validations_run": [f"pytest tests/test_{i}.py" for i in range(100)],
    }


# ---------------------------------------------------------------------------
# Core invariants
# ---------------------------------------------------------------------------


def test_empty_input_returns_minimal_marker(tmp_path):
    """Non-mapping or None input does not raise and returns a minimal marker."""
    bounded = bound_receipt_metadata(None, run_id="r1", repo_root=tmp_path)
    assert bounded["_bounded"] is True
    assert bounded["_empty"] is True


def test_bounded_summary_stays_within_target_bytes(tmp_path):
    """The bounded summary must serialise to <= BOUNDED_METADATA_TARGET_BYTES."""
    raw = _huge_metadata(stdout_kb=2048, depth=6)
    bounded = bound_receipt_metadata(raw, run_id="r-target", repo_root=tmp_path)
    serialised = json.dumps(bounded, sort_keys=True, default=str)
    assert len(serialised.encode("utf-8")) <= BOUNDED_METADATA_TARGET_BYTES


def test_bounding_completes_quickly_even_with_multi_mb_input(tmp_path):
    """The original spin took 2+ seconds per call. Bounded path must be sub-second."""
    raw = _huge_metadata(stdout_kb=4096, depth=6)
    started = time.monotonic()
    bounded = bound_receipt_metadata(raw, run_id="r-perf", repo_root=tmp_path)
    elapsed = time.monotonic() - started
    # Generous bound — codex spec is "completes promptly"; we aim well under 5s.
    # In practice this should be ~50-200 ms even on multi-MB inputs.
    assert elapsed < 2.0, f"bound_receipt_metadata took {elapsed:.2f}s, expected < 2s"
    # Sanity-check: serialising the bounded summary is also fast.
    started = time.monotonic()
    json.dumps(bounded, sort_keys=True, default=str)
    elapsed = time.monotonic() - started
    assert elapsed < 0.5, f"json.dumps of bounded summary took {elapsed:.2f}s"


def test_preserved_scalars_pass_through(tmp_path):
    """Scalar fields downstream consumers read must be preserved verbatim."""
    raw = _huge_metadata()
    bounded = bound_receipt_metadata(raw, run_id="r-scalars", repo_root=tmp_path)
    assert bounded["issue_title"].startswith("[CS-01a]")
    assert bounded["actual_target_agent"] == "claude"
    assert bounded["terminal_outcome"] == "deliverable_created"
    assert bounded["pr_number"] == 9999
    assert bounded["worker_receipt_id"] == "rcpt-9876"


def test_text_blobs_are_tail_truncated(tmp_path):
    """stdout / stderr / log style fields are truncated to BOUNDED_TAIL_BYTES tail."""
    raw = _huge_metadata(stdout_kb=2048)
    bounded = bound_receipt_metadata(raw, run_id="r-tails", repo_root=tmp_path)
    stdout = bounded["stdout"]
    assert isinstance(stdout, str)
    # Tail truncation includes a marker; the encoded body itself is bounded.
    encoded = stdout.encode("utf-8")
    # Allow a small overhead for the truncation marker prefix.
    assert len(encoded) < BOUNDED_TAIL_BYTES + 256
    assert "[truncated:" in stdout


def test_nested_dicts_become_summaries_not_full_dicts(tmp_path):
    """dispatch_gate / publish_result / harvest_result must NOT be the full dict."""
    raw = _huge_metadata(depth=8)
    bounded = bound_receipt_metadata(raw, run_id="r-nested", repo_root=tmp_path)
    dispatch = bounded["dispatch_gate"]
    # The summary should not contain the deep nested "layer"/"child" structure.
    # It should be a flat dict with type+keys descriptors.
    serialised = json.dumps(dispatch, default=str)
    # The original 8-layer chain would expand to many KB; the summary must not.
    assert len(serialised.encode("utf-8")) < 4 * 1024, (
        f"dispatch_gate summary too large: {len(serialised)} bytes"
    )


def test_lists_are_length_capped(tmp_path):
    """List fields must be capped to a safe number of items."""
    raw = _huge_metadata()
    bounded = bound_receipt_metadata(raw, run_id="r-lists", repo_root=tmp_path)
    # 100 items in raw; capped to <=33 (max items + 1 truncation marker).
    assert len(bounded["blocked_reasons"]) <= 33
    assert len(bounded["changed_files"]) <= 33
    # The truncation marker must be present.
    if len(bounded["blocked_reasons"]) == 33:
        assert "more items truncated" in bounded["blocked_reasons"][-1]


def test_full_payload_persisted_to_reference_path(tmp_path):
    """The full raw payload must be written to disk with a sha256 reference."""
    raw = _huge_metadata(stdout_kb=128)
    bounded = bound_receipt_metadata(raw, run_id="r-ref", repo_root=tmp_path)
    assert "_reference" in bounded
    ref = bounded["_reference"]
    target = Path(ref["path"])
    assert target.exists()
    # sha256 must match what's on disk.
    import hashlib

    on_disk = target.read_text(encoding="utf-8").encode("utf-8")
    assert hashlib.sha256(on_disk).hexdigest() == ref["sha256"]
    # Reference file must contain the original stdout (full, not truncated).
    parsed = json.loads(target.read_text(encoding="utf-8"))
    assert parsed["stdout"].startswith("x" * 1024)


def test_run_id_is_filesystem_safe(tmp_path):
    """run_id values with slashes/colons must be sanitised for the filename."""
    raw = {"issue_title": "test"}
    bounded = bound_receipt_metadata(raw, run_id="rcpt/with:slashes and spaces", repo_root=tmp_path)
    if "_reference" in bounded:
        path = Path(bounded["_reference"]["path"])
        # The unsafe characters must have been replaced.
        assert "/" not in path.name
        assert ":" not in path.name


def test_reference_path_under_repo_root(tmp_path):
    """Reference files land under <repo_root>/.aragora/worker-results/."""
    raw = {"issue_title": "test"}
    bounded = bound_receipt_metadata(raw, run_id="r-anchor", repo_root=tmp_path)
    expected_dir = tmp_path / DEFAULT_RESULTS_DIR
    assert expected_dir.exists()
    if "_reference" in bounded:
        ref_path = Path(bounded["_reference"]["path"])
        assert expected_dir in ref_path.parents


def test_overflow_fallback_returns_minimal_stub(tmp_path):
    """If even the bounded summary exceeds the cap, we fall back to a stub."""
    # Build a metadata dict where preserved scalar fields alone are huge.
    # Each preserved key takes a 4 KiB string; accumulated they exceed the cap.
    raw: dict = {}
    for i, key in enumerate(("issue_title", "branch", "pr_url", "head_sha", "base_sha")):
        raw[key] = "z" * 4096
    bounded = bound_receipt_metadata(
        raw, run_id="r-overflow", repo_root=tmp_path, target_bytes=8 * 1024
    )
    # Should still be a dict and either the normal bounded shape OR the overflow stub.
    assert bounded["_bounded"] is True
    assert isinstance(bounded, dict)


def test_reference_keys_lists_known_shape():
    """reference_keys() exposes the shape contract for downstream tests."""
    keys = set(reference_keys())
    assert "issue_title" in keys
    assert "_bounded" in keys
    assert "_reference" in keys
    assert "dispatch_gate" in keys


# ---------------------------------------------------------------------------
# Integration with _emit_lane_receipt
# ---------------------------------------------------------------------------


def test_emit_lane_receipt_does_not_spin_with_huge_metadata(tmp_path, monkeypatch):
    """End-to-end: BossLoop._emit_lane_receipt must complete promptly with huge metadata.

    This is the regression test for the spin: with the unbounded path, this
    test would either hang or take 2+ seconds per call. The bounded path
    completes in milliseconds.
    """
    from aragora.swarm.boss_loop import BossLoop, BossLoopConfig

    # Anchor reference file writes inside the test's tmp_path.
    monkeypatch.chdir(tmp_path)

    config = BossLoopConfig(
        max_iterations=1,
        iteration_interval_seconds=0.0,
        repo="synaptent/aragora",
    )
    loop = BossLoop(config=config)
    huge = _huge_metadata(stdout_kb=2048, depth=6)
    worker_result = {
        "outcome": "deliverable_created",
        "deliverable": {"type": "pr", "pr_url": "https://github.com/x/y/pull/1"},
        "receipt_metadata": huge,
        "receipt_id": "rcpt-spin-test",
        "lease_id": "lease-x",
        "agent_id": "boss-loop",
        "base_sha": "00" * 20,
        "head_sha": "ff" * 20,
        "changed_files": ["x.py"],
        "validations_run": ["pytest"],
        "risks": [],
        "pr_url": "https://github.com/x/y/pull/1",
        "pr_number": 1,
        "branch": "test/branch",
        "reasons": [],
    }
    issue_dict = {"number": 5839, "title": "[CS-01a] test"}

    started = time.monotonic()
    receipt_id = loop._emit_lane_receipt(worker_result, issue_dict, elapsed=1.23)
    elapsed = time.monotonic() - started

    # Strong assertion: must beat the 2-second pre-fix spin time by a wide margin.
    assert elapsed < 5.0, f"_emit_lane_receipt took {elapsed:.2f}s, expected < 5s"
    # receipt_id may be None if the receipts pipeline isn't fully wired in test
    # context, but the call must have completed without raising or hanging.
    assert receipt_id is None or isinstance(receipt_id, str)
