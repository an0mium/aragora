"""RS-03 regression tests — terminal truth benchmark scoring.

Parametrizes over all 14 fixture files and validates that
``classify_from_metrics()`` returns the expected class for every example.
Also tests the ``score_benchmark.py`` script's ``score_fixtures()`` helper
directly and verifies error handling for missing/empty directories.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from aragora.swarm.terminal_truth import (
    TerminalClass,
    classify_from_metrics,
    score_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = REPO_ROOT / "benchmarks" / "fixtures" / "swarm" / "terminal_truth"
SCORE_SCRIPT = REPO_ROOT / "scripts" / "score_benchmark.py"
RESCUE_PRODUCTIZATION_PATH = REPO_ROOT / "docs" / "benchmarks" / "rescue_productization.json"

# Collect fixture files for parametrization
_fixture_files = sorted(FIXTURES_DIR.glob("*.json")) if FIXTURES_DIR.is_dir() else []


# ---------------------------------------------------------------------------
# Parametrized per-fixture classification tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_file",
    _fixture_files,
    ids=[f.stem for f in _fixture_files],
)
def test_fixture_classifies_correctly(fixture_file: Path) -> None:
    """Each example in a fixture file must classify to its expected_class."""
    with fixture_file.open() as fh:
        examples = json.load(fh)

    assert isinstance(examples, list), f"{fixture_file.name}: root must be a JSON array"
    assert 3 <= len(examples) <= 5, (
        f"{fixture_file.name}: expected 3-5 examples, got {len(examples)}"
    )

    for idx, row in enumerate(examples):
        expected = row["expected_class"]
        result = classify_from_metrics(row)
        assert result.value == expected, (
            f"{fixture_file.name}[{idx}]: expected {expected!r}, got {result.value!r}"
        )


@pytest.mark.parametrize(
    "fixture_file",
    _fixture_files,
    ids=[f.stem for f in _fixture_files],
)
def test_fixture_schema_valid(fixture_file: Path) -> None:
    """Every example must contain the required metric fields plus expected_class."""
    required_keys = {
        "worker_status",
        "worker_outcome",
        "elapsed_seconds",
        "files_changed",
        "has_deliverable",
        "publish_action",
        "expected_class",
    }
    with fixture_file.open() as fh:
        examples = json.load(fh)

    for idx, row in enumerate(examples):
        missing = required_keys - set(row.keys())
        assert not missing, f"{fixture_file.name}[{idx}]: missing keys {missing}"


def test_all_14_terminal_classes_covered() -> None:
    """There must be exactly one fixture file per TerminalClass value."""
    expected_stems = {tc.value for tc in TerminalClass}
    actual_stems = {f.stem for f in _fixture_files}
    assert actual_stems == expected_stems, (
        f"Missing: {expected_stems - actual_stems}, Extra: {actual_stems - expected_stems}"
    )


def test_fixtures_cover_all_families() -> None:
    """Fixtures span success, rescue, and blocked families."""
    families = set()
    for fixture_file in _fixture_files:
        with fixture_file.open() as fh:
            examples = json.load(fh)
        for row in examples:
            tc = classify_from_metrics(row)
            families.add(tc.family)
    assert families == {"success", "rescue", "blocked"}


def test_rescue_productization_records_admission_class_corpus_synthesis() -> None:
    """The #7209 admission-class rescue has a durable productization ledger entry."""
    with RESCUE_PRODUCTIZATION_PATH.open() as fh:
        payload = json.load(fh)

    entries = payload["entries"]
    classes = [entry["class"] for entry in entries]
    assert len(classes) == len(set(classes))

    entry = next(item for item in entries if item["class"] == "admission_class_corpus_synthesis_v1")
    assert entry["target"] == "#7209"
    assert entry["target_kind"] == "issue"
    assert "#7225" in entry["notes"]
    assert "#7228" in entry["notes"]


# ---------------------------------------------------------------------------
# score_benchmark() function tests
# ---------------------------------------------------------------------------


def test_score_benchmark_with_fixture_rows() -> None:
    """score_benchmark() returns a correct summary when given all fixture rows."""
    all_rows: list[dict] = []
    for fixture_file in _fixture_files:
        with fixture_file.open() as fh:
            all_rows.extend(json.load(fh))

    summary = score_benchmark(all_rows)

    assert summary["total"] == len(all_rows)
    assert isinstance(summary["successes"], int)
    assert 0.0 <= summary["no_rescue_rate"] <= 1.0
    assert isinstance(summary["meets_30d_target"], bool)
    assert isinstance(summary["families"], dict)
    assert isinstance(summary["classes"], dict)
    # All three families must be represented
    assert set(summary["families"].keys()) == {"success", "rescue", "blocked"}
    # Family counts must sum to total
    assert sum(summary["families"].values()) == summary["total"]
    # Class counts must sum to total
    assert sum(summary["classes"].values()) == summary["total"]
    assert isinstance(summary["actionable_failures"], int)
    assert summary["actionable_failures"] >= 0


def test_score_benchmark_empty_input() -> None:
    """score_benchmark() handles an empty list gracefully."""
    summary = score_benchmark([])
    assert summary["total"] == 0
    assert summary["no_rescue_rate"] == 0.0


def test_score_benchmark_single_success_row() -> None:
    """score_benchmark() correctly scores a single success row."""
    row = {
        "worker_status": "completed",
        "worker_outcome": "pr_adopted",
        "elapsed_seconds": 120.0,
        "files_changed": 5,
        "has_deliverable": True,
        "publish_action": "merged",
    }
    summary = score_benchmark([row])
    assert summary["total"] == 1
    assert summary["successes"] == 1
    assert summary["no_rescue_rate"] == 1.0
    assert summary["meets_30d_target"] is True
    assert summary["families"]["success"] == 1


def test_score_benchmark_mixed_rows() -> None:
    """score_benchmark() correctly computes rates for mixed success/failure rows."""
    success_row = {
        "worker_status": "completed",
        "worker_outcome": "pr_adopted",
        "elapsed_seconds": 120.0,
        "files_changed": 5,
        "has_deliverable": True,
        "publish_action": "merged",
    }
    rescue_row = {
        "worker_status": "running",
        "worker_outcome": "timeout",
        "elapsed_seconds": 3600.0,
        "files_changed": 0,
        "has_deliverable": False,
        "publish_action": "",
    }
    summary = score_benchmark([success_row, rescue_row])
    assert summary["total"] == 2
    assert summary["successes"] == 1
    assert summary["no_rescue_rate"] == 0.5
    assert summary["meets_30d_target"] is True


# ---------------------------------------------------------------------------
# Script CLI tests
# ---------------------------------------------------------------------------


def test_score_script_help_exits_zero() -> None:
    """``python3 scripts/score_benchmark.py --help`` must exit 0."""
    result = subprocess.run(
        [sys.executable, str(SCORE_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"--help exited {result.returncode}: {result.stderr}"


def test_score_script_default_run_exits_zero() -> None:
    """Default invocation must load all fixtures and exit 0."""
    result = subprocess.run(
        [sys.executable, str(SCORE_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Default run exited {result.returncode}:\n{result.stdout}\n{result.stderr}"
    )
    assert "PASS" in result.stdout or "pass" in result.stdout.lower()


def test_score_script_missing_dir_exits_nonzero() -> None:
    """Missing fixtures directory must cause non-zero exit."""
    result = subprocess.run(
        [sys.executable, str(SCORE_SCRIPT), "--fixtures-dir", "/tmp/nonexistent_dir_rs03"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0, "Should exit non-zero for missing dir"


def test_score_script_empty_dir_exits_nonzero() -> None:
    """Empty fixtures directory must cause non-zero exit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, str(SCORE_SCRIPT), "--fixtures-dir", tmpdir],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0, "Should exit non-zero for empty dir"


def test_score_script_mismatch_exits_nonzero() -> None:
    """A fixture with wrong expected_class must cause non-zero exit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_fixture = Path(tmpdir) / "bad_fixture.json"
        bad_fixture.write_text(
            json.dumps(
                [
                    {
                        "worker_status": "completed",
                        "worker_outcome": "pr_adopted",
                        "elapsed_seconds": 100.0,
                        "files_changed": 5,
                        "has_deliverable": True,
                        "publish_action": "merged",
                        # Intentional mismatch to prove benchmark failures stay detectable.
                        "expected_class": "rescue_timeout",
                    },
                    {
                        "worker_status": "completed",
                        "worker_outcome": "pr_adopted",
                        "elapsed_seconds": 200.0,
                        "files_changed": 3,
                        "has_deliverable": True,
                        "publish_action": "merged",
                        "expected_class": "rescue_timeout",
                    },
                    {
                        "worker_status": "completed",
                        "worker_outcome": "pr_adopted",
                        "elapsed_seconds": 300.0,
                        "files_changed": 7,
                        "has_deliverable": True,
                        "publish_action": "merged",
                        "expected_class": "rescue_timeout",
                    },
                ]
            )
        )
        result = subprocess.run(
            [sys.executable, str(SCORE_SCRIPT), "--fixtures-dir", tmpdir],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0, (
            f"Should exit non-zero for mismatch:\n{result.stdout}\n{result.stderr}"
        )


def test_score_script_bad_schema_exits_nonzero() -> None:
    """Schema/type problems in fixture rows must fail the benchmark."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_fixture = Path(tmpdir) / "bad_schema.json"
        bad_fixture.write_text(
            json.dumps(
                [
                    {
                        "worker_status": "completed",
                        "worker_outcome": "pr_adopted",
                        "elapsed_seconds": "100.0",
                        "files_changed": True,
                        "has_deliverable": "yes",
                        "publish_action": "merged",
                        "expected_class": "deliverable_pr_created",
                    }
                ]
            )
        )
        result = subprocess.run(
            [sys.executable, str(SCORE_SCRIPT), "--fixtures-dir", tmpdir],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 1, (
            f"Bad schema should fail benchmark with exit 1:\n{result.stdout}\n{result.stderr}"
        )
        assert "schema error" in result.stdout
