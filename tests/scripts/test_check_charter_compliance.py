from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from scripts.check_charter_compliance import ChangedFile, classify_changes


def _write_charter(path: Path, *, status: str = "DRAFT") -> Path:
    path.write_text(
        textwrap.dedent(
            f"""
            meta:
              status: {status}
            registry:
              - id: CHR-P4A-001
                state: REMOVED
                binding_in_draft: true
                paths: [aragora/server/metrics.py]
                symbols: []
                evidence: metrics moved to observability
              - id: CHR-P4A-004
                state: REMOVED
                binding_in_draft: true
                paths: [aragora/queue/__init__.py]
                symbols: ["aragora.queue:create_default_executor"]
                evidence: queue must not re-export create_default_executor
              - id: CHR-X-007
                state: PENDING
                binding_in_draft: true
                paths: [aragora/metrics/]
                symbols: []
                evidence: no new aragora.metrics imports
              - id: CHR-X-014
                state: PENDING
                paths: [aragora/tasks/]
                symbols: []
                evidence: proposed retire tasks package
              - id: CHR-X-040
                state: PARKED
                paths: [aragora/control_plane/registry.py]
                symbols: ["aragora.control_plane.registry:ControlPlaneRegistry"]
                kept_symbols: ["aragora.control_plane.registry:ControlPlaneHealth"]
                evidence: registry remainder parked but health surface kept
            """
        ).strip()
        + "\n"
    )
    return path


def test_binding_removed_path_blocks(tmp_path: Path):
    charter = _write_charter(tmp_path / "charters.yaml")
    result = classify_changes(
        charter_path=charter,
        repo=tmp_path,
        changed_files=[ChangedFile("aragora/server/metrics.py", ("def metric(): pass",))],
    )
    assert result.has_blocking_findings is True
    assert [(f.entry_id, f.severity, f.reason) for f in result.findings] == [
        ("CHR-P4A-001", "blocking", "removed path changed")
    ]


def test_symbol_scoped_removed_entry_only_blocks_when_symbol_is_touched(tmp_path: Path):
    charter = _write_charter(tmp_path / "charters.yaml")
    unrelated = classify_changes(
        charter_path=charter,
        repo=tmp_path,
        changed_files=[ChangedFile("aragora/queue/__init__.py", ("OTHER = 1",))],
    )
    assert unrelated.has_blocking_findings is False
    assert (
        unrelated.findings[0].reason
        == "symbol-scoped entry path changed but chartered symbol was not touched"
    )

    touched = classify_changes(
        charter_path=charter,
        repo=tmp_path,
        changed_files=[
            ChangedFile(
                "aragora/queue/__init__.py",
                ("from aragora.debate.queue_executor import create_default_executor",),
            )
        ],
    )
    assert touched.has_blocking_findings is True
    assert touched.findings[0].matched_symbol == "aragora.queue:create_default_executor"


def test_binding_pending_new_import_blocks(tmp_path: Path):
    charter = _write_charter(tmp_path / "charters.yaml")
    result = classify_changes(
        charter_path=charter,
        repo=tmp_path,
        changed_files=[
            ChangedFile("aragora/server/foo.py", ("from aragora.metrics import gauge",))
        ],
    )
    assert result.has_blocking_findings is True
    assert result.findings[0].entry_id == "CHR-X-007"
    assert "new importer/caller" in result.findings[0].reason


def test_proposed_pending_is_advisory_in_draft(tmp_path: Path):
    charter = _write_charter(tmp_path / "charters.yaml")
    result = classify_changes(
        charter_path=charter,
        repo=tmp_path,
        changed_files=[ChangedFile("aragora/tasks/router.py", ("def new_router(): pass",))],
    )
    assert result.has_blocking_findings is False
    assert result.findings[0].entry_id == "CHR-X-014"
    assert result.findings[0].severity == "advisory"


def test_proposed_pending_blocks_after_ratification(tmp_path: Path):
    charter = _write_charter(tmp_path / "charters.yaml", status="RATIFIED")
    result = classify_changes(
        charter_path=charter,
        repo=tmp_path,
        changed_files=[ChangedFile("aragora/tasks/router.py", ("def new_router(): pass",))],
    )
    assert result.has_blocking_findings is True
    assert result.findings[0].entry_id == "CHR-X-014"
    assert result.findings[0].severity == "blocking"


def test_kept_symbol_does_not_block_parked_path(tmp_path: Path):
    charter = _write_charter(tmp_path / "charters.yaml")
    result = classify_changes(
        charter_path=charter,
        repo=tmp_path,
        changed_files=[
            ChangedFile("aragora/control_plane/registry.py", ("class ControlPlaneHealth: ...",))
        ],
    )
    assert result.has_blocking_findings is False
    assert result.findings[0].severity == "advisory"


def test_cli_json_reports_blocking(tmp_path: Path):
    charter = _write_charter(tmp_path / "charters.yaml")
    repo = tmp_path / "repo"
    repo.mkdir()
    script = Path(__file__).parents[2] / "scripts" / "check_charter_compliance.py"
    completed = subprocess.run(
        [
            "python",
            str(script),
            "--repo",
            str(repo),
            "--charters",
            str(charter),
            "--changed-path",
            "aragora/server/metrics.py",
            "--json",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["blocking"] is True
    assert payload["summary"]["blocking"] == 1


@pytest.mark.parametrize("state", ["DRAFT", "RATIFIED"])
def test_charter_status_is_reported(tmp_path: Path, state: str):
    charter = _write_charter(tmp_path / "charters.yaml", status=state)
    result = classify_changes(charter_path=charter, repo=tmp_path, changed_files=[])
    assert result.charter_status == state
