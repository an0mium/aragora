"""Tests for scripts/verify_claims.py (DIC-14 / #6024).

Invokes the script via subprocess (system python3 which has aragora
installed).  Tests are skipped if the script is unavailable.
No network access; all manifests use ``kind: manual`` or ``--dry-run``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_claims.py"
_PYTHON = shutil.which("python3") or sys.executable

_MANUAL = """\
schema_version: 1
manifest_id: test_manual
claims:
  - claim_id: test.manual.claim
    statement: Test.
    owner: test
    scope: repo
    confidence: high
    freshness_sla_hours: 24
    evidence:
      - note: present
    verification:
      kind: manual
      command: ""
    failure:
      severity: info
      allowed_action: report_only
    receipts:
      - type: test
"""

_CMD_DRY = (
    _MANUAL.replace("kind: manual", "kind: command")
    .replace('command: ""', 'command: "true"')
    .replace("manifest_id: test_manual", "manifest_id: test_cmd")
)


def _available() -> bool:
    try:
        return (
            subprocess.run(
                [_PYTHON, str(_SCRIPT), "--help"], capture_output=True, timeout=15
            ).returncode
            == 0
        )
    except Exception:
        return False


skip_if_unavailable = pytest.mark.skipif(
    not _available(), reason="script unavailable in this environment"
)


def _run(claims_dir: Path, extra: list[str] | None = None) -> tuple[int, Any, str]:
    cmd = [_PYTHON, str(_SCRIPT), "--claims-dir", str(claims_dir), *(extra or [])]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    payload = json.loads(r.stdout.strip()) if r.stdout.strip() else None
    return r.returncode, payload, r.stderr


def _write(tmp_path: Path, name: str, body: str) -> None:
    (tmp_path / name).write_text(body, encoding="utf-8")


@skip_if_unavailable
class TestBasic:
    def test_empty_dir(self, tmp_path: Path) -> None:
        rc, payload, _ = _run(tmp_path)
        assert rc == 0 and payload["manifests_scanned"] == 0

    def test_manual_claim_unsupported(self, tmp_path: Path) -> None:
        _write(tmp_path, "m.yaml", _MANUAL)
        rc, payload, _ = _run(tmp_path)
        assert rc == 0
        assert payload["results"][0]["status"] == "unsupported"
        assert payload["results"][0]["claim_id"] == "test.manual.claim"

    def test_dry_run_skips_command(self, tmp_path: Path) -> None:
        _write(tmp_path, "m.yaml", _CMD_DRY)
        rc, payload, _ = _run(tmp_path, ["--dry-run"])
        assert rc == 0
        assert payload["results"][0]["status"] == "unsupported"

    def test_invocation_from_outside_checkout_resolves_repo_package(self, tmp_path: Path) -> None:
        _write(tmp_path, "m.yaml", _MANUAL)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [_PYTHON, str(_SCRIPT), "--claims-dir", str(tmp_path)],
            capture_output=True,
            cwd=tmp_path,
            env=env,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["results"][0]["claim_id"] == "test.manual.claim"

    def test_summary_counts_results(self, tmp_path: Path) -> None:
        multi = _MANUAL
        for i in range(1, 3):
            extra = _MANUAL.replace("test.manual.claim", f"test.claim.{i}").replace(
                "manifest_id: test_manual", f"manifest_id: test_{i}"
            )
            _write(tmp_path, f"m{i}.yaml", extra)
        _write(tmp_path, "m0.yaml", multi)
        rc, payload, _ = _run(tmp_path)
        assert rc == 0
        assert payload["manifests_scanned"] == 3
        assert sum(payload["summary"].values()) == 3

    def test_output_has_schema_version(self, tmp_path: Path) -> None:
        _write(tmp_path, "m.yaml", _MANUAL)
        _, payload, _ = _run(tmp_path)
        assert payload["schema_version"] == 1
        assert {"schema_version", "manifests_scanned", "results", "summary"} <= payload.keys()


@skip_if_unavailable
class TestOutputFile:
    def test_file_written_not_stdout(self, tmp_path: Path) -> None:
        _write(tmp_path, "m.yaml", _MANUAL)
        out = tmp_path / "out.json"
        rc, payload, _ = _run(tmp_path, ["--output", str(out)])
        assert rc == 0 and payload is None
        data = json.loads(out.read_text())
        assert data["results"][0]["claim_id"] == "test.manual.claim"


@skip_if_unavailable
class TestExitCode:
    def test_unsupported_is_healthy(self, tmp_path: Path) -> None:
        _write(tmp_path, "m.yaml", _MANUAL)
        rc, _, _ = _run(tmp_path, ["--exit-code"])
        assert rc == 0

    def test_missing_dir_is_fatal(self, tmp_path: Path) -> None:
        rc, _, stderr = _run(tmp_path / "nope")
        assert rc == 2
        assert "not found" in stderr


@skip_if_unavailable
class TestMultiManifest:
    def test_malformed_produces_error(self, tmp_path: Path) -> None:
        (tmp_path / "bad.yaml").write_text("not: a: valid: [", encoding="utf-8")
        _write(tmp_path, "good.yaml", _MANUAL)
        rc, payload, _ = _run(tmp_path)
        assert rc == 0
        statuses = {r["status"] for r in payload["results"]}
        assert "error" in statuses and "unsupported" in statuses
