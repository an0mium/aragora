"""Executable claim verification runner (DIC-14 / #6024).

Loads claim manifests produced by the DIC-13 schema
(docs/status/claims/*.yaml), verifies each claim, and emits a JSON
status report.  No network, no queue mutation, no issue creation.

Verification outcomes
---------------------
pass        — command exited 0 and evidence paths are fresh
fail        — command exited non-zero
stale       — evidence path(s) are older than freshness_sla_hours
unsupported — verification kind is ``workflow`` or ``manual``
error       — an exception prevented verification from completing
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import yaml


class ClaimStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    STALE = "stale"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


@dataclass
class ClaimResult:
    claim_id: str
    status: ClaimStatus
    message: str
    severity: str = "info"
    allowed_action: str = "report_only"
    elapsed_ms: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# Injected command runner type: (args: list[str]) -> (returncode: int, stdout: str, stderr: str)
CommandRunner = Callable[[list[str]], tuple[int, str, str]]


def _default_command_runner(args: list[str]) -> tuple[int, str, str]:
    """Run a subprocess with a 60-second wall-clock timeout."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "command timed out after 60s"
    except FileNotFoundError as exc:
        return 1, "", f"executable not found: {exc}"


class ClaimVerifier:
    """Verifies explicit claim manifests and emits machine-readable results.

    Parameters
    ----------
    repo_root:
        Absolute base path used to resolve relative evidence ``path`` values.
        Defaults to the current working directory.
    command_runner:
        Overrideable hook for running verification commands.  The default
        uses ``subprocess.run``.  Inject a stub for tests.
    dry_run:
        If *True*, skip command execution and return UNSUPPORTED for all
        command-kind claims.  Useful for schema validation passes.
    """

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        command_runner: CommandRunner | None = None,
        dry_run: bool = False,
    ) -> None:
        self._repo_root = repo_root or Path(os.getcwd())
        self._run_command: CommandRunner = command_runner or _default_command_runner
        self._dry_run = dry_run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_manifest(self, manifest_path: Path) -> list[ClaimResult]:
        """Load a YAML manifest and verify every claim in it."""
        with open(manifest_path) as fh:
            manifest = yaml.safe_load(fh)
        claims: list[dict[str, Any]] = manifest.get("claims", [])
        return [self.verify_claim(c) for c in claims]

    def verify_claim(self, claim: dict[str, Any]) -> ClaimResult:
        """Verify a single claim dict and return a ClaimResult."""
        claim_id: str = claim.get("claim_id", "<unknown>")
        failure_cfg: dict[str, Any] = claim.get("failure", {})
        severity = failure_cfg.get("severity", "info")
        allowed_action = failure_cfg.get("allowed_action", "report_only")

        start = time.monotonic()
        try:
            result = self._verify(claim)
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.monotonic() - start) * 1000
            return ClaimResult(
                claim_id=claim_id,
                status=ClaimStatus.ERROR,
                message=f"verification raised an exception: {exc}",
                severity=severity,
                allowed_action=allowed_action,
                elapsed_ms=round(elapsed, 1),
            )

        elapsed = (time.monotonic() - start) * 1000
        return ClaimResult(
            claim_id=claim_id,
            status=result["status"],
            message=result["message"],
            severity=severity,
            allowed_action=allowed_action,
            elapsed_ms=round(elapsed, 1),
            detail=result.get("detail", {}),
        )

    @staticmethod
    def report_json(results: list[ClaimResult], *, indent: int = 2) -> str:
        """Serialise results to a JSON string."""
        payload = {
            "schema_version": 1,
            "results": [r.to_dict() for r in results],
            "summary": {s.value: sum(1 for r in results if r.status == s) for s in ClaimStatus},
        }
        return json.dumps(payload, indent=indent)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _verify(self, claim: dict[str, Any]) -> dict[str, Any]:
        freshness_sla: int = claim.get("freshness_sla_hours", 24)
        evidence: list[dict[str, Any]] = claim.get("evidence", [])

        evidence_paths = self._check_path_evidence(evidence, freshness_sla)
        missing = evidence_paths["missing"]
        if missing:
            return {
                "status": ClaimStatus.ERROR,
                "message": f"evidence path missing: {missing}",
                "detail": {"missing_paths": missing},
            }

        stale = evidence_paths["stale"]
        if stale:
            return {
                "status": ClaimStatus.STALE,
                "message": f"evidence path stale (>{freshness_sla}h): {stale}",
                "detail": {"stale_paths": stale},
            }

        verification: dict[str, Any] = claim.get("verification", {})
        kind: str = verification.get("kind", "")

        if kind in ("workflow", "manual"):
            return {
                "status": ClaimStatus.UNSUPPORTED,
                "message": f"verification kind '{kind}' requires external runner",
            }

        if kind != "command":
            return {
                "status": ClaimStatus.UNSUPPORTED,
                "message": f"unknown verification kind '{kind}'",
            }

        if self._dry_run:
            return {
                "status": ClaimStatus.UNSUPPORTED,
                "message": "dry_run=True; command execution skipped",
            }

        command_str: str = verification.get("command", "")
        if not command_str:
            return {
                "status": ClaimStatus.ERROR,
                "message": "verification.command is empty",
            }

        args = shlex.split(command_str)
        returncode, stdout, stderr = self._run_command(args)

        if returncode == 0:
            return {
                "status": ClaimStatus.PASS,
                "message": "command exited 0",
                "detail": {"stdout": stdout[:500], "stderr": stderr[:200]},
            }
        return {
            "status": ClaimStatus.FAIL,
            "message": f"command exited {returncode}",
            "detail": {"stdout": stdout[:500], "stderr": stderr[:200]},
        }

    def _check_path_evidence(
        self,
        evidence: list[dict[str, Any]],
        freshness_sla_hours: int,
    ) -> dict[str, list[str]]:
        """Return missing and stale path-type evidence items."""
        missing: list[str] = []
        stale: list[str] = []
        threshold_s = freshness_sla_hours * 3600
        now = time.time()
        for item in evidence:
            path_str: str | None = item.get("path")
            if not path_str:
                continue
            p = Path(path_str)
            if not p.is_absolute():
                p = self._repo_root / p
            if not p.exists():
                missing.append(str(item["path"]))
                continue
            age_s = now - p.stat().st_mtime
            if age_s > threshold_s:
                stale.append(str(item["path"]))
        return {"missing": missing, "stale": stale}
