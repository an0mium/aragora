"""Integration tests for the canonical-metrics executable claim manifest.

These tests verify that:
- docs/status/claims/canonical_metrics.yaml is schema-valid
- scripts/check_canonical_metrics.py runs end-to-end without crashing
- At least one claim currently resolves cleanly against the live repo
  (catching the case where the whole manifest has regressed in parsing)

This suite does NOT assert that every claim passes — drift is expected
and the script reports it. The job of these tests is to prove the
verifier itself works, not that the claims are all currently true.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "docs" / "status" / "claims" / "canonical_metrics.yaml"
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check_canonical_metrics.py"


def _run_check(*extra_args: str) -> tuple[int, dict]:
    """Invoke the check script and parse its stdout as JSON."""
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), *extra_args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
    )
    payload = json.loads(result.stdout) if result.stdout.strip() else {}
    return result.returncode, payload


class TestManifestPresent:
    def test_manifest_file_exists(self) -> None:
        assert MANIFEST_PATH.exists(), f"manifest missing at {MANIFEST_PATH}"

    def test_check_script_is_executable(self) -> None:
        assert CHECK_SCRIPT.exists(), f"check script missing at {CHECK_SCRIPT}"


class TestManifestSchema:
    def test_manifest_is_valid_yaml(self) -> None:
        import yaml

        payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        assert payload["manifest_id"] == "canonical_metrics"
        assert payload["schema_version"] == 1
        assert isinstance(payload["claims"], list)
        assert len(payload["claims"]) >= 1

    def test_every_claim_has_required_fields(self) -> None:
        import yaml

        payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        required = {
            "claim_id",
            "statement",
            "owner",
            "scope",
            "confidence",
            "evidence",
            "freshness_sla_hours",
            "verification",
            "failure",
            "receipts",
        }
        for claim in payload["claims"]:
            missing = required - set(claim.keys())
            assert not missing, f"claim {claim.get('claim_id')} missing fields: {missing}"

    def test_every_claim_id_appears_in_check_script(self) -> None:
        """Regression guard: YAML claim ids must be wired in the Python script."""
        import yaml

        payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        claim_ids = {claim["claim_id"] for claim in payload["claims"]}
        script_text = CHECK_SCRIPT.read_text(encoding="utf-8")
        for claim_id in claim_ids:
            assert claim_id in script_text, (
                f"claim {claim_id!r} declared in manifest but not wired into check script"
            )


class TestCheckScriptRunsEndToEnd:
    def test_all_claims_produces_structured_output(self) -> None:
        rc, payload = _run_check("--all")
        # rc is 0 or 1 depending on current drift — both are valid
        # (1 means "drift reported"; not a test failure)
        assert rc in {0, 1}
        assert payload["manifest_id"] == "canonical_metrics"
        assert "results" in payload
        assert "summary" in payload
        assert len(payload["results"]) >= 1

    def test_single_claim_mode_isolates_one_result(self) -> None:
        rc, payload = _run_check("--claim", "canonical.version.matches_pyproject")
        assert rc in {0, 1}
        assert len(payload["results"]) == 1
        assert payload["results"][0]["claim_id"] == "canonical.version.matches_pyproject"

    def test_unknown_claim_returns_usage_error(self) -> None:
        rc, payload = _run_check("--claim", "not_a_real_claim")
        assert rc == 2

    def test_version_claim_currently_passes(self) -> None:
        """The version claim must pass; it's the most constrained check.

        If this fails it means CANONICAL_GOALS.md and pyproject.toml have
        drifted and a release-readiness bug is on main — not a test
        infrastructure problem.
        """
        rc, payload = _run_check("--claim", "canonical.version.matches_pyproject")
        assert payload["results"][0]["status"] == "pass", (
            "version drift between CANONICAL_GOALS.md and pyproject.toml. "
            f"Details: {payload['results'][0]['message']}"
        )

    def test_summary_counts_add_up(self) -> None:
        rc, payload = _run_check("--all")
        total = payload["summary"]["pass"] + payload["summary"]["warn"] + payload["summary"]["fail"]
        assert total == len(payload["results"])


class TestReceiptWriteOption:
    def test_write_receipt_creates_file(self, tmp_path, monkeypatch) -> None:
        """The --write-receipt flag must produce a receipt file."""
        # Run in a temp copy of the repo structure so we don't pollute the real receipt
        # We call the script with --write-receipt and check the output path in repo_root.
        # For safety, we invoke and then check — if the file exists, the write path works.
        rc, _ = _run_check("--all", "--write-receipt")
        receipt_path = (
            REPO_ROOT / "docs" / "status" / "generated" / "canonical_metrics" / "latest.json"
        )
        assert receipt_path.exists()
        content = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert content["manifest_id"] == "canonical_metrics"


class TestSecurityClaimsRegistered:
    """The Phase-14c security.* claims must be wired into the verifier."""

    SECURITY_CLAIM_IDS = (
        "security.gitleaks.dual_stage",
        "security.model_pins.frontier_aligned",
        "security.incident_log.present",
        "security.openrouter_fallback.wired",
    )

    def test_security_claims_appear_in_manifest(self) -> None:
        import yaml

        payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest_ids = {claim["claim_id"] for claim in payload["claims"]}
        for claim_id in self.SECURITY_CLAIM_IDS:
            assert claim_id in manifest_ids, f"security claim {claim_id} missing from manifest"

    def test_security_claims_are_individually_runnable(self) -> None:
        for claim_id in self.SECURITY_CLAIM_IDS:
            rc, payload = _run_check("--claim", claim_id)
            assert rc in {0, 1}, f"verifier crashed on {claim_id} (rc={rc})"
            assert len(payload["results"]) == 1
            assert payload["results"][0]["claim_id"] == claim_id

    def test_openrouter_fallback_currently_passes(self) -> None:
        """OpenRouter wiring must hold on every commit — it's a blocking claim.

        Removing QuotaFallbackMixin (or OpenAICompatibleMixin, which inherits
        from it) regresses the failure mode the OpenRouter-first policy was
        put in place to prevent. If this test fails, the regression is in
        aragora/agents/api_agents/, not in the verifier.
        """
        rc, payload = _run_check("--claim", "security.openrouter_fallback.wired")
        result = payload["results"][0]
        assert result["status"] == "pass", (
            "QuotaFallbackMixin coverage regressed on a frontier provider agent. "
            f"Details: {result['message']}"
        )
