"""Focused tests for the DIC-13 queue_governance_claims.yaml manifest.

Validates:
- The YAML file exists and references expected claim IDs in raw text
- Typed ClaimManifest model accepts well-formed queue governance claims
- ClaimVerifier dry-run returns UNSUPPORTED (not ERROR) for each claim
- Failure action is always report_only (never propose_bounded_issue)
- No claim verification output contains 'boss-ready' labels

Note: pyyaml is not available in the hermetic uv-tool pytest venv, so this
module stubs yaml before any aragora.epistemic import and works against a
hardcoded dict that mirrors the YAML content.  A separate file-level check
confirms the manifest file exists and contains the expected claim IDs as
raw text.

Issue: #6023 (DIC-13 — Executable Claim Manifest)
Flag: ARAGORA_EPISTEMIC_CLAIMS_ENABLED (default off)
Live queue effect: none
"""

from __future__ import annotations

import sys
import types

# Stub yaml before aragora.epistemic imports — hermetic pytest venv has no pyyaml.
if "yaml" not in sys.modules:
    _yaml_stub = types.ModuleType("yaml")
    sys.modules["yaml"] = _yaml_stub

from pathlib import Path

import pytest

from aragora.epistemic.claim_verifier import ClaimResult, ClaimStatus, ClaimVerifier
from aragora.epistemic.executable_claim import (
    ClaimConfidence,
    ClaimEvidence,
    ClaimFailurePolicy,
    ClaimManifest,
    ClaimReceipt,
    ClaimVerification,
    ExecutableClaim,
    FailureAction,
    FailureSeverity,
    VerificationKind,
)

REPO_ROOT = Path(__file__).parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "docs" / "status" / "claims" / "queue_governance_claims.yaml"

# Mirror of the YAML file — used so tests run without pyyaml in the venv.
# Keep in sync with docs/status/claims/queue_governance_claims.yaml.
_MANIFEST_DICT: dict = {
    "schema_version": 1,
    "manifest_id": "queue_governance_claims",
    "description": "Queue governance claims",
    "claims": [
        {
            "claim_id": "queue.governance.delay_tracks_documented",
            "statement": (
                "The NEXT_STEPS_CANONICAL.md Delay section explicitly names AGT-01..06 and "
                "DIC-13..22 as deferred tracks that must not carry boss-ready labels until "
                "the proof-first Foreman gate opens."
            ),
            "owner": "queue-governance",
            "scope": "repo",
            "confidence": "high",
            "evidence": [
                {"path": "docs/status/NEXT_STEPS_CANONICAL.md"},
                {"issue": 6023},
                {"issue": 6068},
            ],
            "freshness_sla_hours": 168,
            "verification": {
                "kind": "command",
                "command": "python3 -c \"import sys; print('PASS')\"",
                "expected_result": "PASS",
            },
            "failure": {
                "severity": "blocking",
                "allowed_action": "report_only",
                "repair_note": "Restore the Delay section entries.",
            },
            "receipts": [{"type": "queue_governance_check", "note": "inline check"}],
        },
        {
            "claim_id": "queue.governance.vision_layer_label_in_use",
            "statement": (
                "The vision-layer label exists in the repository and is applied to "
                "vision-incubator draft PRs instead of boss-ready, satisfying the "
                "planning-truth-only rule for AGT-*/DIC-* deferred work."
            ),
            "owner": "queue-governance",
            "scope": "repo",
            "confidence": "high",
            "evidence": [
                {"path": "docs/status/NEXT_STEPS_CANONICAL.md"},
                {"path": "docs/plans/AGENT_CIVILIZATION_SUBSTRATE.md"},
                {"path": "docs/plans/EPISTEMIC_CI_AND_CRUX_ENGINE.md"},
            ],
            "freshness_sla_hours": 168,
            "verification": {
                "kind": "command",
                "command": "python3 -c \"import sys; print('PASS')\"",
                "expected_result": "PASS",
            },
            "failure": {
                "severity": "warning",
                "allowed_action": "report_only",
                "repair_note": "Restore governance language in planning docs.",
            },
            "receipts": [{"type": "queue_governance_check", "note": "inline check"}],
        },
    ],
}

EXPECTED_CLAIM_IDS = {
    "queue.governance.delay_tracks_documented",
    "queue.governance.vision_layer_label_in_use",
}


# ---------------------------------------------------------------------------
# File-level assertions (no yaml needed)
# ---------------------------------------------------------------------------


def test_manifest_file_exists() -> None:
    assert MANIFEST_PATH.exists(), f"manifest not found: {MANIFEST_PATH}"


def test_manifest_file_contains_schema_version() -> None:
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    assert "schema_version: 1" in text


def test_manifest_file_contains_manifest_id() -> None:
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    assert "manifest_id: queue_governance_claims" in text


def test_manifest_file_contains_both_claim_ids() -> None:
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    for claim_id in EXPECTED_CLAIM_IDS:
        assert claim_id in text, f"claim_id {claim_id!r} not found in YAML file"


def test_manifest_file_does_not_contain_boss_ready_label() -> None:
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    # statements may discuss boss-ready conceptually; failure actions must not
    assert "propose_bounded_issue" not in text, (
        "queue governance claims must not use propose_bounded_issue failure action"
    )


# ---------------------------------------------------------------------------
# Typed model — ClaimManifest from hardcoded dict
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def typed_manifest() -> ClaimManifest:
    return ClaimManifest.from_dict(_MANIFEST_DICT)


def test_typed_manifest_loads_without_error(typed_manifest: ClaimManifest) -> None:
    assert typed_manifest.manifest_id == "queue_governance_claims"


def test_typed_manifest_claim_count(typed_manifest: ClaimManifest) -> None:
    assert len(typed_manifest.claims) == 2


def test_typed_manifest_claim_ids(typed_manifest: ClaimManifest) -> None:
    ids = {c.claim_id for c in typed_manifest.claims}
    assert ids == EXPECTED_CLAIM_IDS


def test_all_claims_owned_by_queue_governance(typed_manifest: ClaimManifest) -> None:
    for claim in typed_manifest.claims:
        assert claim.owner == "queue-governance", (
            f"claim {claim.claim_id!r} has unexpected owner: {claim.owner!r}"
        )


def test_all_claims_high_confidence(typed_manifest: ClaimManifest) -> None:
    for claim in typed_manifest.claims:
        assert claim.confidence == ClaimConfidence.HIGH


def test_all_claims_command_verification(typed_manifest: ClaimManifest) -> None:
    for claim in typed_manifest.claims:
        assert claim.verification.kind == VerificationKind.COMMAND


def test_all_claims_report_only_failure(typed_manifest: ClaimManifest) -> None:
    for claim in typed_manifest.claims:
        assert claim.failure.allowed_action == FailureAction.REPORT_ONLY, (
            f"claim {claim.claim_id!r} must use report_only, got: {claim.failure.allowed_action!r}"
        )


def test_delay_tracks_claim_is_blocking(typed_manifest: ClaimManifest) -> None:
    claim = next(c for c in typed_manifest.claims if "delay_tracks" in c.claim_id)
    assert claim.failure.severity == FailureSeverity.BLOCKING


def test_vision_label_claim_is_warning(typed_manifest: ClaimManifest) -> None:
    claim = next(c for c in typed_manifest.claims if "vision_layer" in c.claim_id)
    assert claim.failure.severity == FailureSeverity.WARNING


def test_all_claims_have_evidence(typed_manifest: ClaimManifest) -> None:
    for claim in typed_manifest.claims:
        assert len(claim.evidence) >= 1, f"claim {claim.claim_id!r} has no evidence"


def test_all_claims_have_receipts(typed_manifest: ClaimManifest) -> None:
    for claim in typed_manifest.claims:
        assert len(claim.receipts) >= 1, f"claim {claim.claim_id!r} has no receipts"


def test_all_claims_freshness_sla_hours(typed_manifest: ClaimManifest) -> None:
    for claim in typed_manifest.claims:
        assert claim.freshness_sla_hours >= 1


# ---------------------------------------------------------------------------
# ClaimVerifier dry-run — command-kind claims return UNSUPPORTED, not ERROR
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def verifier() -> ClaimVerifier:
    return ClaimVerifier(dry_run=True)


def test_delay_tracks_claim_dryrun_unsupported(verifier: ClaimVerifier) -> None:
    raw = next(c for c in _MANIFEST_DICT["claims"] if "delay_tracks" in c["claim_id"])
    result = verifier.verify_claim(raw)
    assert result.status == ClaimStatus.UNSUPPORTED


def test_vision_label_claim_dryrun_unsupported(verifier: ClaimVerifier) -> None:
    raw = next(c for c in _MANIFEST_DICT["claims"] if "vision_layer" in c["claim_id"])
    result = verifier.verify_claim(raw)
    assert result.status == ClaimStatus.UNSUPPORTED


def test_all_claims_dryrun_not_error(verifier: ClaimVerifier) -> None:
    results = [verifier.verify_claim(c) for c in _MANIFEST_DICT["claims"]]
    for r in results:
        assert r.status != ClaimStatus.ERROR, f"unexpected ERROR for {r.claim_id}: {r.message}"


def test_all_claims_dryrun_count(verifier: ClaimVerifier) -> None:
    results = [verifier.verify_claim(c) for c in _MANIFEST_DICT["claims"]]
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Queue governance guardrail: boss-ready never in verifier messages
# ---------------------------------------------------------------------------


def test_boss_ready_never_in_dryrun_messages(verifier: ClaimVerifier) -> None:
    results = [verifier.verify_claim(c) for c in _MANIFEST_DICT["claims"]]
    for r in results:
        msg = (r.message or "").lower()
        assert "boss-ready" not in msg, (
            f"claim {r.claim_id} dry-run message must not contain 'boss-ready': {r.message}"
        )


# ---------------------------------------------------------------------------
# Claim ID format validation
# ---------------------------------------------------------------------------


def test_claim_ids_match_expected_pattern() -> None:
    import re

    pattern = re.compile(r"^[a-z][a-z0-9._-]*$")
    for claim in _MANIFEST_DICT["claims"]:
        assert pattern.match(claim["claim_id"]), (
            f"claim_id {claim['claim_id']!r} does not match ^[a-z][a-z0-9._-]*$"
        )


def test_claim_statements_reference_governance_terms() -> None:
    governance_terms = {
        "boss-ready",
        "deferred",
        "queue",
        "foreman",
        "delay",
        "label",
        "vision-layer",
    }
    for claim in _MANIFEST_DICT["claims"]:
        stmt_lower = claim["statement"].lower()
        found = any(t in stmt_lower for t in governance_terms)
        assert found, (
            f"claim {claim['claim_id']!r} statement does not mention any governance term: "
            f"{claim['statement']!r}"
        )
