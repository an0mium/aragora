from __future__ import annotations

import json
from pathlib import Path

from scripts.check_execution_gate_policy_control import (
    TRACKED_FIELDS,
    check_repo,
    compute_approval_signature,
    compute_defaults_checksum,
    validate_policy_document,
)


def _valid_post_debate_source() -> str:
    return """
from dataclasses import dataclass

@dataclass
class PostDebateConfig:
    enforce_execution_safety_gate: bool = True
    execution_gate_require_verified_signed_receipt: bool = True
    execution_gate_enforce_receipt_signer_allowlist: bool = False
    execution_gate_allowed_receipt_signer_keys: tuple[str, ...] = ()
    execution_gate_require_signed_receipt_timestamp: bool = True
    execution_gate_receipt_max_age_seconds: int = 86400
    execution_gate_receipt_max_future_skew_seconds: int = 120
    execution_gate_min_provider_diversity: int = 2
    execution_gate_min_model_family_diversity: int = 2
    execution_gate_block_on_context_taint: bool = True
    execution_gate_block_on_high_severity_dissent: bool = True
    execution_gate_high_severity_dissent_threshold: float = 0.7
"""


def _source_defaults() -> dict[str, object]:
    return {
        "enforce_execution_safety_gate": True,
        "execution_gate_require_verified_signed_receipt": True,
        "execution_gate_enforce_receipt_signer_allowlist": False,
        "execution_gate_allowed_receipt_signer_keys": [],
        "execution_gate_require_signed_receipt_timestamp": True,
        "execution_gate_receipt_max_age_seconds": 86400,
        "execution_gate_receipt_max_future_skew_seconds": 120,
        "execution_gate_min_provider_diversity": 2,
        "execution_gate_min_model_family_diversity": 2,
        "execution_gate_block_on_context_taint": True,
        "execution_gate_block_on_high_severity_dissent": True,
        "execution_gate_high_severity_dissent_threshold": 0.7,
    }


def _valid_policy() -> dict[str, object]:
    defaults = _source_defaults()
    checksum = compute_defaults_checksum(defaults)
    approval = {
        "approved_by": ["platform-security@aragora.ai"],
        "approved_at": "2026-03-05T00:00:00Z",
        "change_ticket": "SEC-123",
    }
    signature = compute_approval_signature(
        policy_id="execution_safety_gate_defaults",
        version="2026.03.05.1",
        defaults_checksum=checksum,
        approval=approval,
    )
    return {
        "policy_id": "execution_safety_gate_defaults",
        "version": "2026.03.05.1",
        "effective_date": "2026-03-05",
        "defaults": defaults,
        "defaults_checksum": checksum,
        "approval": approval,
        "approval_signature": signature,
    }


def _write_repo_layout(
    repo_root: Path,
    *,
    config_relpath: str,
) -> None:
    config_path = repo_root / config_relpath
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_valid_post_debate_source(), encoding="utf-8")

    policy_path = repo_root / "security/policies/execution_gate_defaults_policy.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(_valid_policy()), encoding="utf-8")


def test_validate_policy_document_accepts_valid_payload() -> None:
    policy = _valid_policy()
    errors = validate_policy_document(policy, _source_defaults())
    assert errors == []


def test_validate_policy_document_rejects_checksum_mismatch() -> None:
    policy = _valid_policy()
    policy["defaults_checksum"] = "sha256:deadbeef"
    errors = validate_policy_document(policy, _source_defaults())
    assert errors
    assert any("defaults_checksum" in error for error in errors)


def test_validate_policy_document_rejects_missing_tracked_key() -> None:
    policy = _valid_policy()
    defaults = dict(policy["defaults"])
    defaults.pop(TRACKED_FIELDS[0], None)
    policy["defaults"] = defaults
    policy["defaults_checksum"] = compute_defaults_checksum(defaults)
    policy["approval_signature"] = compute_approval_signature(
        policy_id=str(policy["policy_id"]),
        version=str(policy["version"]),
        defaults_checksum=str(policy["defaults_checksum"]),
        approval=policy["approval"],  # type: ignore[arg-type]
    )
    errors = validate_policy_document(policy, _source_defaults())
    assert errors
    assert any("missing tracked keys" in error for error in errors)


def test_repo_policy_control_passes_for_current_tree() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    violations = check_repo(repo_root)
    assert violations == []


def test_repo_policy_control_supports_extracted_config_module(tmp_path: Path) -> None:
    _write_repo_layout(tmp_path, config_relpath="aragora/debate/post_debate_config.py")
    violations = check_repo(tmp_path)
    assert violations == []


def test_repo_policy_control_falls_back_to_legacy_coordinator_path(tmp_path: Path) -> None:
    _write_repo_layout(tmp_path, config_relpath="aragora/debate/post_debate_coordinator.py")
    violations = check_repo(tmp_path)
    assert violations == []
