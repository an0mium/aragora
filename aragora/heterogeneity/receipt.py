"""Deterministic receipt helpers for heterogeneity probes."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

HETEROGENEITY_RECEIPT_SCHEMA_VERSION = "heterogeneity_probe_receipt.v1"
SOURCE_ARTIFACT_HASH_SPEC = "sha256(raw file bytes)"
TRANSCRIPT_SIDECAR_ROLE = "transcript_sidecar"


class MissingProvenanceError(ValueError):
    """Raised when a settlement-grade receipt lacks required source artifacts."""


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Return canonical JSON used for receipt hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_receipt_id(receipt: Mapping[str, Any]) -> str:
    """Compute a stable receipt ID, excluding volatile receipt fields."""
    body = copy.deepcopy(dict(receipt))
    body.pop("receipt_id", None)
    body.pop("produced_at", None)
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for the exact bytes at ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path, *, root: str | Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def build_source_artifact(
    path: str | Path,
    *,
    role: str = TRANSCRIPT_SIDECAR_ROLE,
    format: str,
    root: str | Path | None = None,
    required_for_rejudge: bool = True,
    text_capture: str = "full",
    created_before_receipt_id: bool = True,
) -> dict[str, Any]:
    """Build a source-artifact binding for a receipt."""
    artifact_path = Path(path)
    return {
        "role": role,
        "path": _portable_path(artifact_path, root=root),
        "sha256": sha256_file(artifact_path),
        "bytes": artifact_path.stat().st_size,
        "format": format,
        "hash_spec": SOURCE_ARTIFACT_HASH_SPEC,
        "required_for_rejudge": required_for_rejudge,
        "text_capture": text_capture,
        "created_before_receipt_id": created_before_receipt_id,
    }


def _artifact_path(artifact: Mapping[str, Any], *, base_dir: str | Path | None) -> Path | None:
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute() or base_dir is None:
        return path
    return Path(base_dir) / path


def source_artifact_status(
    receipt: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return source-artifact binding status without rejecting legacy receipts."""
    artifacts = receipt.get("source_artifacts")
    if artifacts is None:
        return {
            "canonical": False,
            "status": "legacy_unbound",
            "source_artifact_count": 0,
            "problems": ["missing_source_artifacts"],
        }
    if not isinstance(artifacts, list):
        return {
            "canonical": False,
            "status": "malformed_source_artifacts",
            "source_artifact_count": 0,
            "problems": ["source_artifacts_not_list"],
        }

    problems: list[str] = []
    checked = 0
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            problems.append(f"artifact_{index}:not_object")
            continue
        path = _artifact_path(artifact, base_dir=base_dir)
        if path is None:
            problems.append(f"artifact_{index}:missing_path")
            continue
        expected_sha = artifact.get("sha256")
        expected_bytes = artifact.get("bytes")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            problems.append(f"artifact_{index}:missing_sha256")
            continue
        if not path.exists():
            problems.append(f"artifact_{index}:transcript_missing")
            continue
        if isinstance(expected_bytes, int) and path.stat().st_size != expected_bytes:
            problems.append(f"artifact_{index}:bytes_mismatch")
            continue
        if sha256_file(path) != expected_sha:
            problems.append(f"artifact_{index}:hash_mismatch")
            continue
        checked += 1

    if not artifacts:
        problems.append("empty_source_artifacts")
    if problems:
        if any(problem.endswith(":transcript_missing") for problem in problems):
            status = "transcript_missing"
        elif any(
            problem.endswith(":hash_mismatch") or problem.endswith(":bytes_mismatch")
            for problem in problems
        ):
            status = "hash_mismatch"
        else:
            status = "malformed_source_artifacts"
        return {
            "canonical": False,
            "status": status,
            "source_artifact_count": len(artifacts),
            "checked_source_artifact_count": checked,
            "problems": problems,
        }

    return {
        "canonical": True,
        "status": "bound",
        "source_artifact_count": len(artifacts),
        "checked_source_artifact_count": checked,
        "problems": [],
    }


def require_source_artifacts(
    receipt: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> None:
    """Fail closed unless ``receipt`` is bound to existing source artifacts."""
    status = source_artifact_status(receipt, base_dir=base_dir)
    if status.get("canonical") is not True:
        raise MissingProvenanceError(
            "receipt source artifacts are not bound: "
            f"{status.get('status')} ({', '.join(map(str, status.get('problems', [])))})"
        )


def write_receipt(
    receipt: Mapping[str, Any],
    output_dir: str | Path,
    *,
    require_bound_source_artifacts: bool = False,
    artifact_base_dir: str | Path | None = None,
) -> Path:
    """Write a receipt under ``output_dir`` using its receipt ID."""
    receipt_id = receipt.get("receipt_id")
    if not isinstance(receipt_id, str) or not receipt_id:
        raise ValueError("receipt must include a non-empty receipt_id")
    if require_bound_source_artifacts:
        require_source_artifacts(receipt, base_dir=artifact_base_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{receipt_id}.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
