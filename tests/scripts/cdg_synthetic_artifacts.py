"""Synthetic canonical Contract Drift mission artifacts for hermetic CI tests.

The canonical ``library/contract-drift-original-cohort-v1.json`` and
``library/contract-drift-sdk-provenance-v1.json`` artifacts are multi-MB
mission files that are not committed to the repository, so every test of the
standalone authority/boundary validation surface used to ``pytest.skip`` in
CI. This module builds SMALL synthetic artifacts with the exact structural
cardinalities the validators enforce (655 originals = 598 SDK + 57
path-level, 666 projection edges / 9 multi-edge / max 4 edges, 690 provenance
occurrences / 12 multi-atom records / 75 core / 523 extended partition) plus
the required canonical byte serialization, and patches the pinned
digest/byte-length constants in both governance scripts so the full
validation logic runs hermetically.

Copy-through metadata (membership anchor, baseline birth, dependency blob
oids) carries the same pinned values as the real artifacts, so tests that
assert those fields behave identically in real and synthetic modes. Tests
that spawn the scripts in a subprocess cannot see the in-process constant
patches and must keep requiring the real artifacts (skip-if-absent).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import scripts.check_contract_drift_ratchet as ratchet
import scripts.generate_contract_drift_inventory as gen

REPO_ROOT = Path(gen.__file__).resolve().parents[1]

# Byte lengths of the REAL artifacts, captured before any constant patching;
# used as a cheap authenticity check when locating real mission artifacts.
_REAL_COHORT_BYTE_LENGTH = gen.COHORT_BYTE_LENGTH
_REAL_PROVENANCE_BYTE_LENGTH = gen.PROVENANCE_BYTE_LENGTH

# Copy-through metadata mirroring the ratified real artifacts. The validators
# never verify these against git; pinned-fact tests assert them.
MEMBERSHIP_ANCHOR = {"commit_sha": "6c4784330dca2d1709edf50b38f4d1201e92c83a"}
MEMBERSHIP_SOURCES = [
    {"path": "scripts/baselines/verify_sdk_contracts.json"},
    {"path": "scripts/baselines/validate_openapi_routes.json"},
    {"path": "scripts/baselines/check_sdk_parity.json"},
]
BASELINE_BIRTH = {"commit_sha": "af5edb22235d4c40a97fe3faa54168b406ab5696"}
DEPENDENCIES = {
    "verifier": {"git_blob_oid": "2d2a0e866f722dab0fad29f4283d1158aad0c408"},
    "normalizer": {"git_blob_oid": "5710fcd04234f997586a187ff8e2ace42f30dddc"},
}
EXTRACTION_ALGORITHM = {"name": "synthetic-test-fixture", "version": 1}


def _canonical_bytes(doc: Any) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _set_digest(values: list[str], field: str, schema: str) -> str:
    return _sha256(_canonical_bytes({field: sorted(values), "schema": schema}))


@dataclass(frozen=True)
class SyntheticArtifacts:
    cohort_bytes: bytes
    provenance_bytes: bytes
    original_id_set_sha256: str
    provenance_record_set_sha256: str
    projection_record_set_sha256: str
    sdk_id_set_sha256: str
    core_id_set_sha256: str
    extended_id_set_sha256: str


@dataclass(frozen=True)
class CdgArtifactEnvironment:
    real: bool
    cohort_path: Path
    provenance_path: Path
    mission_dir: Path


def build_synthetic_artifacts() -> SyntheticArtifacts:
    """Build byte-canonical cohort/provenance artifacts at full cardinality."""
    category_order = [
        ("python_sdk_drift", 74),
        ("typescript_sdk_drift", 524),
        ("routes_missing_in_spec", 11),
        ("routes_orphaned_in_spec", 17),
        ("sdk_missing_from_both", 29),
    ]
    assert dict(category_order) == gen.RATIFIED_CATEGORY_COUNTS

    original_records: list[dict[str, Any]] = []
    provenance_records: list[dict[str, Any]] = []
    projection_records: list[dict[str, Any]] = []
    original_ids: list[str] = []
    projection_digests: list[str] = []
    provenance_digests: list[str] = []
    core_ids: list[str] = []
    extended_ids: list[str] = []

    sdk_index = 0
    path_index = 0
    source_index = 0
    for category, count in category_order:
        is_sdk = category in gen.SDK_CATEGORIES
        for i in range(count):
            literal = f"GET /synthetic/{category}/{i}"
            payload = _canonical_bytes(
                {
                    "category": category,
                    "exact_historical_literal_record": literal,
                    "schema": "cdg-original-record-id-v1",
                }
            )
            record_id = "cdg1:" + _sha256(payload)
            original_ids.append(record_id)
            record: dict[str, Any] = {
                "category": category,
                "exact_historical_literal_record": literal,
                "id_payload_byte_length": len(payload),
                "id_payload_sha256": _sha256(payload),
                "original_record_id": record_id,
                "sdk_language": list(gen.SDK_LANGUAGE_BY_CATEGORY[category]),
                "method": "GET" if is_sdk else None,
                "source_array_index": source_index,
            }
            if is_sdk:
                # 12 multi-atom records, 75 core (domain atoms), 523 extended.
                if sdk_index < 12:
                    atoms = ["debate", "memory"]
                elif sdk_index < 75:
                    atoms = ["memory"]
                else:
                    atoms = ["synthetic"]
                partition = "core" if sdk_index < 75 else "extended"
                # 92 records carry 2 source occurrences: 598 + 92 == 690.
                occurrences: list[dict[str, Any]] = [
                    {"path": f"sdk/source_{sdk_index}.py", "line": 1}
                ]
                if sdk_index < 92:
                    occurrences.append({"path": f"sdk/source_{sdk_index}.py", "line": 2})
                prov: dict[str, Any] = {
                    "original_record_id": record_id,
                    "category": category,
                    "exact_historical_literal_record": literal,
                    "id_payload_byte_length": record["id_payload_byte_length"],
                    "id_payload_sha256": record["id_payload_sha256"],
                    "source_array_index": source_index,
                    "sdk_language": gen.SDK_LANGUAGE_BY_CATEGORY[category][0],
                    "provenance_atoms": atoms,
                    "source_occurrences": occurrences,
                    "partition": partition,
                }
                prov_digest = _sha256(_canonical_bytes(prov))
                prov["record_sha256"] = prov_digest
                provenance_records.append(prov)
                provenance_digests.append(prov_digest)
                record["sdk_provenance_record_sha256"] = prov_digest
                (core_ids if partition == "core" else extended_ids).append(record_id)
                edge_count = 1  # SDK memberships must carry exactly one edge
                sdk_index += 1
            else:
                # Path-level: one 4-edge and eight 2-edge memberships give
                # 598 + 4 + 16 + 48 == 666 edges, 9 multi-edge, max 4.
                edge_count = 4 if path_index == 0 else (2 if path_index <= 8 else 1)
                path_index += 1
            original_records.append(record)

            edges = [
                {
                    "method": "GET",
                    "normalized_path": f"/synthetic/{category}/{i}/{edge}",
                    "evidence": [f"evidence:{category}:{i}:{edge}"],
                }
                for edge in range(edge_count)
            ]
            projection: dict[str, Any] = {
                "original_record_id": record_id,
                "category": category,
                "operation_edges": edges,
            }
            projection_digest = _sha256(_canonical_bytes(projection))
            projection["record_sha256"] = projection_digest
            projection_records.append(projection)
            projection_digests.append(projection_digest)
            source_index += 1

    id_set_sha256 = _set_digest(
        original_ids, "original_record_ids", "cdg-original-record-id-set-v1"
    )
    projection_set_sha256 = _set_digest(
        projection_digests, "record_sha256_values", "cdg-operation-projection-record-digest-set-v1"
    )
    provenance_set_sha256 = _set_digest(
        provenance_digests, "record_sha256_values", "cdg-sdk-provenance-record-digest-set-v1"
    )
    sdk_id_set_sha256 = _set_digest(
        core_ids + extended_ids, "original_record_ids", "cdg-sdk-original-record-id-set-v1"
    )
    core_id_set_sha256 = _set_digest(
        core_ids, "original_record_ids", "cdg-core-original-record-id-set-v1"
    )
    extended_id_set_sha256 = _set_digest(
        extended_ids, "original_record_ids", "cdg-extended-original-record-id-set-v1"
    )

    cohort_doc = {
        "schema": gen.COHORT_SCHEMA,
        "membership_anchor": MEMBERSHIP_ANCHOR,
        "membership_sources": MEMBERSHIP_SOURCES,
        "original_records": original_records,
        "original_record_id_set": {
            "original_record_ids": sorted(original_ids),
            "sha256": id_set_sha256,
        },
        "counts": {
            "records": 655,
            "by_category": dict(sorted(category_order)),
            "method_bearing_sdk_records": 598,
            "method_null_route_parity_records": 57,
        },
        "operation_projection": {
            "schema": gen.PROJECTION_SCHEMA,
            "records": projection_records,
            "record_digest_set_sha256": projection_set_sha256,
        },
    }
    provenance_doc = {
        "schema": gen.PROVENANCE_SCHEMA,
        "baseline_birth": BASELINE_BIRTH,
        "dependencies": DEPENDENCIES,
        "extraction_algorithm": EXTRACTION_ALGORITHM,
        "records": provenance_records,
        "record_digest_set_sha256": provenance_set_sha256,
        "counts": {
            "records": 598,
            "source_occurrences": 690,
            "records_with_multiple_distinct_atoms": 12,
            "core": 75,
            "extended": 523,
        },
        "partition": {
            "partition_rule_version": gen.PARTITION_RULE_VERSION,
            "core_count": 75,
            "extended_count": 523,
            "sdk_original_record_id_set_sha256": sdk_id_set_sha256,
            "core_original_record_id_set_sha256": core_id_set_sha256,
            "extended_original_record_id_set_sha256": extended_id_set_sha256,
        },
    }
    return SyntheticArtifacts(
        cohort_bytes=_canonical_bytes(cohort_doc) + b"\n",
        provenance_bytes=_canonical_bytes(provenance_doc) + b"\n",
        original_id_set_sha256=id_set_sha256,
        provenance_record_set_sha256=provenance_set_sha256,
        projection_record_set_sha256=projection_set_sha256,
        sdk_id_set_sha256=sdk_id_set_sha256,
        core_id_set_sha256=core_id_set_sha256,
        extended_id_set_sha256=extended_id_set_sha256,
    )


def patch_pinned_constants(mp: pytest.MonkeyPatch, art: SyntheticArtifacts) -> None:
    """Point both scripts' pinned digests at the synthetic artifacts."""
    cohort_len = len(art.cohort_bytes)
    cohort_sha = _sha256(art.cohort_bytes)
    prov_len = len(art.provenance_bytes)
    prov_sha = _sha256(art.provenance_bytes)
    mp.setattr(gen, "COHORT_BYTE_LENGTH", cohort_len)
    mp.setattr(gen, "COHORT_SHA256", cohort_sha)
    mp.setattr(gen, "PROVENANCE_BYTE_LENGTH", prov_len)
    mp.setattr(gen, "PROVENANCE_SHA256", prov_sha)
    mp.setattr(gen, "RATIFIED_ORIGINAL_ID_SET_SHA256", art.original_id_set_sha256)
    mp.setattr(gen, "RATIFIED_PROVENANCE_DIGEST_SET_SHA256", art.provenance_record_set_sha256)
    mp.setattr(gen, "RATIFIED_PROJECTION_DIGEST_SET_SHA256", art.projection_record_set_sha256)
    mp.setattr(gen, "RATIFIED_SDK_ID_SET_SHA256", art.sdk_id_set_sha256)
    mp.setattr(gen, "RATIFIED_CORE_ID_SET_SHA256", art.core_id_set_sha256)
    mp.setattr(gen, "RATIFIED_EXTENDED_ID_SET_SHA256", art.extended_id_set_sha256)
    mp.setattr(
        ratchet,
        "COHORT_ARTIFACT",
        {**ratchet.COHORT_ARTIFACT, "byte_length": cohort_len, "sha256": cohort_sha},
    )
    mp.setattr(
        ratchet,
        "PROVENANCE_ARTIFACT",
        {**ratchet.PROVENANCE_ARTIFACT, "byte_length": prov_len, "sha256": prov_sha},
    )
    mp.setattr(ratchet, "ORIGINAL_ID_SET_SHA256", art.original_id_set_sha256)
    mp.setattr(ratchet, "PROVENANCE_RECORD_SET_SHA256", art.provenance_record_set_sha256)
    mp.setattr(ratchet, "PROJECTION_RECORD_SET_SHA256", art.projection_record_set_sha256)
    mp.setattr(ratchet, "SDK_ID_SET_SHA256", art.sdk_id_set_sha256)
    mp.setattr(ratchet, "CORE_ID_SET_SHA256", art.core_id_set_sha256)
    mp.setattr(ratchet, "EXTENDED_ID_SET_SHA256", art.extended_id_set_sha256)


def find_real_artifacts() -> tuple[Path, Path] | None:
    """Locate the REAL multi-MB canonical mission artifacts, if present.

    Uses the unpatched pinned byte lengths as a cheap authenticity check so
    a synthetic library directory never masquerades as the real one.
    """
    roots: list[Path] = []
    mission_dir = os.environ.get("FACTORY_MISSION_DIR", "").strip()
    if mission_dir:
        roots.append(Path(mission_dir) / "library")
    settings_path = os.environ.get("FACTORY_RUNTIME_SETTINGS_PATH", "").strip()
    if settings_path:
        roots.append(Path(settings_path).parent / "library")
    roots.append(REPO_ROOT / "library")
    for root in roots:
        cohort = root / gen.COHORT_ARTIFACT_FILENAME
        provenance = root / gen.PROVENANCE_ARTIFACT_FILENAME
        if (
            cohort.is_file()
            and provenance.is_file()
            and cohort.stat().st_size == _REAL_COHORT_BYTE_LENGTH
            and provenance.stat().st_size == _REAL_PROVENANCE_BYTE_LENGTH
        ):
            return cohort, provenance
    return None


@contextmanager
def artifact_environment(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[CdgArtifactEnvironment]:
    """Provide mission artifacts: real when available, synthetic otherwise.

    In synthetic mode this also patches the pinned digest constants of both
    scripts (in-process only) and points ``FACTORY_MISSION_DIR`` at the
    synthetic library so the standalone auto-discovery path finds them.
    """
    mp = pytest.MonkeyPatch()
    try:
        real = find_real_artifacts()
        if real is not None:
            cohort, provenance = real
            mission_dir = cohort.parent.parent
            mp.setenv("FACTORY_MISSION_DIR", str(mission_dir))
            yield CdgArtifactEnvironment(
                real=True,
                cohort_path=cohort,
                provenance_path=provenance,
                mission_dir=mission_dir,
            )
            return
        art = build_synthetic_artifacts()
        mission_dir = tmp_path_factory.mktemp("cdg-synthetic-mission")
        library = mission_dir / "library"
        library.mkdir()
        cohort = library / gen.COHORT_ARTIFACT_FILENAME
        provenance = library / gen.PROVENANCE_ARTIFACT_FILENAME
        cohort.write_bytes(art.cohort_bytes)
        provenance.write_bytes(art.provenance_bytes)
        patch_pinned_constants(mp, art)
        mp.setenv("FACTORY_MISSION_DIR", str(mission_dir))
        yield CdgArtifactEnvironment(
            real=False,
            cohort_path=cohort,
            provenance_path=provenance,
            mission_dir=mission_dir,
        )
    finally:
        mp.undo()


def init_authority_repo(
    tmp_path: Path,
    *,
    workflows: dict[str, str] | None = None,
    include_inventory: bool = True,
) -> tuple[Path, str]:
    """Minimal git repo satisfying the Tier-4 authority closure at HEAD.

    Contains a byte-copy of the real canonical policy module (so the
    ref-vs-working-tree policy check passes while classification still uses
    the real imported policy), stub files for every authority root and the
    merge-train mirror, the active inventory, and any supplied workflow
    files (keys are paths relative to ``.github/workflows/``).
    """
    repo = tmp_path / "authority-repo"
    repo.mkdir()
    review_queue = gen._review_queue_module()
    files: dict[str, bytes] = {
        gen.CANONICAL_CLASSIFIER_PATH: (REPO_ROOT / gen.CANONICAL_CLASSIFIER_PATH).read_bytes(),
        gen.MERGE_TRAIN_MIRROR_PATH: b"# synthetic merge-train mirror stub\n",
    }
    for root in review_queue.CONTRACT_DRIFT_AUTHORITY_PREFIXES:
        files.setdefault(str(root), b"# synthetic authority root stub\n")
    if include_inventory:
        inventory_doc = {
            "version": 1,
            "generated_by": "tests/scripts/cdg_synthetic_artifacts.py",
            "cohort_commit": "synthetic",
            "items": [
                {
                    "id": "python_sdk_drift:GET /synthetic",
                    "source": "python_sdk_drift",
                    "class": "start_cohort",
                    "discovered_on": gen.COHORT_DATE,
                    "provenance": gen.COHORT_PROVENANCE,
                    "status": "open",
                }
            ],
        }
        files[gen.DEFAULT_INVENTORY] = (
            json.dumps(inventory_doc, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
    else:
        files.pop(gen.DEFAULT_INVENTORY, None)
    for name, text in (workflows or {}).items():
        files[f"{gen.WORKFLOW_DIR}/{name}"] = text.encode("utf-8")
    for rel, data in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "authority"],
        cwd=repo,
        check=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, sha
