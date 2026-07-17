#!/usr/bin/env python3
"""Enforce contract drift governance: program burn-down and per-PR delta ratchet.

Three modes:

- ``program`` (cron / dispatch / main): per-class scheduled targets over OPEN
  items in the canonical provenance-classified inventory. ``start_cohort``
  burns down from the 2026-04-17 program baseline (655 items, -10%/week,
  read ONLY from scripts/baselines/contract_drift_program.json); each
  ``discovered`` batch burns down from its own batch size and discovery date
  on the same weekly reduction. Fails closed on missing/unparseable inputs,
  inventory desync, unexplained baseline entries, or unknown classes.

- ``pr``: non-worsening delta ratchet with an explained-intake path. Compares
  the five baseline-file counts at HEAD against the merge base (``--base-ref``);
  equal-or-lower counts PASS even while the program schedule is red. A count
  increase PASSES only when EVERY baseline entry that is new vs the base ref
  is born in this PR as an inventory item with ``class=discovered``, a
  provenance containing a PR/issue reference, and a valid ``discovered_on``
  date. The accounting is PER LIST: each increased count needs delta <= the
  distinct new entries in that same list, so a duplicate-entry increase
  cannot hide behind a legitimate new entry in the same or another list. The
  #9325 ruling bans UNEXPLAINED debt absorption; explained intake of newly
  VISIBLE debt (e.g. the canary-probe-exposed orphan routes in #9332) is the
  designed mechanism, and each intake batch immediately starts its own
  weekly burn-down clock in program mode. Increases with any new entry
  missing from the inventory or failing those checks, and increases that
  reopen an item with base-inventory history (a regression, not intake),
  still FAIL. Any duplicated baseline entry is an integrity failure in both
  modes regardless of deltas (#9354: a minted duplicate is slack a later PR
  could cash in as fake burn-down, even delta-neutrally), as is any other
  integrity violation. The program status is still reported (informational)
  so PR authors see the burn-down state.

- ``boundary``: independent schema-versioned validation of a governed
  interval. It binds exact start/end SHAs, the standalone authority manifest,
  immutable cohort/provenance artifacts, reconstructed projection and
  partition facts, and every supplied evidence digest. Inherited conclusions
  and mutating action traces fail closed; unmet immutable-release or rule-suite
  prerequisites report ``blocked`` instead of passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

try:
    from scripts import generate_contract_drift_inventory as inventory_mod
except ImportError:  # executed directly: script dir is on sys.path
    import generate_contract_drift_inventory as inventory_mod  # type: ignore[no-redef]

COUNT_KEYS: tuple[tuple[str, str, str], ...] = (
    ("verify_python_sdk_drift", "verify", "python_sdk_drift"),
    ("verify_typescript_sdk_drift", "verify", "typescript_sdk_drift"),
    ("routes_missing_in_spec", "routes", "missing_in_spec"),
    ("routes_orphaned_in_spec", "routes", "orphaned_in_spec"),
    ("sdk_missing_from_both", "parity", "missing_from_both_sdks"),
)

BOUNDARY_SCHEMA_VERSION = 1
BOUNDARY_NAMES = (
    "corrective_bootstrap",
    "route_truth",
    "core_sdk",
    "extended_sdk",
    "final_seal",
)
BOUNDARY_EVIDENCE_SCHEMA = "contract-drift-boundary-evidence-v1"
BOUNDARY_MANIFEST_SCHEMA = "contract-drift-boundary-manifest-v1"
AUTHORITY_MANIFEST_SCHEMA = "cdg-authority-manifest-v1"
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_SERIALIZATION = (
    "UTF-8; no BOM; object keys sorted; compact separators comma/colon; "
    "declared array orders; one terminal LF"
)
COHORT_ARTIFACT = {
    "path": "library/contract-drift-original-cohort-v1.json",
    "byte_length": 1_692_125,
    "sha256": "565cd84a9a5d266f61b66bd7965e0a036e4817ef5fed32edb8c41a2dea6cc208",
    "schema": "contract-drift-original-cohort-v1",
}
PROVENANCE_ARTIFACT = {
    "path": "library/contract-drift-sdk-provenance-v1.json",
    "byte_length": 898_099,
    "sha256": "21ae1c30200cda6df51dbca7053bbbbde6241ab78a73347b0fe5e4d2ed79f07f",
    "schema": "contract-drift-sdk-provenance-v1",
}
ORIGINAL_ID_SET_SHA256 = "c1235670c183b1887ba3fe4280fa0320f9fd6f4a85b8f346d4332ac2aebbe269"
PROVENANCE_RECORD_SET_SHA256 = "0d30ce3b083344f19949da12ae2d92952757af0aea800b3f99d447458b6eeba0"
PROJECTION_RECORD_SET_SHA256 = "2d6790a6f825c53047639d9433f40e3e10b5bfc9e357bcd161f6b341134775e5"
SDK_ID_SET_SHA256 = "51a963079136a92a86485b56f6cef42aafc7749bfad146ce5fb37293524c5762"
CORE_ID_SET_SHA256 = "b3a1755f027c998d507f13f3ba9093f769cea8720d44bfac12be6beccd626787"
EXTENDED_ID_SET_SHA256 = "bb1fc41548778022dab3041bc05fc40a4da239a1bd4ad8b1ccbcd1007d90b252"
BOUNDARY_DIGEST_FIELDS = (
    "dependency_manifest_sha256",
    "active_inventory_sha256",
    "public_symbol_manifest_sha256",
    "route_boundary_sha256",
    "category_set_sha256",
    "unit_set_sha256",
    "core_set_sha256",
    "extended_set_sha256",
)
MUTATING_HTTP_VERBS = {"POST", "PUT", "PATCH", "DELETE"}
READ_ONLY_GIT_SUBCOMMANDS = {
    "cat-file",
    "diff",
    "diff-tree",
    "for-each-ref",
    "log",
    "ls-files",
    "ls-tree",
    "merge-base",
    "rev-list",
    "rev-parse",
    "show",
    "show-ref",
    "status",
}


def _load_json_strict(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"{label} missing: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} unparseable: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} malformed (expected object): {path}")
    return data


def _canonical_json_bytes(value: Any, *, terminal_lf: bool = False) -> bytes:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return rendered + (b"\n" if terminal_lf else b"")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _duplicate_key_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_canonical_artifact(
    path: Path,
    *,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"canonical artifact missing: {path}")
    if path.is_symlink():
        raise ValueError(f"canonical artifact may not be a symlink: {path}")
    raw = path.read_bytes()
    if len(raw) != expected["byte_length"]:
        raise ValueError(
            f"{expected['path']} byte length mismatch: {len(raw)} != {expected['byte_length']}"
        )
    digest = _sha256_bytes(raw)
    if digest != expected["sha256"]:
        raise ValueError(f"{expected['path']} SHA-256 mismatch: {digest} != {expected['sha256']}")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{expected['path']} has a forbidden UTF-8 BOM")
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError(f"{expected['path']} must have exactly one terminal LF")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{expected['path']} is not UTF-8") from exc
    try:
        payload = json.loads(text, object_pairs_hook=_duplicate_key_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{expected['path']} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{expected['path']} must contain a JSON object")
    if payload.get("schema") != expected["schema"]:
        raise ValueError(
            f"{expected['path']} schema mismatch: "
            f"{payload.get('schema')!r} != {expected['schema']!r}"
        )
    if _canonical_json_bytes(payload, terminal_lf=True) != raw:
        raise ValueError(f"{expected['path']} bytes are not canonical sorted-key JSON")
    return payload, {
        **expected,
        "canonical_bytes_valid": True,
        "canonical_serialization": CANONICAL_SERIALIZATION,
    }


def _digest_set(schema: str, values: list[str], field: str) -> str:
    return _sha256_bytes(_canonical_json_bytes({"schema": schema, field: sorted(values)}))


def _validate_original_cohort(cohort: dict[str, Any]) -> dict[str, Any]:
    records = cohort.get("original_records")
    if not isinstance(records, list) or len(records) != 655:
        raise ValueError("canonical cohort must contain exactly 655 original_records")

    category_counts: dict[str, int] = defaultdict(int)
    original_ids: list[str] = []
    sdk_records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"original record {index} is not an object")
        category = record.get("category")
        literal = record.get("exact_historical_literal_record")
        if not isinstance(category, str) or not isinstance(literal, str):
            raise ValueError(f"original record {index} has malformed identity fields")
        id_payload = {
            "category": category,
            "exact_historical_literal_record": literal,
            "schema": "cdg-original-record-id-v1",
        }
        id_payload_bytes = _canonical_json_bytes(id_payload)
        payload_digest = _sha256_bytes(id_payload_bytes)
        expected_id = f"cdg1:{payload_digest}"
        if record.get("id_payload_byte_length") != len(id_payload_bytes):
            raise ValueError(f"original record {index} id payload length mismatch")
        if record.get("id_payload_sha256") != payload_digest:
            raise ValueError(f"original record {index} id payload digest mismatch")
        if record.get("original_record_id") != expected_id:
            raise ValueError(f"original record {index} original_record_id mismatch")
        original_ids.append(expected_id)
        category_counts[category] += 1
        if category in {"python_sdk_drift", "typescript_sdk_drift"}:
            if not record.get("method"):
                raise ValueError(f"SDK original record {expected_id} has null method")
            sdk_records[expected_id] = record
        elif record.get("method") is not None:
            raise ValueError(f"path-level original record {expected_id} has a method")

    if len(set(original_ids)) != 655:
        raise ValueError("canonical cohort contains duplicate original_record_id values")
    id_set = cohort.get("original_record_id_set")
    if not isinstance(id_set, dict):
        raise ValueError("canonical cohort lacks original_record_id_set")
    if id_set.get("original_record_ids") != sorted(original_ids):
        raise ValueError("canonical cohort original_record_id_set is not the exact sorted ID set")
    id_set_digest = _digest_set(
        "cdg-original-record-id-set-v1",
        original_ids,
        "original_record_ids",
    )
    if id_set.get("sha256") != id_set_digest or id_set_digest != ORIGINAL_ID_SET_SHA256:
        raise ValueError("canonical cohort original-record ID-set digest mismatch")

    declared_counts = cohort.get("counts")
    if not isinstance(declared_counts, dict):
        raise ValueError("canonical cohort lacks counts")
    if declared_counts.get("records") != 655:
        raise ValueError("canonical cohort declared record count mismatch")
    if declared_counts.get("by_category") != dict(sorted(category_counts.items())):
        raise ValueError("canonical cohort category counts mismatch")
    if declared_counts.get("method_bearing_sdk_records") != 598:
        raise ValueError("canonical cohort SDK record count mismatch")
    if declared_counts.get("method_null_route_parity_records") != 57:
        raise ValueError("canonical cohort path-level record count mismatch")

    projection = cohort.get("operation_projection")
    if not isinstance(projection, dict):
        raise ValueError("canonical cohort lacks operation_projection")
    projection_records = projection.get("records")
    if not isinstance(projection_records, list) or len(projection_records) != 655:
        raise ValueError("operation projection must contain 655 membership records")
    projection_ids: list[str] = []
    projection_digests: list[str] = []
    edge_total = 0
    multi_edge = 0
    max_edges = 0
    edge_count_distribution: dict[str, int] = defaultdict(int)
    for index, record in enumerate(projection_records):
        if not isinstance(record, dict):
            raise ValueError(f"operation projection record {index} is not an object")
        original_id = record.get("original_record_id")
        projection_ids.append(str(original_id))
        expected_record_digest = _sha256_bytes(
            _canonical_json_bytes({k: v for k, v in record.items() if k != "record_sha256"})
        )
        if record.get("record_sha256") != expected_record_digest:
            raise ValueError(f"operation projection record {index} digest mismatch")
        projection_digests.append(expected_record_digest)
        edges = record.get("operation_edges")
        if not isinstance(edges, list) or not edges:
            raise ValueError(f"operation projection record {index} has no operation_edges")
        seen_edges: set[tuple[str, str]] = set()
        for edge in edges:
            if not isinstance(edge, dict):
                raise ValueError(f"operation projection record {index} has malformed edge")
            method = edge.get("method")
            normalized_path = edge.get("normalized_path")
            if method not in {
                "GET",
                "HEAD",
                "POST",
                "PUT",
                "DELETE",
                "CONNECT",
                "OPTIONS",
                "TRACE",
                "PATCH",
            }:
                raise ValueError(f"operation projection record {index} has invalid method")
            if not isinstance(normalized_path, str) or not normalized_path.startswith("/"):
                raise ValueError(f"operation projection record {index} has invalid normalized_path")
            if not isinstance(edge.get("evidence"), list) or not edge["evidence"]:
                raise ValueError(f"operation projection record {index} has empty evidence")
            edge_key = (method, normalized_path)
            if edge_key in seen_edges:
                raise ValueError(f"operation projection record {index} has duplicate edge")
            seen_edges.add(edge_key)
        edge_count = len(edges)
        edge_total += edge_count
        multi_edge += int(edge_count > 1)
        max_edges = max(max_edges, edge_count)
        edge_count_distribution[str(edge_count)] += 1

    if sorted(projection_ids) != sorted(original_ids) or len(set(projection_ids)) != 655:
        raise ValueError("operation projection membership does not biject with original IDs")
    projection_digest = _digest_set(
        "cdg-operation-projection-record-digest-set-v1",
        projection_digests,
        "record_sha256_values",
    )
    if (
        projection.get("record_digest_set_sha256") != projection_digest
        or projection_digest != PROJECTION_RECORD_SET_SHA256
    ):
        raise ValueError("operation projection record-digest-set mismatch")
    if (edge_total, multi_edge, max_edges) != (666, 9, 4):
        raise ValueError(
            "operation projection cardinality mismatch "
            f"(edges={edge_total}, multi={multi_edge}, max={max_edges})"
        )
    return {
        "original_record_ids": sorted(original_ids),
        "category_counts": dict(sorted(category_counts.items())),
        "sdk_records": sdk_records,
        "projection_membership_count": len(projection_records),
        "projection_edge_count": edge_total,
        "projection_multi_edge_originals": multi_edge,
        "projection_max_edges": max_edges,
        "projection_edge_count_distribution": dict(sorted(edge_count_distribution.items())),
        "projection_record_digest_set_sha256": projection_digest,
    }


def _validate_sdk_provenance(
    provenance: dict[str, Any],
    cohort_summary: dict[str, Any],
) -> dict[str, Any]:
    records = provenance.get("records")
    if not isinstance(records, list) or len(records) != 598:
        raise ValueError("canonical SDK provenance must contain exactly 598 records")
    cohort_sdk_records = cohort_summary["sdk_records"]
    record_ids: list[str] = []
    record_digests: list[str] = []
    occurrence_count = 0
    multi_atom_count = 0
    partitions: dict[str, list[str]] = {"core": [], "extended": []}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"SDK provenance record {index} is not an object")
        original_id = record.get("original_record_id")
        if not isinstance(original_id, str) or original_id not in cohort_sdk_records:
            raise ValueError(f"SDK provenance record {index} lacks a cohort link")
        expected_digest = _sha256_bytes(
            _canonical_json_bytes({k: v for k, v in record.items() if k != "record_sha256"})
        )
        if record.get("record_sha256") != expected_digest:
            raise ValueError(f"SDK provenance record {index} digest mismatch")
        cohort_record = cohort_sdk_records[original_id]
        for field in (
            "category",
            "exact_historical_literal_record",
            "id_payload_byte_length",
            "id_payload_sha256",
            "source_array_index",
        ):
            if record.get(field) != cohort_record.get(field):
                raise ValueError(f"SDK provenance record {index} {field} link mismatch")
        if cohort_record.get("sdk_language") != [record.get("sdk_language")]:
            raise ValueError(f"SDK provenance record {index} sdk_language link mismatch")
        if cohort_record.get("sdk_provenance_record_sha256") != expected_digest:
            raise ValueError(f"SDK provenance record {index} cohort digest link mismatch")
        atoms = record.get("provenance_atoms")
        occurrences = record.get("source_occurrences")
        if (
            not isinstance(atoms, list)
            or not atoms
            or not all(isinstance(atom, str) and atom for atom in atoms)
        ):
            raise ValueError(f"SDK provenance record {index} has malformed provenance atoms")
        if not isinstance(occurrences, list) or not occurrences:
            raise ValueError(f"SDK provenance record {index} has no source occurrences")
        partition = record.get("partition")
        if not isinstance(partition, str) or partition not in partitions:
            raise ValueError(f"SDK provenance record {index} has invalid partition")
        partitions[partition].append(original_id)
        record_ids.append(original_id)
        record_digests.append(expected_digest)
        occurrence_count += len(occurrences)
        multi_atom_count += int(len(atoms) > 1)

    if sorted(record_ids) != sorted(cohort_sdk_records) or len(set(record_ids)) != 598:
        raise ValueError("SDK provenance records do not biject with cohort SDK records")
    record_set_digest = _digest_set(
        "cdg-sdk-provenance-record-digest-set-v1",
        record_digests,
        "record_sha256_values",
    )
    if (
        provenance.get("record_digest_set_sha256") != record_set_digest
        or record_set_digest != PROVENANCE_RECORD_SET_SHA256
    ):
        raise ValueError("SDK provenance record-digest-set mismatch")
    counts = provenance.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("SDK provenance lacks counts")
    expected_counts = {
        "records": 598,
        "source_occurrences": 690,
        "records_with_multiple_distinct_atoms": 12,
        "core": 75,
        "extended": 523,
    }
    for field, expected_value in expected_counts.items():
        if counts.get(field) != expected_value:
            raise ValueError(f"SDK provenance count {field} mismatch")
    if occurrence_count != 690 or multi_atom_count != 12:
        raise ValueError("SDK provenance reconstructed occurrence/atom counts mismatch")

    all_sdk_ids = sorted(record_ids)
    core_ids = sorted(partitions["core"])
    extended_ids = sorted(partitions["extended"])
    if set(core_ids) & set(extended_ids) or sorted(core_ids + extended_ids) != all_sdk_ids:
        raise ValueError("SDK core/extended partition is not disjoint and exhaustive")
    partition = provenance.get("partition")
    if not isinstance(partition, dict):
        raise ValueError("SDK provenance lacks partition descriptor")
    digest_expectations = (
        (
            "sdk_original_record_id_set_sha256",
            _digest_set("cdg-sdk-original-record-id-set-v1", all_sdk_ids, "original_record_ids"),
            SDK_ID_SET_SHA256,
        ),
        (
            "core_original_record_id_set_sha256",
            _digest_set("cdg-core-original-record-id-set-v1", core_ids, "original_record_ids"),
            CORE_ID_SET_SHA256,
        ),
        (
            "extended_original_record_id_set_sha256",
            _digest_set(
                "cdg-extended-original-record-id-set-v1",
                extended_ids,
                "original_record_ids",
            ),
            EXTENDED_ID_SET_SHA256,
        ),
    )
    for field, reconstructed, expected in digest_expectations:
        if partition.get(field) != reconstructed or reconstructed != expected:
            raise ValueError(f"SDK partition digest {field} mismatch")
    return {
        "record_count": len(records),
        "source_occurrence_count": occurrence_count,
        "multiple_atom_record_count": multi_atom_count,
        "missing_provenance_count": 0,
        "core_count": len(core_ids),
        "extended_count": len(extended_ids),
        "record_digest_set_sha256": record_set_digest,
        "baseline_birth": provenance.get("baseline_birth"),
        "dependencies": provenance.get("dependencies"),
        "extraction_algorithm": provenance.get("extraction_algorithm"),
        "sdk_original_record_id_set_sha256": SDK_ID_SET_SHA256,
        "core_original_record_id_set_sha256": CORE_ID_SET_SHA256,
        "extended_original_record_id_set_sha256": EXTENDED_ID_SET_SHA256,
    }


def _resolve_full_sha(repo_root: Path, ref: str, *, label: str) -> str:
    if not FULL_SHA_RE.fullmatch(ref):
        raise ValueError(f"{label} must be a full 40-character lowercase commit SHA")
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = proc.stdout.strip()
    if resolved != ref:
        raise ValueError(f"{label} did not resolve to the exact supplied commit SHA")
    return resolved


def _is_ancestor(repo_root: Path, start_sha: str, end_sha: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", start_sha, end_sha],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _load_json_no_duplicates(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"{label} missing: {path}")
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symlink: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_key_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _validate_authority_manifest(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
    end_sha: str,
    cohort: dict[str, Any],
    provenance: dict[str, Any],
    cohort_summary: dict[str, Any],
) -> dict[str, Any]:
    if manifest.get("schema") != AUTHORITY_MANIFEST_SCHEMA:
        raise ValueError("authority manifest schema mismatch")
    if manifest.get("ref") != end_sha:
        raise ValueError("authority manifest ref does not equal boundary end SHA")
    repo_files = manifest.get("repo_files")
    if (
        not isinstance(repo_files, list)
        or not repo_files
        or not all(isinstance(path, str) and path for path in repo_files)
    ):
        raise ValueError("authority manifest repo_files must be a nonempty string list")
    if repo_files != sorted(set(repo_files)):
        raise ValueError("authority manifest repo_files must be sorted and unique")
    file_bindings = manifest.get("file_bindings")
    if not isinstance(file_bindings, list) or len(file_bindings) != len(repo_files):
        raise ValueError("authority manifest file_bindings do not cover repo_files")
    bound_paths = []
    for binding in file_bindings:
        if not isinstance(binding, dict):
            raise ValueError("authority manifest contains a malformed file binding")
        path = binding.get("path")
        digest = binding.get("sha256")
        if not isinstance(path, str) or not SHA256_RE.fullmatch(str(digest)):
            raise ValueError("authority manifest file binding lacks path/SHA-256")
        if not isinstance(binding.get("byte_length"), int) or binding["byte_length"] < 0:
            raise ValueError("authority manifest file binding has invalid byte length")
        if binding.get("effective_tier") != 4:
            raise ValueError(f"authority manifest closure member is not Tier 4: {path}")
        if not isinstance(binding.get("matched_authority_rule"), str):
            raise ValueError(f"authority manifest closure member has no matched rule: {path}")
        blob = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "blob", f"{end_sha}:{path}"],
            check=True,
            capture_output=True,
        ).stdout
        oid = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{end_sha}:{path}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if (
            binding.get("git_blob_oid") != oid
            or binding["byte_length"] != len(blob)
            or digest != _sha256_bytes(blob)
        ):
            raise ValueError(f"authority manifest file binding does not match ref: {path}")
        bound_paths.append(path)
    if bound_paths != repo_files:
        raise ValueError("authority manifest file bindings are not in repo_files order")
    policy = manifest.get("policy")
    if (
        not isinstance(policy, dict)
        or policy.get("canonical_policy_module") != "aragora.cli.commands.review_queue"
        or policy.get("tier") != 4
        or not isinstance(policy.get("policy_version"), int)
        or policy["policy_version"] < 1
    ):
        raise ValueError("authority manifest canonical policy binding mismatch")
    authority_roots = policy.get("authority_roots")
    review_queue = inventory_mod._review_queue_module()
    expected_roots = sorted(str(root) for root in review_queue.CONTRACT_DRIFT_AUTHORITY_PREFIXES)
    if (
        not isinstance(authority_roots, list)
        or authority_roots != expected_roots
        or policy.get("policy_version") != review_queue.CONTRACT_DRIFT_AUTHORITY_POLICY_VERSION
        or policy.get("tier") != review_queue.CONTRACT_DRIFT_AUTHORITY_TIER
        or not set(authority_roots) <= set(repo_files)
    ):
        raise ValueError("authority manifest authority roots are incomplete or noncanonical")
    expected_closure = set(authority_roots) | {"aragora/cli/commands/review_queue.py"}
    workflow_names = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-tree",
            "--name-only",
            end_sha,
            ".github/workflows/",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for workflow in workflow_names:
        if not workflow.endswith(".yml"):
            continue
        workflow_blob = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "blob", f"{end_sha}:{workflow}"],
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8", errors="replace")
        if any(root in workflow_blob for root in authority_roots):
            expected_closure.add(workflow)
    if repo_files != sorted(expected_closure):
        raise ValueError("authority manifest repository closure mismatch")
    mirror = manifest.get("merge_train_mirror")
    if (
        not isinstance(mirror, dict)
        or mirror.get("path") != "scripts/tier4_merge_train.py"
        or mirror.get("role") != "dependency_light_mirror"
        or mirror.get("in_repo_files") is not False
    ):
        raise ValueError("authority manifest merge-train mirror binding mismatch")
    mirror_blob = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "cat-file",
            "blob",
            f"{end_sha}:scripts/tier4_merge_train.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    mirror_oid = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "--verify",
            f"{end_sha}:scripts/tier4_merge_train.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        mirror.get("git_blob_oid") != mirror_oid
        or mirror.get("byte_length") != len(mirror_blob)
        or mirror.get("sha256") != _sha256_bytes(mirror_blob)
    ):
        raise ValueError("authority manifest merge-train mirror does not match ref")
    digest = manifest.get("authority_manifest_sha256")
    if not SHA256_RE.fullmatch(str(digest)):
        raise ValueError("authority manifest lacks a valid authority_manifest_sha256")
    reconstructed = _sha256_bytes(
        _canonical_json_bytes(
            {key: value for key, value in manifest.items() if key != "authority_manifest_sha256"}
        )
    )
    if digest != reconstructed:
        raise ValueError("authority manifest digest mismatch")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("authority manifest lacks canonical artifact bindings")
    for key, expected in (
        ("original_cohort", COHORT_ARTIFACT),
        ("sdk_provenance", PROVENANCE_ARTIFACT),
    ):
        descriptor = artifacts.get(key)
        if not isinstance(descriptor, dict):
            raise ValueError(f"authority manifest lacks {key} artifact binding")
        required = {
            "logical_path": expected["path"],
            "byte_length": expected["byte_length"],
            "sha256": expected["sha256"],
            "schema": expected["schema"],
            "canonical_bytes": True,
            "utf8_no_bom": True,
            "single_terminal_lf": True,
        }
        if descriptor != required:
            raise ValueError(f"authority manifest {key} artifact binding mismatch")

    facts = manifest.get("inventory_facts")
    if not isinstance(facts, dict):
        raise ValueError("authority manifest lacks deterministic inventory facts")
    if facts.get("inventory_schema") != "cdg-standalone-inventory-v1" or not SHA256_RE.fullmatch(
        str(facts.get("inventory_sha256"))
    ):
        raise ValueError("authority manifest deterministic inventory digest mismatch")
    if facts.get("membership_anchor") != cohort.get("membership_anchor"):
        raise ValueError("authority manifest membership anchor mismatch")
    if facts.get("membership_sources") != cohort.get("membership_sources"):
        raise ValueError("authority manifest membership sources mismatch")
    expected_facts = {
        "original_record_total": 655,
        "original_record_id_set_sha256": ORIGINAL_ID_SET_SHA256,
        "category_counts": cohort_summary["category_counts"],
        "method_bearing_sdk_records": 598,
        "method_null_route_parity_records": 57,
    }
    for field, expected in expected_facts.items():
        if facts.get(field) != expected:
            raise ValueError(f"authority manifest inventory fact mismatch: {field}")
    projection = facts.get("operation_projection")
    if not isinstance(projection, dict) or (
        projection.get("membership_count"),
        projection.get("operation_edge_count"),
        projection.get("multi_edge_membership_count"),
        projection.get("max_operation_edges"),
        projection.get("record_digest_set_sha256"),
    ) != (655, 666, 9, 4, PROJECTION_RECORD_SET_SHA256):
        raise ValueError("authority manifest operation projection facts mismatch")
    provenance_facts = facts.get("sdk_provenance")
    if not isinstance(provenance_facts, dict):
        raise ValueError("authority manifest lacks SDK provenance facts")
    for field in (
        "baseline_birth",
        "dependencies",
        "extraction_algorithm",
    ):
        if provenance_facts.get(field) != provenance.get(field):
            raise ValueError(f"authority manifest SDK provenance {field} mismatch")
    provenance_expected = {
        "record_count": 598,
        "source_occurrence_count": 690,
        "multi_atom_record_count": 12,
        "missing_record_count": 0,
        "record_digest_set_sha256": PROVENANCE_RECORD_SET_SHA256,
    }
    for field, expected_value in provenance_expected.items():
        if provenance_facts.get(field) != expected_value:
            raise ValueError(f"authority manifest SDK provenance fact mismatch: {field}")
    partition = provenance_facts.get("partition")
    if not isinstance(partition, dict) or (
        partition.get("core_count"),
        partition.get("extended_count"),
        partition.get("sdk_original_record_id_set_sha256"),
        partition.get("core_original_record_id_set_sha256"),
        partition.get("extended_original_record_id_set_sha256"),
    ) != (75, 523, SDK_ID_SET_SHA256, CORE_ID_SET_SHA256, EXTENDED_ID_SET_SHA256):
        raise ValueError("authority manifest SDK partition facts mismatch")

    active = facts.get("active_inventory")
    if not isinstance(active, dict) or active.get("path") != inventory_mod.DEFAULT_INVENTORY:
        raise ValueError("authority manifest active inventory binding mismatch")
    active_blob = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "cat-file",
            "blob",
            f"{end_sha}:{inventory_mod.DEFAULT_INVENTORY}",
        ],
        check=True,
        capture_output=True,
    ).stdout
    active_oid = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "--verify",
            f"{end_sha}:{inventory_mod.DEFAULT_INVENTORY}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    active_doc = json.loads(active_blob)
    active_items = active_doc.get("items")
    if not isinstance(active_items, list):
        raise ValueError("authority manifest active inventory has no items list")
    expected_active = {
        "path": inventory_mod.DEFAULT_INVENTORY,
        "git_blob_oid": active_oid,
        "byte_length": len(active_blob),
        "sha256": _sha256_bytes(active_blob),
        "cohort_commit": str(active_doc.get("cohort_commit", "")),
        "item_total": len(active_items),
        "open_items": sum(
            1 for item in active_items if isinstance(item, dict) and item.get("status") == "open"
        ),
        "resolved_items": sum(
            1
            for item in active_items
            if isinstance(item, dict) and item.get("status") == "resolved"
        ),
    }
    if active != expected_active:
        raise ValueError("authority manifest active inventory does not match ref")
    return {
        "authority_manifest_sha256": digest,
        "authority_repo_file_count": len(repo_files),
        "authority_repo_files": repo_files,
        "active_inventory_sha256": expected_active["sha256"],
    }


def _validate_action_audit(actions: Any) -> list[dict[str, Any]]:
    if actions is None:
        return []
    if not isinstance(actions, list):
        raise ValueError("boundary evidence actions must be a list")
    validated: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ValueError(f"boundary evidence action {index} is malformed")
        kind = action.get("kind")
        if kind == "http":
            method = str(action.get("method") or "").upper()
            if method in MUTATING_HTTP_VERBS:
                raise ValueError(f"mutating HTTP verb rejected: {method}")
            if method not in {"GET", "HEAD"}:
                raise ValueError(f"unsupported HTTP verb in read-only evidence: {method}")
        elif kind == "git":
            argv = action.get("argv")
            if not isinstance(argv, list) or not argv:
                raise ValueError(f"boundary evidence git action {index} lacks argv")
            args = [str(arg) for arg in argv]
            cursor = 1 if args and Path(args[0]).name == "git" else 0
            subcommand = ""
            while cursor < len(args):
                arg = args[cursor]
                if arg in {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}:
                    cursor += 2
                    continue
                if arg.startswith(("--git-dir=", "--work-tree=", "--namespace=")):
                    cursor += 1
                    continue
                if arg.startswith("-"):
                    cursor += 1
                    continue
                subcommand = arg
                break
            if subcommand not in READ_ONLY_GIT_SUBCOMMANDS:
                raise ValueError(f"non-read-only git command rejected: {subcommand or '<missing>'}")
        elif kind == "subprocess":
            operation = str(action.get("operation") or "").lower()
            if operation in {
                "merge",
                "release_upload",
                "rerun",
                "settle",
                "status_mutation",
                "tag_mutation",
                "workflow_dispatch",
            }:
                raise ValueError(f"mutating subprocess action rejected: {operation}")
            if action.get("mutating") is not False:
                raise ValueError(f"subprocess action {index} is not explicitly read-only")
        elif kind == "filesystem":
            operation = str(action.get("operation") or "").lower()
            if operation not in {"read", "stat"}:
                raise ValueError(f"filesystem mutation rejected: {operation}")
        else:
            raise ValueError(f"boundary evidence action {index} has unsupported kind {kind!r}")
        validated.append(action)
    return validated


def _validate_remote_resources(resources: Any) -> tuple[list[dict[str, Any]], str | None]:
    if resources is None:
        return [], None
    if not isinstance(resources, list):
        raise ValueError("boundary evidence remote_resources must be a list")
    validated: list[dict[str, Any]] = []
    moved_reason: str | None = None
    for index, resource in enumerate(resources):
        if not isinstance(resource, dict) or not isinstance(resource.get("resource"), str):
            raise ValueError(f"boundary evidence remote resource {index} is malformed")
        before = resource.get("before")
        after = resource.get("after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError(f"boundary evidence remote resource {index} lacks snapshots")
        stable_fields = ("etag", "updated_at", "sha256")
        if not any(before.get(field) is not None for field in stable_fields):
            raise ValueError(
                f"boundary evidence remote resource {index} lacks ETag, updated_at, or SHA-256"
            )
        for field in stable_fields:
            value = before.get(field)
            if value is not None and after.get(field) != value and moved_reason is None:
                moved_reason = (
                    f"remote resource moved during read-only probe: {resource['resource']}"
                )
        validated.append(resource)
    return validated, moved_reason


def _validate_boundary_evidence(
    evidence: dict[str, Any],
    *,
    boundary: str,
    start_sha: str,
    end_sha: str,
    authority_summary: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    if evidence.get("schema") != BOUNDARY_EVIDENCE_SCHEMA:
        raise ValueError("boundary evidence schema mismatch")
    if evidence.get("boundary") != boundary:
        raise ValueError("boundary evidence boundary name mismatch")
    if evidence.get("start_sha") != start_sha or evidence.get("end_sha") != end_sha:
        raise ValueError("boundary evidence start/end SHA mismatch")
    if evidence.get("inherited_from_boundary") is not None:
        raise ValueError("boundary evidence may not inherit a prior boundary conclusion")
    if evidence.get("prior_boundary_manifest") is not None:
        raise ValueError("boundary evidence may not embed a prior boundary manifest")
    if evidence.get("authority_manifest_sha256") != authority_summary["authority_manifest_sha256"]:
        raise ValueError("boundary evidence authority-manifest digest mismatch")
    if evidence.get("active_inventory_sha256") != authority_summary["active_inventory_sha256"]:
        raise ValueError("boundary evidence active-inventory digest mismatch")

    digests: dict[str, str] = {
        "authority_manifest_sha256": authority_summary["authority_manifest_sha256"]
    }
    for field in BOUNDARY_DIGEST_FIELDS:
        value = evidence.get(field)
        if not SHA256_RE.fullmatch(str(value)):
            raise ValueError(f"boundary evidence missing valid digest: {field}")
        digests[field] = str(value)
    if digests["core_set_sha256"] != CORE_ID_SET_SHA256:
        raise ValueError("boundary evidence core-set digest mismatch")
    if digests["extended_set_sha256"] != EXTENDED_ID_SET_SHA256:
        raise ValueError("boundary evidence extended-set digest mismatch")

    actions = _validate_action_audit(evidence.get("actions"))
    remote_resources, remote_blocked_reason = _validate_remote_resources(
        evidence.get("remote_resources")
    )
    prereqs = evidence.get("external_prerequisites")
    if not isinstance(prereqs, dict):
        raise ValueError("boundary evidence lacks external_prerequisites")
    release_claimed = prereqs.get("future_release_immutability_enabled") is True
    blocked_reason = remote_blocked_reason
    if blocked_reason is None:
        if not release_claimed:
            blocked_reason = "future GitHub Release immutability is not enabled"
        else:
            blocked_reason = (
                "future GitHub Release immutability claim lacks an independent live "
                "read-only GitHub verification"
            )

    governed_prs = evidence.get("governed_prs")
    receipts = evidence.get("first_parent_receipts")
    if not isinstance(governed_prs, list) or not isinstance(receipts, list):
        raise ValueError("boundary evidence must contain governed_prs and first_parent_receipts")
    return {
        "digests": digests,
        "actions": actions,
        "remote_resources": remote_resources,
        "external_prerequisites": prereqs,
        "governed_prs": governed_prs,
        "first_parent_receipts": receipts,
        "source_receipts": evidence.get("source_receipts", []),
        "attestation": evidence.get("attestation"),
    }, blocked_reason


def build_boundary_result(
    *,
    repo_root: Path,
    schema_version: int,
    boundary: str,
    start_ref: str,
    end_ref: str,
    authority_manifest_path: Path | None,
    cohort_artifact_path: Path | None,
    sdk_provenance_artifact_path: Path | None,
    evidence_path: Path | None,
) -> dict[str, Any]:
    if schema_version != BOUNDARY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported boundary schema version {schema_version}; "
            f"expected {BOUNDARY_SCHEMA_VERSION}"
        )
    if boundary not in BOUNDARY_NAMES:
        raise ValueError(f"unsupported Contract Drift boundary: {boundary}")
    start_sha = _resolve_full_sha(repo_root, start_ref, label="--start-ref")
    end_sha = _resolve_full_sha(repo_root, end_ref, label="--end-ref")
    if not _is_ancestor(repo_root, start_sha, end_sha):
        raise ValueError("boundary start SHA is not an ancestor of end SHA")

    try:
        cohort_artifact_path = inventory_mod._discover_artifact_path(
            cohort_artifact_path,
            inventory_mod.COHORT_ARTIFACT_FILENAME,
            repo_root,
            inventory_mod.COHORT_LOGICAL_PATH,
        )
        sdk_provenance_artifact_path = inventory_mod._discover_artifact_path(
            sdk_provenance_artifact_path,
            inventory_mod.PROVENANCE_ARTIFACT_FILENAME,
            repo_root,
            inventory_mod.PROVENANCE_LOGICAL_PATH,
        )
    except inventory_mod.StandaloneError as exc:
        raise ValueError(exc.message) from exc

    cohort, cohort_descriptor = _load_canonical_artifact(
        cohort_artifact_path,
        expected=COHORT_ARTIFACT,
    )
    provenance, provenance_descriptor = _load_canonical_artifact(
        sdk_provenance_artifact_path,
        expected=PROVENANCE_ARTIFACT,
    )
    cohort_summary = _validate_original_cohort(cohort)
    provenance_summary = _validate_sdk_provenance(provenance, cohort_summary)

    if authority_manifest_path is None:
        authority_manifest = inventory_mod._build_authority_manifest(
            repo_root,
            end_sha,
            cohort_artifact_path,
            sdk_provenance_artifact_path,
        )
    else:
        authority_manifest = _load_json_no_duplicates(
            authority_manifest_path,
            label="Authority manifest",
        )
    authority_summary = _validate_authority_manifest(
        authority_manifest,
        repo_root=repo_root,
        end_sha=end_sha,
        cohort=cohort,
        provenance=provenance,
        cohort_summary=cohort_summary,
    )
    evidence_summary: dict[str, Any]
    blocked_reason: str | None
    if evidence_path is None:
        evidence_summary = {
            "status": "unavailable",
            "required_schema": BOUNDARY_EVIDENCE_SCHEMA,
            "authority_manifest_sha256": authority_summary["authority_manifest_sha256"],
            "active_inventory_sha256": authority_summary["active_inventory_sha256"],
        }
        blocked_reason = "independent boundary evidence input is unavailable"
    else:
        evidence = _load_json_no_duplicates(evidence_path, label="Boundary evidence")
        evidence_summary, blocked_reason = _validate_boundary_evidence(
            evidence,
            boundary=boundary,
            start_sha=start_sha,
            end_sha=end_sha,
            authority_summary=authority_summary,
        )

    result: dict[str, Any] = {
        "schema": BOUNDARY_MANIFEST_SCHEMA,
        "schema_version": schema_version,
        "status": "blocked" if blocked_reason else "pass",
        "passing": blocked_reason is None,
        "blocked_reason": blocked_reason,
        "boundary": boundary,
        "start_sha": start_sha,
        "end_sha": end_sha,
        "canonical_artifacts": {
            "original_cohort": cohort_descriptor,
            "sdk_provenance": provenance_descriptor,
        },
        "original_cohort": {
            "record_count": 655,
            "category_counts": cohort_summary["category_counts"],
            "original_record_id_set_sha256": ORIGINAL_ID_SET_SHA256,
            "membership_anchor": cohort.get("membership_anchor"),
            "membership_sources": cohort.get("membership_sources"),
        },
        "sdk_provenance": provenance_summary,
        "operation_projection": {
            "schema": "cdg-operation-projection-v1",
            "membership_count": cohort_summary["projection_membership_count"],
            "edge_count": cohort_summary["projection_edge_count"],
            "multi_edge_originals": cohort_summary["projection_multi_edge_originals"],
            "maximum_edges_per_original": cohort_summary["projection_max_edges"],
            "edge_count_distribution": cohort_summary["projection_edge_count_distribution"],
            "record_digest_set_sha256": cohort_summary["projection_record_digest_set_sha256"],
        },
        "authority": authority_summary,
        "evidence": evidence_summary,
    }
    result["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(result))
    return result


def _counts_from_docs(docs: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = {
        count_key: len(docs.get(alias, {}).get(list_key, []) or [])
        for count_key, alias, list_key in COUNT_KEYS
    }
    counts["total_items"] = sum(counts.values())
    return counts


def _git_doc(repo_root: Path, ref: str, path: Path) -> dict[str, Any]:
    """Baseline file content at a git ref; missing at the ref means empty."""
    rel = path.resolve().relative_to(repo_root.resolve()).as_posix()
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{ref}:{rel}"],
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout) if proc.returncode == 0 else {}


def _target_after_weeks(start_total: int, weekly_reduction: float, weeks: int) -> int:
    # One-shot floored decay. The previous iterative int(round(n * 0.9)) had
    # fixed points at 1-4 (e.g. round(4 * 0.9) == 4), so small per-batch
    # clocks would never be required to reach zero and larger ones stalled
    # at 4. floor(start * factor**weeks) is monotonic to 0.
    factor = (1.0 - weekly_reduction) ** max(0, weeks)
    return max(0, math.floor(start_total * factor))


def _load_program(program_baseline: Path) -> dict[str, Any]:
    program = _load_json_strict(program_baseline, "Program baseline")
    start_date_raw = program.get("start_date")
    start_total = int(program.get("start_total_items", -1))
    weekly_reduction = float(program.get("weekly_reduction", -1.0))
    grace_weeks = int(program.get("grace_weeks", 0))
    if not start_date_raw:
        raise ValueError("Program baseline must include 'start_date'")
    if start_total < 0:
        raise ValueError("Program baseline has invalid 'start_total_items'")
    if not (0.0 < weekly_reduction < 1.0):
        raise ValueError("Program baseline 'weekly_reduction' must be between 0 and 1")
    return {
        "start_date": date.fromisoformat(start_date_raw),
        "start_total_items": start_total,
        "weekly_reduction": weekly_reduction,
        "grace_weeks": grace_weeks,
    }


def _evaluate_classes(
    program: dict[str, Any], items: list[dict[str, Any]], as_of: date
) -> list[dict[str, Any]]:
    """Per-class scheduled targets over OPEN inventory items."""
    weekly_reduction = program["weekly_reduction"]
    weeks_elapsed = max(0, (as_of - program["start_date"]).days) // 7
    effective_weeks = max(0, weeks_elapsed - program["grace_weeks"])

    cohort_open = sum(1 for i in items if i["class"] == "start_cohort" and i["status"] == "open")
    classes = [
        {
            "name": "start_cohort",
            "batch_start": program["start_date"].isoformat(),
            "batch_size": program["start_total_items"],
            "weeks_elapsed": effective_weeks,
            "open_items": cohort_open,
            "target_max": _target_after_weeks(
                program["start_total_items"], weekly_reduction, effective_weeks
            ),
        }
    ]

    batches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item["class"] == "discovered":
            batches[item["discovered_on"]].append(item)
    for discovered_on in sorted(batches):
        batch = batches[discovered_on]
        weeks = max(0, (as_of - date.fromisoformat(discovered_on)).days) // 7
        classes.append(
            {
                "name": f"discovered:{discovered_on}",
                "batch_start": discovered_on,
                "batch_size": len(batch),
                "weeks_elapsed": weeks,
                "open_items": sum(1 for i in batch if i["status"] == "open"),
                "target_max": _target_after_weeks(len(batch), weekly_reduction, weeks),
            }
        )

    for cls in classes:
        cls["passing"] = cls["open_items"] <= cls["target_max"]
    return classes


def _append_only_issues(
    base_items: dict[str, dict[str, Any]], head_items: dict[str, dict[str, Any]]
) -> list[str]:
    """Append-only lifecycle invariants for pr mode: history is immutable.

    For every item present in the inventory at the base ref: ``class``,
    ``discovered_on``, and ``provenance`` may not change (so reopening a
    resolved item cannot reset its burn-down clock), and the item may not be
    deleted. Status transitions (open<->resolved) remain allowed for items
    with base history.

    Birth-state invariant: an item NEW at head (absent from the base
    inventory) must be born ``open``. Resolution requires history — a
    fabricated ``resolved`` item would otherwise inflate its batch_size and
    pad that batch's scheduled target while leaving PR count deltas at zero.
    (New open items are additionally required to be baseline-backed by the
    global sync check, so a fabricated open item fails too.)
    """
    issues: list[str] = []
    for item_id, base_item in base_items.items():
        head_item = head_items.get(item_id)
        if head_item is None:
            issues.append(f"Inventory item deleted (inventory is append-only): {item_id}")
            continue
        for field in ("class", "discovered_on", "provenance"):
            if head_item.get(field) != base_item.get(field):
                issues.append(
                    f"Immutable inventory field {field!r} changed for {item_id}: "
                    f"{base_item.get(field)!r} -> {head_item.get(field)!r}"
                )
    for item_id, head_item in head_items.items():
        if item_id not in base_items and head_item.get("status") != "open":
            issues.append(
                "New inventory item must be born open, not "
                f"{head_item.get('status')!r} (resolution requires history): {item_id}"
            )
    return issues


_LIST_KEY_BY_COUNT_KEY = {count_key: list_key for count_key, _alias, list_key in COUNT_KEYS}


def _unexplained_increase_reasons(
    deltas: dict[str, dict[str, int]],
    new_ids: dict[str, str],
    base_items: dict[str, dict[str, Any]],
    head_items: dict[str, dict[str, Any]],
) -> list[str]:
    """Explained-intake gate for pr-mode count increases.

    An increase is explained iff, PER LIST, every unit of increase is covered
    by a distinct baseline entry new vs the base ref (``delta <= new distinct
    entries in that list`` — repo-wide accounting would let a duplicate-entry
    increase hide behind a legitimate new entry elsewhere), and EVERY new
    entry was born in this PR as an inventory item with ``class=discovered``,
    a provenance containing a PR/issue reference, and a valid
    ``discovered_on`` date. Reopening an item that already has base-inventory
    history is a regression, not intake (its batch clock has already been
    burning, so it may not be re-absorbed as fresh debt). Backdating a
    genuinely new item is self-defeating (it joins an older batch whose
    scheduled target has already decayed) and the metadata invariants bound
    the date to [cohort, as_of].
    """
    increased = sorted(key for key, d in deltas.items() if d["delta"] > 0)
    if not increased:
        return []
    reasons: list[str] = []
    for count_key in increased:
        list_key = _LIST_KEY_BY_COUNT_KEY[count_key]
        delta = deltas[count_key]["delta"]
        distinct_new = sum(1 for lk in new_ids.values() if lk == list_key)
        if delta > distinct_new:
            reasons.append(
                f"{count_key} increased by {delta} with only {distinct_new} distinct new "
                f"baseline entr{'y' if distinct_new == 1 else 'ies'} in {list_key}; every "
                "unit of increase must be its own newly inventoried discovered entry "
                "(duplicate entries are not intake)"
            )
    for item_id in sorted(new_ids):
        item = head_items.get(item_id)
        if item is None:
            reasons.append(f"New baseline entry has no inventory record: {item_id}")
            continue
        if item_id in base_items:
            reasons.append(f"Reopened item is a regression, not discovered intake: {item_id}")
            continue
        if item.get("class") != "discovered":
            reasons.append(
                f"New baseline entry must be class=discovered, not {item.get('class')!r}: {item_id}"
            )
        if not inventory_mod.PROVENANCE_REF.search(item.get("provenance") or ""):
            reasons.append(f"New baseline entry provenance lacks a PR/issue reference: {item_id}")
        try:
            date.fromisoformat(item.get("discovered_on") or "")
        except (TypeError, ValueError):
            reasons.append(
                f"New baseline entry has invalid discovered_on "
                f"{item.get('discovered_on')!r}: {item_id}"
            )
    return reasons


def build_ratchet_result(
    *,
    mode: str,
    program_baseline: Path,
    verify_baseline: Path,
    routes_baseline: Path,
    parity_baseline: Path,
    inventory_path: Path,
    repo_root: Path,
    as_of: date,
    base_ref: str | None = None,
    cohort_commit: str = inventory_mod.COHORT_COMMIT,
) -> dict[str, Any]:
    integrity_issues: list[str] = []

    program: dict[str, Any] | None = None
    try:
        program = _load_program(program_baseline)
    except ValueError as exc:
        integrity_issues.append(str(exc))

    docs: dict[str, dict[str, Any]] = {}
    for label, alias, path in (
        ("verify_sdk_contracts baseline", "verify", verify_baseline),
        ("validate_openapi_routes baseline", "routes", routes_baseline),
        ("check_sdk_parity baseline", "parity", parity_baseline),
    ):
        try:
            docs[alias] = _load_json_strict(path, label)
        except ValueError as exc:
            integrity_issues.append(str(exc))
            docs[alias] = {}
    counts = _counts_from_docs(docs)
    # Duplicated baseline entries inflate the raw counts above while the
    # id-deduped inventory sees one item; fail closed (head docs only —
    # history at the base ref or cohort commit is never re-judged).
    integrity_issues.extend(inventory_mod.find_duplicate_entry_issues(docs))

    cohort_ids: dict[str, str] | None = None
    try:
        cohort_ids = inventory_mod.collect_ids(
            inventory_mod.load_git_docs(repo_root, cohort_commit)
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        integrity_issues.append(
            f"Cohort commit {cohort_commit} unavailable in this checkout "
            "(fetch it before running); cannot verify derivable metadata"
        )

    inventory: dict[str, Any] = {"items": []}
    try:
        inventory = _load_json_strict(inventory_path, "Contract drift inventory")
    except ValueError as exc:
        integrity_issues.append(str(exc))
    else:
        current_ids = inventory_mod.collect_ids(docs)
        integrity_issues.extend(inventory_mod.find_sync_issues(inventory, current_ids))
        if cohort_ids is not None:
            integrity_issues.extend(
                inventory_mod.find_metadata_issues(
                    [i for i in inventory.get("items", []) if isinstance(i, dict)],
                    cohort_ids,
                    as_of=as_of,
                )
            )

    items = [i for i in inventory.get("items", []) if isinstance(i, dict)]
    classes: list[dict[str, Any]] = []
    if program is not None and not integrity_issues:
        classes = _evaluate_classes(program, items, as_of)

    total_open = sum(cls["open_items"] for cls in classes)
    total_target = sum(cls["target_max"] for cls in classes)
    program_passing = bool(classes) and all(cls["passing"] for cls in classes)

    result: dict[str, Any] = {
        "mode": mode,
        "program": {
            "start_date": program["start_date"].isoformat() if program else None,
            "as_of": as_of.isoformat(),
            "days_elapsed": max(0, (as_of - program["start_date"]).days) if program else 0,
            "weeks_elapsed": (max(0, (as_of - program["start_date"]).days) // 7 if program else 0),
            "effective_weeks": (
                max(
                    0,
                    max(0, (as_of - program["start_date"]).days) // 7 - program["grace_weeks"],
                )
                if program
                else 0
            ),
            "grace_weeks": program["grace_weeks"] if program else 0,
            "weekly_reduction": program["weekly_reduction"] if program else None,
            "start_total_items": program["start_total_items"] if program else None,
        },
        "current": counts,
        "target": {"max_open_items": total_target},
        "delta_to_target": total_open - total_target,
        "classes": classes,
        "integrity": {"passing": not integrity_issues, "issues": integrity_issues},
        "program_passing": program_passing and not integrity_issues,
    }

    if mode == "pr":
        if not base_ref:
            raise ValueError("--base-ref is required in pr mode")
        subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{base_ref}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        )
        base_docs = {
            "verify": _git_doc(repo_root, base_ref, verify_baseline),
            "routes": _git_doc(repo_root, base_ref, routes_baseline),
            "parity": _git_doc(repo_root, base_ref, parity_baseline),
        }
        base_counts = _counts_from_docs(base_docs)

        # The schedule parameters themselves are immutable in pr mode: a PR
        # that edits contract_drift_program.json is threshold inflation by
        # definition (the #9325 ruling's banned move) and must be settled as
        # its own operator-approved change over a red gate, never slipped in.
        base_program = _git_doc(repo_root, base_ref, program_baseline)
        head_program = _load_json_strict(program_baseline, "Program baseline")
        for field in ("start_date", "start_total_items", "weekly_reduction", "grace_weeks"):
            if base_program.get(field) != head_program.get(field):
                integrity_issues.append(
                    "Program baseline parameter changed in PR "
                    f"({field}: {base_program.get(field)!r} -> {head_program.get(field)!r}); "
                    "schedule changes require operator settlement, not a PR-mode pass"
                )

        base_inventory = _git_doc(repo_root, base_ref, inventory_path)
        base_items = {
            i["id"]: i for i in base_inventory.get("items", []) if isinstance(i, dict) and "id" in i
        }
        head_items = {i["id"]: i for i in items if "id" in i}
        integrity_issues.extend(_append_only_issues(base_items, head_items))
        result["integrity"] = {
            "passing": not integrity_issues,
            "issues": integrity_issues,
        }
        result["program_passing"] = result["program_passing"] and not integrity_issues

        deltas = {
            key: {
                "base": base_counts[key],
                "head": counts[key],
                "delta": counts[key] - base_counts[key],
            }
            for key, _alias, _list in COUNT_KEYS
        }
        increased = sorted(k for k, d in deltas.items() if d["delta"] > 0)
        base_id_set = set(inventory_mod.collect_ids(base_docs))
        new_ids = {
            item_id: list_key
            for item_id, list_key in inventory_mod.collect_ids(docs).items()
            if item_id not in base_id_set
        }
        unexplained = _unexplained_increase_reasons(deltas, new_ids, base_items, head_items)
        increased_list_keys = {_LIST_KEY_BY_COUNT_KEY[key] for key in increased}
        result["pr_delta"] = {
            "base_ref": base_ref,
            "counts": deltas,
            "increased": increased,
            "new_entries": sorted(new_ids),
            # Only the new entries in lists that actually increased — what the
            # intake allowance is being spent on (the rest belong to net-zero
            # or decreasing lists and explain nothing).
            "intake_entries": sorted(
                item_id for item_id, lk in new_ids.items() if lk in increased_list_keys
            ),
            "unexplained_increase": unexplained,
        }
        result["passing"] = not unexplained and not integrity_issues
    else:
        result["passing"] = result["program_passing"]

    return result


def _print_text(result: dict[str, Any]) -> None:
    program = result["program"]
    current = result["current"]
    print(f"Contract Drift Ratchet [{result['mode']} mode]")
    print("=" * 60)
    print(
        f"As of: {program['as_of']}  |  Start: {program['start_date']}  |  "
        f"Weeks elapsed: {program['weeks_elapsed']} "
        f"(effective: {program['effective_weeks']})"
    )
    print(
        f"Start total: {program['start_total_items']}  |  "
        f"Current total: {current['total_items']}  |  "
        f"Target max: {result['target']['max_open_items']}"
    )
    print("-" * 60)
    print(
        "Source counts: "
        f"py={current['verify_python_sdk_drift']} "
        f"ts={current['verify_typescript_sdk_drift']} "
        f"missing={current['routes_missing_in_spec']} "
        f"orphaned={current['routes_orphaned_in_spec']} "
        f"both={current['sdk_missing_from_both']}"
    )
    for cls in result["classes"]:
        status = "PASS" if cls["passing"] else "FAIL"
        print(
            f"  class {cls['name']}: open={cls['open_items']} "
            f"target<={cls['target_max']} (batch {cls['batch_size']} @ "
            f"{cls['batch_start']}, {cls['weeks_elapsed']}w) [{status}]"
        )
    print(f"Delta to target: {result['delta_to_target']:+d}")
    if result["mode"] == "pr":
        pr_delta = result["pr_delta"]
        print("-" * 60)
        print(f"PR delta vs {pr_delta['base_ref']}:")
        for key, d in pr_delta["counts"].items():
            print(f"  {key}: {d['base']} -> {d['head']} ({d['delta']:+d})")
        if pr_delta["increased"] and not pr_delta["unexplained_increase"]:
            print(
                f"Increase explained as discovered intake "
                f"({len(pr_delta['intake_entries'])} new inventoried entries "
                "in the increased lists)"
            )
        for reason in pr_delta["unexplained_increase"]:
            print(f"  UNEXPLAINED: {reason}")
        print(
            "Program status (informational): " + ("PASS" if result["program_passing"] else "FAIL")
        )
    if not result["integrity"]["passing"]:
        print("Integrity issues (fail closed):")
        for issue in result["integrity"]["issues"]:
            print(f"  - {issue}")
    print("PASS" if result["passing"] else "FAIL")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check contract drift ratchet")
    parser.add_argument("--mode", choices=("program", "pr", "boundary"), default="program")
    parser.add_argument(
        "--schema-version",
        type=int,
        default=None,
        help="Required manifest schema version for boundary mode",
    )
    parser.add_argument("--boundary", choices=BOUNDARY_NAMES, default=None)
    parser.add_argument("--start-ref", default=None, help="Full interval start SHA")
    parser.add_argument("--end-ref", default=None, help="Full interval end SHA")
    parser.add_argument(
        "--authority-manifest",
        type=Path,
        default=None,
        help="Fresh standalone authority-manifest JSON for --end-ref",
    )
    parser.add_argument(
        "--cohort-artifact",
        type=Path,
        default=None,
        help="Canonical contract-drift-original-cohort-v1.json",
    )
    parser.add_argument(
        "--sdk-provenance-artifact",
        type=Path,
        default=None,
        help="Canonical contract-drift-sdk-provenance-v1.json",
    )
    parser.add_argument(
        "--boundary-evidence",
        type=Path,
        default=None,
        help="Independent contract-drift-boundary-evidence-v1 JSON",
    )
    parser.add_argument(
        "--base-ref", default=None, help="Merge base ref for pr mode (e.g. origin/main)"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--program-baseline",
        type=Path,
        default=Path("scripts/baselines/contract_drift_program.json"),
        help="Program baseline config path (sole source of schedule numbers)",
    )
    parser.add_argument(
        "--verify-baseline",
        type=Path,
        default=Path("scripts/baselines/verify_sdk_contracts.json"),
    )
    parser.add_argument(
        "--routes-baseline",
        type=Path,
        default=Path("scripts/baselines/validate_openapi_routes.json"),
    )
    parser.add_argument(
        "--parity-baseline",
        type=Path,
        default=Path("scripts/baselines/check_sdk_parity.json"),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(inventory_mod.DEFAULT_INVENTORY),
        help="Canonical provenance-classified inventory path",
    )
    parser.add_argument(
        "--cohort-commit",
        default=inventory_mod.COHORT_COMMIT,
        help="Commit whose baselines define the start cohort (derivable metadata)",
    )
    parser.add_argument(
        "--as-of",
        default=date.today().isoformat(),
        help="Date for ratchet evaluation (YYYY-MM-DD, default: today)",
    )
    parser.add_argument("--strict", action="store_true", help="Exit 1 when failing")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    if args.mode == "boundary":
        required = {
            "--schema-version": args.schema_version,
            "--boundary": args.boundary,
            "--start-ref": args.start_ref,
            "--end-ref": args.end_ref,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            error = {
                "schema": BOUNDARY_MANIFEST_SCHEMA,
                "schema_version": args.schema_version,
                "status": "fail",
                "passing": False,
                "error_code": "missing_required_boundary_input",
                "error": f"missing required boundary argument(s): {', '.join(missing)}",
            }
            if args.json:
                print(json.dumps(error, sort_keys=True, separators=(",", ":")))
            else:
                print(f"FAIL (closed): {error['error']}", file=sys.stderr)
            return 1
        try:
            result = build_boundary_result(
                repo_root=args.repo_root,
                schema_version=args.schema_version,
                boundary=args.boundary,
                start_ref=args.start_ref,
                end_ref=args.end_ref,
                authority_manifest_path=args.authority_manifest,
                cohort_artifact_path=args.cohort_artifact,
                sdk_provenance_artifact_path=args.sdk_provenance_artifact,
                evidence_path=args.boundary_evidence,
            )
        except (ValueError, subprocess.CalledProcessError) as exc:
            error = {
                "schema": BOUNDARY_MANIFEST_SCHEMA,
                "schema_version": args.schema_version,
                "status": "fail",
                "passing": False,
                "error_code": "boundary_validation_failed",
                "error": str(exc),
            }
            if args.json:
                print(json.dumps(error, sort_keys=True, separators=(",", ":")))
            else:
                print(f"FAIL (closed): {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        else:
            print(
                f"Contract Drift boundary {result['boundary']}: "
                f"{result['status'].upper()} ({result['start_sha']}..{result['end_sha']})"
            )
            if result["blocked_reason"]:
                print(f"Blocked: {result['blocked_reason']}")
            print(f"Manifest SHA-256: {result['manifest_sha256']}")
        return 0 if result["passing"] else 2

    try:
        result = build_ratchet_result(
            mode=args.mode,
            program_baseline=args.program_baseline,
            verify_baseline=args.verify_baseline,
            routes_baseline=args.routes_baseline,
            parity_baseline=args.parity_baseline,
            inventory_path=args.inventory,
            repo_root=args.repo_root,
            as_of=date.fromisoformat(args.as_of),
            base_ref=args.base_ref,
            cohort_commit=args.cohort_commit,
        )
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"FAIL (closed): {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_text(result)

    if not result["integrity"]["passing"]:
        # Integrity violations always fail closed, independent of --strict.
        print("\nFAIL: contract drift integrity violation (fail closed).", file=sys.stderr)
        return 1

    if args.strict and not result["passing"]:
        message = "\nFAIL: Contract drift ratchet is failing."
        print(message, file=sys.stderr if args.json else sys.stdout)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
