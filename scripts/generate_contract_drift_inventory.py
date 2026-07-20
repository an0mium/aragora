#!/usr/bin/env python3
"""Build/refresh the canonical provenance-classified contract drift inventory.

Every entry in the three contract-drift baseline files becomes an inventory
item with a stable id, a provenance class, and a discovery date:

- Items present in the baselines at the 2026-04-17 program cohort commit
  (5cc7a81b77) are classified ``start_cohort``.
- Items that appear later REQUIRE an explicit ``--provenance`` containing a
  PR/issue reference; without it the generator fails closed and lists the
  unexplained items (no silent debt absorption).
- Items that disappear from the baselines are marked ``resolved`` and are
  never deleted (audit trail). Resolution is verifiable because the sibling
  no-regression gates keep the baselines a superset of live violations, so
  baseline pruning only happens when items are actually fixed.

The inventory is deterministic and sorted; ``--check`` fails (exit 1) when
the committed inventory is out of sync with the baselines.

Standalone read-only surfaces (VAL-CDG-001 / VAL-CDG-002)
---------------------------------------------------------

Three additional modes never write to the repository, never touch the
network, and read git state only through ref-pinned plumbing:

- ``--authority-manifest --ref <full-sha> --json`` emits the Tier-4
  authority closure manifest with per-file bindings and a canonical digest.
- ``--classify-tier --changed-file <path> --ref <full-sha> --json`` reports
  the canonical effective tier, matched authority rule, and manifest digest.
- ``--ref <full-sha> --cohort-artifact <p> --sdk-provenance-artifact <p>
  --json`` emits the canonical, digest-bound original-cohort inventory.

Classification policy is imported from the canonical
``aragora.cli.commands.review_queue`` module and is never duplicated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

COHORT_COMMIT = "5cc7a81b77aeb6680613d441f74d72c435c8443c"
COHORT_DATE = "2026-04-17"
COHORT_PROVENANCE = "2026-04-17 program baseline cohort"
DEFAULT_INVENTORY = "scripts/baselines/contract_drift_inventory.json"
VALID_CLASSES = frozenset({"start_cohort", "discovered"})
VALID_STATUSES = frozenset({"open", "resolved"})
PROVENANCE_REF = re.compile(r"#\d+|https?://\S+")

# alias -> (repo-relative baseline path, tracked list keys)
BASELINE_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "verify": (
        "scripts/baselines/verify_sdk_contracts.json",
        ("python_sdk_drift", "typescript_sdk_drift"),
    ),
    "routes": (
        "scripts/baselines/validate_openapi_routes.json",
        ("missing_in_spec", "orphaned_in_spec"),
    ),
    "parity": (
        "scripts/baselines/check_sdk_parity.json",
        ("missing_from_both_sdks",),
    ),
}


def normalize_key(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    return json.dumps(entry, sort_keys=True)


def collect_ids(docs: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Map stable item id -> source list key for all baseline entries."""
    ids: dict[str, str] = {}
    for alias, (_path, list_keys) in BASELINE_SPECS.items():
        doc = docs.get(alias) or {}
        for list_key in list_keys:
            for entry in doc.get(list_key, []) or []:
                ids[f"{list_key}:{normalize_key(entry)}"] = list_key
    return ids


def find_duplicate_entry_issues(docs: dict[str, dict[str, Any]]) -> list[str]:
    """Duplicate entries within a baseline list (fail closed on hand edits).

    The inventory dedupes by normalized id, but the ratchet counts raw list
    lengths — a duplicated entry inflates the count-based ratchet by one and
    becomes a count-decrease freebie when later removed. Every baseline entry
    must be unique.
    """
    issues: list[str] = []
    for alias, (_path, list_keys) in BASELINE_SPECS.items():
        doc = docs.get(alias) or {}
        for list_key in list_keys:
            seen: set[str] = set()
            for entry in doc.get(list_key, []) or []:
                item_id = f"{list_key}:{normalize_key(entry)}"
                if item_id in seen:
                    issues.append(f"Duplicate baseline entry: {item_id}")
                seen.add(item_id)
    return issues


def load_working_docs(repo_root: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for alias, (rel_path, _keys) in BASELINE_SPECS.items():
        path = repo_root / rel_path
        if not path.exists():
            raise ValueError(f"Baseline file missing: {path}")
        docs[alias] = json.loads(path.read_text())
    return docs


def load_git_docs(repo_root: Path, ref: str) -> dict[str, dict[str, Any]]:
    """Read the baseline files at a git ref; a file missing at the ref is empty."""
    subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    docs: dict[str, dict[str, Any]] = {}
    for alias, (rel_path, _keys) in BASELINE_SPECS.items():
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{ref}:{rel_path}"],
            capture_output=True,
            text=True,
        )
        docs[alias] = json.loads(proc.stdout) if proc.returncode == 0 else {}
    return docs


def build_inventory(
    current_ids: dict[str, str],
    cohort_ids: dict[str, str],
    existing_items: list[dict[str, Any]],
    *,
    as_of: date,
    provenance: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (sorted inventory items, unexplained new item ids)."""
    items = {item["id"]: dict(item) for item in existing_items}
    unexplained: list[str] = []

    for item_id, list_key in current_ids.items():
        record = items.get(item_id)
        if record is not None:
            if record.get("status") == "resolved":
                record["status"] = "open"
                record.pop("resolved_on", None)
        elif item_id in cohort_ids:
            items[item_id] = {
                "id": item_id,
                "source": list_key,
                "class": "start_cohort",
                "discovered_on": COHORT_DATE,
                "provenance": COHORT_PROVENANCE,
                "status": "open",
            }
        elif provenance:
            items[item_id] = {
                "id": item_id,
                "source": list_key,
                "class": "discovered",
                "discovered_on": as_of.isoformat(),
                "provenance": provenance,
                "status": "open",
            }
        else:
            unexplained.append(item_id)

    for item_id, record in items.items():
        if item_id not in current_ids and record.get("status") == "open":
            record["status"] = "resolved"
            record["resolved_on"] = as_of.isoformat()

    return sorted(items.values(), key=lambda item: item["id"]), sorted(unexplained)


def find_sync_issues(inventory: dict[str, Any], current_ids: dict[str, str]) -> list[str]:
    """Integrity issues between an inventory document and live baseline ids."""
    issues: list[str] = []
    items = inventory.get("items")
    if not isinstance(items, list):
        return ["Inventory has no 'items' list"]

    open_ids = set()
    seen_ids: set[str] = set()
    for item in items:
        item_id = item.get("id", "<missing id>")
        # Duplicate ids would inflate a discovered batch's size (and thus its
        # scheduled target) while dict-collapsed append-only checks see only
        # one row — every id must be unique.
        if item_id in seen_ids:
            issues.append(f"Duplicate inventory id: {item_id}")
        seen_ids.add(item_id)
        klass = item.get("class")
        if klass not in VALID_CLASSES:
            issues.append(f"Unknown class {klass!r} for inventory item {item_id}")
        provenance = item.get("provenance") or ""
        if not provenance:
            issues.append(f"Inventory item missing provenance: {item_id}")
        elif klass == "discovered" and not PROVENANCE_REF.search(provenance):
            # Hand-edited committed inventories must meet the same bar the
            # generator enforces: discovered items need a PR/issue reference.
            issues.append(f"Discovered item provenance lacks a PR/issue reference: {item_id}")
        if item.get("status") == "open":
            open_ids.add(item_id)

    for item_id in sorted(set(current_ids) - open_ids):
        issues.append(f"Baseline entry not open in inventory (unexplained debt): {item_id}")
    for item_id in sorted(open_ids - set(current_ids)):
        issues.append(f"Inventory item open but absent from baselines (stale): {item_id}")
    return issues


def find_metadata_issues(
    items: list[dict[str, Any]], cohort_ids: dict[str, str], *, as_of: date
) -> list[str]:
    """Derivable-metadata invariants (fail closed on forged classification).

    - Any item whose id is in the cohort set MUST be class=start_cohort with
      discovered_on=COHORT_DATE (cohort reclassification is impossible).
    - No item may claim start_cohort unless its id is in the cohort set.
    - ``discovered`` items must have discovered_on within [COHORT_DATE, as_of].
    - Unknown status values, and resolved items without resolved_on, fail.
    """
    issues: list[str] = []
    cohort_start = date.fromisoformat(COHORT_DATE)
    for item in items:
        item_id = item.get("id", "<missing id>")
        status = item.get("status")
        if status not in VALID_STATUSES:
            issues.append(f"Unknown status {status!r} for inventory item {item_id}")
        elif status == "resolved" and not item.get("resolved_on"):
            issues.append(f"Resolved item missing resolved_on: {item_id}")

        klass = item.get("class")
        raw_discovered = item.get("discovered_on")
        try:
            discovered = date.fromisoformat(raw_discovered or "")
        except (TypeError, ValueError):
            # TypeError: non-string JSON values (numbers, objects) must fail
            # closed as invalid dates, not escape as a traceback.
            issues.append(f"Invalid discovered_on {raw_discovered!r} for inventory item {item_id}")
            continue

        if item_id in cohort_ids:
            if klass != "start_cohort" or raw_discovered != COHORT_DATE:
                issues.append(
                    f"Cohort item reclassified (must be start_cohort @ {COHORT_DATE}): "
                    f"{item_id} (class={klass!r}, discovered_on={raw_discovered!r})"
                )
        elif klass == "start_cohort":
            issues.append(
                f"Item claims start_cohort but is not in the {COHORT_DATE} cohort: {item_id}"
            )
        elif klass == "discovered" and not (cohort_start <= discovered <= as_of):
            issues.append(
                f"discovered_on out of bounds [{COHORT_DATE}, {as_of.isoformat()}]: "
                f"{item_id} ({raw_discovered})"
            )
    return issues


def render_inventory(items: list[dict[str, Any]], cohort_commit: str) -> str:
    doc = {
        "version": 1,
        "generated_by": "scripts/generate_contract_drift_inventory.py",
        "cohort_commit": cohort_commit,
        "items": items,
    }
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# Standalone read-only authority/inventory surfaces (VAL-CDG-001/VAL-CDG-002)
# ---------------------------------------------------------------------------

FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HTTP_METHOD_TOKENS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"}
)

COHORT_ARTIFACT_FILENAME = "contract-drift-original-cohort-v1.json"
PROVENANCE_ARTIFACT_FILENAME = "contract-drift-sdk-provenance-v1.json"
COHORT_LOGICAL_PATH = f"library/{COHORT_ARTIFACT_FILENAME}"
PROVENANCE_LOGICAL_PATH = f"library/{PROVENANCE_ARTIFACT_FILENAME}"
COHORT_SCHEMA = "contract-drift-original-cohort-v1"
PROVENANCE_SCHEMA = "contract-drift-sdk-provenance-v1"
PROJECTION_SCHEMA = "cdg-operation-projection-v1"
COHORT_BYTE_LENGTH = 1692125
COHORT_SHA256 = "565cd84a9a5d266f61b66bd7965e0a036e4817ef5fed32edb8c41a2dea6cc208"
PROVENANCE_BYTE_LENGTH = 898099
PROVENANCE_SHA256 = "21ae1c30200cda6df51dbca7053bbbbde6241ab78a73347b0fe5e4d2ed79f07f"

RATIFIED_ORIGINAL_ID_SET_SHA256 = "c1235670c183b1887ba3fe4280fa0320f9fd6f4a85b8f346d4332ac2aebbe269"
RATIFIED_PROVENANCE_DIGEST_SET_SHA256 = (
    "0d30ce3b083344f19949da12ae2d92952757af0aea800b3f99d447458b6eeba0"
)
RATIFIED_PROJECTION_DIGEST_SET_SHA256 = (
    "2d6790a6f825c53047639d9433f40e3e10b5bfc9e357bcd161f6b341134775e5"
)
RATIFIED_SDK_ID_SET_SHA256 = "51a963079136a92a86485b56f6cef42aafc7749bfad146ce5fb37293524c5762"
RATIFIED_CORE_ID_SET_SHA256 = "b3a1755f027c998d507f13f3ba9093f769cea8720d44bfac12be6beccd626787"
RATIFIED_EXTENDED_ID_SET_SHA256 = "bb1fc41548778022dab3041bc05fc40a4da239a1bd4ad8b1ccbcd1007d90b252"

RATIFIED_CATEGORY_COUNTS = {
    "python_sdk_drift": 74,
    "routes_missing_in_spec": 11,
    "routes_orphaned_in_spec": 17,
    "sdk_missing_from_both": 29,
    "typescript_sdk_drift": 524,
}
SDK_CATEGORIES = ("python_sdk_drift", "typescript_sdk_drift")
SDK_LANGUAGE_BY_CATEGORY = {
    "python_sdk_drift": ["python"],
    "typescript_sdk_drift": ["typescript"],
    "sdk_missing_from_both": ["python", "typescript"],
    "routes_missing_in_spec": [],
    "routes_orphaned_in_spec": [],
}
PARTITION_RULE_VERSION = "cdg-sdk-partition-rule-v1"
PARTITION_DOMAINS = frozenset(
    {
        "agents",
        "debate",
        "evaluation",
        "evidence",
        "explainability",
        "knowledge",
        "learning",
        "memory",
        "ml",
        "ranking",
        "reasoning",
    }
)

EXPECTED_ORIGINAL_TOTAL = 655
EXPECTED_SDK_RECORD_TOTAL = 598
EXPECTED_ROUTE_PARITY_TOTAL = 57
EXPECTED_SOURCE_OCCURRENCES = 690
EXPECTED_MULTI_ATOM_RECORDS = 12
EXPECTED_CORE_UNITS = 75
EXPECTED_EXTENDED_UNITS = 523
EXPECTED_OPERATION_EDGES = 666
EXPECTED_MULTI_EDGE_MEMBERSHIPS = 9
EXPECTED_MAX_OPERATION_EDGES = 4

CANONICAL_CLASSIFIER_PATH = "aragora/cli/commands/review_queue.py"
MERGE_TRAIN_MIRROR_PATH = "scripts/tier4_merge_train.py"
WORKFLOW_DIR = ".github/workflows"

INVENTORY_OUTPUT_SCHEMA = "cdg-standalone-inventory-v1"
AUTHORITY_MANIFEST_SCHEMA = "cdg-authority-manifest-v1"
CLASSIFICATION_OUTPUT_SCHEMA = "cdg-tier-classification-v1"
ERROR_OUTPUT_SCHEMA = "cdg-standalone-error-v1"


class StandaloneError(Exception):
    """Fail-closed error with a machine-readable code for the standalone modes."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _set_digest(values: list[str], key: str, schema: str) -> str:
    return _sha256_hex(_canonical_json_bytes({key: sorted(values), "schema": schema}))


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise StandaloneError(code, message)


def _review_queue_module() -> Any:
    # The sibling checkout's canonical policy must win over any installed copy
    # so the manifest, classifier, and mirror are read at the same version.
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from aragora.cli.commands import review_queue

    return review_queue


def _git_read(repo_root: Path, *argv: str) -> bytes | None:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *argv],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def _resolve_exact_ref(repo_root: Path, ref: Any) -> str:
    _require(
        isinstance(ref, str) and bool(FULL_SHA_RE.fullmatch(ref)),
        "ref_invalid",
        "--ref must be an exact 40-character lowercase hex commit SHA",
    )
    out = _git_read(repo_root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    resolved = (out or b"").decode("ascii", errors="replace").strip()
    _require(
        resolved == ref,
        "ref_unresolved",
        f"--ref does not resolve to itself as a commit object: {ref}",
    )
    return ref


def _validate_repo_relative_path(path: Any) -> str:
    _require(
        isinstance(path, str) and path != "",
        "changed_file_invalid",
        "changed file path must be a non-empty string",
    )
    _require(
        path == path.strip() and "\\" not in path and "\x00" not in path,
        "changed_file_invalid",
        f"non-normalized changed file path rejected: {path!r}",
    )
    _require(
        not path.startswith("/"),
        "changed_file_invalid",
        f"absolute changed file path rejected: {path!r}",
    )
    _require(
        all(segment not in ("", ".", "..") for segment in path.split("/")),
        "changed_file_invalid",
        f"traversal or non-normalized changed file path rejected: {path!r}",
    )
    return path


def _git_blob_at_ref(repo_root: Path, ref: str, path: str) -> bytes | None:
    return _git_read(repo_root, "cat-file", "blob", f"{ref}:{path}")


def _git_blob_oid(repo_root: Path, ref: str, path: str) -> str | None:
    out = _git_read(repo_root, "rev-parse", "--verify", "--quiet", f"{ref}:{path}")
    if out is None:
        return None
    return out.decode("ascii", errors="replace").strip()


def _workflow_paths_at_ref(repo_root: Path, ref: str) -> list[str]:
    out = _git_read(repo_root, "ls-tree", "--name-only", ref, f"{WORKFLOW_DIR}/")
    if out is None:
        return []
    names = out.decode("utf-8", errors="replace").splitlines()
    return sorted(name for name in names if name.endswith(".yml"))


def _discover_artifact_path(
    explicit: Path | None, filename: str, repo_root: Path, logical_path: str
) -> Path:
    if explicit is not None:
        return Path(explicit)
    candidates: list[Path] = []
    mission_dir = os.environ.get("FACTORY_MISSION_DIR", "").strip()
    if mission_dir:
        candidates.append(Path(mission_dir) / "library" / filename)
    settings_path = os.environ.get("FACTORY_RUNTIME_SETTINGS_PATH", "").strip()
    if settings_path:
        candidates.append(Path(settings_path).parent / "library" / filename)
    candidates.append(Path(repo_root) / "library" / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise StandaloneError("artifact_missing", f"canonical artifact not found: {logical_path}")


def _load_canonical_artifact(
    path: Path,
    *,
    logical_path: str,
    expected_length: int,
    expected_sha256: str,
    expected_schema: str,
) -> dict[str, Any]:
    _require(
        Path(path).is_file(),
        "artifact_missing",
        f"canonical artifact not found: {logical_path}",
    )
    _require(
        not Path(path).is_symlink(),
        "artifact_symlink_forbidden",
        f"{logical_path}: symbolic links are forbidden",
    )
    raw = Path(path).read_bytes()
    _require(
        len(raw) == expected_length,
        "artifact_byte_length_mismatch",
        f"{logical_path}: expected {expected_length} bytes, found {len(raw)}",
    )
    _require(
        _sha256_hex(raw) == expected_sha256,
        "artifact_sha256_mismatch",
        f"{logical_path}: SHA-256 does not match the canonical descriptor",
    )
    _require(
        not raw.startswith(b"\xef\xbb\xbf"),
        "artifact_noncanonical_bytes",
        f"{logical_path}: UTF-8 BOM is forbidden",
    )
    _require(
        raw.endswith(b"\n") and raw.count(b"\n") == 1,
        "artifact_noncanonical_bytes",
        f"{logical_path}: must contain exactly one terminal LF",
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StandaloneError("artifact_not_utf8", f"{logical_path}: {exc}") from exc

    def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StandaloneError(
                    "artifact_duplicate_key", f"{logical_path}: duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        doc = json.loads(text, object_pairs_hook=_reject_duplicates)
    except json.JSONDecodeError as exc:
        raise StandaloneError("artifact_invalid_json", f"{logical_path}: {exc}") from exc
    _require(
        _canonical_json_bytes(doc) + b"\n" == raw,
        "artifact_noncanonical_bytes",
        f"{logical_path}: bytes are not compact sorted-key JSON plus one terminal LF",
    )
    _require(
        isinstance(doc, dict) and doc.get("schema") == expected_schema,
        "artifact_schema_mismatch",
        f"{logical_path}: schema must be {expected_schema!r}",
    )
    return doc


def _verify_original_records(cohort: dict[str, Any]) -> dict[str, Any]:
    records = cohort.get("original_records")
    _require(
        isinstance(records, list) and len(records) == EXPECTED_ORIGINAL_TOTAL,
        "cohort_count_mismatch",
        f"expected exactly {EXPECTED_ORIGINAL_TOTAL} original records",
    )
    assert isinstance(records, list)
    ids: list[str] = []
    category_counts: dict[str, int] = {}
    method_bearing = 0
    method_null = 0
    for record in records:
        _require(
            isinstance(record, dict),
            "cohort_record_malformed",
            "original record must be a JSON object",
        )
        category = record.get("category")
        literal = record.get("exact_historical_literal_record")
        _require(
            category in RATIFIED_CATEGORY_COUNTS,
            "cohort_unknown_category",
            f"unknown original record category: {category!r}",
        )
        _require(
            isinstance(literal, str) and literal != "",
            "cohort_record_malformed",
            "original record is missing its exact historical literal",
        )
        payload = _canonical_json_bytes(
            {
                "category": category,
                "exact_historical_literal_record": literal,
                "schema": "cdg-original-record-id-v1",
            }
        )
        expected_id = "cdg1:" + _sha256_hex(payload)
        _require(
            record.get("id_payload_byte_length") == len(payload)
            and record.get("id_payload_sha256") == _sha256_hex(payload)
            and record.get("original_record_id") == expected_id,
            "original_id_mismatch",
            f"original_record_id does not reproduce for literal {literal!r}",
        )
        _require(
            record.get("sdk_language") == SDK_LANGUAGE_BY_CATEGORY[category],
            "cohort_record_malformed",
            f"sdk_language must be category-derived provenance: {expected_id}",
        )
        method = record.get("method")
        if category in SDK_CATEGORIES:
            _require(
                isinstance(method, str) and method in HTTP_METHOD_TOKENS,
                "cohort_record_malformed",
                f"SDK original must carry an exact uppercase method token: {expected_id}",
            )
            method_bearing += 1
        else:
            _require(
                method is None,
                "cohort_record_malformed",
                f"path-level original must carry method=null: {expected_id}",
            )
            method_null += 1
        category_counts[category] = category_counts.get(category, 0) + 1
        ids.append(expected_id)
    _require(
        len(set(ids)) == EXPECTED_ORIGINAL_TOTAL,
        "cohort_duplicate_original_id",
        "duplicate original_record_id in cohort artifact",
    )
    _require(
        category_counts == RATIFIED_CATEGORY_COUNTS,
        "cohort_category_counts_mismatch",
        f"category counts {category_counts} do not match the ratified schema",
    )
    _require(
        method_bearing == EXPECTED_SDK_RECORD_TOTAL and method_null == EXPECTED_ROUTE_PARITY_TOTAL,
        "cohort_method_plane_mismatch",
        "method-bearing/path-level split must be exactly 598/57",
    )
    id_set_sha256 = _set_digest(ids, "original_record_ids", "cdg-original-record-id-set-v1")
    _require(
        id_set_sha256 == RATIFIED_ORIGINAL_ID_SET_SHA256,
        "original_id_set_digest_mismatch",
        f"recomputed original ID set digest {id_set_sha256} is not the ratified value",
    )
    return {
        "ids": sorted(ids),
        "category_counts": category_counts,
        "id_set_sha256": id_set_sha256,
    }


def _atom_matches_domain(atom: Any, record_id: Any) -> bool:
    _require(
        isinstance(atom, str) and atom != "",
        "provenance_partition_malformed",
        f"provenance atom must be a non-empty string: {record_id}",
    )
    normalized = atom.replace("-", "_")
    if normalized in PARTITION_DOMAINS:
        return True
    return normalized.endswith("s") and normalized[:-1] in PARTITION_DOMAINS


def _verify_sdk_provenance(provenance: dict[str, Any], cohort: dict[str, Any]) -> dict[str, Any]:
    records = provenance.get("records")
    _require(
        isinstance(records, list) and len(records) == EXPECTED_SDK_RECORD_TOTAL,
        "provenance_count_mismatch",
        f"expected exactly {EXPECTED_SDK_RECORD_TOTAL} SDK provenance records",
    )
    assert isinstance(records, list)
    digest_by_id: dict[str, str] = {}
    digests: list[str] = []
    core_ids: list[str] = []
    extended_ids: list[str] = []
    occurrence_total = 0
    multi_atom = 0
    for record in records:
        _require(
            isinstance(record, dict),
            "provenance_record_malformed",
            "SDK provenance record must be a JSON object",
        )
        record_id = record.get("original_record_id")
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        digest = _sha256_hex(_canonical_json_bytes(body))
        _require(
            record.get("record_sha256") == digest,
            "provenance_record_digest_mismatch",
            f"record_sha256 does not reproduce for {record_id}",
        )
        atoms = record.get("provenance_atoms")
        _require(
            isinstance(atoms, list) and len(atoms) > 0,
            "provenance_partition_malformed",
            f"provenance_atoms must be a non-empty array: {record_id}",
        )
        matches = [_atom_matches_domain(atom, record_id) for atom in atoms]
        expected_partition = "core" if any(matches) else "extended"
        _require(
            record.get("partition") == expected_partition,
            "provenance_partition_mismatch",
            f"independently recomputed partition disagrees for {record_id}",
        )
        occurrences = record.get("source_occurrences")
        _require(
            isinstance(occurrences, list) and len(occurrences) > 0,
            "provenance_occurrence_missing",
            f"source_occurrences must be a non-empty array: {record_id}",
        )
        occurrence_total += len(occurrences)
        if len(set(atoms)) > 1:
            multi_atom += 1
        _require(
            isinstance(record_id, str) and record_id not in digest_by_id,
            "provenance_duplicate_link",
            f"duplicate or malformed original_record_id link: {record_id!r}",
        )
        digest_by_id[record_id] = digest
        digests.append(digest)
        if expected_partition == "core":
            core_ids.append(record_id)
        else:
            extended_ids.append(record_id)
    _require(
        occurrence_total == EXPECTED_SOURCE_OCCURRENCES,
        "provenance_occurrence_count_mismatch",
        f"expected {EXPECTED_SOURCE_OCCURRENCES} source occurrences, found {occurrence_total}",
    )
    _require(
        multi_atom == EXPECTED_MULTI_ATOM_RECORDS,
        "provenance_multi_atom_count_mismatch",
        f"expected {EXPECTED_MULTI_ATOM_RECORDS} multi-atom records, found {multi_atom}",
    )
    _require(
        len(core_ids) == EXPECTED_CORE_UNITS and len(extended_ids) == EXPECTED_EXTENDED_UNITS,
        "partition_count_mismatch",
        f"core/extended partition must be exactly {EXPECTED_CORE_UNITS}/{EXPECTED_EXTENDED_UNITS}",
    )
    digest_set_sha256 = _set_digest(
        digests, "record_sha256_values", "cdg-sdk-provenance-record-digest-set-v1"
    )
    _require(
        digest_set_sha256 == RATIFIED_PROVENANCE_DIGEST_SET_SHA256,
        "provenance_digest_set_mismatch",
        "recomputed provenance record-digest-set is not the ratified value",
    )
    sdk_set_sha256 = _set_digest(
        list(digest_by_id), "original_record_ids", "cdg-sdk-original-record-id-set-v1"
    )
    core_set_sha256 = _set_digest(
        core_ids, "original_record_ids", "cdg-core-original-record-id-set-v1"
    )
    extended_set_sha256 = _set_digest(
        extended_ids, "original_record_ids", "cdg-extended-original-record-id-set-v1"
    )
    _require(
        sdk_set_sha256 == RATIFIED_SDK_ID_SET_SHA256
        and core_set_sha256 == RATIFIED_CORE_ID_SET_SHA256
        and extended_set_sha256 == RATIFIED_EXTENDED_ID_SET_SHA256,
        "partition_set_digest_mismatch",
        "recomputed SDK/core/extended ID set digests are not the pinned values",
    )
    cohort_sdk = [
        record for record in cohort["original_records"] if record.get("category") in SDK_CATEGORIES
    ]
    _require(
        {record["original_record_id"] for record in cohort_sdk} == set(digest_by_id),
        "provenance_link_mismatch",
        "SDK provenance records must map one-to-one onto cohort SDK originals",
    )
    for record in cohort_sdk:
        _require(
            record.get("sdk_provenance_record_sha256")
            == digest_by_id[record["original_record_id"]],
            "provenance_link_digest_mismatch",
            f"cohort sdk_provenance_record_sha256 mismatch: {record['original_record_id']}",
        )
    return {
        "record_count": EXPECTED_SDK_RECORD_TOTAL,
        "source_occurrence_count": occurrence_total,
        "multi_atom_record_count": multi_atom,
        "missing_record_count": 0,
        "record_digest_set_sha256": digest_set_sha256,
        "baseline_birth": provenance.get("baseline_birth"),
        "dependencies": provenance.get("dependencies"),
        "extraction_algorithm": provenance.get("extraction_algorithm"),
        "records": records,
        "partition": {
            "partition_rule_version": PARTITION_RULE_VERSION,
            "core_count": len(core_ids),
            "extended_count": len(extended_ids),
            "intersection_count": 0,
            "union_count": EXPECTED_SDK_RECORD_TOTAL,
            "sdk_original_record_id_set_sha256": sdk_set_sha256,
            "core_original_record_id_set_sha256": core_set_sha256,
            "extended_original_record_id_set_sha256": extended_set_sha256,
            "core_unit_ids": sorted(core_ids),
            "extended_unit_ids": sorted(extended_ids),
        },
    }


def _verify_operation_projection(cohort: dict[str, Any], original_ids: list[str]) -> dict[str, Any]:
    projection = cohort.get("operation_projection")
    _require(
        isinstance(projection, dict) and projection.get("schema") == PROJECTION_SCHEMA,
        "projection_schema_mismatch",
        f"operation projection schema must be {PROJECTION_SCHEMA!r}",
    )
    assert isinstance(projection, dict)
    records = projection.get("records")
    _require(
        isinstance(records, list) and len(records) == EXPECTED_ORIGINAL_TOTAL,
        "projection_membership_mismatch",
        f"expected exactly {EXPECTED_ORIGINAL_TOTAL} projection memberships",
    )
    assert isinstance(records, list)
    membership_ids: list[str] = []
    digests: list[str] = []
    edge_total = 0
    multi_edge = 0
    max_edges = 0
    edge_count_distribution: dict[str, int] = {}
    for record in records:
        _require(
            isinstance(record, dict),
            "projection_record_malformed",
            "projection membership must be a JSON object",
        )
        record_id = record.get("original_record_id")
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        digest = _sha256_hex(_canonical_json_bytes(body))
        _require(
            record.get("record_sha256") == digest,
            "projection_record_digest_mismatch",
            f"projection record_sha256 does not reproduce for {record_id}",
        )
        digests.append(digest)
        edges = record.get("operation_edges")
        _require(
            isinstance(edges, list) and len(edges) > 0,
            "projection_edges_missing",
            f"projection membership must carry at least one edge: {record_id}",
        )
        for edge in edges:
            _require(
                isinstance(edge, dict)
                and isinstance(edge.get("method"), str)
                and edge.get("method") in HTTP_METHOD_TOKENS,
                "projection_edge_method_invalid",
                f"projection edge must carry an exact uppercase method token: {record_id}",
            )
            _require(
                isinstance(edge.get("normalized_path"), str)
                and edge.get("normalized_path") != ""
                and isinstance(edge.get("evidence"), list)
                and len(edge.get("evidence")) > 0,
                "projection_edge_malformed",
                f"projection edge must carry a normalized path and evidence: {record_id}",
            )
        if record.get("category") in SDK_CATEGORIES:
            _require(
                len(edges) == 1,
                "projection_sdk_membership_edge_count_mismatch",
                f"SDK projection membership must carry exactly one edge: {record_id}",
            )
        edge_total += len(edges)
        if len(edges) > 1:
            multi_edge += 1
        max_edges = max(max_edges, len(edges))
        key = str(len(edges))
        edge_count_distribution[key] = edge_count_distribution.get(key, 0) + 1
        membership_ids.append(record_id)
    _require(
        sorted(membership_ids) == original_ids,
        "projection_membership_mismatch",
        "projection must contain exactly one membership per original record",
    )
    _require(
        edge_total == EXPECTED_OPERATION_EDGES
        and multi_edge == EXPECTED_MULTI_EDGE_MEMBERSHIPS
        and max_edges == EXPECTED_MAX_OPERATION_EDGES,
        "projection_edge_counts_mismatch",
        f"projection edge counts must be {EXPECTED_OPERATION_EDGES}/"
        f"{EXPECTED_MULTI_EDGE_MEMBERSHIPS}/max {EXPECTED_MAX_OPERATION_EDGES}",
    )
    digest_set_sha256 = _set_digest(
        digests, "record_sha256_values", "cdg-operation-projection-record-digest-set-v1"
    )
    _require(
        digest_set_sha256 == RATIFIED_PROJECTION_DIGEST_SET_SHA256,
        "projection_digest_set_mismatch",
        "recomputed projection record-digest-set is not the ratified value",
    )
    return {
        "schema": PROJECTION_SCHEMA,
        "membership_count": EXPECTED_ORIGINAL_TOTAL,
        "operation_edge_count": edge_total,
        "multi_edge_membership_count": multi_edge,
        "max_operation_edges": max_edges,
        "edge_count_distribution": edge_count_distribution,
        "record_digest_set_sha256": digest_set_sha256,
        "records": records,
    }


def _active_inventory_binding(repo_root: Path, ref: str) -> dict[str, Any]:
    raw = _git_blob_at_ref(repo_root, ref, DEFAULT_INVENTORY)
    _require(
        raw is not None,
        "active_inventory_missing",
        f"{DEFAULT_INVENTORY} not found at --ref",
    )
    assert raw is not None
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StandaloneError(
            "active_inventory_invalid", f"{DEFAULT_INVENTORY} at --ref is not valid JSON: {exc}"
        ) from exc
    items = doc.get("items") if isinstance(doc, dict) else None
    _require(
        isinstance(items, list),
        "active_inventory_invalid",
        f"{DEFAULT_INVENTORY} at --ref has no items list",
    )
    assert isinstance(doc, dict)
    assert isinstance(items, list)
    open_items = sum(1 for item in items if isinstance(item, dict) and item.get("status") == "open")
    resolved_items = sum(
        1 for item in items if isinstance(item, dict) and item.get("status") == "resolved"
    )
    return {
        "path": DEFAULT_INVENTORY,
        "git_blob_oid": _git_blob_oid(repo_root, ref, DEFAULT_INVENTORY),
        "byte_length": len(raw),
        "sha256": _sha256_hex(raw),
        "cohort_commit": str(doc.get("cohort_commit", "")),
        "item_total": len(items),
        "open_items": open_items,
        "resolved_items": resolved_items,
    }


def _artifact_descriptor(
    logical_path: str, byte_length: int, sha256: str, schema: str
) -> dict[str, Any]:
    return {
        "logical_path": logical_path,
        "byte_length": byte_length,
        "sha256": sha256,
        "schema": schema,
        "canonical_bytes": True,
        "utf8_no_bom": True,
        "single_terminal_lf": True,
    }


def _build_inventory_document(
    repo_root: Path, ref: str, cohort_path: Path, provenance_path: Path
) -> dict[str, Any]:
    cohort = _load_canonical_artifact(
        cohort_path,
        logical_path=COHORT_LOGICAL_PATH,
        expected_length=COHORT_BYTE_LENGTH,
        expected_sha256=COHORT_SHA256,
        expected_schema=COHORT_SCHEMA,
    )
    provenance = _load_canonical_artifact(
        provenance_path,
        logical_path=PROVENANCE_LOGICAL_PATH,
        expected_length=PROVENANCE_BYTE_LENGTH,
        expected_sha256=PROVENANCE_SHA256,
        expected_schema=PROVENANCE_SCHEMA,
    )
    original = _verify_original_records(cohort)
    provenance_facts = _verify_sdk_provenance(provenance, cohort)
    projection_facts = _verify_operation_projection(cohort, original["ids"])
    document = {
        "schema": INVENTORY_OUTPUT_SCHEMA,
        "status": "ok",
        "ref": ref,
        "artifacts": {
            "original_cohort": _artifact_descriptor(
                COHORT_LOGICAL_PATH, COHORT_BYTE_LENGTH, COHORT_SHA256, COHORT_SCHEMA
            ),
            "sdk_provenance": _artifact_descriptor(
                PROVENANCE_LOGICAL_PATH,
                PROVENANCE_BYTE_LENGTH,
                PROVENANCE_SHA256,
                PROVENANCE_SCHEMA,
            ),
        },
        "membership_anchor": cohort.get("membership_anchor"),
        "membership_sources": cohort.get("membership_sources"),
        "original_record_total": EXPECTED_ORIGINAL_TOTAL,
        "original_record_ids": original["ids"],
        "original_record_id_set_sha256": original["id_set_sha256"],
        "category_counts": original["category_counts"],
        "method_bearing_sdk_records": EXPECTED_SDK_RECORD_TOTAL,
        "method_null_route_parity_records": EXPECTED_ROUTE_PARITY_TOTAL,
        "original_records": cohort["original_records"],
        "operation_projection": projection_facts,
        "sdk_provenance": provenance_facts,
        "active_inventory": _active_inventory_binding(repo_root, ref),
    }
    document["inventory_sha256"] = _sha256_hex(_canonical_json_bytes(document))
    return document


def _inventory_facts(document: dict[str, Any]) -> dict[str, Any]:
    projection = {
        key: value for key, value in document["operation_projection"].items() if key != "records"
    }
    provenance = {
        key: value for key, value in document["sdk_provenance"].items() if key != "records"
    }
    return {
        "inventory_schema": document["schema"],
        "inventory_sha256": document["inventory_sha256"],
        "membership_anchor": document["membership_anchor"],
        "membership_sources": document["membership_sources"],
        "original_record_total": document["original_record_total"],
        "original_record_id_set_sha256": document["original_record_id_set_sha256"],
        "category_counts": document["category_counts"],
        "method_bearing_sdk_records": document["method_bearing_sdk_records"],
        "method_null_route_parity_records": document["method_null_route_parity_records"],
        "operation_projection": projection,
        "sdk_provenance": provenance,
        "active_inventory": document["active_inventory"],
    }


def _matched_authority_rule(review_queue: Any, path: str) -> str | None:
    for prefix in review_queue.TIER_4_PREFIXES:
        if review_queue._matches_prefix(path, (prefix,)):
            return prefix
    return None


def _authority_closure(review_queue: Any, repo_root: Path, ref: str) -> dict[str, str]:
    roots = [str(prefix) for prefix in review_queue.CONTRACT_DRIFT_AUTHORITY_PREFIXES]
    reasons = dict.fromkeys(roots, "canonical_authority_root")
    reasons.setdefault(CANONICAL_CLASSIFIER_PATH, "canonical_classification_policy")
    for workflow in _workflow_paths_at_ref(repo_root, ref):
        blob = _git_blob_at_ref(repo_root, ref, workflow)
        if blob is None:
            continue
        text = blob.decode("utf-8", errors="replace")
        if any(root in text for root in roots):
            reasons.setdefault(workflow, "workflow_references_authority_root")
    return reasons


def _file_binding(
    repo_root: Path, ref: str, path: str, review_queue: Any, reason: str
) -> dict[str, Any]:
    blob = _git_blob_at_ref(repo_root, ref, path)
    _require(
        blob is not None,
        "closure_member_missing",
        f"authority closure member missing at --ref: {path}",
    )
    assert blob is not None
    tier, tier_name, _tier_reason = review_queue._classify_model_review_tier([path])
    _require(
        tier == review_queue.CONTRACT_DRIFT_AUTHORITY_TIER,
        "closure_member_not_tier4",
        f"authority closure member does not classify Tier 4: {path}",
    )
    return {
        "path": path,
        "git_blob_oid": _git_blob_oid(repo_root, ref, path),
        "byte_length": len(blob),
        "sha256": _sha256_hex(blob),
        "closure_reason": reason,
        "effective_tier": tier,
        "tier_name": tier_name,
        "matched_authority_rule": _matched_authority_rule(review_queue, path),
    }


def _require_policy_module_matches_ref(repo_root: Path, ref: str) -> None:
    """The manifest binds file blobs at --ref but classifies them with the
    imported live policy module; certifying across a mismatch would let the
    digest attest a closure that was never the policy at the governed ref."""
    blob = _git_blob_at_ref(repo_root, ref, CANONICAL_CLASSIFIER_PATH)
    _require(
        blob is not None,
        "policy_module_missing_at_ref",
        f"canonical policy module missing at --ref: {CANONICAL_CLASSIFIER_PATH}",
    )
    assert blob is not None
    live_path = repo_root / CANONICAL_CLASSIFIER_PATH
    _require(
        live_path.exists(),
        "policy_module_missing_live",
        f"canonical policy module missing from working tree: {CANONICAL_CLASSIFIER_PATH}",
    )
    _require(
        _sha256_hex(live_path.read_bytes()) == _sha256_hex(blob),
        "policy_module_ref_mismatch",
        "working-tree policy module differs from the --ref blob: "
        f"{CANONICAL_CLASSIFIER_PATH}; run from a checkout of the governed ref",
    )


def _build_authority_manifest(
    repo_root: Path, ref: str, cohort_path: Path, provenance_path: Path
) -> dict[str, Any]:
    _require_policy_module_matches_ref(repo_root, ref)
    review_queue = _review_queue_module()
    inventory_document = _build_inventory_document(repo_root, ref, cohort_path, provenance_path)
    reasons = _authority_closure(review_queue, repo_root, ref)
    repo_files = sorted(reasons)
    file_bindings = [
        _file_binding(repo_root, ref, path, review_queue, reasons[path]) for path in repo_files
    ]
    mirror_blob = _git_blob_at_ref(repo_root, ref, MERGE_TRAIN_MIRROR_PATH)
    _require(
        mirror_blob is not None,
        "mirror_missing",
        f"merge-train mirror missing at --ref: {MERGE_TRAIN_MIRROR_PATH}",
    )
    assert mirror_blob is not None
    manifest = {
        "schema": AUTHORITY_MANIFEST_SCHEMA,
        "status": "ok",
        "ref": ref,
        "policy": {
            "canonical_policy_module": "aragora.cli.commands.review_queue",
            "policy_version": review_queue.CONTRACT_DRIFT_AUTHORITY_POLICY_VERSION,
            "tier": review_queue.CONTRACT_DRIFT_AUTHORITY_TIER,
            "authority_roots": sorted(
                str(prefix) for prefix in review_queue.CONTRACT_DRIFT_AUTHORITY_PREFIXES
            ),
        },
        "repo_files": repo_files,
        "file_bindings": file_bindings,
        "merge_train_mirror": {
            "path": MERGE_TRAIN_MIRROR_PATH,
            "git_blob_oid": _git_blob_oid(repo_root, ref, MERGE_TRAIN_MIRROR_PATH),
            "byte_length": len(mirror_blob),
            "sha256": _sha256_hex(mirror_blob),
            "role": "dependency_light_mirror",
            "in_repo_files": False,
            "reason": (
                "canonical policy does not classify the mirror itself as Tier 4; "
                "bound as metadata only"
            ),
        },
        "artifacts": inventory_document["artifacts"],
        "inventory_facts": _inventory_facts(inventory_document),
    }
    manifest["authority_manifest_sha256"] = _sha256_hex(_canonical_json_bytes(manifest))
    return manifest


def _build_tier_classification(
    repo_root: Path,
    ref: str,
    changed_files: list[str],
    cohort_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    _require_policy_module_matches_ref(repo_root, ref)
    review_queue = _review_queue_module()
    manifest = _build_authority_manifest(repo_root, ref, cohort_path, provenance_path)
    tier, tier_name, tier_reason = review_queue._classify_model_review_tier(list(changed_files))
    matched: str | None = None
    for path in changed_files:
        matched = _matched_authority_rule(review_queue, path)
        if matched is not None:
            break
    return {
        "schema": CLASSIFICATION_OUTPUT_SCHEMA,
        "status": "ok",
        "ref": ref,
        "changed_files": list(changed_files),
        "effective_tier": tier,
        "tier_name": tier_name,
        "tier_reason": tier_reason,
        "matched_authority_path": matched,
        "authority_manifest_sha256": manifest["authority_manifest_sha256"],
        "inventory_facts": manifest["inventory_facts"],
    }


def _emit_standalone(document: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")


def run_standalone(args: argparse.Namespace) -> int:
    if args.classify_tier and args.authority_manifest:
        mode = "invalid"
    elif args.classify_tier:
        mode = "classify-tier"
    elif args.authority_manifest:
        mode = "authority-manifest"
    else:
        mode = "inventory"
    try:
        _require(
            mode != "invalid",
            "mode_conflict",
            "--classify-tier and --authority-manifest are mutually exclusive",
        )
        repo_root = Path(args.repo_root)
        ref = _resolve_exact_ref(repo_root, args.ref)
        changed_files: list[str] = []
        if mode == "classify-tier":
            _require(
                bool(args.changed_file),
                "changed_file_required",
                "--classify-tier requires at least one --changed-file",
            )
            changed_files = [_validate_repo_relative_path(path) for path in args.changed_file]
        cohort_path = _discover_artifact_path(
            args.cohort_artifact, COHORT_ARTIFACT_FILENAME, repo_root, COHORT_LOGICAL_PATH
        )
        provenance_path = _discover_artifact_path(
            args.sdk_provenance_artifact,
            PROVENANCE_ARTIFACT_FILENAME,
            repo_root,
            PROVENANCE_LOGICAL_PATH,
        )
        if mode == "inventory":
            document = _build_inventory_document(repo_root, ref, cohort_path, provenance_path)
        elif mode == "authority-manifest":
            document = _build_authority_manifest(repo_root, ref, cohort_path, provenance_path)
        else:
            document = _build_tier_classification(
                repo_root, ref, changed_files, cohort_path, provenance_path
            )
    except StandaloneError as exc:
        _emit_standalone(
            {
                "schema": ERROR_OUTPUT_SCHEMA,
                "status": "fail",
                "mode": mode,
                "error": {"code": exc.code, "message": exc.message},
            }
        )
        return 1
    except Exception as exc:  # fail closed: never emit a partial success artifact
        _emit_standalone(
            {
                "schema": ERROR_OUTPUT_SCHEMA,
                "status": "fail",
                "mode": mode,
                "error": {"code": "unhandled_error", "message": f"{type(exc).__name__}: {exc}"},
            }
        )
        return 1
    _emit_standalone(document)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--cohort-commit", default=COHORT_COMMIT)
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument(
        "--provenance",
        default=None,
        help="Required provenance (with a PR/issue link) for any post-cohort items",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail (exit 1) if the inventory is out of sync with the baselines",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Exact full commit SHA for the standalone read-only modes",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (standalone read-only modes)",
    )
    parser.add_argument(
        "--authority-manifest",
        action="store_true",
        help="Emit the Tier-4 authority closure manifest at --ref (read-only)",
    )
    parser.add_argument(
        "--classify-tier",
        action="store_true",
        help="Classify --changed-file paths with the canonical review_queue policy (read-only)",
    )
    parser.add_argument(
        "--changed-file",
        action="append",
        default=None,
        help="Repo-relative changed file path for --classify-tier (repeatable)",
    )
    parser.add_argument(
        "--cohort-artifact",
        type=Path,
        default=None,
        help=f"Path to the canonical {COHORT_LOGICAL_PATH} artifact",
    )
    parser.add_argument(
        "--sdk-provenance-artifact",
        type=Path,
        default=None,
        help=f"Path to the canonical {PROVENANCE_LOGICAL_PATH} artifact",
    )
    args = parser.parse_args()

    if args.authority_manifest or args.classify_tier or args.ref is not None:
        return run_standalone(args)

    repo_root = args.repo_root
    inventory_path = args.inventory or repo_root / DEFAULT_INVENTORY
    as_of = date.fromisoformat(args.as_of)

    if args.provenance is not None and not PROVENANCE_REF.search(args.provenance):
        print("ERROR: --provenance must include a PR/issue reference (#123 or URL)")
        return 1

    try:
        working_docs = load_working_docs(repo_root)
        current_ids = collect_ids(working_docs)
        cohort_ids = collect_ids(load_git_docs(repo_root, args.cohort_commit))
    except (ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}")
        return 1

    duplicate_issues = find_duplicate_entry_issues(working_docs)
    if duplicate_issues:
        print("FAIL: duplicate baseline entries (dedupe the baseline lists first):")
        for issue in duplicate_issues:
            print(f"  - {issue}")
        return 1

    existing: dict[str, Any] = {"items": []}
    if inventory_path.exists():
        existing = json.loads(inventory_path.read_text())

    if args.check:
        if not inventory_path.exists():
            print(f"FAIL: inventory missing: {inventory_path}")
            return 1
        issues = find_sync_issues(existing, current_ids)
        issues += find_metadata_issues(existing.get("items", []), cohort_ids, as_of=as_of)
        if issues:
            print("FAIL: inventory out of sync with baselines:")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        print(f"OK: inventory in sync ({len(current_ids)} open items)")
        return 0

    items, unexplained = build_inventory(
        current_ids,
        cohort_ids,
        existing.get("items", []),
        as_of=as_of,
        provenance=args.provenance,
    )
    if unexplained:
        print(
            "FAIL: post-cohort items without provenance (pass --provenance with a PR/issue link):"
        )
        for item_id in unexplained:
            print(f"  - {item_id}")
        return 1

    metadata_issues = find_metadata_issues(items, cohort_ids, as_of=as_of)
    # Birth-state guard: the generator must never mint a resolved item that
    # was not already present in the existing inventory file. Resolution is
    # only ever a transition of recorded history, never a creation.
    existing_ids = {i.get("id") for i in existing.get("items", []) if isinstance(i, dict)}
    metadata_issues += [
        f"Resolved item minted without prior history (must be born open): {item['id']}"
        for item in items
        if item.get("status") == "resolved" and item["id"] not in existing_ids
    ]
    if metadata_issues:
        print("FAIL: inventory metadata invariants violated (refusing to write):")
        for issue in metadata_issues:
            print(f"  - {issue}")
        return 1

    inventory_path.write_text(render_inventory(items, args.cohort_commit))
    open_count = sum(1 for item in items if item["status"] == "open")
    resolved = len(items) - open_count
    print(f"Wrote {inventory_path}: {open_count} open, {resolved} resolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
