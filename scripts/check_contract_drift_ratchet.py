#!/usr/bin/env python3
"""Enforce contract drift governance: program burn-down and per-PR delta ratchet.

Two modes:

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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, cast

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
BOUNDARY_EVIDENCE_INDEX_SCHEMA = "contract-drift-boundary-evidence-index-v1"
BOUNDARY_MANIFEST_SCHEMA = "contract-drift-boundary-manifest-v1"
BOUNDARY_CAPSULE_MANIFEST_SCHEMA = "contract-drift-boundary-capsule-manifest-v1"
BOUNDARY_CAPSULE_PAYLOAD_SCHEMA = "contract-drift-boundary-capsule-payload-v1"
CANONICAL_SERIALIZATION = (
    "UTF-8; no BOM; object keys sorted; compact separators comma/colon; "
    "declared array orders; exactly one terminal LF"
)
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

COHORT_ARTIFACT: dict[str, Any] = {
    "byte_length": 1_692_125,
    "filename": "contract-drift-original-cohort-v1.json",
    "logical_path": "library/contract-drift-original-cohort-v1.json",
    "sha256": "565cd84a9a5d266f61b66bd7965e0a036e4817ef5fed32edb8c41a2dea6cc208",
}
PROVENANCE_ARTIFACT: dict[str, Any] = {
    "byte_length": 898_099,
    "filename": "contract-drift-sdk-provenance-v1.json",
    "logical_path": "library/contract-drift-sdk-provenance-v1.json",
    "sha256": "21ae1c30200cda6df51dbca7053bbbbde6241ab78a73347b0fe5e4d2ed79f07f",
}
ORIGINAL_ID_SET_SHA256 = "c1235670c183b1887ba3fe4280fa0320f9fd6f4a85b8f346d4332ac2aebbe269"
PROJECTION_RECORD_SET_SHA256 = "2d6790a6f825c53047639d9433f40e3e10b5bfc9e357bcd161f6b341134775e5"
PROVENANCE_RECORD_SET_SHA256 = "0d30ce3b083344f19949da12ae2d92952757af0aea800b3f99d447458b6eeba0"
SDK_ID_SET_SHA256 = "51a963079136a92a86485b56f6cef42aafc7749bfad146ce5fb37293524c5762"
CORE_ID_SET_SHA256 = "b3a1755f027c998d507f13f3ba9093f769cea8720d44bfac12be6beccd626787"
EXTENDED_ID_SET_SHA256 = "bb1fc41548778022dab3041bc05fc40a4da239a1bd4ad8b1ccbcd1007d90b252"
EXPECTED_CATEGORY_COUNTS = {
    "python_sdk_drift": 74,
    "routes_missing_in_spec": 11,
    "routes_orphaned_in_spec": 17,
    "sdk_missing_from_both": 29,
    "typescript_sdk_drift": 524,
}
CORE_DOMAINS = frozenset(
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
STAGE1_TEST_MATRIX = (
    "tests/governance/test_contract_drift_measurement_authority_tier.py",
    "tests/scripts/test_generate_contract_drift_inventory.py",
    "tests/scripts/test_tier4_merge_train.py",
)
STAGE1_REQUIRED_TESTS = tuple(
    sorted(
        {
            "test_all_loaded_repository_modules_are_under_exact_ref_extraction_root",
            "test_authority_roots_are_tier4",
            "test_canonical_tier_cli_is_read_only_and_digest_bound",
            "test_classifier_and_merge_train_closure_match",
            "test_deterministic_bounded_authority_dependency_closure_has_incoming_edges_and_exact_ref_digests",
            "test_local_reusable_workflows_and_composite_actions_join_closure",
            "test_measured_sdk_handler_openapi_subjects_are_not_authority_dependencies",
            "test_merge_train_mirror_is_normal_repo_file_authority_member",
            "test_standalone_classifier_extracts_and_calls_exact_ref_canonical_review_queue_policy_under_I_S",
            "test_workflows_yml_and_yaml_recurse_through_structural_run_uses_and_path_filters",
        }
    )
)

_READ_ONLY_GIT_SUBCOMMANDS = frozenset(
    {
        "cat-file",
        "diff",
        "for-each-ref",
        "ls-files",
        "ls-tree",
        "log",
        "merge-base",
        "rev-list",
        "rev-parse",
        "show",
        "show-ref",
        "status",
    }
)
_FORBIDDEN_CALLER_FIELDS = frozenset(
    {
        "actions",
        "caller_objects",
        "caller_summary",
        "counts",
        "digests",
        "inherited_from_boundary",
        "object_list",
        "operation_log",
        "parsed_objects",
        "prior_boundary_manifest",
        "results",
        "summary",
        "summaries",
    }
)
_RESOURCE_SCHEMAS = {
    "boundary_chronology": "contract-drift-boundary-chronology-v1",
    "corrective_bootstrap": "contract-drift-corrective-bootstrap-proof-v1",
    "route_truth": "contract-drift-route-truth-proof-v1",
    "core_sdk": "contract-drift-core-sdk-proof-v1",
    "extended_sdk": "contract-drift-extended-sdk-proof-v1",
    "final_seal": "contract-drift-final-seal-proof-v1",
    "external_prerequisites": "contract-drift-external-prerequisites-v1",
    "durable_capsule": "contract-drift-durable-capsule-v1",
    "governed_prs": "contract-drift-governed-prs-v1",
    "first_parent_receipts": "contract-drift-first-parent-receipts-v1",
}


class BoundaryBlocked(RuntimeError):
    """Authenticated external unavailability or independently observed movement."""


def _canonical_json_bytes(value: Any, *, terminal_lf: bool = False) -> bytes:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if terminal_lf:
        rendered += "\n"
    return rendered.encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_set(schema: str, values: Iterable[str], field: str) -> str:
    return _sha256_bytes(_canonical_json_bytes({"schema": schema, field: sorted(values)}))


def _duplicate_key_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _stable_file_bytes(path: Path, *, attempts: int = 3) -> bytes:
    last_identity: tuple[int, int, int, int, int, int] | None = None
    for _attempt in range(attempts):
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError as exc:
            raise ValueError(f"external input missing: {path}") from exc
        except OSError as exc:
            raise ValueError(
                f"external input cannot be opened without following links: {path}"
            ) from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError(f"external input is not a uniquely linked regular file: {path}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        )
        if before_identity == after_identity and len(raw) == after.st_size:
            return raw
        last_identity = after_identity
    raise BoundaryBlocked(
        f"authenticated local resource moved concurrently: {path} ({last_identity})"
    )


def _read_canonical_json_bytes(
    path: Path,
    *,
    label: str,
    expected_byte_length: int | None,
    expected_sha256: str | None,
    terminal_lf: bool,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    if expected_byte_length is None or expected_byte_length < 0:
        raise ValueError(f"{label} requires an independently supplied byte length")
    if expected_sha256 is None or not SHA256_RE.fullmatch(expected_sha256):
        raise ValueError(f"{label} requires an independently supplied SHA-256")
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symlink: {path}")
    raw = _stable_file_bytes(path)
    if len(raw) != expected_byte_length:
        raise ValueError(
            f"{label} byte-length mismatch: expected {expected_byte_length}, got {len(raw)}"
        )
    digest = _sha256_bytes(raw)
    if digest != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{label} has a UTF-8 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not exact UTF-8") from exc
    if terminal_lf:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\n" in raw[:-1]:
            raise ValueError(f"{label} must have exactly one terminal LF")
    elif b"\n" in raw:
        raise ValueError(f"{label} must not contain a terminal LF")
    try:
        parsed = json.loads(text, object_pairs_hook=_duplicate_key_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid duplicate-free JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    if raw != _canonical_json_bytes(parsed, terminal_lf=terminal_lf):
        raise ValueError(
            f"{label} bytes are not canonical compact sorted-key JSON; "
            "parse-reserialize equivalence is not proof"
        )
    return (
        parsed,
        {
            "byte_length": len(raw),
            "canonical_bytes_valid": True,
            "canonical_serialization": CANONICAL_SERIALIZATION,
            "path": path.name,
            "sha256": digest,
        },
        raw,
    )


def _load_canonical_json_bytes(
    path: Path,
    *,
    label: str,
    expected_byte_length: int | None,
    expected_sha256: str | None,
    terminal_lf: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed, descriptor, _raw = _read_canonical_json_bytes(
        path,
        label=label,
        expected_byte_length=expected_byte_length,
        expected_sha256=expected_sha256,
        terminal_lf=terminal_lf,
    )
    return parsed, descriptor


def _parse_canonical_json_raw(
    raw: bytes,
    *,
    label: str,
    terminal_lf: bool,
) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{label} has a UTF-8 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not exact UTF-8") from exc
    if terminal_lf:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\n" in raw[:-1]:
            raise ValueError(f"{label} must have exactly one terminal LF")
    elif b"\n" in raw:
        raise ValueError(f"{label} must not contain a terminal LF")
    try:
        parsed = json.loads(text, object_pairs_hook=_duplicate_key_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid duplicate-free JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    if raw != _canonical_json_bytes(parsed, terminal_lf=terminal_lf):
        raise ValueError(f"{label} is not canonical compact sorted-key JSON")
    return parsed


def _append_operation(
    operation_log: list[dict[str, Any]],
    *,
    kind: str,
    resource: str,
    identifier: str,
    raw: bytes,
    response_identity: dict[str, Any] | None = None,
    movement: bool = False,
) -> None:
    operation_log.append(
        {
            "authentication": "pass",
            "byte_length": len(raw),
            "identifier": identifier,
            "kind": kind,
            "movement_observed": movement,
            "resource": resource,
            "response_identity": response_identity or {},
            "sequence": len(operation_log) + 1,
            "sha256": _sha256_bytes(raw),
        }
    )


def _git_subcommand(argv: list[str]) -> str:
    index = 1
    while index < len(argv):
        item = argv[index]
        if item == "-C":
            index += 2
            continue
        if item.startswith("--git-dir=") or item.startswith("--work-tree="):
            index += 1
            continue
        if item.startswith("-"):
            index += 1
            continue
        return item
    return ""


def _guard_http_method(method: str) -> None:
    normalized = method.upper()
    if normalized not in {"GET", "HEAD"}:
        raise ValueError(f"mutating HTTP verb rejected: {normalized}")


def _guard_subprocess_argv(argv: list[str]) -> None:
    if not argv:
        raise ValueError("unsupported empty subprocess action")
    executable = Path(argv[0]).name
    if executable == "git":
        subcommand = _git_subcommand(argv)
        if subcommand not in _READ_ONLY_GIT_SUBCOMMANDS:
            raise ValueError(f"mutating or unsupported git action rejected: {subcommand}")
        return
    if executable == "gh":
        if len(argv) < 2:
            raise ValueError("unsupported gh action")
        group = argv[1]
        if group == "api":
            method = "GET"
            api_args = argv[2:]
            for index, item in enumerate(api_args):
                if item in {"-f", "-F", "--field", "--raw-field", "--input"} or item.startswith(
                    ("-f=", "-F=", "--field=", "--raw-field=", "--input=")
                ):
                    raise ValueError("mutating gh api field/input action rejected")
                if item in {"--method", "-X"}:
                    if index + 1 >= len(api_args):
                        raise ValueError("unsupported gh api method action")
                    method = api_args[index + 1]
                elif item.startswith("--method="):
                    method = item.split("=", 1)[1]
                elif item.startswith("-X") and item != "-X":
                    method = item[2:].removeprefix("=")
            _guard_http_method(method)
            return
        allowed = {
            ("attestation", "verify"),
            ("pr", "checks"),
            ("pr", "list"),
            ("pr", "view"),
            ("release", "verify"),
            ("release", "verify-asset"),
            ("release", "view"),
            ("run", "list"),
            ("run", "view"),
        }
        action = (group, argv[2] if len(argv) > 2 else "")
        if action not in allowed:
            raise ValueError(f"mutating or unsupported gh action rejected: {' '.join(action)}")
        return
    raise ValueError(f"unsupported subprocess action rejected: {executable}")


def _operation_argv(argv: list[str]) -> list[str]:
    normalized = list(argv)
    if normalized and Path(normalized[0]).name == "git":
        index = 1
        while index < len(normalized):
            if normalized[index] == "-C" and index + 1 < len(normalized):
                normalized[index + 1] = "<repository>"
                index += 2
                continue
            index += 1
    return normalized


def _run_read_only(
    argv: list[str],
    *,
    operation_log: list[dict[str, Any]],
    resource: str,
    check: bool = True,
    log_operation: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    _guard_subprocess_argv(argv)
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    executable = Path(argv[0]).name
    try:
        if executable == "git":
            proc = subprocess.run(["git", *argv[1:]], capture_output=True, env=env, check=False)
        elif executable == "gh":
            proc = subprocess.run(["gh", *argv[1:]], capture_output=True, env=env, check=False)
        else:  # guarded above; retain a fail-closed branch for static analysis
            raise ValueError(f"unsupported subprocess action rejected: {executable}")
    except OSError as exc:
        raise ValueError(f"read-only subprocess could not execute: {executable}: {exc}") from exc
    raw = proc.stdout if proc.returncode == 0 else proc.stderr
    if log_operation:
        _append_operation(
            operation_log,
            kind="subprocess",
            resource=resource,
            identifier=" ".join(_operation_argv(argv)),
            raw=raw,
            response_identity={"returncode": proc.returncode},
        )
    if check and proc.returncode != 0:
        raise ValueError(
            f"read-only subprocess failed ({proc.returncode}): {' '.join(argv)}: "
            f"{proc.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return proc


def _resolve_full_sha(
    repo_root: Path,
    ref: str,
    *,
    label: str,
    operation_log: list[dict[str, Any]],
) -> str:
    if not FULL_SHA_RE.fullmatch(ref):
        raise ValueError(f"{label} must be a full lowercase 40-hex commit SHA")
    proc = _run_read_only(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        operation_log=operation_log,
        resource=label,
    )
    resolved = proc.stdout.decode("ascii").strip()
    if resolved != ref:
        raise ValueError(f"{label} did not resolve to the exact supplied SHA")
    return resolved


def _is_ancestor(
    repo_root: Path,
    ancestor: str,
    descendant: str,
    operation_log: list[dict[str, Any]],
) -> bool:
    proc = _run_read_only(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", ancestor, descendant],
        operation_log=operation_log,
        resource="git-ancestry",
        check=False,
    )
    if proc.returncode not in {0, 1}:
        raise ValueError("git merge-base ancestry probe failed")
    return proc.returncode == 0


def _path_manifest(
    path: Path,
    *,
    content: bool,
    exclude_top_level: frozenset[str] = frozenset(),
) -> bytes:
    if not path.exists():
        return _canonical_json_bytes([])
    if path.is_file():
        raw = path.read_bytes()
        return _canonical_json_bytes(
            [{"path": path.name, "size": len(raw), "sha256": _sha256_bytes(raw)}]
        )
    entries: list[dict[str, Any]] = []
    for child in sorted(path.rglob("*")):
        relative = child.relative_to(path).as_posix()
        if child.relative_to(path).parts[0] in exclude_top_level:
            continue
        if child.is_symlink():
            entries.append({"kind": "symlink", "path": relative, "target": os.readlink(child)})
        elif child.is_file():
            stat_result = child.stat()
            entry: dict[str, Any] = {
                "kind": "file",
                "path": relative,
                "size": stat_result.st_size,
            }
            if content:
                entry["sha256"] = _sha256_bytes(child.read_bytes())
            entries.append(entry)
        elif child.is_dir():
            entries.append({"kind": "directory", "path": relative})
    return _canonical_json_bytes(entries)


def _snapshot_repository(
    repo_root: Path,
    operation_log: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    def git_output(args: list[str], resource: str) -> bytes:
        return _run_read_only(
            ["git", "-C", str(repo_root), *args],
            operation_log=operation_log,
            resource=resource,
        ).stdout

    status = git_output(
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        "snapshot-worktree-status",
    )
    tracked = git_output(["ls-files", "-s", "-z"], "snapshot-index-entries")
    unstaged = git_output(
        ["diff", "--no-ext-diff", "--binary", "--"],
        "snapshot-unstaged-diff",
    )
    staged = git_output(
        ["diff", "--cached", "--no-ext-diff", "--binary", "--"],
        "snapshot-staged-diff",
    )
    untracked_names = git_output(
        ["ls-files", "--others", "--exclude-standard", "-z"],
        "snapshot-untracked-files",
    )
    untracked: list[dict[str, Any]] = []
    for raw_name in untracked_names.split(b"\0"):
        if not raw_name:
            continue
        relative = raw_name.decode("utf-8")
        candidate = repo_root / relative
        if candidate.is_symlink():
            untracked.append(
                {"path": relative, "sha256": _sha256_bytes(os.readlink(candidate).encode())}
            )
        elif candidate.is_file():
            untracked.append({"path": relative, "sha256": _sha256_bytes(candidate.read_bytes())})
    worktree_raw = b"\0".join(
        (
            status,
            tracked,
            unstaged,
            staged,
            _canonical_json_bytes(untracked),
        )
    )
    git_dir_raw = git_output(["rev-parse", "--path-format=absolute", "--git-dir"], "git-dir")
    common_dir_raw = git_output(
        ["rev-parse", "--path-format=absolute", "--git-common-dir"],
        "git-common-dir",
    )
    git_dir = Path(git_dir_raw.decode().strip())
    common_dir = Path(common_dir_raw.decode().strip())
    index_path = Path(
        git_output(
            ["rev-parse", "--path-format=absolute", "--git-path", "index"],
            "git-index-path",
        )
        .decode()
        .strip()
    )
    refs_raw = git_output(
        ["for-each-ref", "--format=%(objectname)%00%(refname)%00"],
        "snapshot-refs",
    )
    packed_refs = common_dir / "packed-refs"
    if packed_refs.exists():
        refs_raw += packed_refs.read_bytes()
    objects_raw = git_output(
        [
            "cat-file",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
            "--batch-all-objects",
        ],
        "snapshot-object-database",
    )
    components = {
        "common_git_dir": _path_manifest(
            common_dir,
            content=True,
            exclude_top_level=frozenset({"logs", "objects", "refs", "worktrees"}),
        ),
        "index": _path_manifest(index_path, content=True),
        "object_database": objects_raw,
        "refs": refs_raw,
        "reflogs": _path_manifest(common_dir / "logs", content=True),
        "worktree": worktree_raw,
        "worktree_git_dir": _path_manifest(git_dir, content=True),
    }
    snapshot = {
        name: {"byte_length": len(raw), "sha256": _sha256_bytes(raw)}
        for name, raw in components.items()
    }
    _append_operation(
        operation_log,
        kind="local_snapshot",
        resource="repository",
        identifier=".",
        raw=_canonical_json_bytes(snapshot),
    )
    return snapshot


def _guard_write_path(path: Path, scratch_root: Path, output_root: Path) -> None:
    resolved = path.resolve()
    allowed = (scratch_root.resolve(), output_root.resolve())
    if not any(resolved == root or root in resolved.parents for root in allowed):
        raise ValueError(f"write outside explicit scratch/output roots rejected: {path}")


def _remote_identity_moved(
    before: dict[str, Any],
    after: dict[str, Any],
) -> bool:
    keys = ("etag", "updated_at", "sha256", "byte_length")
    return any(before.get(key) != after.get(key) for key in keys)


def _retry_stable_remote_probe(
    probe: Callable[[], tuple[dict[str, Any], dict[str, Any]]],
    *,
    attempts: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for _attempt in range(attempts):
        before, after = probe()
        if not _remote_identity_moved(before, after):
            return before, after
    raise BoundaryBlocked("authenticated remote resource moved concurrently")


def _discover_canonical_artifact(
    explicit_path: Path | None,
    descriptor: dict[str, Any],
    repo_root: Path,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    mission_dir = os.environ.get("FACTORY_MISSION_DIR")
    if mission_dir:
        candidates.append(Path(mission_dir) / descriptor["logical_path"])
    runtime_settings = os.environ.get("FACTORY_RUNTIME_SETTINGS_PATH")
    if runtime_settings:
        candidates.append(
            Path(runtime_settings).resolve().parent / "library" / descriptor["filename"]
        )
    candidates.extend(
        (
            repo_root / descriptor["logical_path"],
            repo_root.parent / "library" / descriptor["filename"],
        )
    )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    raise ValueError(f"canonical mission artifact unavailable: {descriptor['logical_path']}")


def _validate_original_cohort(cohort: dict[str, Any]) -> dict[str, Any]:
    if cohort.get("schema") != "contract-drift-original-cohort-v1":
        raise ValueError("canonical cohort schema mismatch")
    records = cohort.get("original_records")
    if not isinstance(records, list) or len(records) != 655:
        raise ValueError("canonical cohort must contain exactly 655 original records")
    category_counts: dict[str, int] = defaultdict(int)
    original_ids: list[str] = []
    sdk_records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"canonical cohort record {index} is malformed")
        category = record.get("category")
        literal = record.get("exact_historical_literal_record")
        if not isinstance(category, str) or not isinstance(literal, str):
            raise ValueError(f"canonical cohort record {index} lacks identity fields")
        payload = {
            "category": category,
            "exact_historical_literal_record": literal,
            "schema": "cdg-original-record-id-v1",
        }
        payload_bytes = _canonical_json_bytes(payload)
        payload_sha256 = _sha256_bytes(payload_bytes)
        original_id = f"cdg1:{payload_sha256}"
        if record.get("id_payload_byte_length") != len(payload_bytes):
            raise ValueError(f"canonical cohort record {index} ID payload length mismatch")
        if record.get("id_payload_sha256") != payload_sha256:
            raise ValueError(f"canonical cohort record {index} ID payload digest mismatch")
        if record.get("original_record_id") != original_id:
            raise ValueError(f"canonical cohort record {index} ID mismatch")
        category_counts[category] += 1
        original_ids.append(original_id)
        if category in {"python_sdk_drift", "typescript_sdk_drift"}:
            method = record.get("method")
            if not isinstance(method, str) or not method:
                raise ValueError(f"SDK cohort record {index} lacks a method")
            sdk_records[original_id] = record
        elif record.get("method") is not None:
            raise ValueError(f"path-level cohort record {index} carries a method")
    if len(set(original_ids)) != 655:
        raise ValueError("canonical cohort contains duplicate original record IDs")
    id_set = cohort.get("original_record_id_set")
    if not isinstance(id_set, dict):
        raise ValueError("canonical cohort lacks the original-record ID set")
    if id_set.get("original_record_ids") != sorted(original_ids):
        raise ValueError("canonical cohort original-record ID set is incomplete or unsorted")
    id_set_digest = _digest_set(
        "cdg-original-record-id-set-v1",
        original_ids,
        "original_record_ids",
    )
    if id_set.get("sha256") != id_set_digest or id_set_digest != ORIGINAL_ID_SET_SHA256:
        raise ValueError("canonical cohort original-record ID-set digest mismatch")
    if dict(sorted(category_counts.items())) != EXPECTED_CATEGORY_COUNTS:
        raise ValueError("canonical cohort category counts mismatch")
    counts = cohort.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("canonical cohort lacks counts")
    if counts.get("records") != 655 or counts.get("by_category") != EXPECTED_CATEGORY_COUNTS:
        raise ValueError("canonical cohort declared counts mismatch")
    if (
        counts.get("method_bearing_sdk_records") != 598
        or counts.get("method_null_route_parity_records") != 57
    ):
        raise ValueError("canonical cohort method-plane counts mismatch")

    projection = cohort.get("operation_projection")
    if not isinstance(projection, dict):
        raise ValueError("canonical cohort lacks operation projection")
    projection_records = projection.get("records")
    if not isinstance(projection_records, list) or len(projection_records) != 655:
        raise ValueError("operation projection must contain 655 membership records")
    projection_ids: list[str] = []
    projection_digests: list[str] = []
    edge_count = 0
    multi_edge = 0
    max_edges = 0
    edge_distribution: dict[str, int] = defaultdict(int)
    valid_methods = {
        "CONNECT",
        "DELETE",
        "GET",
        "HEAD",
        "OPTIONS",
        "PATCH",
        "POST",
        "PUT",
        "TRACE",
    }
    for index, record in enumerate(projection_records):
        if not isinstance(record, dict):
            raise ValueError(f"operation projection record {index} is malformed")
        projection_original_id = record.get("original_record_id")
        if not isinstance(projection_original_id, str):
            raise ValueError(f"operation projection record {index} lacks an original ID")
        projection_ids.append(projection_original_id)
        digest = _sha256_bytes(
            _canonical_json_bytes(
                {key: value for key, value in record.items() if key != "record_sha256"}
            )
        )
        if record.get("record_sha256") != digest:
            raise ValueError(f"operation projection record {index} digest mismatch")
        projection_digests.append(digest)
        edges = record.get("operation_edges")
        if not isinstance(edges, list) or not edges:
            raise ValueError(f"operation projection record {index} has no edges")
        seen: set[tuple[str, str]] = set()
        for edge in edges:
            if not isinstance(edge, dict):
                raise ValueError(f"operation projection record {index} has a malformed edge")
            method = edge.get("method")
            path = edge.get("normalized_path")
            if method not in valid_methods:
                raise ValueError(f"operation projection record {index} has an invalid method")
            if not isinstance(path, str) or not path.startswith("/"):
                raise ValueError(f"operation projection record {index} has an invalid path")
            if not isinstance(edge.get("evidence"), list) or not edge["evidence"]:
                raise ValueError(f"operation projection record {index} lacks evidence")
            edge_key = (method, path)
            if edge_key in seen:
                raise ValueError(f"operation projection record {index} has duplicate edges")
            seen.add(edge_key)
        size = len(edges)
        edge_count += size
        multi_edge += int(size > 1)
        max_edges = max(max_edges, size)
        edge_distribution[str(size)] += 1
    if sorted(projection_ids) != sorted(original_ids) or len(set(projection_ids)) != 655:
        raise ValueError("operation projection does not biject with the original cohort")
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
    if (edge_count, multi_edge, max_edges) != (666, 9, 4):
        raise ValueError("operation projection cardinality mismatch")
    return {
        "category_counts": EXPECTED_CATEGORY_COUNTS,
        "category_counts_sha256": _sha256_bytes(
            _canonical_json_bytes(
                {
                    "counts": EXPECTED_CATEGORY_COUNTS,
                    "schema": "cdg-original-category-counts-v1",
                }
            )
        ),
        "id_encoding": cohort.get("id_encoding"),
        "membership_anchor": cohort.get("membership_anchor"),
        "membership_sources": cohort.get("membership_sources"),
        "original_record_id_set_sha256": id_set_digest,
        "original_record_ids": sorted(original_ids),
        "record_count": 655,
        "sdk_records": sdk_records,
        "operation_projection": {
            "edge_count": edge_count,
            "edge_count_distribution": dict(sorted(edge_distribution.items())),
            "max_edges": max_edges,
            "membership_count": len(projection_records),
            "multi_edge_originals": multi_edge,
            "one_to_many_rule": projection.get("one_to_many_rule"),
            "record_digest_set_sha256": projection_digest,
            "schema": projection.get("schema"),
            "witness_dependencies": projection.get("witness_dependencies"),
        },
    }


def _partition_from_atoms(atoms: list[str]) -> tuple[str, list[dict[str, str]]]:
    if not atoms or not all(isinstance(atom, str) and atom for atom in atoms):
        raise ValueError("SDK provenance atoms must be a nonempty string array")
    matches: list[dict[str, str]] = []
    for atom in atoms:
        normalized = atom.replace("-", "_")
        if normalized in CORE_DOMAINS:
            matches.append(
                {
                    "atom": atom,
                    "domain": normalized,
                    "match_rule": "exact",
                    "normalized_atom": normalized,
                }
            )
        elif normalized.endswith("s") and normalized[:-1] in CORE_DOMAINS:
            matches.append(
                {
                    "atom": atom,
                    "domain": normalized[:-1],
                    "match_rule": "remove_exactly_one_trailing_s",
                    "normalized_atom": normalized,
                }
            )
    return ("core" if matches else "extended"), matches


def _validate_sdk_provenance(
    provenance: dict[str, Any],
    cohort_summary: dict[str, Any],
) -> dict[str, Any]:
    if provenance.get("schema") != "contract-drift-sdk-provenance-v1":
        raise ValueError("canonical SDK provenance schema mismatch")
    records = provenance.get("records")
    if not isinstance(records, list) or len(records) != 598:
        raise ValueError("canonical SDK provenance must contain 598 records")
    cohort_sdk = cohort_summary["sdk_records"]
    record_ids: list[str] = []
    record_digests: list[str] = []
    source_occurrences = 0
    multi_atom_records = 0
    partitions: dict[str, list[str]] = {"core": [], "extended": []}
    record_links: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"SDK provenance record {index} is malformed")
        original_id = record.get("original_record_id")
        if not isinstance(original_id, str) or original_id not in cohort_sdk:
            raise ValueError(f"SDK provenance record {index} lacks a cohort link")
        digest = _sha256_bytes(
            _canonical_json_bytes(
                {key: value for key, value in record.items() if key != "record_sha256"}
            )
        )
        if record.get("record_sha256") != digest:
            raise ValueError(f"SDK provenance record {index} digest mismatch")
        cohort_record = cohort_sdk[original_id]
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
            raise ValueError(f"SDK provenance record {index} language link mismatch")
        if cohort_record.get("sdk_provenance_record_sha256") != digest:
            raise ValueError(f"SDK provenance record {index} cohort digest link mismatch")
        atoms = record.get("provenance_atoms")
        if not isinstance(atoms, list):
            raise ValueError(f"SDK provenance record {index} lacks provenance atoms")
        reconstructed_partition, reconstructed_matches = _partition_from_atoms(atoms)
        if record.get("partition") != reconstructed_partition:
            raise ValueError(f"SDK provenance record {index} partition mismatch")
        if record.get("matched_domains") != reconstructed_matches:
            raise ValueError(f"SDK provenance record {index} matched-domain proof mismatch")
        occurrences = record.get("source_occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            raise ValueError(f"SDK provenance record {index} lacks source occurrences")
        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                raise ValueError(f"SDK provenance record {index} has malformed occurrence")
            if occurrence.get("provenance_atom") not in atoms:
                raise ValueError(f"SDK provenance record {index} occurrence atom is not declared")
            if occurrence.get("sdk_language") != record.get("sdk_language"):
                raise ValueError(f"SDK provenance record {index} occurrence language mismatch")
        record_ids.append(original_id)
        record_digests.append(digest)
        partitions[reconstructed_partition].append(original_id)
        source_occurrences += len(occurrences)
        multi_atom_records += int(len(atoms) > 1)
        record_links.append(
            {
                "original_record_id": original_id,
                "partition": reconstructed_partition,
                "record_sha256": digest,
                "source_occurrence_count": len(occurrences),
            }
        )
    if sorted(record_ids) != sorted(cohort_sdk) or len(set(record_ids)) != 598:
        raise ValueError("SDK provenance does not biject with cohort SDK records")
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
    core_ids = sorted(partitions["core"])
    extended_ids = sorted(partitions["extended"])
    if set(core_ids) & set(extended_ids) or sorted(core_ids + extended_ids) != sorted(record_ids):
        raise ValueError("SDK provenance partition is not disjoint and exhaustive")
    digest_expectations = {
        "sdk_original_record_id_set_sha256": (
            _digest_set(
                "cdg-sdk-original-record-id-set-v1",
                record_ids,
                "original_record_ids",
            ),
            SDK_ID_SET_SHA256,
        ),
        "core_original_record_id_set_sha256": (
            _digest_set(
                "cdg-core-original-record-id-set-v1",
                core_ids,
                "original_record_ids",
            ),
            CORE_ID_SET_SHA256,
        ),
        "extended_original_record_id_set_sha256": (
            _digest_set(
                "cdg-extended-original-record-id-set-v1",
                extended_ids,
                "original_record_ids",
            ),
            EXTENDED_ID_SET_SHA256,
        ),
    }
    partition = provenance.get("partition")
    if not isinstance(partition, dict):
        raise ValueError("SDK provenance lacks partition descriptors")
    for field, (reconstructed, ratified) in digest_expectations.items():
        if partition.get(field) != reconstructed or reconstructed != ratified:
            raise ValueError(f"SDK provenance partition digest mismatch: {field}")
    counts = provenance.get("counts")
    expected_counts = {
        "core": 75,
        "extended": 523,
        "python_sdk_drift": 74,
        "records": 598,
        "records_with_multiple_distinct_atoms": 12,
        "source_occurrences": 690,
        "typescript_sdk_drift": 524,
    }
    if counts != expected_counts:
        raise ValueError("SDK provenance declared counts mismatch")
    if (
        source_occurrences != 690
        or multi_atom_records != 12
        or len(core_ids) != 75
        or len(extended_ids) != 523
    ):
        raise ValueError("SDK provenance reconstructed counts mismatch")
    return {
        "baseline_birth": provenance.get("baseline_birth"),
        "core_count": len(core_ids),
        "core_original_record_id_set_sha256": CORE_ID_SET_SHA256,
        "core_original_record_ids": core_ids,
        "dependencies": provenance.get("dependencies"),
        "extended_count": len(extended_ids),
        "extended_original_record_id_set_sha256": EXTENDED_ID_SET_SHA256,
        "extended_original_record_ids": extended_ids,
        "extraction_algorithm": provenance.get("extraction_algorithm"),
        "missing_provenance_count": 0,
        "multiple_atom_record_count": multi_atom_records,
        "record_count": 598,
        "record_digest_set_sha256": record_set_digest,
        "record_links": sorted(record_links, key=lambda item: item["original_record_id"]),
        "sdk_original_record_id_set_sha256": SDK_ID_SET_SHA256,
        "sdk_original_record_ids": sorted(record_ids),
        "source_occurrence_count": source_occurrences,
    }


def _authenticate_canonical_artifacts(
    *,
    repo_root: Path,
    cohort_artifact_path: Path | None,
    sdk_provenance_artifact_path: Path | None,
    scratch_root: Path,
    operation_log: list[dict[str, Any]],
) -> dict[str, Any]:
    cohort_path = _discover_canonical_artifact(
        cohort_artifact_path,
        COHORT_ARTIFACT,
        repo_root,
    )
    provenance_path = _discover_canonical_artifact(
        sdk_provenance_artifact_path,
        PROVENANCE_ARTIFACT,
        repo_root,
    )
    cohort, cohort_descriptor, cohort_raw = _read_canonical_json_bytes(
        cohort_path,
        label="canonical original-cohort artifact",
        expected_byte_length=COHORT_ARTIFACT["byte_length"],
        expected_sha256=COHORT_ARTIFACT["sha256"],
        terminal_lf=True,
    )
    _append_operation(
        operation_log,
        kind="canonical_artifact",
        resource="original_cohort",
        identifier=COHORT_ARTIFACT["logical_path"],
        raw=cohort_raw,
    )
    provenance, provenance_descriptor, provenance_raw = _read_canonical_json_bytes(
        provenance_path,
        label="canonical SDK-provenance artifact",
        expected_byte_length=PROVENANCE_ARTIFACT["byte_length"],
        expected_sha256=PROVENANCE_ARTIFACT["sha256"],
        terminal_lf=True,
    )
    _append_operation(
        operation_log,
        kind="canonical_artifact",
        resource="sdk_provenance",
        identifier=PROVENANCE_ARTIFACT["logical_path"],
        raw=provenance_raw,
    )
    cohort_summary = _validate_original_cohort(cohort)
    provenance_summary = _validate_sdk_provenance(provenance, cohort_summary)
    return {
        "operation_projection": cohort_summary["operation_projection"],
        "original_cohort": {
            **cohort_descriptor,
            "category_counts": cohort_summary["category_counts"],
            "category_counts_sha256": cohort_summary["category_counts_sha256"],
            "id_encoding": cohort_summary["id_encoding"],
            "logical_path": COHORT_ARTIFACT["logical_path"],
            "membership_anchor": cohort_summary["membership_anchor"],
            "membership_sources": cohort_summary["membership_sources"],
            "original_record_id_set_sha256": cohort_summary["original_record_id_set_sha256"],
            "original_record_ids": cohort_summary["original_record_ids"],
            "record_count": cohort_summary["record_count"],
        },
        "sdk_provenance": {
            **provenance_descriptor,
            **provenance_summary,
            "logical_path": PROVENANCE_ARTIFACT["logical_path"],
        },
    }


def _git_blob_binding(
    repo_root: Path,
    end_sha: str,
    item: dict[str, Any],
    operation_log: list[dict[str, Any]],
) -> None:
    path = item.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("authority manifest repository file lacks a path")
    proc = _run_read_only(
        ["git", "-C", str(repo_root), "show", f"{end_sha}:{path}"],
        operation_log=operation_log,
        resource=f"authority-repo-file:{path}",
    )
    raw = proc.stdout
    if item.get("byte_length") != len(raw) or item.get("sha256") != _sha256_bytes(raw):
        raise ValueError(f"authority manifest repository binding mismatch: {path}")
    oid = (
        _run_read_only(
            ["git", "-C", str(repo_root), "rev-parse", f"{end_sha}:{path}"],
            operation_log=operation_log,
            resource=f"authority-blob-oid:{path}",
        )
        .stdout.decode("ascii")
        .strip()
    )
    if item.get("git_blob_oid") != oid:
        raise ValueError(f"authority manifest blob OID mismatch: {path}")


def _authority_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    repo_files = manifest["repo_files"]
    inventory = manifest["inventory"]
    public_members = [
        item for item in repo_files if "public" in item["path"] and "symbol" in item["path"]
    ]
    route_members = [
        item for item in repo_files if "route" in item["path"] or "openapi" in item["path"]
    ]
    verifier = next(
        (item for item in repo_files if item["path"] == "scripts/check_contract_drift_ratchet.py"),
        None,
    )
    if verifier is None:
        raise ValueError("authority manifest omits the boundary verifier")
    return {
        "authority_manifest_sha256": manifest["authority_manifest_sha256"],
        "authority_roots_sha256": _sha256_bytes(_canonical_json_bytes(manifest["authority_roots"])),
        "dependency_manifest_sha256": _sha256_bytes(_canonical_json_bytes(repo_files)),
        "boundary_verifier_sha256": verifier["sha256"],
        "inventory_sha256": _sha256_bytes(_canonical_json_bytes(inventory)),
        "public_symbol_sha256": _sha256_bytes(_canonical_json_bytes(public_members)),
        "repo_file_count": len(repo_files),
        "route_authority_member_count": len(route_members),
        "route_boundary_sha256": _sha256_bytes(_canonical_json_bytes(route_members)),
    }


def _validate_authority_manifest(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
    end_sha: str,
    operation_log: list[dict[str, Any]],
) -> dict[str, Any]:
    if manifest.get("schema") != inventory_mod.AUTHORITY_MANIFEST_SCHEMA:
        raise ValueError("authority manifest schema mismatch")
    if manifest.get("ref") != end_sha:
        raise ValueError("authority manifest exact-ref mismatch")
    self_digest = manifest.get("authority_manifest_sha256")
    if not isinstance(self_digest, str):
        raise ValueError("authority manifest lacks its semantic digest")
    payload = {key: value for key, value in manifest.items() if key != "authority_manifest_sha256"}
    reconstructed = _sha256_bytes(_canonical_json_bytes(payload, terminal_lf=True))
    if self_digest != reconstructed:
        raise ValueError("authority manifest semantic digest mismatch")
    roots = manifest.get("authority_roots")
    repo_files = manifest.get("repo_files")
    inventory = manifest.get("inventory")
    if not isinstance(roots, list) or not roots or len(roots) != len(set(roots)):
        raise ValueError("authority manifest roots are empty or duplicated")
    if not isinstance(repo_files, list) or not repo_files:
        raise ValueError("authority manifest repository closure is empty")
    if not isinstance(inventory, dict):
        raise ValueError("authority manifest inventory is malformed")
    paths = [item.get("path") for item in repo_files if isinstance(item, dict)]
    if (
        len(paths) != len(repo_files)
        or not all(isinstance(path, str) for path in paths)
        or paths != sorted(cast(list[str], paths))
        or len(paths) != len(set(paths))
    ):
        raise ValueError("authority manifest repository closure is not sorted and unique")
    root_set = set(roots)
    for item in repo_files:
        if not isinstance(item, dict):
            raise ValueError("authority manifest repository closure member is malformed")
        if item.get("authority_root") != (item.get("path") in root_set):
            raise ValueError("authority manifest authority-root membership mismatch")
        incoming = item.get("incoming_edges")
        if not item.get("authority_root") and (not isinstance(incoming, list) or not incoming):
            raise ValueError("authority manifest closure member lacks an incoming edge")
        if item.get("tier") != 4:
            raise ValueError("authority manifest contains a below-Tier-4 closure member")
        if item.get("matched_rule") != item.get("merge_train_matched_rule"):
            raise ValueError("authority manifest classifier/mirror disagreement")
        _git_blob_binding(repo_root, end_sha, item, operation_log)
    return _authority_summary(manifest)


def _authenticate_authority_manifest(
    *,
    repo_root: Path,
    end_sha: str,
    authority_manifest_path: Path | None,
    authority_manifest_byte_length: int | None,
    authority_manifest_sha256: str | None,
    cohort_artifact_path: Path | None,
    sdk_provenance_artifact_path: Path | None,
    scratch_root: Path,
    operation_log: list[dict[str, Any]],
) -> dict[str, Any]:
    external_artifacts = tuple(
        path for path in (cohort_artifact_path, sdk_provenance_artifact_path) if path is not None
    )
    regenerated = inventory_mod.build_authority_manifest(
        repo_root,
        end_sha,
        scratch_root=scratch_root,
        external_artifacts=external_artifacts,
    )
    regenerated_raw = _canonical_json_bytes(regenerated, terminal_lf=True)
    _append_operation(
        operation_log,
        kind="authority_reconstruction",
        resource="authority_manifest",
        identifier=end_sha,
        raw=regenerated_raw,
    )
    if authority_manifest_path is None:
        manifest = regenerated
        manifest_raw = regenerated_raw
    else:
        manifest, _descriptor, manifest_raw = _read_canonical_json_bytes(
            authority_manifest_path,
            label="external authority manifest",
            expected_byte_length=authority_manifest_byte_length,
            expected_sha256=authority_manifest_sha256,
            terminal_lf=True,
        )
        if manifest_raw != regenerated_raw:
            raise ValueError(
                "external authority manifest does not equal the independently "
                "reconstructed exact-ref manifest"
            )
    _append_operation(
        operation_log,
        kind="external_authority_manifest",
        resource="authority_manifest",
        identifier=authority_manifest_path.name if authority_manifest_path else end_sha,
        raw=manifest_raw,
    )
    summary = _validate_authority_manifest(
        manifest,
        repo_root=repo_root,
        end_sha=end_sha,
        operation_log=operation_log,
    )
    summary["authenticated_manifest_bytes"] = {
        "byte_length": len(manifest_raw),
        "sha256": _sha256_bytes(manifest_raw),
    }
    return summary


def _reject_caller_authority(value: Any, *, label: str) -> None:
    if isinstance(value, dict):
        forbidden = sorted(set(value) & _FORBIDDEN_CALLER_FIELDS)
        if forbidden:
            raise ValueError(
                f"{label} contains caller-supplied or inherited authority: " + ", ".join(forbidden)
            )
        for key, child in value.items():
            _reject_caller_authority(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_caller_authority(child, label=f"{label}[{index}]")


def _load_evidence_resources(
    *,
    evidence_index_path: Path,
    evidence_index_byte_length: int | None,
    evidence_index_sha256: str | None,
    boundary: str,
    start_sha: str,
    end_sha: str,
    operation_log: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    index, index_descriptor, index_raw = _read_canonical_json_bytes(
        evidence_index_path,
        label="external boundary evidence index",
        expected_byte_length=evidence_index_byte_length,
        expected_sha256=evidence_index_sha256,
        terminal_lf=True,
    )
    _append_operation(
        operation_log,
        kind="evidence_index",
        resource="boundary_evidence_index",
        identifier=evidence_index_path.name,
        raw=index_raw,
    )
    allowed_index_fields = {
        "boundary",
        "end_sha",
        "resources",
        "schema",
        "start_sha",
    }
    unknown = sorted(set(index) - allowed_index_fields)
    if unknown:
        raise ValueError(
            "boundary evidence index contains caller-supplied or unsupported fields: "
            + ", ".join(unknown)
        )
    if index.get("schema") != BOUNDARY_EVIDENCE_INDEX_SCHEMA:
        raise ValueError("boundary evidence index schema mismatch")
    if (
        index.get("boundary") != boundary
        or index.get("start_sha") != start_sha
        or index.get("end_sha") != end_sha
    ):
        raise ValueError("boundary evidence index interval mismatch")
    descriptors = index.get("resources")
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError("boundary evidence index resources are missing")
    names = [descriptor.get("name") for descriptor in descriptors if isinstance(descriptor, dict)]
    if (
        len(names) != len(descriptors)
        or not all(isinstance(name, str) for name in names)
        or names != sorted(cast(list[str], names))
        or len(names) != len(set(names))
    ):
        raise ValueError("boundary evidence index resources are not sorted and unique")
    resources: dict[str, dict[str, Any]] = {}
    resource_digests: set[str] = set()
    authenticated_resources: list[dict[str, Any]] = []
    root = evidence_index_path.resolve().parent
    for descriptor in descriptors:
        if set(descriptor) != {"byte_length", "name", "path", "sha256"}:
            raise ValueError("boundary evidence resource descriptor is noncanonical")
        name = descriptor["name"]
        relative = descriptor["path"]
        if name not in _RESOURCE_SCHEMAS:
            raise ValueError(f"unsupported boundary evidence resource: {name}")
        if not isinstance(relative, str):
            raise ValueError(f"boundary evidence resource {name} has an invalid path")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"boundary evidence resource {name} escapes the evidence root")
        resolved = (root / path).resolve()
        if root != resolved and root not in resolved.parents:
            raise ValueError(f"boundary evidence resource {name} escapes the evidence root")
        payload, authenticated, resource_raw = _read_canonical_json_bytes(
            resolved,
            label=f"boundary evidence resource {name}",
            expected_byte_length=descriptor["byte_length"],
            expected_sha256=descriptor["sha256"],
            terminal_lf=True,
        )
        _append_operation(
            operation_log,
            kind="external_resource",
            resource=name,
            identifier=relative,
            raw=resource_raw,
        )
        _reject_caller_authority(payload, label=f"boundary evidence resource {name}")
        if payload.get("schema") != _RESOURCE_SCHEMAS[name]:
            raise ValueError(f"boundary evidence resource {name} schema mismatch")
        if name in BOUNDARY_NAMES:
            if (
                payload.get("predicate") != name
                or payload.get("proof_for_boundary") != boundary
                or payload.get("proof_start_sha") != start_sha
                or payload.get("proof_end_sha") != end_sha
            ):
                raise ValueError(
                    f"boundary evidence predicate {name} interval or identity mismatch"
                )
        elif (
            payload.get("boundary") != boundary
            or payload.get("start_sha") != start_sha
            or payload.get("end_sha") != end_sha
        ):
            raise ValueError(f"boundary evidence resource {name} interval mismatch")
        digest = authenticated["sha256"]
        if digest in resource_digests:
            raise ValueError("boundary evidence resources reuse identical authenticated bytes")
        resource_digests.add(digest)
        authenticated_resources.append(
            {
                "byte_length": authenticated["byte_length"],
                "name": name,
                "path": relative,
                "sha256": digest,
            }
        )
        resources[name] = payload
    selected = BOUNDARY_NAMES[: BOUNDARY_NAMES.index(boundary) + 1]
    required = {
        "boundary_chronology",
        "durable_capsule",
        "external_prerequisites",
        "first_parent_receipts",
        "governed_prs",
        *selected,
    }
    missing = sorted(required - set(resources))
    if missing:
        raise ValueError(
            "boundary evidence index is incomplete; missing resources: " + ", ".join(missing)
        )
    return resources, {
        "index": index_descriptor,
        "resource_count": len(resources),
        "resources": authenticated_resources,
        "resource_sha256s": sorted(resource_digests),
        "source": "external_evidence_index",
    }


def _parse_http_response(raw: bytes) -> tuple[dict[str, str], bytes]:
    def split_header_block(payload: bytes) -> tuple[bytes, bytes]:
        candidates = [
            (index, separator)
            for separator in (b"\r\n\r\n", b"\n\n")
            if (index := payload.find(separator)) >= 0
        ]
        if not candidates:
            raise ValueError("GitHub response did not contain authenticated HTTP headers")
        index, separator = min(candidates, key=lambda item: item[0])
        return payload[:index], payload[index + len(separator) :]

    if not raw.startswith(b"HTTP/"):
        raise ValueError("GitHub response did not contain authenticated HTTP headers")
    header_block, body = split_header_block(raw)
    while True:
        header_lines = header_block.decode("utf-8", errors="strict").splitlines()
        if not header_lines or not header_lines[0].startswith("HTTP/"):
            raise ValueError("GitHub response did not contain authenticated HTTP headers")
        status_parts = header_lines[0].split()
        if (
            len(status_parts) >= 2
            and status_parts[1].isdigit()
            and 100 <= int(status_parts[1]) < 200
            and body.startswith(b"HTTP/")
        ):
            header_block, body = split_header_block(body)
            continue
        break
    headers: dict[str, str] = {}
    for line in header_lines[1:]:
        if ":" in line:
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
    return headers, body


def _gh_api_get(
    endpoint: str,
    *,
    operation_log: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    proc = _run_read_only(
        ["gh", "api", "--method", "GET", "-i", endpoint],
        operation_log=operation_log,
        resource=f"github:{endpoint}",
        check=False,
        log_operation=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"authenticated GitHub GET failed for {endpoint}: "
            f"{proc.stderr.decode('utf-8', errors='replace').strip()}"
        )
    headers, body = _parse_http_response(proc.stdout)
    try:
        payload = json.loads(body, object_pairs_hook=_duplicate_key_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"GitHub response for {endpoint} is malformed: {exc}") from exc
    identity = {
        "byte_length": len(body),
        "etag": headers.get("etag"),
        "sha256": _sha256_bytes(body),
        "updated_at": payload.get("updated_at") if isinstance(payload, dict) else None,
    }
    _append_operation(
        operation_log,
        kind="remote_resource",
        resource=endpoint,
        identifier=endpoint,
        raw=body,
        response_identity=identity,
    )
    return payload, identity


def _gh_api_get_stable(
    endpoint: str,
    *,
    operation_log: list[dict[str, Any]],
    attempts: int = 3,
) -> tuple[Any, dict[str, Any]]:
    for _attempt in range(attempts):
        before_payload, before_identity = _gh_api_get(
            endpoint,
            operation_log=operation_log,
        )
        after_payload, after_identity = _gh_api_get(
            endpoint,
            operation_log=operation_log,
        )
        moved = _remote_identity_moved(before_identity, after_identity)
        operation_log[-2]["movement_observed"] = moved
        operation_log[-1]["movement_observed"] = moved
        if not moved:
            if before_payload != after_payload:
                raise ValueError(
                    f"authenticated GitHub resource contradicted stable identity: {endpoint}"
                )
            return after_payload, after_identity
    raise BoundaryBlocked(f"authenticated GitHub resource moved concurrently: {endpoint}")


def _gh_api_get_raw(
    endpoint: str,
    *,
    operation_log: list[dict[str, Any]],
) -> tuple[bytes, dict[str, Any]]:
    proc = _run_read_only(
        [
            "gh",
            "api",
            "--method",
            "GET",
            "-i",
            "-H",
            "Accept: application/octet-stream",
            endpoint,
        ],
        operation_log=operation_log,
        resource=f"github-raw:{endpoint}",
        check=False,
        log_operation=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"authenticated GitHub asset GET failed for {endpoint}: "
            f"{proc.stderr.decode('utf-8', errors='replace').strip()}"
        )
    headers, body = _parse_http_response(proc.stdout)
    identity = {
        "byte_length": len(body),
        "etag": headers.get("etag"),
        "sha256": _sha256_bytes(body),
        "updated_at": headers.get("last-modified"),
    }
    _append_operation(
        operation_log,
        kind="remote_asset",
        resource=endpoint,
        identifier=endpoint,
        raw=body,
        response_identity=identity,
    )
    return body, identity


def _gh_api_get_raw_stable(
    endpoint: str,
    *,
    operation_log: list[dict[str, Any]],
    attempts: int = 3,
) -> tuple[bytes, dict[str, Any]]:
    for _attempt in range(attempts):
        before_body, before_identity = _gh_api_get_raw(
            endpoint,
            operation_log=operation_log,
        )
        after_body, after_identity = _gh_api_get_raw(
            endpoint,
            operation_log=operation_log,
        )
        moved = _remote_identity_moved(before_identity, after_identity)
        operation_log[-2]["movement_observed"] = moved
        operation_log[-1]["movement_observed"] = moved
        if not moved:
            if before_body != after_body:
                raise ValueError(
                    f"authenticated GitHub asset contradicted stable identity: {endpoint}"
                )
            return after_body, after_identity
    raise BoundaryBlocked(f"authenticated GitHub asset moved concurrently: {endpoint}")


def _gh_api_paginated(
    endpoint: str,
    *,
    operation_log: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    identities: dict[str, dict[str, Any]] = {}
    page = 1
    while True:
        separator = "&" if "?" in endpoint else "?"
        page_endpoint = f"{endpoint}{separator}per_page=100&page={page}"
        payload, identity = _gh_api_get_stable(
            page_endpoint,
            operation_log=operation_log,
        )
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ValueError(
                f"authenticated paginated GitHub response is malformed: {page_endpoint}"
            )
        identities[page_endpoint] = identity
        records.extend(payload)
        if len(payload) < 100:
            break
        page += 1
        if page > 10_000:
            raise ValueError("authenticated GitHub pagination did not terminate")
    record_ids = [item.get("id") for item in records]
    concrete_ids = [item for item in record_ids if item is not None]
    if len(concrete_ids) != len(set(concrete_ids)):
        raise ValueError("authenticated GitHub pagination returned duplicate record IDs")
    return records, identities


def _run_live_verification(
    argv: list[str],
    *,
    operation_log: list[dict[str, Any]],
    resource: str,
) -> tuple[Any, dict[str, Any]]:
    proc = _run_read_only(
        argv,
        operation_log=operation_log,
        resource=resource,
    )
    raw = proc.stdout
    try:
        payload = json.loads(raw, object_pairs_hook=_duplicate_key_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{resource} returned malformed JSON: {exc}") from exc
    identity = {
        "byte_length": len(raw),
        "sha256": _sha256_bytes(raw),
    }
    return payload, identity


def _collect_live_evidence(
    *,
    github_repository: str,
    github_branch: str,
    boundary: str,
    start_sha: str,
    end_sha: str,
    scratch_root: Path,
    operation_log: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    endpoint_identities: dict[str, dict[str, Any]] = {}
    repository, repository_identity = _gh_api_get_stable(
        f"repos/{github_repository}",
        operation_log=operation_log,
    )
    endpoint_identities[f"repos/{github_repository}"] = repository_identity
    if not isinstance(repository, dict) or repository.get("full_name") != github_repository:
        raise ValueError("authenticated GitHub repository identity mismatch")
    protection_endpoint = f"repos/{github_repository}/branches/{github_branch}/protection"
    protection, protection_identity = _gh_api_get_stable(
        protection_endpoint,
        operation_log=operation_log,
    )
    endpoint_identities[protection_endpoint] = protection_identity
    if not isinstance(protection, dict):
        raise ValueError("authenticated branch-protection response is malformed")
    immutability_endpoint = f"repos/{github_repository}/immutable-releases"
    immutability, immutability_identity = _gh_api_get_stable(
        immutability_endpoint,
        operation_log=operation_log,
    )
    endpoint_identities[immutability_endpoint] = immutability_identity
    if not isinstance(immutability, dict) or not isinstance(immutability.get("enabled"), bool):
        raise ValueError("immutable-release setting response is malformed")
    before_snapshot = {
        "endpoints": endpoint_identities,
        "repository": github_repository,
    }
    _append_operation(
        operation_log,
        kind="remote_snapshot",
        resource="github:prerequisite",
        identifier=github_repository,
        raw=_canonical_json_bytes(before_snapshot),
    )
    if not immutability["enabled"]:
        _append_operation(
            operation_log,
            kind="remote_snapshot",
            resource="github:before",
            identifier=github_repository,
            raw=_canonical_json_bytes(before_snapshot),
        )
        after_identities: dict[str, dict[str, Any]] = {}
        for endpoint, before_identity in endpoint_identities.items():
            _payload, after_identity = _gh_api_get_stable(
                endpoint,
                operation_log=operation_log,
            )
            if _remote_identity_moved(before_identity, after_identity):
                raise BoundaryBlocked(
                    f"authenticated GitHub resource moved concurrently: {endpoint}"
                )
            after_identities[endpoint] = after_identity
        _append_operation(
            operation_log,
            kind="remote_snapshot",
            resource="github:after",
            identifier=github_repository,
            raw=_canonical_json_bytes(
                {
                    "endpoints": after_identities,
                    "repository": github_repository,
                }
            ),
        )
        raise BoundaryBlocked("future GitHub Release immutability is authenticated and unavailable")

    releases, release_page_identities = _gh_api_paginated(
        f"repos/{github_repository}/releases",
        operation_log=operation_log,
    )
    endpoint_identities.update(release_page_identities)
    exact = [
        release
        for release in releases
        if isinstance(release, dict) and release.get("tag_name") == end_sha
    ]
    if not exact:
        raise BoundaryBlocked(
            f"immutable release capsule for {boundary} at exact tag {end_sha} "
            "is authenticated and unavailable"
        )
    if len(exact) != 1:
        raise ValueError("exact-SHA immutable release capsule identity is ambiguous")
    release_id = exact[0].get("id")
    if not isinstance(release_id, int) or release_id <= 0:
        raise ValueError("immutable release capsule API ID is malformed")
    release_endpoint = f"repos/{github_repository}/releases/{release_id}"
    release, release_identity = _gh_api_get_stable(
        release_endpoint,
        operation_log=operation_log,
    )
    endpoint_identities[release_endpoint] = release_identity
    if (
        not isinstance(release, dict)
        or release.get("tag_name") != end_sha
        or release.get("draft") is not False
        or release.get("prerelease") is not False
        or release.get("immutable") is not True
    ):
        raise ValueError("exact-SHA release is not a published immutable capsule")
    assets = release.get("assets")
    if not isinstance(assets, list) or not all(isinstance(item, dict) for item in assets):
        raise ValueError("immutable release capsule asset list is malformed")
    expected_asset_names = ("checksums.txt", "manifest.json", "payload.json")
    assets_by_name = {
        item.get("name"): item for item in assets if item.get("name") in expected_asset_names
    }
    if (
        len(assets) != len(expected_asset_names)
        or sorted(assets_by_name) != list(expected_asset_names)
        or len(assets_by_name) != 3
    ):
        raise ValueError("immutable release capsule assets are incomplete or duplicated")

    asset_bytes: dict[str, bytes] = {}
    asset_identities: dict[str, dict[str, Any]] = {}
    for name in expected_asset_names:
        asset_id = assets_by_name[name].get("id")
        if not isinstance(asset_id, int) or asset_id <= 0:
            raise ValueError(f"immutable release capsule asset ID is malformed: {name}")
        endpoint = f"repos/{github_repository}/releases/assets/{asset_id}"
        raw, identity = _gh_api_get_raw_stable(
            endpoint,
            operation_log=operation_log,
        )
        declared_digest = assets_by_name[name].get("digest")
        if declared_digest not in {None, f"sha256:{identity['sha256']}"}:
            raise ValueError(f"immutable release capsule asset digest mismatch: {name}")
        asset_bytes[name] = raw
        asset_identities[endpoint] = identity

    checksum_raw = asset_bytes["checksums.txt"]
    if checksum_raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("immutable release checksum asset has a UTF-8 BOM")
    try:
        checksum_text = checksum_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("immutable release checksum asset is not UTF-8") from exc
    if not checksum_text.endswith("\n") or checksum_text.endswith("\n\n"):
        raise ValueError("immutable release checksum asset has noncanonical terminal LF")
    checksum_lines = checksum_text[:-1].splitlines()
    expected_lines = [
        f"{_sha256_bytes(asset_bytes[name])}  {name}" for name in ("manifest.json", "payload.json")
    ]
    if checksum_lines != expected_lines:
        raise ValueError("immutable release checksum asset is incomplete or noncanonical")

    manifest = _parse_canonical_json_raw(
        asset_bytes["manifest.json"],
        label="immutable release capsule manifest",
        terminal_lf=True,
    )
    payload = _parse_canonical_json_raw(
        asset_bytes["payload.json"],
        label="immutable release capsule payload",
        terminal_lf=True,
    )
    _require_exact_fields(
        manifest,
        {
            "boundary",
            "end_sha",
            "payload_byte_length",
            "payload_sha256",
            "schema",
            "start_sha",
        },
        label="immutable release capsule manifest",
    )
    if (
        manifest.get("schema") != BOUNDARY_CAPSULE_MANIFEST_SCHEMA
        or manifest.get("boundary") != boundary
        or manifest.get("start_sha") != start_sha
        or manifest.get("end_sha") != end_sha
        or manifest.get("payload_byte_length") != len(asset_bytes["payload.json"])
        or manifest.get("payload_sha256") != _sha256_bytes(asset_bytes["payload.json"])
    ):
        raise ValueError("immutable release capsule manifest binding mismatch")
    _require_exact_fields(
        payload,
        {"boundary", "end_sha", "resources", "schema", "start_sha"},
        label="immutable release capsule payload",
    )
    if (
        payload.get("schema") != BOUNDARY_CAPSULE_PAYLOAD_SCHEMA
        or payload.get("boundary") != boundary
        or payload.get("start_sha") != start_sha
        or payload.get("end_sha") != end_sha
    ):
        raise ValueError("immutable release capsule payload interval mismatch")
    payload_resources = payload.get("resources")
    if not isinstance(payload_resources, list) or not all(
        isinstance(item, dict) for item in payload_resources
    ):
        raise ValueError("immutable release capsule payload resources are malformed")
    payload_resource_names = [item.get("name") for item in payload_resources]
    if (
        not all(isinstance(name, str) for name in payload_resource_names)
        or payload_resource_names != sorted(payload_resource_names)
        or len(payload_resource_names) != len(set(payload_resource_names))
    ):
        raise ValueError("immutable release capsule payload resources are not sorted and unique")

    resources: dict[str, dict[str, Any]] = {}
    resource_descriptors: list[dict[str, Any]] = []
    resource_digests: set[str] = set()
    for item in payload_resources:
        _require_exact_fields(item, {"name", "value"}, label="capsule payload resource")
        name = item.get("name")
        value = item.get("value")
        if name not in _RESOURCE_SCHEMAS or not isinstance(value, dict) or name in resources:
            raise ValueError("capsule payload resource identity is invalid or duplicated")
        _reject_caller_authority(value, label=f"capsule payload resource {name}")
        if value.get("schema") != _RESOURCE_SCHEMAS[name]:
            raise ValueError(f"capsule payload resource schema mismatch: {name}")
        if name in BOUNDARY_NAMES:
            if (
                value.get("predicate") != name
                or value.get("proof_for_boundary") != boundary
                or value.get("proof_start_sha") != start_sha
                or value.get("proof_end_sha") != end_sha
            ):
                raise ValueError(f"capsule payload predicate interval mismatch: {name}")
        elif (
            value.get("boundary") != boundary
            or value.get("start_sha") != start_sha
            or value.get("end_sha") != end_sha
        ):
            raise ValueError(f"capsule payload resource interval mismatch: {name}")
        raw = _canonical_json_bytes(value, terminal_lf=True)
        digest = _sha256_bytes(raw)
        if digest in resource_digests:
            raise ValueError("capsule payload resources reuse identical canonical bytes")
        resource_digests.add(digest)
        resources[name] = value
        resource_descriptors.append(
            {
                "byte_length": len(raw),
                "name": name,
                "path": f"payload.json#{name}",
                "sha256": digest,
            }
        )
        _append_operation(
            operation_log,
            kind="capsule_resource",
            resource=name,
            identifier=f"release:{release_id}:payload.json#{name}",
            raw=raw,
        )
    selected = BOUNDARY_NAMES[: BOUNDARY_NAMES.index(boundary) + 1]
    required = {
        "boundary_chronology",
        "durable_capsule",
        "external_prerequisites",
        "first_parent_receipts",
        "governed_prs",
        *selected,
    }
    missing = sorted(required - set(resources))
    if missing:
        raise ValueError(
            "immutable release capsule payload is incomplete; missing resources: "
            + ", ".join(missing)
        )

    release_verification, release_verification_identity = _run_live_verification(
        ["gh", "release", "verify", end_sha, "-R", github_repository, "--format", "json"],
        operation_log=operation_log,
        resource="release-attestation",
    )
    if not release_verification:
        raise ValueError("GitHub release verification returned no attestations")
    verification_commands: list[tuple[list[str], dict[str, Any], str]] = [
        (
            ["gh", "release", "verify", end_sha, "-R", github_repository, "--format", "json"],
            release_verification_identity,
            "release-attestation",
        )
    ]
    local_asset_identities: dict[str, dict[str, Any]] = {}
    for name in expected_asset_names:
        local_path = scratch_root / f"contract-drift-{release_id}-{name}"
        _guard_write_path(local_path, scratch_root, scratch_root)
        local_path.write_bytes(asset_bytes[name])
        _append_operation(
            operation_log,
            kind="scratch_write",
            resource=name,
            identifier=local_path.name,
            raw=asset_bytes[name],
        )
        local_asset_identities[str(local_path)] = {
            "byte_length": len(asset_bytes[name]),
            "sha256": _sha256_bytes(asset_bytes[name]),
        }
        _asset_verification, identity = _run_live_verification(
            [
                "gh",
                "release",
                "verify-asset",
                end_sha,
                str(local_path),
                "-R",
                github_repository,
                "--format",
                "json",
            ],
            operation_log=operation_log,
            resource=f"release-asset-attestation:{name}",
        )
        verification_commands.append(
            (
                [
                    "gh",
                    "release",
                    "verify-asset",
                    end_sha,
                    str(local_path),
                    "-R",
                    github_repository,
                    "--format",
                    "json",
                ],
                identity,
                f"release-asset-attestation:{name}",
            )
        )
        _attestation, identity = _run_live_verification(
            [
                "gh",
                "attestation",
                "verify",
                str(local_path),
                "-R",
                github_repository,
                "--signer-workflow",
                f"{github_repository}/.github/workflows/contract-drift-boundary.yml",
                "--format",
                "json",
            ],
            operation_log=operation_log,
            resource=f"sigstore-attestation:{name}",
        )
        verification_commands.append(
            (
                [
                    "gh",
                    "attestation",
                    "verify",
                    str(local_path),
                    "-R",
                    github_repository,
                    "--signer-workflow",
                    f"{github_repository}/.github/workflows/contract-drift-boundary.yml",
                    "--format",
                    "json",
                ],
                identity,
                f"sigstore-attestation:{name}",
            )
        )
        if (
            local_path.is_symlink()
            or local_path.stat().st_nlink != 1
            or local_path.read_bytes() != asset_bytes[name]
        ):
            raise ValueError(f"scratch release asset moved during verification: {name}")

    prerequisite = resources["external_prerequisites"]
    rule_suite = prerequisite.get("rule_suite")
    if not isinstance(rule_suite, dict) or not isinstance(rule_suite.get("id"), int):
        raise ValueError("capsule rule-suite identifier is malformed")
    rule_suite_endpoint = f"repos/{github_repository}/rulesets/rule-suites/{rule_suite['id']}"
    observed_rule_suite, rule_suite_identity = _gh_api_get_stable(
        rule_suite_endpoint,
        operation_log=operation_log,
    )
    endpoint_identities[rule_suite_endpoint] = rule_suite_identity
    if (
        not isinstance(observed_rule_suite, dict)
        or observed_rule_suite.get("id") != rule_suite["id"]
        or observed_rule_suite.get("result") != "pass"
    ):
        raise ValueError("authenticated GitHub rule suite is missing, bypassed, or nonpassing")
    if rule_suite != {
        "authenticated": True,
        "available": True,
        "bypassed": False,
        "id": observed_rule_suite["id"],
        "result": "pass",
    }:
        raise ValueError("capsule rule-suite claim contradicts authenticated GitHub evidence")
    if prerequisite.get("administration") != {
        "authenticated": True,
        "available": True,
    }:
        raise ValueError("capsule Administration claim is false or malformed")
    if prerequisite.get("future_release_immutability") != {
        "authenticated": True,
        "available": True,
        "enabled": True,
    }:
        raise ValueError("capsule immutable-release claim contradicts repository settings")

    capsule = resources["durable_capsule"]
    expected_release_claim = {
        "asset_api_ids": [
            assets_by_name[name]["id"]
            for name in ("manifest.json", "payload.json", "checksums.txt")
        ],
        "asset_names": ["manifest.json", "payload.json", "checksums.txt"],
        "exact_full_sha_tag": end_sha,
        "immutable": True,
        "release_api_id": release_id,
        "verified": True,
    }
    attestation_digest = _sha256_bytes(
        _canonical_json_bytes([identity for _argv, identity, _resource in verification_commands])
    )
    expected_attestation_claim = {
        "bundle_sha256": attestation_digest,
        "verified": True,
        "workflow": "actions/attest@v4",
    }
    if (
        capsule.get("release") != expected_release_claim
        or capsule.get("attestation") != expected_attestation_claim
    ):
        raise ValueError("capsule release or attestation claim contradicts live verification")
    publication = resources.get("final_seal", {}).get("publication")
    if isinstance(publication, dict) and isinstance(publication.get("fact"), dict):
        expected_publication_fields = {
            "attestation_bundle_sha256": attestation_digest,
            "release_api_id": release_id,
            "rule_suite_id": rule_suite["id"],
        }
        if any(
            publication["fact"].get(key) != value
            for key, value in expected_publication_fields.items()
        ):
            raise ValueError("capsule final publication contradicts live verification")

    governed = resources["governed_prs"].get("records")
    receipts = resources["first_parent_receipts"].get("records")
    if not isinstance(governed, list) or not isinstance(receipts, list):
        raise ValueError("capsule governed PR or receipt records are malformed")
    receipt_by_pr = {item.get("pr"): item for item in receipts if isinstance(item, dict)}
    for record in governed:
        if not isinstance(record, dict) or not isinstance(record.get("pr"), int):
            raise ValueError("capsule governed PR record is malformed")
        number = record["pr"]
        pr_endpoint = f"repos/{github_repository}/pulls/{number}"
        observed_pr, pr_identity = _gh_api_get_stable(
            pr_endpoint,
            operation_log=operation_log,
        )
        endpoint_identities[pr_endpoint] = pr_identity
        if (
            not isinstance(observed_pr, dict)
            or observed_pr.get("number") != number
            or observed_pr.get("merged_at") is None
            or not isinstance(observed_pr.get("base"), dict)
            or not isinstance(observed_pr.get("head"), dict)
            or observed_pr["base"].get("sha") != record.get("base_sha")
            or observed_pr["head"].get("sha") != record.get("head_sha")
            or observed_pr.get("merge_commit_sha") != receipt_by_pr.get(number, {}).get("merge_sha")
        ):
            raise ValueError(f"authenticated governed PR #{number} contradicts capsule evidence")
        files, file_identities = _gh_api_paginated(
            f"repos/{github_repository}/pulls/{number}/files",
            operation_log=operation_log,
        )
        endpoint_identities.update(file_identities)
        if len(files) != observed_pr.get("changed_files"):
            raise ValueError(f"authenticated governed PR #{number} file discovery is incomplete")
        if record.get("changed_files_complete") is not True:
            raise ValueError(f"capsule governed PR #{number} denies complete file discovery")
        receipt = receipt_by_pr.get(number)
        if not isinstance(receipt, dict):
            raise ValueError(f"capsule governed PR #{number} lacks a first-parent receipt")
        observed_commits: dict[str, dict[str, Any]] = {}
        for label, sha in (
            ("head", record["head_sha"]),
            ("merge", receipt["merge_sha"]),
        ):
            commit_endpoint = f"repos/{github_repository}/git/commits/{sha}"
            commit, commit_identity = _gh_api_get_stable(
                commit_endpoint,
                operation_log=operation_log,
            )
            endpoint_identities[commit_endpoint] = commit_identity
            if (
                not isinstance(commit, dict)
                or commit.get("sha") != sha
                or not isinstance(commit.get("tree"), dict)
                or not isinstance(commit["tree"].get("sha"), str)
            ):
                raise ValueError(f"authenticated governed PR #{number} {label} commit is malformed")
            observed_commits[label] = commit
        head_tree = observed_commits["head"]["tree"]["sha"]
        merge_tree = observed_commits["merge"]["tree"]["sha"]
        merge_parents = observed_commits["merge"].get("parents")
        if (
            record.get("head_tree_sha") != head_tree
            or receipt.get("head_tree_sha") != head_tree
            or receipt.get("merge_tree_sha") != merge_tree
            or head_tree != merge_tree
            or not isinstance(merge_parents, list)
            or not merge_parents
            or not isinstance(merge_parents[0], dict)
            or merge_parents[0].get("sha") != record["base_sha"]
        ):
            raise ValueError(
                f"authenticated governed PR #{number} lacks first-parent or tree equality"
            )

    live_before_snapshot = {
        "assets": asset_identities,
        "endpoints": endpoint_identities,
        "local_assets": {
            Path(path).name: identity for path, identity in sorted(local_asset_identities.items())
        },
        "repository": github_repository,
        "verifications": [identity for _argv, identity, _resource in verification_commands],
    }
    _append_operation(
        operation_log,
        kind="remote_snapshot",
        resource="github:before",
        identifier=github_repository,
        raw=_canonical_json_bytes(live_before_snapshot),
    )
    return (
        resources,
        {
            "index": {
                "byte_length": len(asset_bytes["manifest.json"]),
                "canonical_bytes_valid": True,
                "canonical_serialization": CANONICAL_SERIALIZATION,
                "path": "manifest.json",
                "sha256": _sha256_bytes(asset_bytes["manifest.json"]),
            },
            "resource_count": len(resources),
            "resources": sorted(resource_descriptors, key=lambda item: item["name"]),
            "resource_sha256s": sorted(resource_digests),
            "source": "immutable_github_release",
        },
        {
            "asset_identities": asset_identities,
            "endpoint_identities": endpoint_identities,
            "github_repository": github_repository,
            "local_asset_identities": local_asset_identities,
            "verification_commands": verification_commands,
        },
    )


def _reauthenticate_live_context(
    context: dict[str, Any],
    *,
    operation_log: list[dict[str, Any]],
) -> dict[str, Any]:
    after_endpoints: dict[str, dict[str, Any]] = {}
    for endpoint, before_identity in sorted(context["endpoint_identities"].items()):
        _payload, after_identity = _gh_api_get_stable(
            endpoint,
            operation_log=operation_log,
        )
        if _remote_identity_moved(before_identity, after_identity):
            raise BoundaryBlocked(f"authenticated GitHub resource moved concurrently: {endpoint}")
        after_endpoints[endpoint] = after_identity
    after_assets: dict[str, dict[str, Any]] = {}
    for endpoint, before_identity in sorted(context["asset_identities"].items()):
        _raw, after_identity = _gh_api_get_raw_stable(
            endpoint,
            operation_log=operation_log,
        )
        if _remote_identity_moved(before_identity, after_identity):
            raise BoundaryBlocked(f"authenticated GitHub asset moved concurrently: {endpoint}")
        after_assets[endpoint] = after_identity
    after_local_assets: dict[str, dict[str, Any]] = {}
    for raw_path, before_identity in sorted(context["local_asset_identities"].items()):
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise BoundaryBlocked(f"authenticated scratch release asset moved: {path.name}")
        raw = path.read_bytes()
        after_identity = {
            "byte_length": len(raw),
            "sha256": _sha256_bytes(raw),
        }
        if _remote_identity_moved(before_identity, after_identity):
            raise BoundaryBlocked(f"authenticated scratch release asset moved: {path.name}")
        after_local_assets[path.name] = after_identity
        _append_operation(
            operation_log,
            kind="scratch_read_after",
            resource=path.name,
            identifier=path.name,
            raw=raw,
        )
    verification_identities: list[dict[str, Any]] = []
    for argv, before_identity, resource in context["verification_commands"]:
        _payload, after_identity = _run_live_verification(
            argv,
            operation_log=operation_log,
            resource=f"{resource}:after",
        )
        if _remote_identity_moved(before_identity, after_identity):
            raise BoundaryBlocked(
                f"authenticated GitHub verification moved concurrently: {resource}"
            )
        verification_identities.append(after_identity)
    snapshot = {
        "assets": after_assets,
        "endpoints": after_endpoints,
        "local_assets": after_local_assets,
        "repository": context["github_repository"],
        "verifications": verification_identities,
    }
    _append_operation(
        operation_log,
        kind="remote_snapshot",
        resource="github:after",
        identifier=context["github_repository"],
        raw=_canonical_json_bytes(snapshot),
    )
    return snapshot


def _require_bool(value: Any, *, label: str) -> None:
    if value is not True:
        raise ValueError(f"{label} is false or missing")


def _require_sha256(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} is not a lowercase SHA-256")


def _require_exact_fields(
    value: dict[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields are incomplete or noncanonical")


def _fact_digest(schema: str, fact: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes({"fact": fact, "schema": schema}))


def _validate_fact_digest(
    value: dict[str, Any],
    *,
    schema: str,
    label: str,
) -> None:
    _require_exact_fields(value, {"fact", "sha256"}, label=label)
    fact = value.get("fact")
    if not isinstance(fact, dict):
        raise ValueError(f"{label} fact is malformed")
    if value.get("sha256") != _fact_digest(schema, fact):
        raise ValueError(f"{label} fact digest mismatch")


def _git_json_at_ref(
    repo_root: Path,
    ref: str,
    path: str,
    *,
    operation_log: list[dict[str, Any]],
) -> dict[str, Any]:
    proc = _run_read_only(
        ["git", "-C", str(repo_root), "show", f"{ref}:{path}"],
        operation_log=operation_log,
        resource=f"git-json:{ref}:{path}",
    )
    _append_operation(
        operation_log,
        kind="git_blob",
        resource=path,
        identifier=ref,
        raw=proc.stdout,
    )
    try:
        parsed = json.loads(proc.stdout, object_pairs_hook=_duplicate_key_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path} at {ref} is malformed JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} at {ref} is not a JSON object")
    return parsed


def _baseline_category_counts_at_ref(
    repo_root: Path,
    ref: str,
    *,
    operation_log: list[dict[str, Any]],
) -> dict[str, int]:
    paths = {
        "parity": "scripts/baselines/check_sdk_parity.json",
        "routes": "scripts/baselines/validate_openapi_routes.json",
        "verify": "scripts/baselines/verify_sdk_contracts.json",
    }
    docs = {
        name: _git_json_at_ref(
            repo_root,
            ref,
            path,
            operation_log=operation_log,
        )
        for name, path in paths.items()
    }
    return {
        "python_sdk_drift": len(docs["verify"].get("python_sdk_drift", [])),
        "routes_missing_in_spec": len(docs["routes"].get("missing_in_spec", [])),
        "routes_orphaned_in_spec": len(docs["routes"].get("orphaned_in_spec", [])),
        "sdk_missing_from_both": len(docs["parity"].get("missing_from_both_sdks", [])),
        "typescript_sdk_drift": len(docs["verify"].get("typescript_sdk_drift", [])),
    }


def _sdk_debt_original_ids_at_ref(
    repo_root: Path,
    ref: str,
    *,
    operation_log: list[dict[str, Any]],
) -> set[str]:
    baseline = _git_json_at_ref(
        repo_root,
        ref,
        "scripts/baselines/verify_sdk_contracts.json",
        operation_log=operation_log,
    )
    original_ids: set[str] = set()
    for category in ("python_sdk_drift", "typescript_sdk_drift"):
        literals = baseline.get(category)
        if not isinstance(literals, list) or not all(isinstance(item, str) for item in literals):
            raise ValueError(
                f"scripts/baselines/verify_sdk_contracts.json at {ref} has malformed {category}"
            )
        for literal in literals:
            original_ids.add(
                "cdg1:"
                + _sha256_bytes(
                    _canonical_json_bytes(
                        {
                            "category": category,
                            "exact_historical_literal_record": literal,
                            "schema": "cdg-original-record-id-v1",
                        }
                    )
                )
            )
    return original_ids


def _validate_boundary_chronology(
    resource: dict[str, Any],
    *,
    repo_root: Path,
    boundary: str,
    start_sha: str,
    end_sha: str,
    operation_log: list[dict[str, Any]],
) -> dict[str, str]:
    _require_exact_fields(
        resource,
        {"boundaries", "boundary", "end_sha", "schema", "start_sha"},
        label="boundary chronology",
    )
    records = resource.get("boundaries")
    if not isinstance(records, list) or not records:
        raise ValueError("boundary chronology is empty")
    selected_names = BOUNDARY_NAMES[: BOUNDARY_NAMES.index(boundary) + 1]
    names = [record.get("boundary") for record in records if isinstance(record, dict)]
    if names != list(selected_names):
        raise ValueError("boundary chronology is not the exact selected ordered prefix")
    chronology: dict[str, str] = {}
    prior = start_sha
    for record in records:
        if set(record) != {"boundary", "sha"}:
            raise ValueError("boundary chronology record is malformed")
        name = record["boundary"]
        sha = record["sha"]
        if not isinstance(sha, str) or not FULL_SHA_RE.fullmatch(sha):
            raise ValueError(f"boundary chronology SHA is malformed: {name}")
        _resolve_full_sha(
            repo_root,
            sha,
            label=f"boundary chronology {name}",
            operation_log=operation_log,
        )
        if sha == prior or not _is_ancestor(repo_root, prior, sha, operation_log):
            raise ValueError(f"boundary chronology is not strictly ordered at {name}")
        chronology[name] = sha
        prior = sha
    if chronology[boundary] != end_sha:
        raise ValueError("selected boundary chronology SHA does not equal end SHA")
    return chronology


def _validate_corrective_bootstrap(
    proof: dict[str, Any],
    chronology: dict[str, str],
    *,
    authority: dict[str, Any],
    repo_root: Path,
    operation_log: list[dict[str, Any]],
) -> list[str]:
    _require_exact_fields(
        proof,
        {
            "accepted_stage1_closure",
            "corrective_transition",
            "predicate",
            "proof_end_sha",
            "proof_for_boundary",
            "proof_start_sha",
            "schema",
            "stage2_verifier_chronology",
        },
        label="corrective_bootstrap proof",
    )
    closure = proof.get("accepted_stage1_closure")
    verifier = proof.get("stage2_verifier_chronology")
    transition = proof.get("corrective_transition")
    if not all(isinstance(value, dict) for value in (closure, verifier, transition)):
        raise ValueError("corrective_bootstrap proof is malformed")
    closure = cast(dict[str, Any], closure)
    verifier = cast(dict[str, Any], verifier)
    transition = cast(dict[str, Any], transition)
    _validate_fact_digest(
        closure,
        schema="contract-drift-stage1-closure-fact-v1",
        label="accepted Stage-1 closure",
    )
    expected_closure = {
        "authority_manifest_sha256": authority["authority_manifest_sha256"],
        "boundary_verifier_sha256": authority["boundary_verifier_sha256"],
        "dependency_manifest_sha256": authority["dependency_manifest_sha256"],
        "inventory_sha256": authority["inventory_sha256"],
        "repo_file_count": authority["repo_file_count"],
    }
    if closure["fact"] != expected_closure:
        raise ValueError("accepted Stage-1 closure does not match reconstructed authority")
    _validate_fact_digest(
        verifier,
        schema="contract-drift-stage2-verifier-chronology-fact-v1",
        label="Stage-2 verifier chronology",
    )
    expected_verifier = {
        "corrective_boundary_sha": chronology["corrective_bootstrap"],
        "ordered_after_stage1": True,
        "start_sha": proof["proof_start_sha"],
        "verifier_sha256": authority["boundary_verifier_sha256"],
    }
    if verifier["fact"] != expected_verifier:
        raise ValueError("Stage-2 verifier chronology does not match reconstructed authority")
    _validate_fact_digest(
        transition,
        schema="contract-drift-corrective-transition-fact-v1",
        label="corrective transition",
    )
    commit_count = int(
        _run_read_only(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-list",
                "--count",
                f"{proof['proof_start_sha']}..{chronology['corrective_bootstrap']}",
            ],
            operation_log=operation_log,
            resource="corrective-transition-count",
        ).stdout.decode("ascii")
    )
    expected_transition = {
        "commit_count": commit_count,
        "end_sha": chronology["corrective_bootstrap"],
        "start_sha": proof["proof_start_sha"],
    }
    if transition["fact"] != expected_transition or commit_count <= 0:
        raise ValueError("corrective transition does not match the exact git interval")
    return [
        "accepted_stage1_closure",
        "stage2_verifier_chronology",
        "corrective_transition",
    ]


def _validate_route_truth(
    proof: dict[str, Any],
    *,
    authority: dict[str, Any],
    chronology: dict[str, str],
    repo_root: Path,
    operation_log: list[dict[str, Any]],
) -> list[str]:
    _require_exact_fields(
        proof,
        {
            "openapi_truth",
            "predicate",
            "proof_end_sha",
            "proof_for_boundary",
            "proof_start_sha",
            "route_truth",
            "schema",
        },
        label="route_truth proof",
    )
    route = proof.get("route_truth")
    openapi = proof.get("openapi_truth")
    if not isinstance(route, dict) or not isinstance(openapi, dict):
        raise ValueError("route_truth proof is malformed")
    _validate_fact_digest(
        route,
        schema="contract-drift-route-truth-fact-v1",
        label="route truth",
    )
    expected_route = {
        "authority_route_member_count": authority["route_authority_member_count"],
        "boundary_sha": chronology["route_truth"],
        "complete": True,
        "method_aware": True,
        "route_boundary_sha256": authority["route_boundary_sha256"],
    }
    if route["fact"] != expected_route:
        raise ValueError("route truth does not match reconstructed route authority")
    _validate_fact_digest(
        openapi,
        schema="contract-drift-openapi-truth-fact-v1",
        label="OpenAPI truth",
    )
    expected_openapi = {
        "boundary_sha": chronology["route_truth"],
        "complete": True,
        "route_boundary_sha256": authority["route_boundary_sha256"],
    }
    if openapi["fact"] != expected_openapi:
        raise ValueError("OpenAPI truth does not match reconstructed route authority")
    counts = _baseline_category_counts_at_ref(
        repo_root,
        chronology["route_truth"],
        operation_log=operation_log,
    )
    if counts["routes_missing_in_spec"] != 0 or counts["routes_orphaned_in_spec"] != 0:
        raise ValueError("route truth is contradicted by exact-ref route baselines")
    return ["route_truth", "openapi_truth"]


def _validate_sdk_paydown(
    proof: dict[str, Any],
    *,
    boundary: str,
    chronology: dict[str, str],
    canonical_artifacts: dict[str, Any],
    repo_root: Path,
    operation_log: list[dict[str, Any]],
) -> list[str]:
    zero_key = "zero_core_debt" if boundary == "core_sdk" else "zero_sdk_debt"
    _require_exact_fields(
        proof,
        {
            "predicate",
            "proof_end_sha",
            "proof_for_boundary",
            "proof_start_sha",
            "qualifying_paydown",
            "schema",
            zero_key,
        },
        label=f"{boundary} proof",
    )
    paydown = proof.get("qualifying_paydown")
    zero = proof.get(zero_key)
    if not isinstance(paydown, dict) or not isinstance(zero, dict):
        raise ValueError(f"{boundary} proof is malformed")
    _validate_fact_digest(
        paydown,
        schema=f"contract-drift-{boundary.replace('_', '-')}-paydown-fact-v1",
        label=f"{boundary} qualifying paydown",
    )
    paydown_fact = paydown["fact"]
    if paydown_fact.get("boundary_sha") != chronology[boundary]:
        raise ValueError(f"{boundary} paydown boundary SHA mismatch")
    removed = paydown_fact.get("removed_original_record_ids")
    if (
        not isinstance(removed, list)
        or not removed
        or not all(
            isinstance(item, str) and item.startswith("cdg1:") and SHA256_RE.fullmatch(item[5:])
            for item in removed
        )
    ):
        raise ValueError(f"{boundary} lacks a qualifying immutable-unit removal")
    expected_removed = (
        canonical_artifacts["sdk_provenance"]["core_original_record_ids"]
        if boundary == "core_sdk"
        else canonical_artifacts["sdk_provenance"]["extended_original_record_ids"]
    )
    if removed != expected_removed:
        raise ValueError(
            f"{boundary} qualifying paydown does not equal its independently "
            "reconstructed immutable partition"
        )
    for field in ("added_units", "category_growth", "replacement_units"):
        if paydown_fact.get(field) != []:
            raise ValueError(f"{boundary} has forbidden {field}")
    max_pr_delta = paydown_fact.get("max_pr_delta")
    if not isinstance(max_pr_delta, int) or not (0 < max_pr_delta <= 800):
        raise ValueError(f"{boundary} violates the per-PR size cap")
    zero_schema = (
        "contract-drift-zero-core-debt-fact-v1"
        if boundary == "core_sdk"
        else "contract-drift-zero-sdk-debt-fact-v1"
    )
    _validate_fact_digest(zero, schema=zero_schema, label=f"{boundary} zero debt")
    expected_partition_digest = (
        canonical_artifacts["sdk_provenance"]["core_original_record_id_set_sha256"]
        if boundary == "core_sdk"
        else canonical_artifacts["sdk_provenance"]["sdk_original_record_id_set_sha256"]
    )
    expected_zero = {
        "boundary_sha": chronology[boundary],
        "partition_set_sha256": expected_partition_digest,
        "remaining_original_units": 0,
    }
    if zero["fact"] != expected_zero:
        raise ValueError(f"{boundary} original-cohort debt is not zero")
    remaining_ids = _sdk_debt_original_ids_at_ref(
        repo_root,
        chronology[boundary],
        operation_log=operation_log,
    )
    target_ids = set(
        canonical_artifacts["sdk_provenance"]["core_original_record_ids"]
        if boundary == "core_sdk"
        else canonical_artifacts["sdk_provenance"]["sdk_original_record_ids"]
    )
    if remaining_ids & target_ids:
        raise ValueError(
            f"{boundary} zero debt is contradicted by exact-ref SDK category baselines"
        )
    return ["qualifying_paydown", zero_key]


def _validate_final_seal(
    proof: dict[str, Any],
    *,
    chronology: dict[str, str],
    capsule: dict[str, Any],
    prerequisites: dict[str, Any],
    repo_root: Path,
    operation_log: list[dict[str, Any]],
) -> list[str]:
    _require_exact_fields(
        proof,
        {
            "complete_paydown",
            "dated_trajectory",
            "final_zero",
            "predicate",
            "proof_end_sha",
            "proof_for_boundary",
            "proof_start_sha",
            "publication",
            "schema",
        },
        label="final_seal proof",
    )
    publication = proof.get("publication")
    paydown = proof.get("complete_paydown")
    trajectory = proof.get("dated_trajectory")
    final_zero = proof.get("final_zero")
    if not all(isinstance(value, dict) for value in (publication, paydown, trajectory, final_zero)):
        raise ValueError("final_seal proof is malformed")
    publication = cast(dict[str, Any], publication)
    paydown = cast(dict[str, Any], paydown)
    trajectory = cast(dict[str, Any], trajectory)
    final_zero = cast(dict[str, Any], final_zero)
    _validate_fact_digest(
        publication,
        schema="contract-drift-publication-fact-v1",
        label="final publication",
    )
    expected_publication = {
        "attestation_bundle_sha256": capsule["attestation"]["bundle_sha256"],
        "boundary_sha": chronology["final_seal"],
        "release_api_id": capsule["release"]["release_api_id"],
        "rule_suite_id": prerequisites["rule_suite"]["id"],
    }
    if publication["fact"] != expected_publication:
        raise ValueError("final publication does not match authenticated capsule evidence")
    _validate_fact_digest(
        paydown,
        schema="contract-drift-complete-paydown-fact-v1",
        label="final complete paydown",
    )
    if paydown["fact"] != {
        "boundary_sha": chronology["final_seal"],
        "remaining_original_units": 0,
    }:
        raise ValueError("final complete paydown is not zero")
    _validate_fact_digest(
        trajectory,
        schema="contract-drift-dated-trajectory-fact-v1",
        label="final dated trajectory",
    )
    trajectory_fact = trajectory["fact"]
    if (
        trajectory_fact.get("boundary_sha") != chronology["final_seal"]
        or not isinstance(trajectory_fact.get("as_of"), str)
        or trajectory_fact.get("target") != 0
        or trajectory_fact.get("total") != 0
    ):
        raise ValueError("final dated trajectory is false or malformed")
    try:
        date.fromisoformat(trajectory_fact["as_of"])
    except ValueError as exc:
        raise ValueError("final dated trajectory date is malformed") from exc
    _validate_fact_digest(
        final_zero,
        schema="contract-drift-final-zero-fact-v1",
        label="final zero",
    )
    if final_zero["fact"] != {
        "all_categories_zero": True,
        "boundary_sha": chronology["final_seal"],
    }:
        raise ValueError("final category zero is false or malformed")
    counts = _baseline_category_counts_at_ref(
        repo_root,
        chronology["final_seal"],
        operation_log=operation_log,
    )
    if any(counts.values()):
        raise ValueError("final zero is contradicted by exact-ref category baselines")
    if trajectory_fact["total"] != sum(counts.values()):
        raise ValueError("final dated trajectory total contradicts exact-ref baselines")
    return ["publication", "complete_paydown", "dated_trajectory", "final_zero"]


def _validate_external_prerequisites(resource: dict[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "administration",
        "boundary",
        "end_sha",
        "future_release_immutability",
        "rule_suite",
        "schema",
        "start_sha",
    }
    if resource.get("mutation_tainted"):
        raise ValueError("external prerequisite evidence is mutation-tainted")
    _require_exact_fields(
        resource,
        expected_fields,
        label="external prerequisite evidence",
    )
    administration = resource.get("administration")
    release = resource.get("future_release_immutability")
    rule_suite = resource.get("rule_suite")
    if not all(isinstance(value, dict) for value in (administration, release, rule_suite)):
        raise ValueError("external prerequisite evidence is malformed")
    administration = cast(dict[str, Any], administration)
    release = cast(dict[str, Any], release)
    rule_suite = cast(dict[str, Any], rule_suite)
    for name, prerequisite in (
        ("Administration/rule-suite access", administration),
        ("future GitHub Release immutability", release),
        ("passing GitHub rule suite", rule_suite),
    ):
        if prerequisite.get("authenticated") is not True:
            raise ValueError(f"{name} is not independently authenticated")
        if prerequisite.get("available") is not True:
            raise BoundaryBlocked(f"{name} is authenticated and unavailable")
    if release.get("enabled") is not True:
        raise BoundaryBlocked("future GitHub Release immutability is authenticated and unavailable")
    if rule_suite.get("bypassed") is not False:
        raise ValueError("GitHub rule suite was bypassed")
    if rule_suite.get("result") != "pass":
        raise ValueError("GitHub rule suite did not pass")
    rule_suite_id = rule_suite.get("id")
    if not isinstance(rule_suite_id, int) or rule_suite_id <= 0:
        raise ValueError("GitHub rule-suite ID is malformed")
    return {
        "administration_read_verified": True,
        "future_release_immutability_enabled": True,
        "rule_suite": {
            "bypassed": False,
            "id": rule_suite_id,
            "result": "pass",
        },
    }


def _validate_durable_capsule(
    resource: dict[str, Any],
    *,
    end_sha: str,
) -> dict[str, Any]:
    _require_exact_fields(
        resource,
        {
            "attestation",
            "boundary",
            "end_sha",
            "release",
            "schema",
            "start_sha",
        },
        label="durable release capsule evidence",
    )
    release = resource.get("release")
    attestation = resource.get("attestation")
    if not isinstance(release, dict) or not isinstance(attestation, dict):
        raise ValueError("durable release capsule evidence is malformed")
    for field in ("immutable", "verified"):
        _require_bool(release.get(field), label=f"durable release {field}")
    if release.get("exact_full_sha_tag") != end_sha:
        raise ValueError("durable release capsule tag is not the exact end SHA")
    release_id = release.get("release_api_id")
    asset_ids = release.get("asset_api_ids")
    asset_names = release.get("asset_names")
    if not isinstance(release_id, int) or release_id <= 0:
        raise ValueError("durable release API ID is malformed")
    if (
        not isinstance(asset_ids, list)
        or len(asset_ids) != 3
        or len(set(asset_ids)) != 3
        or not all(isinstance(item, int) and item > 0 for item in asset_ids)
    ):
        raise ValueError("durable release asset API IDs are incomplete")
    if asset_names != ["manifest.json", "payload.json", "checksums.txt"]:
        raise ValueError("durable release asset names are incomplete or noncanonical")
    _require_bool(attestation.get("verified"), label="Sigstore attestation")
    if attestation.get("workflow") != "actions/attest@v4":
        raise ValueError("Sigstore attestation workflow identity mismatch")
    _require_sha256(
        attestation.get("bundle_sha256"),
        label="Sigstore attestation bundle digest",
    )
    return {
        "attestation": attestation,
        "release": release,
    }


def _validate_governed_prs(
    resource: dict[str, Any],
    *,
    repo_root: Path,
    operation_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _require_exact_fields(
        resource,
        {"boundary", "end_sha", "records", "schema", "start_sha"},
        label="governed PR evidence",
    )
    records = resource.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("governed PR evidence is missing")
    seen: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("governed PR record is malformed")
        _require_exact_fields(
            record,
            {
                "base_sha",
                "changed_files_complete",
                "head_sha",
                "head_tree_sha",
                "pr",
            },
            label="governed PR record",
        )
        number = record.get("pr")
        base = record.get("base_sha")
        head = record.get("head_sha")
        if not isinstance(number, int) or number <= 0 or number in seen:
            raise ValueError("governed PR number is invalid or duplicated")
        if not isinstance(base, str) or not isinstance(head, str):
            raise ValueError("governed PR SHA binding is malformed")
        if not FULL_SHA_RE.fullmatch(base) or not FULL_SHA_RE.fullmatch(head):
            raise ValueError("governed PR SHA binding is noncanonical")
        if base == head:
            raise ValueError("governed PR interval is empty")
        if not isinstance(record.get("head_tree_sha"), str) or not FULL_SHA_RE.fullmatch(
            record["head_tree_sha"]
        ):
            raise ValueError(f"governed PR #{number} head tree is malformed")
        _require_bool(
            record.get("changed_files_complete"),
            label=f"governed PR #{number} file discovery",
        )
        seen.add(number)
    return records


def _validate_first_parent_receipts(
    resource: dict[str, Any],
) -> list[dict[str, Any]]:
    _require_exact_fields(
        resource,
        {"boundary", "end_sha", "records", "schema", "start_sha"},
        label="first-parent receipt evidence",
    )
    records = resource.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("first-parent receipt evidence is missing")
    seen: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("first-parent receipt is malformed")
        _require_exact_fields(
            record,
            {
                "base_sha",
                "first_parent_sha",
                "head_sha",
                "head_tree_sha",
                "merge_sha",
                "merge_tree_sha",
                "pr",
            },
            label="first-parent receipt",
        )
        number = record.get("pr")
        if not isinstance(number, int) or number <= 0 or number in seen:
            raise ValueError("first-parent receipt PR is invalid or duplicated")
        for field in ("base_sha", "first_parent_sha", "head_sha", "merge_sha"):
            value = record.get(field)
            if not isinstance(value, str) or not FULL_SHA_RE.fullmatch(value):
                raise ValueError(f"first-parent receipt {field} is malformed")
        if record["base_sha"] != record["first_parent_sha"]:
            raise ValueError("first-parent receipt does not equal the frozen base SHA")
        for field in ("head_tree_sha", "merge_tree_sha"):
            if not isinstance(record.get(field), str) or not FULL_SHA_RE.fullmatch(record[field]):
                raise ValueError(f"first-parent receipt #{number} {field} is malformed")
        seen.add(number)
    return records


def _reconcile_prs_and_receipts(
    governed_prs: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    *,
    repo_root: Path,
    start_sha: str,
    end_sha: str,
    operation_log: list[dict[str, Any]],
) -> None:
    prs_by_number = {record["pr"]: record for record in governed_prs}
    receipts_by_number = {record["pr"]: record for record in receipts}
    if set(prs_by_number) != set(receipts_by_number):
        raise ValueError("governed PRs and first-parent receipts do not reconcile")
    ordered = sorted(receipts, key=lambda record: record["merge_sha"])
    ordered = sorted(
        ordered,
        key=lambda record: int(
            _run_read_only(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "rev-list",
                    "--count",
                    f"{start_sha}..{record['merge_sha']}",
                ],
                operation_log=operation_log,
                resource=f"receipt-order:{record['pr']}",
            ).stdout.decode("ascii")
        ),
    )
    prior = start_sha
    for receipt in ordered:
        governed = prs_by_number[receipt["pr"]]
        if (
            receipt["base_sha"] != governed["base_sha"]
            or receipt["head_sha"] != governed["head_sha"]
            or receipt["head_tree_sha"] != governed["head_tree_sha"]
        ):
            raise ValueError(f"governed PR #{receipt['pr']} and first-parent receipt disagree")
        if receipt["base_sha"] != prior:
            raise ValueError("governed PR and receipt intervals are not contiguous")
        first_parent = (
            _run_read_only(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "rev-parse",
                    f"{receipt['merge_sha']}^",
                ],
                operation_log=operation_log,
                resource=f"receipt-first-parent:{receipt['pr']}",
            )
            .stdout.decode("ascii")
            .strip()
        )
        if first_parent != receipt["base_sha"]:
            raise ValueError(f"first-parent receipt for PR #{receipt['pr']} does not match git")
        merge_tree = (
            _run_read_only(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "rev-parse",
                    f"{receipt['merge_sha']}^{{tree}}",
                ],
                operation_log=operation_log,
                resource=f"receipt-merge-tree:{receipt['pr']}",
            )
            .stdout.decode("ascii")
            .strip()
        )
        if (
            merge_tree != receipt["merge_tree_sha"]
            or receipt["merge_tree_sha"] != receipt["head_tree_sha"]
        ):
            raise ValueError(
                f"first-parent receipt for PR #{receipt['pr']} lacks squash tree equality"
            )
        prior = receipt["merge_sha"]
    if prior != end_sha:
        raise ValueError("governed PR and receipt coverage does not reach the boundary end SHA")


def _evaluate_boundary_evidence(
    resources: dict[str, dict[str, Any]],
    *,
    repo_root: Path,
    boundary: str,
    start_sha: str,
    end_sha: str,
    authority: dict[str, Any],
    canonical_artifacts: dict[str, Any],
    operation_log: list[dict[str, Any]],
) -> dict[str, Any]:
    chronology = _validate_boundary_chronology(
        resources["boundary_chronology"],
        repo_root=repo_root,
        boundary=boundary,
        start_sha=start_sha,
        end_sha=end_sha,
        operation_log=operation_log,
    )
    prerequisites = _validate_external_prerequisites(resources["external_prerequisites"])
    capsule = _validate_durable_capsule(
        resources["durable_capsule"],
        end_sha=end_sha,
    )
    governed_prs = _validate_governed_prs(
        resources["governed_prs"],
        repo_root=repo_root,
        operation_log=operation_log,
    )
    receipts = _validate_first_parent_receipts(resources["first_parent_receipts"])
    _reconcile_prs_and_receipts(
        governed_prs,
        receipts,
        repo_root=repo_root,
        start_sha=start_sha,
        end_sha=end_sha,
        operation_log=operation_log,
    )
    selected = BOUNDARY_NAMES[: BOUNDARY_NAMES.index(boundary) + 1]
    predicates: dict[str, dict[str, Any]] = {}
    for name in selected:
        proof = resources[name]
        if name == "corrective_bootstrap":
            checks = _validate_corrective_bootstrap(
                proof,
                chronology,
                authority=authority,
                repo_root=repo_root,
                operation_log=operation_log,
            )
        elif name == "route_truth":
            checks = _validate_route_truth(
                proof,
                authority=authority,
                chronology=chronology,
                repo_root=repo_root,
                operation_log=operation_log,
            )
        elif name in {"core_sdk", "extended_sdk"}:
            checks = _validate_sdk_paydown(
                proof,
                boundary=name,
                chronology=chronology,
                canonical_artifacts=canonical_artifacts,
                repo_root=repo_root,
                operation_log=operation_log,
            )
        else:
            checks = _validate_final_seal(
                proof,
                chronology=chronology,
                capsule=capsule,
                prerequisites=prerequisites,
                repo_root=repo_root,
                operation_log=operation_log,
            )
        predicates[name] = {
            "checks": checks,
            "proof_sha256": _sha256_bytes(_canonical_json_bytes(proof, terminal_lf=True)),
            "proven": True,
        }
    if len({item["proof_sha256"] for item in predicates.values()}) != len(predicates):
        raise ValueError("boundary predicates reuse identical proof bytes")
    return {
        "boundary_chronology": chronology,
        "durable_capsule": capsule,
        "external_prerequisites": prerequisites,
        "first_parent_receipts": receipts,
        "governed_prs": governed_prs,
        "predicates": predicates,
    }


def _reauthenticate_evidence_resources(
    *,
    evidence_index_path: Path,
    evidence_summary: dict[str, Any],
    operation_log: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        _index, index_descriptor, index_raw = _read_canonical_json_bytes(
            evidence_index_path,
            label="external boundary evidence index after verification",
            expected_byte_length=evidence_summary["index"]["byte_length"],
            expected_sha256=evidence_summary["index"]["sha256"],
            terminal_lf=True,
        )
        _append_operation(
            operation_log,
            kind="evidence_index_after",
            resource="boundary_evidence_index",
            identifier=evidence_index_path.name,
            raw=index_raw,
        )
        root = evidence_index_path.resolve().parent
        authenticated_resources: list[dict[str, Any]] = []
        for descriptor in evidence_summary["resources"]:
            resolved = (root / descriptor["path"]).resolve()
            _payload, authenticated, resource_raw = _read_canonical_json_bytes(
                resolved,
                label=f"boundary evidence resource {descriptor['name']} after verification",
                expected_byte_length=descriptor["byte_length"],
                expected_sha256=descriptor["sha256"],
                terminal_lf=True,
            )
            _append_operation(
                operation_log,
                kind="external_resource_after",
                resource=descriptor["name"],
                identifier=descriptor["path"],
                raw=resource_raw,
            )
            authenticated_resources.append(
                {
                    "byte_length": authenticated["byte_length"],
                    "name": descriptor["name"],
                    "path": descriptor["path"],
                    "sha256": authenticated["sha256"],
                }
            )
    except ValueError as exc:
        raise BoundaryBlocked(f"authenticated evidence resource moved concurrently: {exc}") from exc
    return {
        "index": index_descriptor,
        "resources": authenticated_resources,
    }


def _reauthenticate_canonical_input(
    path: Path,
    *,
    descriptor: dict[str, Any],
    label: str,
    operation_log: list[dict[str, Any]],
) -> None:
    try:
        _payload, authenticated, raw = _read_canonical_json_bytes(
            path,
            label=f"{label} after verification",
            expected_byte_length=descriptor["byte_length"],
            expected_sha256=descriptor["sha256"],
            terminal_lf=True,
        )
    except ValueError as exc:
        raise BoundaryBlocked(f"authenticated {label} moved concurrently: {exc}") from exc
    _append_operation(
        operation_log,
        kind="canonical_input_after",
        resource=label,
        identifier=path.name,
        raw=raw,
    )
    if (
        authenticated["byte_length"] != descriptor["byte_length"]
        or authenticated["sha256"] != descriptor["sha256"]
    ):
        raise BoundaryBlocked(f"authenticated {label} moved concurrently")


def _record_remote_snapshot(
    operation_log: list[dict[str, Any]],
    *,
    label: str,
    resource: str,
    identifier: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    raw = _canonical_json_bytes(snapshot)
    _append_operation(
        operation_log,
        kind="remote_snapshot",
        resource=f"{resource}:{label}",
        identifier=identifier,
        raw=raw,
    )
    return {
        "byte_length": len(raw),
        "sha256": _sha256_bytes(raw),
    }


def _remote_snapshots_from_log(
    operation_log: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    entries = [entry for entry in operation_log if entry["kind"] == "remote_snapshot"]
    if len(entries) < 2:
        return None
    before, after = entries[-2:]
    return (
        {"byte_length": before["byte_length"], "sha256": before["sha256"]},
        {"byte_length": after["byte_length"], "sha256": after["sha256"]},
    )


def _finalize_boundary_result(result: dict[str, Any]) -> dict[str, Any]:
    result["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(result))
    return result


def build_boundary_result(
    *,
    repo_root: Path,
    schema_version: int,
    boundary: str,
    start_ref: str,
    end_ref: str,
    authority_manifest_path: Path | None = None,
    authority_manifest_byte_length: int | None = None,
    authority_manifest_sha256: str | None = None,
    cohort_artifact_path: Path | None = None,
    sdk_provenance_artifact_path: Path | None = None,
    github_repository: str = "synaptent/aragora",
    github_branch: str = "main",
    scratch_root: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_scratch = (scratch_root or Path(tempfile.gettempdir())).resolve()
    resolved_output = (output_root or resolved_scratch).resolve()
    operation_log: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "blocked_reason": None,
        "boundary": boundary,
        "operation_log": operation_log,
        "passing": False,
        "repository_root": ".",
        "schema": BOUNDARY_MANIFEST_SCHEMA,
        "schema_version": schema_version,
        "stage1_test_matrix": list(STAGE1_TEST_MATRIX),
        "status": "fail",
        "write_allowlist": ["output_root", "scratch_root"],
    }
    before_snapshot: dict[str, Any] | None = None
    try:
        if schema_version != BOUNDARY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported boundary schema version {schema_version}; "
                f"expected {BOUNDARY_SCHEMA_VERSION}"
            )
        if boundary not in BOUNDARY_NAMES:
            raise ValueError(f"unsupported Contract Drift boundary: {boundary}")
        if not resolved_scratch.is_dir() or not resolved_output.is_dir():
            raise ValueError("declared scratch/output roots must already exist")
        _guard_write_path(resolved_scratch, resolved_scratch, resolved_output)
        _guard_write_path(resolved_output, resolved_scratch, resolved_output)
        _append_operation(
            operation_log,
            kind="write_allowlist",
            resource="filesystem",
            identifier="scratch_root,output_root",
            raw=_canonical_json_bytes(result["write_allowlist"]),
        )
        start_sha = _resolve_full_sha(
            repo_root,
            start_ref,
            label="--start-ref",
            operation_log=operation_log,
        )
        end_sha = _resolve_full_sha(
            repo_root,
            end_ref,
            label="--end-ref",
            operation_log=operation_log,
        )
        result.update({"end_sha": end_sha, "start_sha": start_sha})
        if start_sha == end_sha:
            raise ValueError("boundary start SHA must differ from end SHA")
        if not _is_ancestor(repo_root, start_sha, end_sha, operation_log):
            raise ValueError("boundary start SHA is not an ancestor of end SHA")

        before_snapshot = _snapshot_repository(repo_root, operation_log)
        result["local_snapshot_before"] = before_snapshot
        artifacts = _authenticate_canonical_artifacts(
            repo_root=repo_root,
            cohort_artifact_path=cohort_artifact_path,
            sdk_provenance_artifact_path=sdk_provenance_artifact_path,
            scratch_root=resolved_scratch,
            operation_log=operation_log,
        )
        result["canonical_artifacts"] = artifacts
        resolved_cohort = _discover_canonical_artifact(
            cohort_artifact_path,
            COHORT_ARTIFACT,
            repo_root,
        )
        resolved_provenance = _discover_canonical_artifact(
            sdk_provenance_artifact_path,
            PROVENANCE_ARTIFACT,
            repo_root,
        )
        authority = _authenticate_authority_manifest(
            repo_root=repo_root,
            end_sha=end_sha,
            authority_manifest_path=authority_manifest_path,
            authority_manifest_byte_length=authority_manifest_byte_length,
            authority_manifest_sha256=authority_manifest_sha256,
            cohort_artifact_path=resolved_cohort,
            sdk_provenance_artifact_path=resolved_provenance,
            scratch_root=resolved_scratch,
            operation_log=operation_log,
        )
        result["authority"] = authority
        resources, evidence_summary, live_context = _collect_live_evidence(
            github_repository=github_repository,
            github_branch=github_branch,
            boundary=boundary,
            start_sha=start_sha,
            end_sha=end_sha,
            scratch_root=resolved_scratch,
            operation_log=operation_log,
        )
        result["evidence"] = evidence_summary
        result["remote_snapshot_before"] = _record_remote_snapshot(
            operation_log,
            label="before",
            resource="authenticated_evidence",
            identifier=boundary,
            snapshot={
                "index": evidence_summary["index"],
                "resources": evidence_summary["resources"],
            },
        )
        evaluation = _evaluate_boundary_evidence(
            resources,
            repo_root=repo_root,
            boundary=boundary,
            start_sha=start_sha,
            end_sha=end_sha,
            authority=authority,
            canonical_artifacts=artifacts,
            operation_log=operation_log,
        )
        result.update(evaluation)
        _reauthenticate_canonical_input(
            resolved_cohort,
            descriptor=artifacts["original_cohort"],
            label="canonical original-cohort artifact",
            operation_log=operation_log,
        )
        _reauthenticate_canonical_input(
            resolved_provenance,
            descriptor=artifacts["sdk_provenance"],
            label="canonical SDK-provenance artifact",
            operation_log=operation_log,
        )
        if authority_manifest_path is not None:
            _reauthenticate_canonical_input(
                authority_manifest_path,
                descriptor=authority["authenticated_manifest_bytes"],
                label="external authority manifest",
                operation_log=operation_log,
            )
        _reauthenticate_live_context(
            live_context,
            operation_log=operation_log,
        )
        after_evidence = {
            "index": evidence_summary["index"],
            "resources": evidence_summary["resources"],
        }
        result["remote_snapshot_after"] = _record_remote_snapshot(
            operation_log,
            label="after",
            resource="authenticated_evidence",
            identifier=boundary,
            snapshot=after_evidence,
        )
        if result["remote_snapshot_before"] != result["remote_snapshot_after"]:
            raise BoundaryBlocked("authenticated evidence resources moved concurrently")
        after_snapshot = _snapshot_repository(repo_root, operation_log)
        result["local_snapshot_after"] = after_snapshot
        if before_snapshot != after_snapshot:
            raise BoundaryBlocked("independently observed local repository movement")
        result.update({"passing": True, "status": "pass"})
    except BoundaryBlocked as exc:
        result.update(
            {
                "blocked_reason": str(exc),
                "passing": False,
                "status": "blocked",
            }
        )
        remote_snapshots = _remote_snapshots_from_log(operation_log)
        if remote_snapshots is not None and "remote_snapshot_before" not in result:
            result["remote_snapshot_before"], result["remote_snapshot_after"] = remote_snapshots
        if before_snapshot is not None and "local_snapshot_after" not in result:
            try:
                result["local_snapshot_after"] = _snapshot_repository(
                    repo_root,
                    operation_log,
                )
            except (ValueError, OSError):
                pass
    except (ValueError, OSError, inventory_mod.AuthorityClosureError) as exc:
        result.update(
            {
                "error": str(exc),
                "error_code": "boundary_validation_failed",
                "passing": False,
                "status": "fail",
            }
        )
        remote_snapshots = _remote_snapshots_from_log(operation_log)
        if remote_snapshots is not None and "remote_snapshot_before" not in result:
            result["remote_snapshot_before"], result["remote_snapshot_after"] = remote_snapshots
        if before_snapshot is not None and "local_snapshot_after" not in result:
            try:
                result["local_snapshot_after"] = _snapshot_repository(
                    repo_root,
                    operation_log,
                )
            except (ValueError, OSError):
                pass
    return _finalize_boundary_result(result)


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
    parser.add_argument(
        "--mode",
        choices=("program", "pr", "boundary"),
        default="program",
    )
    parser.add_argument(
        "--schema-version",
        type=int,
        default=None,
        help="Required manifest schema version for boundary mode",
    )
    parser.add_argument("--boundary", choices=BOUNDARY_NAMES, default=None)
    parser.add_argument("--start-ref", default=None)
    parser.add_argument("--end-ref", default=None)
    parser.add_argument("--authority-manifest", type=Path, default=None)
    parser.add_argument("--authority-manifest-byte-length", type=int, default=None)
    parser.add_argument("--authority-manifest-sha256", default=None)
    parser.add_argument("--cohort-artifact", type=Path, default=None)
    parser.add_argument("--sdk-provenance-artifact", type=Path, default=None)
    parser.add_argument("--github-repository", default="synaptent/aragora")
    parser.add_argument("--github-branch", default="main")
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
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
            "--boundary": args.boundary,
            "--end-ref": args.end_ref,
            "--schema-version": args.schema_version,
            "--start-ref": args.start_ref,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            error = _finalize_boundary_result(
                {
                    "blocked_reason": None,
                    "error": "missing required boundary argument(s): " + ", ".join(missing),
                    "error_code": "missing_required_boundary_input",
                    "passing": False,
                    "schema": BOUNDARY_MANIFEST_SCHEMA,
                    "schema_version": args.schema_version,
                    "status": "fail",
                }
            )
            if args.json:
                print(
                    json.dumps(
                        error,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            else:
                print(f"FAIL (closed): {error['error']}", file=sys.stderr)
            return 1
        result = build_boundary_result(
            repo_root=args.repo_root,
            schema_version=args.schema_version,
            boundary=args.boundary,
            start_ref=args.start_ref,
            end_ref=args.end_ref,
            authority_manifest_path=args.authority_manifest,
            authority_manifest_byte_length=args.authority_manifest_byte_length,
            authority_manifest_sha256=args.authority_manifest_sha256,
            cohort_artifact_path=args.cohort_artifact,
            sdk_provenance_artifact_path=args.sdk_provenance_artifact,
            github_repository=args.github_repository,
            github_branch=args.github_branch,
            scratch_root=args.scratch_root,
            output_root=args.output_root,
        )
        if args.json:
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        else:
            print(f"Contract Drift boundary {args.boundary}: {result['status'].upper()}")
            if result.get("blocked_reason"):
                print(f"Blocked: {result['blocked_reason']}")
            if result.get("error"):
                print(f"Failure: {result['error']}")
            print(f"Manifest SHA-256: {result['manifest_sha256']}")
        if result["status"] == "pass":
            return 0
        if result["status"] == "blocked":
            return 2
        return 1

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
