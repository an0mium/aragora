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
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict, deque
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator

COHORT_COMMIT = "5cc7a81b77aeb6680613d441f74d72c435c8443c"
COHORT_DATE = "2026-04-17"
COHORT_PROVENANCE = "2026-04-17 program baseline cohort"
DEFAULT_INVENTORY = "scripts/baselines/contract_drift_inventory.json"
VALID_CLASSES = frozenset({"start_cohort", "discovered"})
VALID_STATUSES = frozenset({"open", "resolved"})
PROVENANCE_REF = re.compile(r"#\d+|https?://\S+")
FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")
AUTHORITY_MANIFEST_SCHEMA = "contract-drift-authority-manifest-v2"
AUTHORITY_CLASSIFICATION_SCHEMA = "contract-drift-authority-classification-v2"
EXACT_REF_INVENTORY_SCHEMA = "contract-drift-exact-ref-inventory-v2"
POLICY_MODULE = "aragora.cli.commands.review_queue"
POLICY_PATH = "aragora/cli/commands/review_queue.py"
MERGE_TRAIN_PATH = "scripts/tier4_merge_train.py"
PYTHON_FLAGS = ("-I", "-S")
EXACT_REF_POLICY_TIMEOUT_SECONDS = 30
WORKFLOW_SUFFIXES = (".yml", ".yaml")
EXECUTABLE_SUFFIXES = (".py", ".sh")
MEASUREMENT_SUBJECT_PREFIXES = (
    "sdk/",
    "openapi/",
    "docs/api/openapi",
    "aragora/server/handler_registry",
    "aragora/server/handlers/",
    "aragora/server/openapi/",
)
DYNAMIC_MEASUREMENT_IMPORT_ARGUMENTS = frozenset({("scripts/check_sdk_parity.py", "module_path")})
FUNCTION_BODY_IMPORT_SOURCES = frozenset(
    {
        "scripts/check_contract_drift_ratchet.py",
        "scripts/check_sdk_parity.py",
        "scripts/generate_contract_drift_inventory.py",
        "scripts/sdk_path_normalize.py",
        "scripts/tier4_merge_train.py",
        "scripts/validate_openapi_routes.py",
        "scripts/verify_sdk_contracts.py",
    }
)
PYTHON_OPTIONS_WITHOUT_VALUES = frozenset(
    {
        "-B",
        "-E",
        "-I",
        "-O",
        "-OO",
        "-P",
        "-R",
        "-S",
        "-V",
        "-b",
        "-bb",
        "-d",
        "-i",
        "-q",
        "-s",
        "-u",
        "-v",
        "-x",
    }
)
PYTHON_OPTIONS_WITH_VALUES = frozenset({"-W", "-X"})
PYTHON_EXECUTABLE_ARGUMENT_BINDINGS = frozenset(
    {
        ("scripts/generate_contract_drift_inventory.py", "interpreter"),
        ("scripts/generate_contract_drift_inventory.py", "interpreter_path"),
    }
)
NON_REPOSITORY_COMMAND_ARGUMENT_BINDINGS = {
    ("scripts/generate_contract_drift_inventory.py", "runner_path"): (
        "__contract_drift_exact_ref_runner.py"
    )
}
DYNAMIC_EXTERNAL_SUBPROCESS_COMMAND_SOURCES = frozenset(
    {
        "aragora/swarm/quorum_evidence.py",
        "scripts/generate_sdk_types.py",
        "scripts/run_pip_audit_gate.py",
    }
)
DYNAMIC_INTERNAL_SUBPROCESS_HELPERS = {
    "aragora/swarm/merge_quorum_io.py": ("aragora/cli/main.py",),
    "scripts/pre_release_check.py": (
        "scripts/check_cross_sdk_parity.py",
        "scripts/check_pentest_findings.py",
        "scripts/check_sdk_namespace_parity.py",
        "scripts/check_sdk_parity.py",
        "scripts/reconcile_status_docs.py",
        "scripts/run_pip_audit_gate.py",
        "scripts/smoke_test.py",
    ),
}
REPOSITORY_MODULE_PREFIXES = ("aragora", "scripts")
LOCAL_REFERENCE_PREFIXES = (
    "./.github/",
    ".github/",
    "./scripts/",
    "scripts/",
    "./aragora/",
    "aragora/",
)

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


class AuthorityClosureError(ValueError):
    """Fail-closed authority closure or exact-ref validation error."""


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_git(
    repo_root: Path,
    args: list[str],
    *,
    check: bool = True,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=check,
        capture_output=True,
        text=text,
    )


def _resolve_full_commit(repo_root: Path, ref: str) -> str:
    if not FULL_SHA_RE.fullmatch(ref):
        raise AuthorityClosureError("--ref must be a full lowercase 40-hex commit SHA")
    proc = _run_git(repo_root, ["rev-parse", "--verify", f"{ref}^{{commit}}"], text=True)
    resolved = proc.stdout.strip()
    if resolved != ref:
        raise AuthorityClosureError(
            f"requested ref did not resolve byte-for-byte to itself: {ref} -> {resolved}"
        )
    return resolved


def _safe_archive_target(root: Path, name: str) -> Path:
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AuthorityClosureError(f"unsafe git archive member: {name!r}")
    target = (root / candidate).resolve(strict=False)
    if target != root and root not in target.parents:
        raise AuthorityClosureError(f"git archive member escapes extraction root: {name!r}")
    return target


def _extract_exact_ref(repo_root: Path, sha: str, extraction_root: Path) -> None:
    """Extract the complete exact-ref tree without consulting the worktree."""
    archive = _run_git(repo_root, ["archive", "--format=tar", sha]).stdout
    extraction_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar.getmembers():
            target = _safe_archive_target(extraction_root, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.isfile():
                source = tar.extractfile(member)
                if source is None:
                    raise AuthorityClosureError(f"could not read git archive member: {member.name}")
                target.write_bytes(source.read())
                target.chmod(stat.S_IMODE(member.mode))
                continue
            if member.issym():
                link = Path(member.linkname)
                if link.is_absolute():
                    raise AuthorityClosureError(f"absolute symlink in git archive: {member.name!r}")
                resolved_link = (target.parent / link).resolve(strict=False)
                if (
                    resolved_link != extraction_root
                    and extraction_root not in resolved_link.parents
                ):
                    raise AuthorityClosureError(f"symlink escapes extraction root: {member.name!r}")
                target.symlink_to(member.linkname)
                continue
            raise AuthorityClosureError(f"unsupported git archive member type for {member.name!r}")


_EXACT_REF_RUNNER = r"""
import importlib
import importlib.util
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
request_path = pathlib.Path(sys.argv[2]).resolve()
if request_path.parent != root:
    raise SystemExit("request must be inside extraction root")
sys.dont_write_bytecode = True
sys.path[:] = [str(root), *[entry for entry in sys.path if pathlib.Path(entry or ".").resolve() != root]]
stdlib_roots = tuple(
    pathlib.Path(entry).resolve()
    for entry in sys.path
    if entry and pathlib.Path(entry).resolve() != root
)

def under_root(value):
    try:
        path = pathlib.Path(value).resolve(strict=False)
    except (TypeError, ValueError):
        return False
    return path == root or root in path.parents

def allowed_read(value):
    try:
        path = pathlib.Path(value).resolve(strict=False)
    except (TypeError, ValueError):
        return False
    return under_root(path) or any(path == base or base in path.parents for base in stdlib_roots)

def audit(event, args):
    if event == "open":
        path = args[0]
        if isinstance(path, int):
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writes = (
            isinstance(mode, str) and any(marker in mode for marker in ("w", "a", "x", "+"))
        ) or (
            isinstance(flags, int)
            and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        )
        if writes:
            raise RuntimeError("forbidden write action in exact-ref classifier")
        if not allowed_read(path):
            raise RuntimeError("forbidden read outside extraction and standard-library roots")
    if event in {"os.listdir", "os.scandir"} and args and not allowed_read(args[0]):
        raise RuntimeError("forbidden directory read outside allowed roots")
    if event in {
        "os.system",
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "subprocess.Popen",
    } or event.startswith(("os.spawn", "ctypes.")):
        raise RuntimeError(f"forbidden subprocess action: {event}")
    if event.startswith(("socket.", "http.client.", "urllib.Request")):
        raise RuntimeError(f"forbidden network action: {event}")
    if event in {
        "os.chmod",
        "os.chflags",
        "os.chown",
        "os.fchmod",
        "os.fchown",
        "os.ftruncate",
        "os.lchmod",
        "os.lchflags",
        "os.lchown",
        "os.link",
        "os.lremovexattr",
        "os.lsetxattr",
        "os.mkfifo",
        "os.mkdir",
        "os.makedirs",
        "os.mknod",
        "os.remove",
        "os.removedirs",
        "os.removexattr",
        "os.rename",
        "os.renames",
        "os.replace",
        "os.rmdir",
        "os.symlink",
        "os.setxattr",
        "os.truncate",
        "os.unlink",
        "os.utime",
        "shutil.chown",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.copymode",
        "shutil.copystat",
        "shutil.move",
        "shutil.rmtree",
        "sqlite3.connect",
        "sqlite3.connect/handle",
    }:
        raise RuntimeError(f"forbidden filesystem mutation: {event}")

sys.addaudithook(audit)
request = json.loads(request_path.read_text(encoding="utf-8"))
review_queue = importlib.import_module("aragora.cli.commands.review_queue")
policy_file = pathlib.Path(review_queue.__file__).resolve()
expected_policy = (root / "aragora/cli/commands/review_queue.py").resolve()
if policy_file != expected_policy:
    raise SystemExit(f"noncanonical policy module: {policy_file}")
for package_name, relative_init in (
    ("aragora", "aragora/__init__.py"),
    ("aragora.cli", "aragora/cli/__init__.py"),
    ("aragora.cli.commands", "aragora/cli/commands/__init__.py"),
):
    package = sys.modules.get(package_name)
    expected_init = (root / relative_init).resolve()
    package_file = pathlib.Path(getattr(package, "__file__", "") or "").resolve()
    package_paths = [
        pathlib.Path(item).resolve() for item in (getattr(package, "__path__", ()) or ())
    ]
    if package_file != expected_init or package_paths != [expected_init.parent]:
        raise SystemExit(
            f"namespace-blended or noncanonical package: "
            f"{package_name}={package_file},{package_paths}"
        )

spec = importlib.util.spec_from_file_location(
    "_exact_ref_tier4_merge_train",
    root / "scripts/tier4_merge_train.py",
)
if spec is None or spec.loader is None:
    raise SystemExit("exact-ref merge-train mirror is unavailable")
merge_train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = merge_train
spec.loader.exec_module(merge_train)

def matched_rule(path):
    for rule in review_queue.TIER_4_PREFIXES:
        if review_queue._matches_prefix(path, (rule,)):
            return rule
    return None

classifications = []
for raw_path in request.get("paths", []):
    path = str(raw_path)
    tier, tier_name, tier_reason = review_queue._classify_model_review_tier([path])
    classifications.append(
        {
            "path": path,
            "tier": tier,
            "tier_name": tier_name,
            "tier_reason": tier_reason,
            "matched_rule": matched_rule(path),
            "merge_train_matched_rule": merge_train.matches_serialized_path(path),
        }
    )

scripts_package = sys.modules.get("scripts")
if scripts_package is not None:
    scripts_root = (root / "scripts").resolve()
    scripts_init = scripts_root / "__init__.py"
    raw_package_file = getattr(scripts_package, "__file__", None)
    package_file = pathlib.Path(raw_package_file).resolve() if raw_package_file else None
    package_paths = [
        pathlib.Path(item).resolve()
        for item in (getattr(scripts_package, "__path__", ()) or ())
    ]
    expected_file = scripts_init if scripts_init.is_file() else None
    if package_file != expected_file or package_paths != [scripts_root]:
        raise SystemExit(
            f"namespace-blended or noncanonical package: scripts={package_file},{package_paths}"
        )

loaded = []
for name, module in sorted(sys.modules.items()):
    repository_named = name.startswith(
        ("aragora", "scripts", "_exact_ref_tier4_merge_train")
    )
    locations = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        locations.append(pathlib.Path(module_file).resolve())
    module_path = getattr(module, "__path__", None)
    if module_path:
        locations.extend(pathlib.Path(item).resolve() for item in module_path)
    for location in locations:
        if location == (root / "__contract_drift_exact_ref_runner.py").resolve():
            continue
        if under_root(location):
            relative = location.relative_to(root).as_posix()
            loaded.append({"module": name, "path": relative})
        elif repository_named:
            raise SystemExit(f"repository module escaped extraction root: {name}={location}")

result = {
    "authority_roots": list(review_queue.CONTRACT_DRIFT_AUTHORITY_PREFIXES),
    "authority_tier": review_queue.CONTRACT_DRIFT_AUTHORITY_TIER,
    "classifications": classifications,
    "loaded_repository_modules": loaded,
    "policy_module": "aragora.cli.commands.review_queue",
    "policy_module_path": policy_file.relative_to(root).as_posix(),
    "policy_version": review_queue.CONTRACT_DRIFT_AUTHORITY_POLICY_VERSION,
    "tier4_prefixes": list(review_queue.TIER_4_PREFIXES),
}
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
"""


def _minimal_subprocess_env(scratch_root: Path) -> dict[str, str]:
    home = scratch_root / "home"
    tmp = scratch_root / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.defpath,
        "TMPDIR": str(tmp),
        "TZ": "UTC",
    }


def _invoke_exact_ref_policy(
    extraction_root: Path,
    paths: Iterable[str],
    *,
    scratch_root: Path,
    interpreter: Path | None = None,
) -> dict[str, Any]:
    interpreter_path = (interpreter or Path(sys.executable)).resolve()
    if not interpreter_path.is_file():
        raise AuthorityClosureError(f"explicit interpreter is unavailable: {interpreter_path}")
    policy_path = extraction_root / POLICY_PATH
    mirror_path = extraction_root / MERGE_TRAIN_PATH
    if not policy_path.is_file():
        raise AuthorityClosureError(f"canonical exact-ref policy is unavailable: {POLICY_PATH}")
    if policy_path.is_symlink():
        raise AuthorityClosureError(
            f"canonical exact-ref policy cannot be a symlink: {POLICY_PATH}"
        )
    if not mirror_path.is_file():
        raise AuthorityClosureError(
            f"exact-ref merge-train mirror is unavailable: {MERGE_TRAIN_PATH}"
        )
    if mirror_path.is_symlink():
        raise AuthorityClosureError(
            f"exact-ref merge-train mirror cannot be a symlink: {MERGE_TRAIN_PATH}"
        )

    runner_path = extraction_root / "__contract_drift_exact_ref_runner.py"
    request_path = extraction_root / "__contract_drift_exact_ref_request.json"
    if runner_path.exists() or request_path.exists():
        raise AuthorityClosureError("exact-ref repository collides with classifier scratch paths")
    runner_path.write_text(_EXACT_REF_RUNNER, encoding="utf-8")
    request_path.write_bytes(_canonical_json_bytes({"paths": sorted(set(paths))}))
    try:
        proc = subprocess.run(
            [
                str(interpreter_path),
                *PYTHON_FLAGS,
                str(runner_path),
                str(extraction_root),
                str(request_path),
            ],
            cwd=extraction_root,
            env=_minimal_subprocess_env(scratch_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=EXACT_REF_POLICY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AuthorityClosureError(
            f"exact-ref canonical classifier exceeded {EXACT_REF_POLICY_TIMEOUT_SECONDS}s timeout"
        ) from exc
    finally:
        runner_path.unlink(missing_ok=True)
        request_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise AuthorityClosureError(f"exact-ref canonical classifier failed: {detail}")
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AuthorityClosureError("exact-ref canonical classifier returned invalid JSON") from exc

    if result.get("policy_module") != POLICY_MODULE:
        raise AuthorityClosureError("exact-ref classifier did not load the canonical policy module")
    if result.get("policy_module_path") != POLICY_PATH:
        raise AuthorityClosureError("exact-ref classifier policy path is noncanonical")
    loaded_paths = {
        entry.get("path")
        for entry in result.get("loaded_repository_modules", [])
        if isinstance(entry, dict)
    }
    if POLICY_PATH not in loaded_paths:
        raise AuthorityClosureError("canonical policy module is absent from loaded-module proof")
    return result


def _module_parts_for_source(source_path: str) -> tuple[str, ...]:
    path = Path(source_path)
    if path.suffix != ".py":
        return ()
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return tuple(parts)


def _is_type_checking_guard(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id == "TYPE_CHECKING"
        or isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
        and node.attr == "TYPE_CHECKING"
    )


def _iter_authority_import_nodes(
    nodes: Iterable[ast.stmt], *, include_function_bodies: bool
) -> Iterator[ast.AST]:
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from node.decorator_list
            yield from node.args.defaults
            yield from (default for default in node.args.kw_defaults if default is not None)
            yield from (
                argument.annotation
                for argument in [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
                if argument.annotation is not None
            )
            if node.args.vararg is not None and node.args.vararg.annotation is not None:
                yield node.args.vararg.annotation
            if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
                yield node.args.kwarg.annotation
            if node.returns is not None:
                yield node.returns
            if include_function_bodies:
                yield from _iter_authority_import_nodes(
                    node.body, include_function_bodies=include_function_bodies
                )
            continue
        if isinstance(node, ast.ClassDef):
            yield from node.decorator_list
            yield from node.bases
            yield from (keyword.value for keyword in node.keywords)
            yield from _iter_authority_import_nodes(
                node.body, include_function_bodies=include_function_bodies
            )
            continue
        if isinstance(node, ast.If):
            if _is_type_checking_guard(node.test):
                yield from _iter_authority_import_nodes(
                    node.orelse, include_function_bodies=include_function_bodies
                )
            else:
                yield from _iter_authority_import_nodes(
                    node.body, include_function_bodies=include_function_bodies
                )
                yield from _iter_authority_import_nodes(
                    node.orelse, include_function_bodies=include_function_bodies
                )
            continue
        if isinstance(node, (ast.Try, ast.TryStar)):
            yield from _iter_authority_import_nodes(
                node.body, include_function_bodies=include_function_bodies
            )
            for handler in node.handlers:
                yield from _iter_authority_import_nodes(
                    handler.body, include_function_bodies=include_function_bodies
                )
            yield from _iter_authority_import_nodes(
                node.orelse, include_function_bodies=include_function_bodies
            )
            yield from _iter_authority_import_nodes(
                node.finalbody, include_function_bodies=include_function_bodies
            )
            continue
        if isinstance(node, (ast.With, ast.AsyncWith)):
            yield from _iter_authority_import_nodes(
                node.body, include_function_bodies=include_function_bodies
            )
            continue
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            yield from _iter_authority_import_nodes(
                node.body, include_function_bodies=include_function_bodies
            )
            yield from _iter_authority_import_nodes(
                node.orelse, include_function_bodies=include_function_bodies
            )
            continue
        if isinstance(node, ast.Match):
            for case in node.cases:
                yield from _iter_authority_import_nodes(
                    case.body, include_function_bodies=include_function_bodies
                )
            continue
        yield node


def _resolve_module_path(extraction_root: Path, module: str) -> str | None:
    if not module:
        return None
    module_path = Path(*module.split("."))
    file_candidate = extraction_root / module_path.with_suffix(".py")
    package_candidate = extraction_root / module_path / "__init__.py"
    # Python's FileFinder resolves a package directory before a same-name
    # module file. Mirror that canonical order rather than blending both trees.
    if package_candidate.is_file():
        return package_candidate.relative_to(extraction_root).as_posix()
    if file_candidate.is_file():
        return file_candidate.relative_to(extraction_root).as_posix()
    return None


def _required_parent_packages(extraction_root: Path, module: str) -> list[str]:
    parts = module.split(".")
    parents: list[str] = []
    for index in range(1, len(parts)):
        candidate = extraction_root / Path(*parts[:index]) / "__init__.py"
        if candidate.is_file():
            parents.append(candidate.relative_to(extraction_root).as_posix())
    return parents


def _absolute_import_module(source_path: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    source_parts = list(_module_parts_for_source(source_path))
    package_parts = source_parts if Path(source_path).name == "__init__.py" else source_parts[:-1]
    remove = node.level - 1
    if remove > len(package_parts):
        raise AuthorityClosureError(f"relative import escapes repository package: {source_path}")
    prefix = package_parts[: len(package_parts) - remove]
    if node.module:
        prefix.extend(node.module.split("."))
    return ".".join(prefix)


def _python_import_edges(extraction_root: Path, source_path: str) -> list[tuple[str, str]]:
    path = extraction_root / source_path
    if path.suffix != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=source_path)
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise AuthorityClosureError(
            f"cannot parse authority Python member {source_path}: {exc}"
        ) from exc

    static_bindings = _static_path_bindings(tree)
    modules: set[str] = set()
    for node in _iter_authority_import_nodes(
        tree.body,
        include_function_bodies=source_path in FUNCTION_BODY_IMPORT_SOURCES,
    ):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_import_module(source_path, node)
            if base:
                modules.add(base)
            for alias in node.names:
                candidate = f"{base}.{alias.name}" if base and alias.name != "*" else ""
                if candidate and _resolve_module_path(extraction_root, candidate) is not None:
                    modules.add(candidate)
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            is_import_module = (
                isinstance(child.func, ast.Attribute) and child.func.attr == "import_module"
            ) or (
                isinstance(child.func, ast.Name)
                and child.func.id in {"__import__", "import_module"}
            )
            is_run_module = (
                isinstance(child.func, ast.Attribute) and child.func.attr == "run_module"
            ) or (isinstance(child.func, ast.Name) and child.func.id == "run_module")
            if not (is_import_module or is_run_module) or not child.args:
                continue
            argument = child.args[0]
            module = _command_token(argument, static_bindings)
            if module is None:
                if (
                    isinstance(argument, ast.Name)
                    and (source_path, argument.id) in DYNAMIC_MEASUREMENT_IMPORT_ARGUMENTS
                ):
                    continue
                raise AuthorityClosureError(
                    f"dynamic repository import is forbidden: {source_path}"
                )
            modules.add(module)

    edges: set[tuple[str, str]] = set()
    for module in sorted(modules):
        top_level = module.split(".", 1)[0]
        resolved = _resolve_module_path(extraction_root, module)
        if resolved is None and top_level not in REPOSITORY_MODULE_PREFIXES:
            if "/" not in module and "." not in module and source_path.startswith("scripts/"):
                script_candidate = extraction_root / "scripts" / f"{module}.py"
                if script_candidate.is_file():
                    resolved = script_candidate.relative_to(extraction_root).as_posix()
            if resolved is None:
                continue
        if resolved is None:
            raise AuthorityClosureError(
                f"repository-local import is unavailable: {source_path} -> {module}"
            )
        if _is_measurement_subject(resolved):
            continue
        for parent in _required_parent_packages(extraction_root, module):
            edges.add((parent, "python_parent_package"))
        edges.add((resolved, "python_repository_import"))
    return sorted(edges)


def _static_path_parts(
    node: ast.AST, bindings: dict[str, list[str]] | None = None
) -> list[str] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        and node.attr == "executable"
    ):
        return ["{sys.executable}"]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "str":
        if len(node.args) == 1:
            return _static_path_parts(node.args[0], bindings)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _static_path_parts(node.left, bindings)
        right = _static_path_parts(node.right, bindings)
        if left is None or right is None:
            return None
        return [*left, *right]
    if isinstance(node, ast.Name):
        if bindings is not None and node.id in bindings:
            return bindings[node.id]
        if node.id in {"REPO_ROOT", "ROOT", "repo_root"}:
            return []
    return None


def _static_path_bindings(tree: ast.Module) -> dict[str, list[str]]:
    bindings: dict[str, list[str]] = {
        "REPO_ROOT": [],
        "ROOT": [],
        "repo_root": [],
    }
    assignments: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assignments.append((node.targets[0].id, node.value))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            assignments.append((node.target.id, node.value))
    for _ in range(len(assignments) + 1):
        changed = False
        for name, value in assignments:
            parts = _static_path_parts(value, bindings)
            if parts is not None and bindings.get(name) != parts:
                bindings[name] = parts
                changed = True
        if not changed:
            break
    return bindings


def _static_sequence_bindings(tree: ast.Module) -> dict[str, list[ast.AST]]:
    bindings: dict[str, list[ast.AST]] = {}
    for node in tree.body:
        target: ast.Name | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if target is not None and isinstance(value, (ast.List, ast.Tuple)):
            bindings[target.id] = list(value.elts)
    return bindings


def _expanded_command_elements(
    command: ast.List | ast.Tuple, sequence_bindings: dict[str, list[ast.AST]]
) -> list[ast.AST]:
    elements: list[ast.AST] = []
    for element in command.elts:
        if isinstance(element, ast.Starred):
            if isinstance(element.value, ast.Name) and element.value.id in sequence_bindings:
                elements.extend(sequence_bindings[element.value.id])
                continue
        elements.append(element)
    return elements


def _command_token(node: ast.AST, bindings: dict[str, list[str]] | None = None) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        and node.attr == "executable"
    ):
        return "{sys.executable}"
    parts = _static_path_parts(node, bindings)
    if parts is None:
        return None
    return "/".join(part.strip("/") for part in parts if part)


def _is_python_command(token: str) -> bool:
    basename = Path(token).name
    return (
        token == "{sys.executable}" or re.fullmatch(r"python(?:3(?:\.\d+)?)?", basename) is not None
    )


def _subprocess_program_index(
    elements: list[ast.AST], bindings: dict[str, list[str]]
) -> int | None:
    index = 0
    token = _command_token(elements[index], bindings)
    if token is None:
        return None
    if Path(token).name == "env":
        index += 1
        while index < len(elements):
            token = _command_token(elements[index], bindings)
            if token is None:
                return None
            if "=" not in token:
                break
            index += 1
    if index < len(elements) and Path(_command_token(elements[index], bindings) or "").name == "uv":
        index += 1
        if index < len(elements) and _command_token(elements[index], bindings) == "run":
            index += 1
    if index >= len(elements):
        return None
    return index


def _local_command_binding(
    tree: ast.Module, call: ast.Call, name: str
) -> ast.List | ast.Tuple | None:
    scopes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.lineno <= call.lineno <= (node.end_lineno or node.lineno)
    ]
    if not scopes:
        scope: ast.AST = tree
    else:
        scope = min(scopes, key=lambda node: (node.end_lineno or node.lineno) - node.lineno)

    def scope_nodes(node: ast.AST) -> Iterator[ast.AST]:
        for child in ast.iter_child_nodes(node):
            yield child
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            yield from scope_nodes(child)

    def binds_name(target: ast.AST) -> bool:
        return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(target))

    def contains_name(node: ast.AST) -> bool:
        return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))

    literal_binding: ast.List | ast.Tuple | None = None
    binding_events = 0
    mutated = False
    for node in scope_nodes(scope):
        if getattr(node, "lineno", call.lineno) >= call.lineno:
            continue
        if isinstance(node, ast.Assign) and any(binds_name(target) for target in node.targets):
            binding_events += 1
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, (ast.List, ast.Tuple))
            ):
                literal_binding = node.value
            else:
                mutated = True
        elif isinstance(node, ast.AnnAssign) and binds_name(node.target):
            binding_events += 1
            if isinstance(node.target, ast.Name) and isinstance(node.value, (ast.List, ast.Tuple)):
                literal_binding = node.value
            else:
                mutated = True
        elif isinstance(node, (ast.AugAssign, ast.NamedExpr)) and binds_name(node.target):
            mutated = True
        elif isinstance(node, (ast.For, ast.AsyncFor)) and binds_name(node.target):
            mutated = True
        elif isinstance(node, (ast.With, ast.AsyncWith)) and any(
            item.optional_vars is not None and binds_name(item.optional_vars) for item in node.items
        ):
            mutated = True
        elif isinstance(node, ast.ExceptHandler) and node.name == name:
            mutated = True
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (alias.asname or alias.name.split(".", 1)[0]) == name for alias in node.names
        ):
            mutated = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                mutated = True
        elif isinstance(node, ast.Delete) and any(binds_name(target) for target in node.targets):
            mutated = True
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == name
        ):
            mutated = True
        elif isinstance(node, ast.Call) and any(
            contains_name(argument)
            for argument in [*node.args, *(keyword.value for keyword in node.keywords)]
        ):
            mutated = True
    if binding_events != 1 or mutated:
        return None
    return literal_binding


def _subprocess_helper_edges(extraction_root: Path, source_path: str) -> list[tuple[str, str]]:
    path = extraction_root / source_path
    if path.suffix != ".py":
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=source_path)
    static_bindings = _static_path_bindings(tree)
    sequence_bindings = _static_sequence_bindings(tree)
    for binding_source, name in PYTHON_EXECUTABLE_ARGUMENT_BINDINGS:
        if binding_source == source_path:
            static_bindings[name] = ["{sys.executable}"]
    for (binding_source, name), value in NON_REPOSITORY_COMMAND_ARGUMENT_BINDINGS.items():
        if binding_source == source_path:
            static_bindings[name] = [value]
    helpers: set[str] = set()
    subprocess_calls = {"run", "Popen", "call", "check_call", "check_output"}
    subprocess_module_aliases = {"subprocess"}
    subprocess_function_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    subprocess_module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            for alias in node.names:
                if alias.name in subprocess_calls:
                    subprocess_function_aliases.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_subprocess_call = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in subprocess_module_aliases
            and node.func.attr in subprocess_calls
        ) or (isinstance(node.func, ast.Name) and node.func.id in subprocess_function_aliases)
        if not is_subprocess_call or not node.args:
            continue
        command = node.args[0]
        if isinstance(command, ast.Name):
            command = _local_command_binding(tree, node, command.id) or command
        if any(keyword.arg is None for keyword in node.keywords):
            raise AuthorityClosureError(
                f"dynamic subprocess keyword arguments are forbidden: {source_path}"
            )
        shell_keywords = [keyword.value for keyword in node.keywords if keyword.arg == "shell"]
        if shell_keywords and not all(
            isinstance(value, ast.Constant) and isinstance(value.value, bool)
            for value in shell_keywords
        ):
            raise AuthorityClosureError(
                f"dynamic subprocess shell mode is forbidden: {source_path}"
            )
        shell_enabled = any(
            isinstance(value, ast.Constant) and value.value is True for value in shell_keywords
        )
        if shell_enabled:
            if not isinstance(command, ast.Constant) or not isinstance(command.value, str):
                raise AuthorityClosureError(f"dynamic shell subprocess is forbidden: {source_path}")
            for helper in _run_script_references(extraction_root, source_path, command.value):
                if not _is_measurement_subject(helper):
                    if not (extraction_root / helper).is_file():
                        raise AuthorityClosureError(
                            f"literal shell helper is unavailable: {source_path} -> {helper}"
                        )
                    helpers.add(helper)
            continue
        elements: list[ast.AST]
        if not isinstance(command, (ast.List, ast.Tuple)):
            if _command_token(command, static_bindings) is None:
                if source_path in DYNAMIC_EXTERNAL_SUBPROCESS_COMMAND_SOURCES:
                    continue
                bound_helpers = DYNAMIC_INTERNAL_SUBPROCESS_HELPERS.get(source_path)
                if bound_helpers:
                    for helper in bound_helpers:
                        if not (extraction_root / helper).is_file():
                            raise AuthorityClosureError(
                                f"bound subprocess wrapper helper is unavailable: "
                                f"{source_path} -> {helper}"
                            )
                        if not _is_measurement_subject(helper):
                            helpers.add(helper)
                    continue
                raise AuthorityClosureError(
                    f"dynamic subprocess command is forbidden: {source_path}"
                )
            elements = [command]
        else:
            elements = _expanded_command_elements(command, sequence_bindings)
        if isinstance(command, (ast.List, ast.Tuple)) and elements:
            program_index = _subprocess_program_index(elements, static_bindings)
            if program_index is None:
                raise AuthorityClosureError(
                    f"dynamic subprocess executable is forbidden: {source_path}"
                )
            program = _command_token(elements[program_index], static_bindings)
            assert program is not None
            if _is_python_command(program):
                argument_index = program_index + 1
                while argument_index < len(elements):
                    argument = _command_token(elements[argument_index], static_bindings)
                    if argument is None:
                        raise AuthorityClosureError(
                            f"dynamic Python subprocess target is forbidden: {source_path}"
                        )
                    if argument == "-m":
                        module_index = argument_index + 1
                        if module_index >= len(elements):
                            raise AuthorityClosureError(
                                f"Python subprocess -m target is unavailable: {source_path}"
                            )
                        module = _command_token(elements[module_index], static_bindings)
                        if module is None:
                            raise AuthorityClosureError(
                                f"dynamic Python subprocess module is forbidden: {source_path}"
                            )
                        if any(
                            module == prefix or module.startswith(f"{prefix}.")
                            for prefix in REPOSITORY_MODULE_PREFIXES
                        ):
                            helpers.add(_repository_module_runner_path(extraction_root, module))
                        break
                    if argument == "-c":
                        raise AuthorityClosureError(
                            f"Python subprocess -c is forbidden in authority closure: {source_path}"
                        )
                    if (
                        argument in PYTHON_OPTIONS_WITHOUT_VALUES
                        or argument.startswith(("-W", "-X"))
                        and argument not in PYTHON_OPTIONS_WITH_VALUES
                    ):
                        argument_index += 1
                        continue
                    if argument in PYTHON_OPTIONS_WITH_VALUES:
                        value_index = argument_index + 1
                        if (
                            value_index >= len(elements)
                            or _command_token(elements[value_index], static_bindings) is None
                        ):
                            raise AuthorityClosureError(
                                f"dynamic Python subprocess option is forbidden: {source_path}"
                            )
                        argument_index += 2
                        continue
                    if argument.startswith("-"):
                        raise AuthorityClosureError(
                            f"unknown Python subprocess option is forbidden: "
                            f"{source_path} -> {argument}"
                        )
                    break
                if argument_index >= len(elements):
                    continue
            elif Path(program).name in {"bash", "sh"}:
                script_index = program_index + 1
                while script_index < len(elements):
                    script = _command_token(elements[script_index], static_bindings)
                    if script is None:
                        raise AuthorityClosureError(
                            f"dynamic shell subprocess target is forbidden: {source_path}"
                        )
                    if script == "-c":
                        command_index = script_index + 1
                        if command_index >= len(elements):
                            raise AuthorityClosureError(
                                f"shell subprocess -c command is unavailable: {source_path}"
                            )
                        shell_command = _command_token(elements[command_index], static_bindings)
                        if shell_command is None:
                            raise AuthorityClosureError(
                                f"dynamic shell subprocess command is forbidden: {source_path}"
                            )
                        for helper in _run_script_references(
                            extraction_root, source_path, shell_command
                        ):
                            if not (extraction_root / helper).is_file():
                                raise AuthorityClosureError(
                                    f"literal shell helper is unavailable: "
                                    f"{source_path} -> {helper}"
                                )
                            if not _is_measurement_subject(helper):
                                helpers.add(helper)
                        break
                    if script in {
                        "--noprofile",
                        "--norc",
                        "-e",
                        "-eu",
                        "-eux",
                        "-u",
                        "-x",
                    }:
                        script_index += 1
                        continue
                    if script.startswith("-"):
                        raise AuthorityClosureError(
                            f"unknown shell subprocess option is forbidden: "
                            f"{source_path} -> {script}"
                        )
                    else:
                        break
        for element in elements:
            parts = _static_path_parts(element, static_bindings)
            if parts is None:
                continue
            literal = _strip_optional_dot_slash("/".join(part.strip("/") for part in parts if part))
            if not literal.endswith(EXECUTABLE_SUFFIXES):
                continue
            if not literal.startswith(("scripts/", "aragora/", ".github/")):
                continue
            target = extraction_root / literal
            if not target.is_file():
                raise AuthorityClosureError(
                    f"literal repository helper is unavailable: {source_path} -> {literal}"
                )
            if not _is_measurement_subject(literal):
                helpers.add(literal)
    return [(helper, "literal_subprocess_helper") for helper in sorted(helpers)]


def _strip_yaml_scalar(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    quote = value[0] if value[0] in {"'", '"'} else ""
    if quote:
        escaped = False
        for index in range(1, len(value)):
            character = value[index]
            if quote == '"' and character == "\\" and not escaped:
                escaped = True
                continue
            if character == quote and not escaped:
                return value[1:index]
            escaped = False
        raise AuthorityClosureError(f"unterminated YAML scalar: {raw!r}")
    return value.split(" #", 1)[0].strip()


def _workflow_structure(path: Path) -> dict[str, list[str]]:
    """Read only structural workflow/action keys needed by the closure."""
    lines = path.read_text(encoding="utf-8").splitlines()
    uses: list[str] = []
    runs: list[str] = []
    path_filters: list[str] = []
    path_ignores: list[str] = []
    key_stack: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.lstrip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        indent = len(raw) - len(stripped)
        while key_stack and key_stack[-1][0] >= indent:
            key_stack.pop()
        ancestor_keys = [key for _key_indent, key in key_stack]
        uses_match = re.match(r"(?:-\s*)?uses:\s*(.*)$", stripped)
        run_match = re.match(r"(?:-\s*)?run:\s*(.*)$", stripped)
        paths_match = re.match(r"(paths|paths-ignore):\s*(.*)$", stripped)
        paths_is_trigger_filter = (
            len(ancestor_keys) >= 2
            and ancestor_keys[0] in {"on", "'on'", '"on"'}
            and ancestor_keys[1] in {"push", "pull_request", "pull_request_target"}
        )
        if uses_match:
            uses.append(_strip_yaml_scalar(uses_match.group(1)))
        elif run_match:
            marker = run_match.group(1).strip()
            if marker in {"|", "|-", "|+", ">", ">-", ">+"}:
                block: list[str] = []
                index += 1
                while index < len(lines):
                    block_raw = lines[index]
                    block_stripped = block_raw.lstrip()
                    block_indent = len(block_raw) - len(block_stripped)
                    if block_stripped and block_indent <= indent:
                        index -= 1
                        break
                    block.append(block_raw[indent + 1 :] if len(block_raw) > indent else "")
                    index += 1
                runs.append("\n".join(block))
            else:
                runs.append(_strip_yaml_scalar(marker))
        elif paths_match and paths_is_trigger_filter:
            destination = path_filters if paths_match.group(1) == "paths" else path_ignores
            inline = paths_match.group(2).strip()
            if inline.startswith("[") and inline.endswith("]"):
                for item in inline[1:-1].split(","):
                    if item.strip():
                        destination.append(_strip_yaml_scalar(item))
            elif not inline:
                index += 1
                while index < len(lines):
                    item_raw = lines[index]
                    item_stripped = item_raw.lstrip()
                    item_indent = len(item_raw) - len(item_stripped)
                    if item_stripped and item_indent <= indent:
                        index -= 1
                        break
                    item_match = re.match(r"-\s*(.*)$", item_stripped)
                    if item_match:
                        destination.append(_strip_yaml_scalar(item_match.group(1)))
                    index += 1
        key_match = re.match(r"(?:-\s*)?([A-Za-z0-9_\"'-]+):\s*$", stripped)
        if key_match:
            key_stack.append((indent, key_match.group(1)))
        index += 1
    return {
        "path_filters": [value for value in path_filters if value],
        "path_ignores": [value for value in path_ignores if value],
        "runs": [value for value in runs if value],
        "uses": [value for value in uses if value],
    }


def _normalize_local_reference(source_path: str, raw_reference: str) -> str | None:
    reference = raw_reference.strip()
    if not reference.startswith("."):
        return None
    if "${{" in reference or "}}" in reference:
        raise AuthorityClosureError(
            f"dynamic local reference is forbidden: {source_path} -> {reference}"
        )
    if not reference.startswith("./"):
        raise AuthorityClosureError(
            f"path-traversing local reference is forbidden: {source_path} -> {reference}"
        )
    candidate = Path(reference[2:])
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AuthorityClosureError(
            f"path-traversing local reference is forbidden: {source_path} -> {reference}"
        )
    return candidate.as_posix()


def _resolve_local_uses(
    extraction_root: Path, source_path: str, reference: str
) -> tuple[str, str] | None:
    relative = _normalize_local_reference(source_path, reference)
    if relative is None:
        return None
    target = extraction_root / relative
    if target.is_dir():
        action_files = [
            candidate
            for candidate in (target / "action.yml", target / "action.yaml")
            if candidate.is_file()
        ]
        if len(action_files) != 1:
            raise AuthorityClosureError(
                f"local composite action must resolve to exactly one action file: "
                f"{source_path} -> {reference}"
            )
        resolved = action_files[0].relative_to(extraction_root).as_posix()
        return (resolved, "local_composite_action")
    if target.is_file() and target.suffix in WORKFLOW_SUFFIXES:
        return (relative, "local_reusable_workflow")
    raise AuthorityClosureError(
        f"unresolved local workflow/action reference: {source_path} -> {reference}"
    )


def _strip_optional_dot_slash(value: str) -> str:
    return value[2:] if value.startswith("./") else value


def _repository_module_runner_path(extraction_root: Path, module: str) -> str:
    module_path = Path(*module.split("."))
    file_candidate = extraction_root / module_path.with_suffix(".py")
    package_candidate = extraction_root / module_path / "__main__.py"
    candidates = [
        candidate.relative_to(extraction_root).as_posix()
        for candidate in (file_candidate, package_candidate)
        if candidate.is_file()
    ]
    if len(candidates) != 1:
        raise AuthorityClosureError(
            f"repository python -m target must resolve exactly once: {module}"
        )
    return candidates[0]


def _run_script_references(extraction_root: Path, source_path: str, run: str) -> list[str]:
    normalized = run.replace("\\\n", " ")
    dynamic_local_reference = re.compile(
        r"(?<![A-Za-z0-9_./-])"
        r"((?:\./)?(?:scripts|aragora|\.github)/[^\s;&|]*[$`*][^\s;&|]*)"
    )
    dynamic_match = dynamic_local_reference.search(normalized)
    if dynamic_match:
        raise AuthorityClosureError(
            f"dynamic local run target is forbidden: {source_path} -> {dynamic_match.group(1)}"
        )
    command_invocation = re.compile(
        r"(?:^[ \t]*|[;&|]\s*|\$\(\s*)"
        r"(?:python(?:3(?:\.\d+)?)?|bash|sh)\s+"
        r"(?:(?:-[A-Za-z]+\s+)*)"
        r"([^\s;&|]+)",
        re.MULTILINE,
    )
    for match in command_invocation.finditer(normalized):
        target = match.group(1).strip("\"'")
        local_target = _strip_optional_dot_slash(target).startswith(
            ("scripts/", "aragora/", ".github/")
        )
        dynamic_executable_glob = "*" in target and target.endswith(EXECUTABLE_SUFFIXES)
        if (
            target.startswith(("$", "`"))
            or (local_target and any(marker in target for marker in ("$", "`", "*")))
            or dynamic_executable_glob
        ):
            raise AuthorityClosureError(
                f"dynamic local run target is forbidden: {source_path} -> {target}"
            )
    module_invocation = re.compile(
        r"(?:^[ \t]*|[;&|]\s*|\$\(\s*)"
        r"python(?:3(?:\.\d+)?)?\s+(?:-[A-Za-z]+\s+)*-m\s+([^\s;&|)]+)",
        re.MULTILINE,
    )
    for match in module_invocation.finditer(normalized):
        module = match.group(1).strip("\"'")
        if any(marker in module for marker in ("$", "`", "*")):
            raise AuthorityClosureError(
                f"dynamic repository python -m target is forbidden: {source_path} -> {module}"
            )
    references = set(_literal_run_script_references(normalized))
    for match in module_invocation.finditer(normalized):
        module = match.group(1).strip("\"'")
        if any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in REPOSITORY_MODULE_PREFIXES
        ):
            references.add(_repository_module_runner_path(extraction_root, module))
    return sorted(references)


def _literal_run_script_references(run: str) -> list[str]:
    pattern = re.compile(
        r"(?<![A-Za-z0-9_./-])"
        r"((?:\./)?(?:scripts|aragora|\.github)/[A-Za-z0-9_./-]+\.(?:py|sh))"
        r"(?![A-Za-z0-9_.-])"
    )
    return sorted({_strip_optional_dot_slash(match.group(1)) for match in pattern.finditer(run)})


def _shell_helper_edges(extraction_root: Path, source_path: str) -> list[tuple[str, str]]:
    references = _run_script_references(
        extraction_root,
        source_path,
        (extraction_root / source_path).read_text(encoding="utf-8"),
    )
    edges: list[tuple[str, str]] = []
    for reference in references:
        target = extraction_root / reference
        if not target.is_file():
            raise AuthorityClosureError(
                f"unresolved shell helper reference: {source_path} -> {reference}"
            )
        if not _is_measurement_subject(reference):
            edges.append((reference, "literal_shell_helper"))
    return edges


def _is_measurement_subject(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in MEASUREMENT_SUBJECT_PREFIXES)


def _workflow_direct_matches(
    extraction_root: Path, workflow_path: str, authority_roots: Iterable[str]
) -> list[tuple[str, str]]:
    structure = _workflow_structure(extraction_root / workflow_path)
    matches: set[tuple[str, str]] = set()
    literal_run_references = set(_literal_run_script_references("\n".join(structure["runs"])))
    for root in authority_roots:
        selected_by_path = False
        for pattern in structure["path_filters"]:
            negative = pattern.startswith("!")
            candidate = pattern[1:] if negative else pattern
            if fnmatch.fnmatchcase(root, candidate):
                selected_by_path = not negative
        if selected_by_path:
            matches.add((root, "workflow_path_filter"))
        if structure["path_ignores"] and not any(
            fnmatch.fnmatchcase(root, pattern) for pattern in structure["path_ignores"]
        ):
            matches.add((root, "workflow_path_ignore"))
        if root in literal_run_references:
            matches.add((root, "workflow_run"))
    return sorted(matches)


def _workflow_member_edges(extraction_root: Path, source_path: str) -> list[tuple[str, str]]:
    structure = _workflow_structure(extraction_root / source_path)
    edges: set[tuple[str, str]] = set()
    for reference in structure["uses"]:
        resolved = _resolve_local_uses(extraction_root, source_path, reference)
        if resolved is not None:
            edges.add(resolved)
    for run in structure["runs"]:
        for reference in _run_script_references(extraction_root, source_path, run):
            target = extraction_root / reference
            if not target.is_file():
                raise AuthorityClosureError(
                    f"unresolved local executable reference: {source_path} -> {reference}"
                )
            if not _is_measurement_subject(reference):
                edges.add((reference, "workflow_run_executable"))
    return sorted(edges)


def _git_blob_binding(
    repo_root: Path, sha: str, extraction_root: Path, path: str
) -> dict[str, Any]:
    oid_proc = _run_git(
        repo_root,
        ["rev-parse", "--verify", f"{sha}:{path}"],
        check=False,
        text=True,
    )
    if oid_proc.returncode != 0:
        raise AuthorityClosureError(f"authority closure member missing at exact ref: {path}")
    oid = oid_proc.stdout.strip()
    type_proc = _run_git(repo_root, ["cat-file", "-t", oid], text=True)
    if type_proc.stdout.strip() != "blob":
        raise AuthorityClosureError(f"authority closure member is not a blob: {path}")
    blob = _run_git(repo_root, ["cat-file", "blob", oid]).stdout
    extracted_path = extraction_root / path
    if not extracted_path.is_file():
        raise AuthorityClosureError(f"authority closure member was not extracted: {path}")
    extracted = extracted_path.read_bytes()
    if extracted != blob:
        raise AuthorityClosureError(
            f"extracted authority member differs from exact-ref blob: {path}"
        )
    return {
        "byte_length": len(blob),
        "git_blob_oid": oid,
        "sha256": _sha256(blob),
    }


def _exact_ref_inventory_facts(
    repo_root: Path,
    sha: str,
    extraction_root: Path,
    *,
    external_artifacts: Iterable[Path] = (),
) -> dict[str, Any]:
    docs: dict[str, dict[str, Any]] = {}
    baseline_bindings: list[dict[str, Any]] = []
    for alias, (path, _keys) in BASELINE_SPECS.items():
        extracted = extraction_root / path
        if not extracted.is_file():
            raise AuthorityClosureError(f"exact-ref inventory baseline is unavailable: {path}")
        raw = extracted.read_bytes()
        try:
            docs[alias] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuthorityClosureError(f"invalid exact-ref inventory baseline: {path}") from exc
        baseline_bindings.append(
            {
                "alias": alias,
                "path": path,
                **_git_blob_binding(repo_root, sha, extraction_root, path),
            }
        )
    ids = collect_ids(docs)
    category_counts: dict[str, int] = defaultdict(int)
    for category in ids.values():
        category_counts[category] += 1

    inventory_path = extraction_root / DEFAULT_INVENTORY
    inventory_binding: dict[str, Any] | None = None
    inventory_counts: dict[str, int] = {}
    if inventory_path.is_file():
        inventory_doc = json.loads(inventory_path.read_bytes())
        items = inventory_doc.get("items", [])
        if not isinstance(items, list):
            raise AuthorityClosureError("exact-ref inventory has no items list")
        inventory_counts = {
            "items": len(items),
            "open": sum(
                1 for item in items if isinstance(item, dict) and item.get("status") == "open"
            ),
            "resolved": sum(
                1 for item in items if isinstance(item, dict) and item.get("status") == "resolved"
            ),
        }
        inventory_binding = {
            "path": DEFAULT_INVENTORY,
            **_git_blob_binding(repo_root, sha, extraction_root, DEFAULT_INVENTORY),
        }

    artifacts = [_canonical_external_artifact(path) for path in external_artifacts]
    artifact_names = [item["path"] for item in artifacts]
    if len(artifact_names) != len(set(artifact_names)):
        raise AuthorityClosureError("external artifact basenames must be unique")
    return {
        "baseline_bindings": sorted(baseline_bindings, key=lambda item: item["path"]),
        "baseline_category_counts": dict(sorted(category_counts.items())),
        "baseline_item_count": len(ids),
        "external_artifacts": sorted(artifacts, key=lambda item: item["path"]),
        "inventory_binding": inventory_binding,
        "inventory_counts": inventory_counts,
    }


def _canonical_external_artifact(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise AuthorityClosureError(f"external artifact has a UTF-8 BOM: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuthorityClosureError(f"external artifact is not UTF-8: {path}") from exc
    if not text.endswith("\n") or text.endswith("\n\n") or "\n" in text[:-1]:
        raise AuthorityClosureError(
            f"external artifact must contain compact JSON and exactly one terminal LF: {path}"
        )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AuthorityClosureError(f"external artifact is not valid JSON: {path}") from exc
    if raw != _canonical_json_bytes(parsed):
        raise AuthorityClosureError(f"external artifact bytes are noncanonical: {path}")
    return {
        "byte_length": len(raw),
        "canonical_bytes": True,
        "path": path.name,
        "sha256": _sha256(raw),
    }


def _loaded_repository_module_paths(policy: dict[str, Any]) -> set[str]:
    loaded: set[str] = set()
    for entry in policy.get("loaded_repository_modules", []):
        if not isinstance(entry, dict):
            raise AuthorityClosureError("exact-ref classifier returned malformed module evidence")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise AuthorityClosureError("exact-ref classifier returned an invalid module path")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise AuthorityClosureError(
                f"exact-ref classifier returned a non-repository module path: {relative}"
            )
        if path.suffix == ".py":
            loaded.add(path.as_posix())
    return loaded


def build_authority_manifest(
    repo_root: Path,
    ref: str,
    *,
    scratch_root: Path | None = None,
    interpreter: Path | None = None,
    external_artifacts: Iterable[Path] = (),
) -> dict[str, Any]:
    """Build the exact-ref, digest-bound executable authority closure."""
    repo_root = repo_root.resolve()
    sha = _resolve_full_commit(repo_root, ref)
    with tempfile.TemporaryDirectory(
        prefix="contract-drift-authority-",
        dir=str(scratch_root.resolve()) if scratch_root else None,
    ) as temporary:
        temporary_root = Path(temporary).resolve()
        extraction_root = temporary_root / "exact-ref"
        child_scratch = temporary_root / "child-scratch"
        _extract_exact_ref(repo_root, sha, extraction_root)
        policy = _invoke_exact_ref_policy(
            extraction_root,
            (),
            scratch_root=child_scratch,
            interpreter=interpreter,
        )
        authority_roots = tuple(policy.get("authority_roots", []))
        if len(authority_roots) != len(set(authority_roots)):
            raise AuthorityClosureError("canonical authority policy contains duplicate roots")
        if not authority_roots:
            raise AuthorityClosureError("canonical authority policy has no authority roots")

        incoming: dict[str, set[tuple[str, str]]] = defaultdict(set)
        roots = set(authority_roots)
        members: set[str] = set(roots)
        for root in authority_roots:
            if not (extraction_root / root).is_file():
                raise AuthorityClosureError(f"authority root is unavailable at exact ref: {root}")

        incoming[POLICY_PATH].add(
            ("scripts/generate_contract_drift_inventory.py", "exact_ref_canonical_policy")
        )
        incoming[MERGE_TRAIN_PATH].add(
            ("scripts/generate_contract_drift_inventory.py", "merge_train_mirror")
        )
        members.update((POLICY_PATH, MERGE_TRAIN_PATH))

        workflow_paths = sorted(
            path.relative_to(extraction_root).as_posix()
            for suffix in WORKFLOW_SUFFIXES
            for path in (extraction_root / ".github" / "workflows").rglob(f"*{suffix}")
            if path.is_file()
        )
        for workflow_path in workflow_paths:
            for root, kind in _workflow_direct_matches(
                extraction_root, workflow_path, authority_roots
            ):
                members.add(workflow_path)
                incoming[workflow_path].add((root, kind))

        queue = deque(sorted(members))
        scanned: set[str] = set()

        def scan_pending_members() -> None:
            while queue:
                source_path = queue.popleft()
                if source_path in scanned:
                    continue
                scanned.add(source_path)
                path = extraction_root / source_path
                edges: list[tuple[str, str]] = []
                if path.suffix == ".py":
                    edges.extend(_python_import_edges(extraction_root, source_path))
                    edges.extend(_subprocess_helper_edges(extraction_root, source_path))
                if path.suffix == ".sh":
                    edges.extend(_shell_helper_edges(extraction_root, source_path))
                if path.suffix in WORKFLOW_SUFFIXES:
                    edges.extend(_workflow_member_edges(extraction_root, source_path))
                for target, kind in edges:
                    if target == source_path or _is_measurement_subject(target):
                        continue
                    incoming[target].add((source_path, kind))
                    if target not in members:
                        members.add(target)
                        queue.append(target)

        while True:
            scan_pending_members()
            exact_policy = _invoke_exact_ref_policy(
                extraction_root,
                members,
                scratch_root=child_scratch,
                interpreter=interpreter,
            )
            loaded_paths = _loaded_repository_module_paths(exact_policy)
            loaded_dependencies = sorted(loaded_paths - members - {POLICY_PATH, MERGE_TRAIN_PATH})
            if not loaded_dependencies:
                break
            for target in loaded_dependencies:
                if _is_measurement_subject(target):
                    raise AuthorityClosureError(
                        f"exact-ref policy loaded a measurement subject: {target}"
                    )
                if not (extraction_root / target).is_file():
                    raise AuthorityClosureError(
                        f"exact-ref policy loaded an unavailable repository module: {target}"
                    )
                members.add(target)
                incoming[target].add((POLICY_PATH, "classifier_runtime_import"))
                queue.append(target)

        for member in sorted(members - roots):
            if not incoming[member]:
                raise AuthorityClosureError(f"unreachable authority closure member: {member}")

        classifications = {item["path"]: item for item in exact_policy.get("classifications", [])}
        if set(classifications) != members:
            raise AuthorityClosureError("exact-ref classifier omitted authority closure members")

        repo_files: list[dict[str, Any]] = []
        below_tier4: list[str] = []
        parity_failures: list[str] = []
        for member in sorted(members):
            classification = classifications[member]
            if classification.get("tier") != 4:
                below_tier4.append(member)
            if classification.get("matched_rule") != classification.get("merge_train_matched_rule"):
                parity_failures.append(member)
            repo_files.append(
                {
                    "authority_root": member in roots,
                    "incoming_edges": [
                        {"from": source, "kind": kind}
                        for source, kind in sorted(incoming.get(member, set()))
                    ],
                    "matched_rule": classification.get("matched_rule"),
                    "merge_train_matched_rule": classification.get("merge_train_matched_rule"),
                    "path": member,
                    "tier": classification.get("tier"),
                    **_git_blob_binding(repo_root, sha, extraction_root, member),
                }
            )
        if below_tier4:
            raise AuthorityClosureError(
                "authority closure members below Tier 4: " + ", ".join(below_tier4)
            )
        if parity_failures:
            raise AuthorityClosureError(
                "canonical classifier and merge-train mirror disagree: "
                + ", ".join(parity_failures)
            )

        inventory = _exact_ref_inventory_facts(
            repo_root,
            sha,
            extraction_root,
            external_artifacts=external_artifacts,
        )
        payload = {
            "authority_roots": list(authority_roots),
            "authority_tier": exact_policy["authority_tier"],
            "inventory": inventory,
            "policy": {
                "interpreter_flags": list(PYTHON_FLAGS),
                "loaded_repository_modules": exact_policy["loaded_repository_modules"],
                "module": POLICY_MODULE,
                "module_path": POLICY_PATH,
                "version": exact_policy["policy_version"],
            },
            "ref": sha,
            "repo_files": repo_files,
            "schema": AUTHORITY_MANIFEST_SCHEMA,
        }
        return {
            **payload,
            "authority_manifest_sha256": _sha256(_canonical_json_bytes(payload)),
        }


def classify_exact_ref_path(
    repo_root: Path,
    ref: str,
    changed_file: str,
    *,
    scratch_root: Path | None = None,
    interpreter: Path | None = None,
    external_artifacts: Iterable[Path] = (),
) -> dict[str, Any]:
    manifest = build_authority_manifest(
        repo_root,
        ref,
        scratch_root=scratch_root,
        interpreter=interpreter,
        external_artifacts=external_artifacts,
    )
    sha = manifest["ref"]
    with tempfile.TemporaryDirectory(
        prefix="contract-drift-classify-",
        dir=str(scratch_root.resolve()) if scratch_root else None,
    ) as temporary:
        temporary_root = Path(temporary).resolve()
        extraction_root = temporary_root / "exact-ref"
        _extract_exact_ref(repo_root.resolve(), sha, extraction_root)
        policy = _invoke_exact_ref_policy(
            extraction_root,
            (changed_file,),
            scratch_root=temporary_root / "child-scratch",
            interpreter=interpreter,
        )
    result = policy["classifications"][0]
    return {
        "authority_manifest_sha256": manifest["authority_manifest_sha256"],
        "changed_file": changed_file,
        "inventory": manifest["inventory"],
        "matched_rule": result["matched_rule"],
        "merge_train_matched_rule": result["merge_train_matched_rule"],
        "ref": sha,
        "schema": AUTHORITY_CLASSIFICATION_SCHEMA,
        "tier": result["tier"],
        "tier_name": result["tier_name"],
        "tier_reason": result["tier_reason"],
    }


def exact_ref_inventory(
    repo_root: Path,
    ref: str,
    *,
    scratch_root: Path | None = None,
    interpreter: Path | None = None,
    external_artifacts: Iterable[Path] = (),
) -> dict[str, Any]:
    manifest = build_authority_manifest(
        repo_root,
        ref,
        scratch_root=scratch_root,
        interpreter=interpreter,
        external_artifacts=external_artifacts,
    )
    return {
        "authority_manifest_sha256": manifest["authority_manifest_sha256"],
        "inventory": manifest["inventory"],
        "ref": manifest["ref"],
        "schema": EXACT_REF_INVENTORY_SCHEMA,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--cohort-commit", default=COHORT_COMMIT)
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--ref", default=None, help="Exact full commit SHA for read-only modes")
    parser.add_argument(
        "--authority-manifest",
        action="store_true",
        help="Emit the canonical exact-ref authority manifest",
    )
    parser.add_argument(
        "--classify-tier",
        action="store_true",
        help="Classify one --changed-file with the exact-ref canonical policy",
    )
    parser.add_argument("--changed-file", default=None)
    parser.add_argument("--json", action="store_true", help="Emit canonical compact JSON")
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--interpreter", type=Path, default=None)
    parser.add_argument("--cohort-artifact", type=Path, default=None)
    parser.add_argument("--sdk-provenance-artifact", type=Path, default=None)
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
    args = parser.parse_args()

    repo_root = args.repo_root
    external_artifacts = [
        path for path in (args.cohort_artifact, args.sdk_provenance_artifact) if path is not None
    ]
    read_only_modes = int(args.authority_manifest) + int(args.classify_tier)
    if read_only_modes > 1:
        print("ERROR: choose only one of --authority-manifest or --classify-tier")
        return 1
    if args.authority_manifest or args.classify_tier or args.ref is not None:
        if args.check or args.inventory is not None or args.provenance is not None:
            print("ERROR: inventory mutation/check options cannot be combined with --ref")
            return 1
        if args.ref is None:
            print("ERROR: exact-ref read-only modes require --ref")
            return 1
        if args.classify_tier != (args.changed_file is not None):
            print("ERROR: --classify-tier requires exactly one --changed-file")
            return 1
        try:
            if args.authority_manifest:
                output = build_authority_manifest(
                    repo_root,
                    args.ref,
                    scratch_root=args.scratch_root,
                    interpreter=args.interpreter,
                    external_artifacts=external_artifacts,
                )
            elif args.classify_tier:
                output = classify_exact_ref_path(
                    repo_root,
                    args.ref,
                    args.changed_file,
                    scratch_root=args.scratch_root,
                    interpreter=args.interpreter,
                    external_artifacts=external_artifacts,
                )
            else:
                output = exact_ref_inventory(
                    repo_root,
                    args.ref,
                    scratch_root=args.scratch_root,
                    interpreter=args.interpreter,
                    external_artifacts=external_artifacts,
                )
        except (
            AuthorityClosureError,
            FileNotFoundError,
            json.JSONDecodeError,
            OSError,
            subprocess.CalledProcessError,
        ) as exc:
            print(f"ERROR: {exc}")
            return 1
        sys.stdout.buffer.write(_canonical_json_bytes(output))
        return 0

    if args.changed_file is not None:
        print("ERROR: --changed-file is valid only with --classify-tier")
        return 1
    if args.scratch_root is not None or args.interpreter is not None or external_artifacts:
        print("ERROR: exact-ref options require --ref")
        return 1

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
