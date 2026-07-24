#!/usr/bin/env python3
"""Fail on new Boundary 2 receipts+verifier dependency violations.

Exit codes:
    0 -- no new violations
    1 -- one or more new forbidden edges or mirror mismatches
    2 -- invalid usage, map, baseline, or source input

Default mode is non-mutating. ``--freeze`` may shrink the baseline; initial
adoption or later growth additionally requires ``--adopt``.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP = REPO_ROOT / "scripts" / "ci" / "boundary_maps" / "receipts_verifier.json"
DEFAULT_BASELINE = REPO_ROOT / "scripts" / "baselines" / "boundary2_edges_baseline.json"


class CheckerError(RuntimeError):
    """A usage or configuration error that maps to exit code 2."""


@dataclass(frozen=True)
class BoundaryConfig:
    boundary_id: int
    boundary_name: str
    map_path: Path
    members: tuple[Path, ...]
    sources: tuple[Path, ...]
    allowed_internal_prefixes: tuple[str, ...]
    standalone_source: Path
    standalone_project_file: Path
    standalone_package_root: str
    allowed_external_roots: frozenset[str]
    mirror_pairs: tuple[tuple[Path, Path], ...]


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CheckerError(f"{label} does not exist: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckerError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CheckerError(f"{label} must contain a JSON object: {path}")
    return data


def _relative_path(root: Path, raw: object, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise CheckerError(f"{label} must be a non-empty relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise CheckerError(f"{label} must stay within the repository: {raw}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise CheckerError(f"{label} resolves outside the repository: {raw}") from exc
    return resolved


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CheckerError(f"{label} must be a non-empty JSON array")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise CheckerError(f"{label} entries must be non-empty strings")
    items = tuple(value)
    if len(items) != len(set(items)):
        raise CheckerError(f"{label} must not contain duplicate entries")
    return items


def load_boundary_map(repo_root: Path, map_path: Path) -> BoundaryConfig:
    root = repo_root.resolve()
    data = _load_json_object(map_path, "Boundary map")
    if data.get("schema_version") != 1:
        raise CheckerError("Boundary map schema_version must be 1")

    boundary = data.get("boundary")
    if not isinstance(boundary, dict):
        raise CheckerError("Boundary map must define a boundary object")
    if boundary.get("id") != 2 or boundary.get("name") != "receipts+verifier":
        raise CheckerError("This checker accepts only Boundary 2 receipts+verifier")
    if not isinstance(boundary.get("provenance"), str) or not boundary["provenance"]:
        raise CheckerError("Boundary map must record non-empty provenance")

    members = tuple(
        _relative_path(root, raw, f"members[{index}]")
        for index, raw in enumerate(_string_list(data.get("members"), "members"))
    )

    import_policy = data.get("python_import_policy")
    if not isinstance(import_policy, dict):
        raise CheckerError("Boundary map must define python_import_policy")
    sources = tuple(
        _relative_path(root, raw, f"python_import_policy.sources[{index}]")
        for index, raw in enumerate(
            _string_list(import_policy.get("sources"), "python_import_policy.sources")
        )
    )
    allowed_internal = _string_list(
        import_policy.get("allowed_internal_prefixes"),
        "python_import_policy.allowed_internal_prefixes",
    )
    if any(not prefix.startswith("aragora.") for prefix in allowed_internal):
        raise CheckerError("allowed_internal_prefixes entries must start with 'aragora.'")

    standalone = import_policy.get("standalone")
    if not isinstance(standalone, dict):
        raise CheckerError("python_import_policy must define standalone")
    standalone_source = _relative_path(
        root, standalone.get("source"), "python_import_policy.standalone.source"
    )
    standalone_project_file = _relative_path(
        root,
        standalone.get("project_file"),
        "python_import_policy.standalone.project_file",
    )
    package_root = standalone.get("package_root")
    if not isinstance(package_root, str) or not package_root.isidentifier():
        raise CheckerError("standalone.package_root must be a Python identifier")
    allowed_external = frozenset(
        _string_list(
            standalone.get("allowed_external_roots"),
            "python_import_policy.standalone.allowed_external_roots",
        )
    )
    if standalone_source not in sources:
        raise CheckerError("standalone.source must also appear in python_import_policy.sources")
    if not standalone_project_file.is_file():
        raise CheckerError(
            f"Standalone project file does not exist: {standalone_project_file.relative_to(root)}"
        )

    raw_pairs = data.get("mirror_pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise CheckerError("mirror_pairs must be a non-empty JSON array")
    mirror_pairs: list[tuple[Path, Path]] = []
    for index, pair in enumerate(raw_pairs):
        if not isinstance(pair, dict):
            raise CheckerError(f"mirror_pairs[{index}] must be a JSON object")
        left = _relative_path(root, pair.get("left"), f"mirror_pairs[{index}].left")
        right = _relative_path(root, pair.get("right"), f"mirror_pairs[{index}].right")
        mirror_pairs.append((left, right))

    for label, paths in (("member", members), ("source", sources)):
        for path in paths:
            if not path.exists():
                raise CheckerError(f"Boundary {label} does not exist: {path.relative_to(root)}")
    for left, right in mirror_pairs:
        if not left.is_file() or not right.is_file():
            raise CheckerError(
                "Mirror inputs must be files: "
                f"{left.relative_to(root)} and {right.relative_to(root)}"
            )

    return BoundaryConfig(
        boundary_id=2,
        boundary_name="receipts+verifier",
        map_path=map_path.resolve(),
        members=members,
        sources=sources,
        allowed_internal_prefixes=allowed_internal,
        standalone_source=standalone_source,
        standalone_project_file=standalone_project_file,
        standalone_package_root=package_root,
        allowed_external_roots=allowed_external,
        mirror_pairs=tuple(mirror_pairs),
    )


def _python_files(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix != ".py":
            raise CheckerError(f"Python source entry is not a .py file: {source}")
        return [source]
    if not source.is_dir():
        raise CheckerError(f"Python source entry is neither a file nor directory: {source}")
    return sorted(path for path in source.rglob("*.py") if "__pycache__" not in path.parts)


def _module_name(path: Path, repo_root: Path, config: BoundaryConfig) -> tuple[str, bool]:
    is_package = path.name == "__init__.py"
    if path.is_relative_to(config.standalone_source):
        relative = path.relative_to(config.standalone_source.parent).with_suffix("")
    else:
        relative = path.relative_to(repo_root).with_suffix("")
    parts = list(relative.parts)
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _resolve_relative_import(
    module_name: str,
    is_package: bool,
    level: int,
    imported_module: str | None,
    imported_names: list[str],
) -> list[str]:
    package_parts = module_name.split(".") if is_package else module_name.split(".")[:-1]
    keep = len(package_parts) - (level - 1)
    if keep < 1:
        return []
    base = package_parts[:keep]
    if imported_module:
        return [".".join([*base, *imported_module.split(".")])]
    return [".".join([*base, name]) for name in imported_names if name != "*"]


def _import_targets(tree: ast.AST, module_name: str, is_package: bool) -> set[str]:
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                targets.update(
                    _resolve_relative_import(
                        module_name,
                        is_package,
                        node.level,
                        node.module,
                        [alias.name for alias in node.names],
                    )
                )
            elif node.module:
                imported_names = [alias.name for alias in node.names if alias.name != "*"]
                if "." not in node.module and imported_names:
                    targets.update(f"{node.module}.{name}" for name in imported_names)
                else:
                    targets.add(node.module)
    return targets


def _prefix_allowed(target: str, prefixes: tuple[str, ...]) -> bool:
    return any(target == prefix or target.startswith(f"{prefix}.") for prefix in prefixes)


_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_toml_object(content: str, label: str) -> dict[str, Any]:
    try:
        import tomllib as tomllib_module
    except ModuleNotFoundError:
        try:
            import tomli as tomllib_module  # type: ignore[import-not-found,no-redef]
        except ModuleNotFoundError as exc:
            raise CheckerError(
                f"Reading {label} requires Python 3.11+ or the tomli package"
            ) from exc
    try:
        data = tomllib_module.loads(content)
    except ValueError as exc:
        raise CheckerError(f"Cannot parse {label}: {exc}") from exc
    if not isinstance(data, dict):
        raise CheckerError(f"{label} must contain a TOML object")
    return data


def _load_toml_object(path: Path, label: str) -> dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CheckerError(f"Cannot read {label} {path}: {exc}") from exc
    return _parse_toml_object(content, f"{label} {path}")


def _requirement_names(raw_dependencies: object, label: str) -> set[str]:
    if not isinstance(raw_dependencies, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw_dependencies
    ):
        raise CheckerError(f"{label} must be a TOML string array")

    dependencies: set[str] = set()
    for requirement in raw_dependencies:
        match = _REQUIREMENT_NAME.match(requirement)
        if not match:
            raise CheckerError(f"Cannot parse {label} requirement: {requirement!r}")
        dependencies.add(_normalize_distribution_name(match.group(1)))
    return dependencies


def _declared_project_dependencies_from_data(
    data: dict[str, Any],
    label: str,
) -> set[str]:
    project = data.get("project")
    if not isinstance(project, dict):
        raise CheckerError(f"{label} must define a [project] table")

    dependencies = _requirement_names(
        project.get("dependencies", []),
        f"{label} [project].dependencies",
    )
    optional_dependencies = project.get("optional-dependencies", {})
    if not isinstance(optional_dependencies, dict):
        raise CheckerError(f"{label} [project.optional-dependencies] must be a TOML table")
    for extra, requirements in optional_dependencies.items():
        if not isinstance(extra, str) or not extra:
            raise CheckerError(
                f"{label} [project.optional-dependencies] keys must be non-empty strings"
            )
        dependencies.update(
            _requirement_names(
                requirements,
                f"{label} [project.optional-dependencies].{extra}",
            )
        )
    return dependencies


def _declared_project_dependencies(path: Path) -> set[str]:
    return _declared_project_dependencies_from_data(
        _load_toml_object(path, "Standalone project file"),
        "Standalone project file",
    )


def _frozen_project_dependencies(
    repo_root: Path,
    config: BoundaryConfig,
    baseline_path: Path,
) -> set[str]:
    if not baseline_path.is_file():
        return set()
    baseline = _load_json_object(baseline_path, "Boundary baseline")
    frozen_ref = baseline.get("frozen_from_ref")
    if not isinstance(frozen_ref, str) or not frozen_ref:
        return set()
    try:
        project_path = config.standalone_project_file.relative_to(repo_root)
    except ValueError as exc:
        raise CheckerError("Standalone project file must stay within the repository") from exc
    result = subprocess.run(
        ["git", "show", f"{frozen_ref}:{project_path.as_posix()}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CheckerError(
            "Cannot read frozen standalone project file "
            f"{project_path} at {frozen_ref}: {result.stderr.strip()}"
        )
    data = _parse_toml_object(
        result.stdout,
        f"Frozen standalone project file {project_path} at {frozen_ref}",
    )
    return _declared_project_dependencies_from_data(
        data,
        f"Frozen standalone project file {project_path} at {frozen_ref}",
    )


def _scan_python_imports(repo_root: Path, config: BoundaryConfig) -> set[str]:
    violations: set[str] = set()
    seen: set[Path] = set()
    for source in config.sources:
        for path in _python_files(source):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            module_name, is_package = _module_name(path, repo_root, config)
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError, UnicodeError) as exc:
                raise CheckerError(f"Cannot parse Python source {path}: {exc}") from exc

            standalone = path.is_relative_to(config.standalone_source)
            for target in _import_targets(tree, module_name, is_package):
                if standalone:
                    root = target.split(".", 1)[0]
                    allowed = (
                        root == config.standalone_package_root
                        or root in sys.stdlib_module_names
                        or root in config.allowed_external_roots
                    )
                    if not allowed:
                        violations.add(f"offline {module_name} -> {target}")
                elif target == "aragora" or (
                    target.startswith("aragora.")
                    and not _prefix_allowed(target, config.allowed_internal_prefixes)
                ):
                    violations.add(f"import {module_name} -> {target}")
    return violations


def compute_violations(
    repo_root: Path,
    config: BoundaryConfig,
    *,
    grandfathered_dependencies: set[str] | None = None,
    baseline_violations: set[str] | None = None,
) -> set[str]:
    root = repo_root.resolve()
    violations = _scan_python_imports(root, config)
    allowed_dependencies = {
        _normalize_distribution_name(name) for name in config.allowed_external_roots
    }
    grandfathered = grandfathered_dependencies or set()
    existing_violations = baseline_violations or set()
    project_name = config.standalone_project_file.parent.name
    for dependency in _declared_project_dependencies(config.standalone_project_file):
        violation = f"offline dependency {project_name} -> {dependency}"
        if dependency not in allowed_dependencies and (
            dependency not in grandfathered or violation in existing_violations
        ):
            violations.add(violation)
    for left, right in config.mirror_pairs:
        try:
            matches = left.read_bytes() == right.read_bytes()
        except OSError as exc:
            raise CheckerError(f"Cannot compare schema mirrors: {exc}") from exc
        if not matches:
            violations.add(f"mirror {left.relative_to(root)} != {right.relative_to(root)}")
    return violations


def load_baseline(path: Path, config: BoundaryConfig, repo_root: Path) -> set[str]:
    data = _load_json_object(path, "Boundary baseline")
    if data.get("schema_version") != 1:
        raise CheckerError("Boundary baseline schema_version must be 1")
    boundary = data.get("boundary")
    if boundary != {
        "id": config.boundary_id,
        "name": config.boundary_name,
    }:
        raise CheckerError("Boundary baseline identity does not match the boundary map")
    map_value = data.get("map")
    try:
        expected_map = str(config.map_path.relative_to(repo_root.resolve()))
    except ValueError:
        expected_map = str(config.map_path)
    if map_value != expected_map:
        raise CheckerError(f"Boundary baseline map is {map_value!r}; expected {expected_map!r}")
    raw_violations = data.get("violations")
    if not isinstance(raw_violations, list) or any(
        not isinstance(item, str) or not item for item in raw_violations
    ):
        raise CheckerError("Boundary baseline violations must be a JSON string array")
    if len(raw_violations) != len(set(raw_violations)):
        raise CheckerError("Boundary baseline contains duplicate violations")
    return set(raw_violations)


def _git_ref(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def write_baseline(
    path: Path,
    violations: set[str],
    config: BoundaryConfig,
    repo_root: Path,
) -> None:
    root = repo_root.resolve()
    try:
        map_value = str(config.map_path.relative_to(root))
    except ValueError:
        map_value = str(config.map_path)
    data = {
        "schema_version": 1,
        "_comment": (
            "Shrink-only Boundary 2 receipts+verifier violations. New entries fail "
            "scripts/ci/check_boundary_edges.py; baseline growth requires the explicit "
            "--freeze --adopt command and review."
        ),
        "boundary": {
            "id": config.boundary_id,
            "name": config.boundary_name,
        },
        "map": map_value,
        "frozen_from_ref": _git_ref(root),
        "frozen_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "violations": sorted(violations),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check Boundary 2 edges against a shrink-only baseline."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Write the current set; growth also requires --adopt.",
    )
    parser.add_argument(
        "--adopt",
        action="store_true",
        help="With --freeze, explicitly permit initial creation or baseline growth.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON result.")
    return parser


def _run_freeze(
    args: argparse.Namespace,
    config: BoundaryConfig,
    current: set[str],
) -> int:
    if args.baseline.exists():
        existing = load_baseline(args.baseline, config, args.repo_root)
        added = current - existing
        if added and not args.adopt:
            for item in sorted(added):
                print(f"REFUSED (would grow baseline): {item}", file=sys.stderr)
            raise CheckerError(
                "--freeze would add violations to the shrink-only baseline; "
                "resolve them or use --freeze --adopt for an explicit re-adoption"
            )
    elif not args.adopt:
        raise CheckerError(
            "Initial baseline creation requires the explicit --freeze --adopt command"
        )
    write_baseline(args.baseline, current, config, args.repo_root)
    print(f"Froze {len(current)} Boundary 2 violation(s) -> {args.baseline}")
    return 0


def _emit_result(
    *,
    current: set[str],
    baseline: set[str],
    new: set[str],
    resolved: set[str],
    as_json: bool,
) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "ok": not new,
                    "current_violations": sorted(current),
                    "baseline_violations": sorted(baseline),
                    "new_violations": sorted(new),
                    "resolved_violations": sorted(resolved),
                },
                indent=2,
            )
        )
        return
    if new:
        print("FAIL: new Boundary 2 receipts+verifier violation(s) detected:")
        for item in sorted(new):
            print(f"  NEW {item}")
    else:
        print("OK: no new Boundary 2 receipts+verifier violations.")
        print(f"    baseline holds {len(baseline)} grandfathered violation(s).")
    if resolved:
        print(
            f"    {len(resolved)} baselined violation(s) are resolved; "
            "run --freeze to shrink the baseline."
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.adopt and not args.freeze:
        print("ERROR: --adopt is valid only with --freeze", file=sys.stderr)
        return 2
    try:
        repo_root = args.repo_root.resolve()
        map_path = args.map if args.map.is_absolute() else repo_root / args.map
        baseline_path = args.baseline if args.baseline.is_absolute() else repo_root / args.baseline
        args.map = map_path
        args.baseline = baseline_path
        args.repo_root = repo_root
        config = load_boundary_map(repo_root, map_path)
        baseline = (
            load_baseline(baseline_path, config, repo_root) if baseline_path.exists() else set()
        )
        current = compute_violations(
            repo_root,
            config,
            grandfathered_dependencies=_frozen_project_dependencies(
                repo_root,
                config,
                baseline_path,
            ),
            baseline_violations=baseline,
        )
        if args.freeze:
            return _run_freeze(args, config, current)
        if not baseline_path.exists():
            raise CheckerError(f"Boundary baseline does not exist: {baseline_path}")
    except CheckerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    new = current - baseline
    resolved = baseline - current
    _emit_result(
        current=current,
        baseline=baseline,
        new=new,
        resolved=resolved,
        as_json=args.json,
    )
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
