#!/usr/bin/env python3
"""Plan the phase-1 typecheck gate for changed Aragora Python files."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

PlanMode = Literal["skip", "changed", "full"]

FORCE_FULL_EXACT_PATHS = {
    "pyproject.toml",
    ".github/workflows/lint.yml",
    "scripts/run_typecheck_gate.py",
    "scripts/test_tiers.sh",
}
FORCE_FULL_PREFIXES = (".github/actions/pr-scope-classifier/",)


@dataclass(frozen=True)
class TypecheckPlan:
    mode: PlanMode
    targets: list[str]
    changed_files: list[str]
    reasons: list[str]

    @property
    def summary(self) -> str:
        if self.mode == "changed":
            return f"Run mypy on {len(self.targets)} touched aragora Python file(s)."
        if self.mode == "full":
            return "Run the full typecheck tier due to config or structural changes."
        return "Skip typecheck: no touched aragora Python files require the phase-1 gate."


def _normalize_paths(paths: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        value = raw.strip().replace("\\", "/")
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _is_requirement_file(path: str) -> bool:
    name = Path(path).name
    return name.startswith("requirements") and name.endswith(".txt")


def _forces_full_typecheck(path: str) -> bool:
    if path in FORCE_FULL_EXACT_PATHS or _is_requirement_file(path):
        return True
    return any(path.startswith(prefix) for prefix in FORCE_FULL_PREFIXES)


def _is_aragora_python_target(path: str) -> bool:
    return path.startswith("aragora/") and path.endswith(".py")


def build_typecheck_plan(*, repo_root: Path, changed_files: list[str]) -> TypecheckPlan:
    normalized = _normalize_paths(changed_files)
    reasons: list[str] = []

    force_full_paths = [path for path in normalized if _forces_full_typecheck(path)]
    if force_full_paths:
        reasons.extend(f"force_full:{path}" for path in force_full_paths)
        return TypecheckPlan(
            mode="full",
            targets=[],
            changed_files=normalized,
            reasons=reasons,
        )

    deleted_targets = [
        path
        for path in normalized
        if _is_aragora_python_target(path) and not (repo_root / path).exists()
    ]
    if deleted_targets:
        reasons.extend(f"deleted_target:{path}" for path in deleted_targets)
        return TypecheckPlan(
            mode="full",
            targets=[],
            changed_files=normalized,
            reasons=reasons,
        )

    targets = [
        path
        for path in normalized
        if _is_aragora_python_target(path) and (repo_root / path).exists()
    ]
    if targets:
        reasons.extend(f"target:{path}" for path in targets)
        return TypecheckPlan(
            mode="changed",
            targets=targets,
            changed_files=normalized,
            reasons=reasons,
        )

    reasons.append("no_aragora_python_targets")
    return TypecheckPlan(
        mode="skip",
        targets=[],
        changed_files=normalized,
        reasons=reasons,
    )


def get_changed_files(*, repo_root: Path, base_ref: str, head_ref: str = "HEAD") -> list[str]:
    base = base_ref if base_ref.startswith("origin/") else f"origin/{base_ref}"
    proc = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head_ref}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def write_github_outputs(*, plan: TypecheckPlan, output_path: Path) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(f"mode={plan.mode}\n")
        handle.write(f"target_count={len(plan.targets)}\n")
        handle.write(f"summary={plan.summary}\n")
        handle.write("reasons<<EOF\n")
        handle.write("\n".join(plan.reasons))
        handle.write("\nEOF\n")


def write_targets_file(*, plan: TypecheckPlan, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(plan.targets)
    output_path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan the truthful phase-1 typecheck gate for changed Aragora Python files.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current working directory).",
    )
    parser.add_argument(
        "--base-ref",
        help="Base git ref for diff planning (example: main). Required unless --files is supplied.",
    )
    parser.add_argument(
        "--head-ref",
        default="HEAD",
        help="Head git ref for diff planning (default: HEAD).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Explicit changed files to classify instead of querying git.",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        help="Optional GitHub Actions output file to append plan metadata to.",
    )
    parser.add_argument(
        "--targets-file",
        type=Path,
        help="Optional file path for newline-delimited mypy targets.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the computed plan as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()

    if args.files is not None:
        changed_files = list(args.files)
    else:
        if not args.base_ref:
            raise SystemExit("--base-ref is required unless --files is provided.")
        changed_files = get_changed_files(
            repo_root=repo_root,
            base_ref=args.base_ref,
            head_ref=args.head_ref,
        )

    plan = build_typecheck_plan(repo_root=repo_root, changed_files=changed_files)

    if args.github_output is not None:
        write_github_outputs(plan=plan, output_path=args.github_output)
    if args.targets_file is not None:
        write_targets_file(plan=plan, output_path=args.targets_file)

    if args.json:
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))
    else:
        print(plan.summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
