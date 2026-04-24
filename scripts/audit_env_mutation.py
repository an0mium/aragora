#!/usr/bin/env python3
"""Audit ``os.environ`` mutation inside concurrent feature surfaces.

Writing to ``os.environ`` from a module that runs in a concurrent request
path or feature-gated epistemic surface shares credentials / config as
mutable global state across requests. The panel flagged this pattern on
#6454, #6447, #6442, and #6472 as a correctness defect.

This script flags new occurrences under the default target paths and
fails CI when one appears. Known intentional or legacy mutations are
listed in ``_ALLOWLIST`` with a short rationale so future readers can see
why they aren't caught.

Pattern detection uses AST so comments and string literals can't
trigger false positives and nested attribute access (``os.environ[x]
= ...``) is matched regardless of whether ``os`` is imported at
module scope or inside a function.

Usage:
    python scripts/audit_env_mutation.py                # scan default targets
    python scripts/audit_env_mutation.py PATH [PATH...]  # scan given files
    python scripts/audit_env_mutation.py --json          # machine-readable
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = [
    REPO_ROOT / "aragora" / "server" / "handlers",
    REPO_ROOT / "aragora" / "epistemic",
]

# Files where an ``os.environ[...] = ...`` (or equivalent) is the
# current product behavior and should not fail the audit. Add with a
# short rationale. Prefer repairing over adding entries here.
_ALLOWLIST: dict[str, str] = {
    # Feature flags in Aragora are stored in ``os.environ`` by design.
    # The admin collection PUT handler writes them after two-phase
    # validation so partial-update state is not possible (#6447).
    "aragora/server/handlers/admin/feature_flags.py": (
        "feature flags are env-stored by design; writes are post-validation atomic"
    ),
    # Context-budget admin endpoint writes two config keys into env;
    # flagged for a future repair to the non-mutating per-request
    # pattern established in #6454. Tracked by panel review.
    "aragora/server/handlers/context_budget.py": (
        "legacy env-based config write; TODO repair to non-mutating pattern"
    ),
    # Existing DIC-17 bridge helper writes an opt-in flag globally. New
    # epistemic helpers should follow the non-mutating override/config pattern
    # used by DIC-19 proof-unit scanning instead of adding more entries here.
    "aragora/epistemic/__init__.py": (
        "legacy DIC-17 env opt-in helper; keep until replaced by injected config"
    ),
    "aragora/epistemic/crux_receipt.py": (
        "legacy test/process opt-in helper; future DIC repairs should avoid env writes"
    ),
    "aragora/epistemic/repair.py": (
        "legacy test/process opt-in helper; future DIC repairs should avoid env writes"
    ),
}

_ENVIRON_ATTR = "environ"
_ENVIRON_MUTATING_METHODS: frozenset[str] = frozenset(
    {"update", "setdefault", "pop", "clear", "popitem", "__setitem__", "__delitem__"}
)


class _EnvMutationVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[tuple[int, str]] = []

    # os.environ[key] = value
    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if _is_environ_subscript(target):
                self.findings.append((node.lineno, "os.environ[...] = ..."))
        self.generic_visit(node)

    # del os.environ[key]
    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if _is_environ_subscript(target):
                self.findings.append((node.lineno, "del os.environ[...]"))
        self.generic_visit(node)

    # os.environ.update(...), .setdefault(...), .pop(...), .clear(), etc.
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in _ENVIRON_MUTATING_METHODS
            and _is_environ_ref(func.value)
        ):
            self.findings.append((node.lineno, f"os.environ.{func.attr}(...)"))
        self.generic_visit(node)


def _is_environ_ref(node: ast.AST) -> bool:
    """True if the node refers to ``os.environ`` (attribute access)."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == _ENVIRON_ATTR
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _is_environ_subscript(node: ast.AST) -> bool:
    """True if the node is a subscript of ``os.environ``."""
    return isinstance(node, ast.Subscript) and _is_environ_ref(node.value)


def scan_file(path: Path) -> list[tuple[int, str]]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    visitor = _EnvMutationVisitor(path)
    visitor.visit(tree)
    return visitor.findings


def iter_targets(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
    return files


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to scan (default: aragora/server/handlers/).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON findings on stdout.")
    args = parser.parse_args()

    targets = iter_targets(args.paths or DEFAULT_TARGETS)

    all_findings: list[dict[str, object]] = []
    for path in targets:
        rel = relative(path)
        if rel in _ALLOWLIST:
            continue
        for line, description in scan_file(path):
            all_findings.append({"file": rel, "line": line, "pattern": description})

    if args.json:
        print(json.dumps({"findings": all_findings, "allowlist": _ALLOWLIST}, indent=2))
    else:
        if all_findings:
            print("env-mutation audit: violations found (concurrent-surface os.environ writes)")
            print("")
            for finding in all_findings:
                print(f"  {finding['file']}:{finding['line']}  {finding['pattern']}")
            print("")
            print("If this mutation is intentional product behavior, add the file to the")
            print("_ALLOWLIST in scripts/audit_env_mutation.py with a short rationale.")
            print("If the mutation is an accidental leak of config state into a concurrent")
            print("request or feature path, repair to a non-mutating per-call/config")
            print("pattern (see #6454 and #6472 for worked examples).")
        else:
            print("env-mutation audit: no new violations.")

    return 1 if all_findings else 0


if __name__ == "__main__":
    sys.exit(main())
