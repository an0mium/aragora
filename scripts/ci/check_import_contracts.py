#!/usr/bin/env python3
"""Shrink-only baseline checker for the Aragora import-linter layer contracts.

Purpose
-------
The repo's macro-architecture is enforced by ``.importlinter`` at the repo root,
which declares the five mission layers (interface > application > domain >
infrastructure > foundation). Because ``main`` currently has many upward imports
and 176 mutual cycles, the layer contracts cannot pass clean at adoption.

This wrapper runs import-linter (the same engine as the ``lint-imports`` CLI) and
compares the *current* set of layer violations against a frozen adoption baseline
at ``scripts/baselines/import_contracts_baseline.json``. It follows the same
shrink-only model as ``.mypy-baseline``:

  * exits 0 when the current violations are a subset of the baseline
    (clean origin/main, or violations resolved);
  * exits 1 on any NEW violation not present in the baseline (fail-on-new);
  * the baseline may only shrink -- ``--freeze`` refuses to add violations to an
    existing baseline unless ``--adopt`` is given.

Violations are recorded at layer-package granularity (``importer -> imported``,
e.g. ``aragora.config -> aragora.server``), which is robust to day-to-day churn
within an already-offending package pair while still catching a brand-new
upward import from a previously-clean package (e.g. the foundation package
``aragora.config`` importing the interface package ``aragora.server``).

Usage
-----
    python3 scripts/ci/check_import_contracts.py                  # check vs baseline
    python3 scripts/ci/check_import_contracts.py --layers foundation,infrastructure
    python3 scripts/ci/check_import_contracts.py --json
    python3 scripts/ci/check_import_contracts.py --freeze         # shrink baseline
    python3 scripts/ci/check_import_contracts.py --freeze --adopt # initial/grow freeze

Requirements
------------
Verified against ``import-linter==2.11`` + ``grimp==3.14``. The checker uses one
private import-linter symbol (``use_cases._register_contract_types``); access is
guarded by ``_resolve_register_contract_types`` so a version that renames or
removes it raises a clear CheckerError (exit 2) instead of an AttributeError. If
that error appears, pin import-linter to the verified version
(``python3 -m pip install 'import-linter==2.11'``).

Exit codes
----------
    0 -- no new violations (clean, or only resolved violations).
    1 -- one or more NEW violations (fail-on-new).
    2 -- a usage/environment error (missing config, missing baseline, import-linter
         not installed, an import-linter API/version mismatch, a layer-count
         mismatch between .importlinter and LAYER_ORDER, or a shrink-only
         violation on --freeze).
"""

from __future__ import annotations

import argparse
import configparser
import datetime as dt
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / ".importlinter"
# NOTE: literal filename below is greppable on purpose (VAL-P0-002 wiring check).
BASELINE_PATH = REPO_ROOT / "scripts" / "baselines" / "import_contracts_baseline.json"

# Canonical mission layers, highest first. This order MUST match the order of the
# layer lines declared in .importlinter so the positional mapping
# (layer line i -> LAYER_ORDER[i]) used by parse_layer_membership() holds.
LAYER_ORDER: tuple[str, ...] = (
    "interface",
    "application",
    "domain",
    "infrastructure",
    "foundation",
)
ROOT_PACKAGE = "aragora"


class CheckerError(RuntimeError):
    """Raised for usage/environment errors that map to exit code 2."""


# --- Config parsing ---------------------------------------------------------


def parse_layer_membership(config_path: Path) -> dict[str, str]:
    """Map each layered module to its canonical layer name by parsing .importlinter.

    Returns a dict keyed by BOTH the fully-qualified module name
    (``aragora.config``) and its tail (``config``), each mapping to one of
    LAYER_ORDER. Layer lines are read in declaration order and zipped against
    LAYER_ORDER positionally; the number of declared layer lines MUST equal
    len(LAYER_ORDER) or a CheckerError is raised (a mismatch would silently
    misalign the positional mapping and drop the extra layers).
    """
    parser = configparser.ConfigParser()
    # Section names contain colons (e.g. [importlinter:contract:aragora-layers]);
    # ConfigParser handles those fine. Values keep their colons verbatim.
    parser.read(config_path, encoding="utf-8")

    container = ROOT_PACKAGE
    layers_raw: str | None = None
    for section in parser.sections():
        if parser.get(section, "type", fallback="").strip() == "layers":
            layers_raw = parser.get(section, "layers", fallback="")
            containers = parser.get(section, "containers", fallback="").strip()
            if containers:
                container = containers.splitlines()[0].strip() or ROOT_PACKAGE
            break
    if layers_raw is None:
        raise CheckerError(f"No 'layers' contract found in {config_path}")

    layer_lines = [ln.strip() for ln in layers_raw.splitlines() if ln.strip()]
    if len(layer_lines) != len(LAYER_ORDER):
        raise CheckerError(
            f"{config_path} declares {len(layer_lines)} layer line(s) but the checker's "
            f"LAYER_ORDER expects exactly {len(LAYER_ORDER)} "
            f"({', '.join(LAYER_ORDER)}). The positional layer -> LAYER_ORDER mapping "
            "would be misaligned; update LAYER_ORDER and .importlinter together."
        )

    membership: dict[str, str] = {}
    for index, line in enumerate(layer_lines):
        layer_name = LAYER_ORDER[index]
        delimiter = ":" if ":" in line else "|"
        for raw_tail in line.split(delimiter):
            tail = raw_tail.strip().strip("()").strip()
            if not tail:
                continue
            full = f"{container}.{tail}"
            membership[full] = layer_name
            membership[tail] = layer_name
    return membership


def layer_of(module_full: str, membership: dict[str, str]) -> str | None:
    """Return the layer for a fully-qualified layer-package name, or None."""
    if module_full in membership:
        return membership[module_full]
    tail = module_full.split(".", 1)[1] if "." in module_full else module_full
    return membership.get(tail)


# --- Running import-linter --------------------------------------------------

# Verified against this import-linter version (see module docstring "Requirements").
_VERIFIED_IMPORT_LINTER = "2.11"


def _resolve_register_contract_types(use_cases_module: object) -> Callable[..., None]:
    """Return import-linter's contract-type registrar, or raise on a version mismatch.

    ``_register_contract_types`` is a private import-linter symbol. Accessing it
    through this guard converts a future-version AttributeError into a clear,
    actionable CheckerError instead of an opaque crash.
    """
    register = getattr(use_cases_module, "_register_contract_types", None)
    if not callable(register):
        raise CheckerError(
            "import-linter API mismatch: 'use_cases._register_contract_types' is "
            f"unavailable. This checker is verified against import-linter=={_VERIFIED_IMPORT_LINTER}; "
            f"pin it (python3 -m pip install 'import-linter=={_VERIFIED_IMPORT_LINTER}') or update "
            "scripts/ci/check_import_contracts.py for the installed import-linter version."
        )
    return register


def compute_current_violations(config_path: Path) -> dict[str, set[str]]:
    """Run import-linter and return {contract_name: {"importer -> imported", ...}}.

    The repo root (the directory containing .importlinter) is inserted at the
    front of sys.path so grimp resolves the LOCAL ``aragora`` package rather than
    an installed/editable copy or the SDK namespace package. Caching is disabled
    so a freshly-added (tamper) module is always picked up.
    """
    root = config_path.resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        import importlinter.api  # noqa: F401  (triggers configuration.configure())
        from importlinter.application import use_cases
    except ImportError as exc:  # pragma: no cover - exercised via validator install path
        raise CheckerError(
            "import-linter/grimp not installed; run: python3 -m pip install import-linter grimp"
        ) from exc

    user_options = use_cases.read_user_options(config_filename=str(config_path))
    _resolve_register_contract_types(use_cases)(user_options)
    report = use_cases.create_report(user_options, cache_dir=None)

    if report.could_not_run or report.invalid_contract_options:
        detail = "; ".join(
            f"{name}: {exc}" for name, exc in report.invalid_contract_options.items()
        )
        raise CheckerError(f"import-linter could not run: {detail or 'unknown error'}")

    violations: dict[str, set[str]] = {}
    for contract, check in report.get_contracts_and_checks():
        deps = check.metadata.get("invalid_dependencies", []) if check.metadata else []
        violations[contract.name] = {f"{dep['importer']} -> {dep['imported']}" for dep in deps}
    return violations


# --- Baseline I/O -----------------------------------------------------------


def load_baseline(path: Path) -> dict[str, set[str]]:
    """Load the baseline file into {contract_name: {violation, ...}}."""
    if not path.exists():
        raise CheckerError(
            f"Baseline {path} does not exist. Freeze it first with "
            "'python3 scripts/ci/check_import_contracts.py --freeze --adopt'."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckerError(f"Invalid JSON in {path}: {exc}") from exc
    contracts = data.get("contracts", {})
    return {name: set(entry.get("violations", [])) for name, entry in contracts.items()}


def _git_ref(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def write_baseline(path: Path, violations: dict[str, set[str]], config_path: Path) -> None:
    """Serialize the violation set to the baseline JSON (sorted, deterministic)."""
    data = {
        "_comment": (
            "Shrink-only baseline of import-linter layer-contract violations "
            "(fail-on-new). Frozen from origin/main by "
            "scripts/ci/check_import_contracts.py --freeze. New violations fail the "
            "checker; this baseline may only shrink. Layer membership was "
            "cross-checked against aragora/module_tiers.yaml (see .importlinter)."
        ),
        "config": config_path.name,
        "frozen_from_ref": _git_ref(config_path.resolve().parent),
        "frozen_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "layers": list(LAYER_ORDER),
        "contracts": {
            name: {"violations": sorted(items)} for name, items in sorted(violations.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# --- Diff / filter ----------------------------------------------------------


def diff_against_baseline(
    current: dict[str, set[str]], baseline: dict[str, set[str]]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return (new_violations, resolved_violations) per contract.

    Iterates the UNION of current and baseline contract names so a contract that
    is present in the baseline but absent from ``current`` (e.g. renamed) still
    has its baselined violations accounted for as resolved rather than silently
    dropped.
    """
    new: dict[str, set[str]] = {}
    resolved: dict[str, set[str]] = {}
    for name in current.keys() | baseline.keys():
        cur = current.get(name, set())
        base = baseline.get(name, set())
        new[name] = cur - base
        resolved[name] = base - cur
    return new, resolved


def filter_by_layers(
    violations: set[str], selected: set[str], membership: dict[str, str]
) -> set[str]:
    """Keep only ``importer -> imported`` whose importer is in a selected layer."""
    kept: set[str] = set()
    for violation in violations:
        importer = violation.split(" -> ", 1)[0]
        if layer_of(importer, membership) in selected:
            kept.add(violation)
    return kept


# --- CLI --------------------------------------------------------------------


def _parse_layers_arg(raw: str) -> set[str]:
    selected = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = selected - set(LAYER_ORDER)
    if unknown:
        raise CheckerError(
            f"Unknown layer(s): {', '.join(sorted(unknown))}. "
            f"Valid layers: {', '.join(LAYER_ORDER)}."
        )
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the import-linter layer contracts (lint-imports) and compare "
            "violations against the shrink-only baseline "
            "scripts/baselines/import_contracts_baseline.json. Exits non-zero on "
            "any NEW layer violation (fail-on-new)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the .importlinter config (default: repo-root .importlinter).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_PATH,
        help=(
            "Path to import_contracts_baseline.json "
            "(default: scripts/baselines/import_contracts_baseline.json)."
        ),
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="",
        help=(
            "Comma-separated layer names to restrict the check to, e.g. "
            "--layers foundation,infrastructure. Only violations whose IMPORTING "
            "module belongs to one of these layers are considered for fail-on-new. "
            f"Valid layers (highest to lowest): {', '.join(LAYER_ORDER)}."
        ),
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help=(
            "Recompute current violations and write them to the baseline. "
            "Refuses to ADD violations to an existing baseline unless --adopt is "
            "given (shrink-only guard)."
        ),
    )
    parser.add_argument(
        "--adopt",
        action="store_true",
        help="With --freeze, permit the baseline to grow (initial adoption only).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON summary to stdout.",
    )
    return parser


def _run_freeze(args: argparse.Namespace) -> int:
    current = compute_current_violations(args.config)
    if args.baseline.exists() and not args.adopt:
        existing = load_baseline(args.baseline)
        new, _ = diff_against_baseline(current, existing)
        added = {n: v for n, v in new.items() if v}
        if added:
            for name, items in added.items():
                for item in sorted(items):
                    print(f"REFUSED (would grow baseline): {name}: {item}", file=sys.stderr)
            raise CheckerError(
                "--freeze would ADD new violations to the baseline (shrink-only). "
                "Resolve them, or pass --adopt for an intentional re-adoption."
            )
    write_baseline(args.baseline, current, args.config)
    total = sum(len(v) for v in current.values())
    print(f"Froze {total} violation(s) across {len(current)} contract(s) -> {args.baseline}")
    return 0


def _emit_json(
    new: dict[str, set[str]],
    resolved: dict[str, set[str]],
    selected: set[str],
    has_new: bool,
) -> None:
    payload = {
        "ok": not has_new,
        "layers_filter": sorted(selected),
        "new_violations": {n: sorted(v) for n, v in new.items() if v},
        "resolved_violations": {n: sorted(v) for n, v in resolved.items() if v},
    }
    print(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.freeze:
            return _run_freeze(args)

        selected = _parse_layers_arg(args.layers) if args.layers else set(LAYER_ORDER)
        membership = parse_layer_membership(args.config)
        current = compute_current_violations(args.config)
        baseline = load_baseline(args.baseline)
        new, resolved = diff_against_baseline(current, baseline)

        if args.layers:
            new = {n: filter_by_layers(v, selected, membership) for n, v in new.items()}
            resolved = {n: filter_by_layers(v, selected, membership) for n, v in resolved.items()}
    except CheckerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    has_new = any(v for v in new.values())

    if args.json:
        _emit_json(new, resolved, selected, has_new)
    else:
        scope = f" (layers: {', '.join(sorted(selected))})" if args.layers else ""
        if has_new:
            print(f"FAIL: NEW import-linter layer violation(s) detected{scope}:")
            for name, items in new.items():
                for item in sorted(items):
                    importer = item.split(" -> ", 1)[0]
                    print(f"  NEW [{layer_of(importer, membership)}] {item}  ({name})")
            print(
                "\nThese imports are not in the frozen baseline. Either remove the "
                "upward import, or (only if intentional) re-freeze with "
                "scripts/ci/check_import_contracts.py --freeze."
            )
        else:
            total_base = sum(len(v) for v in baseline.values())
            print(f"OK: no new import-linter layer violations{scope}.")
            print(f"     baseline holds {total_base} grandfathered violation(s).")
        resolved_total = sum(len(v) for v in resolved.values())
        if resolved_total:
            print(
                f"     {resolved_total} baselined violation(s) now resolved; "
                "run --freeze to shrink the baseline."
            )

    return 1 if has_new else 0


if __name__ == "__main__":
    sys.exit(main())
