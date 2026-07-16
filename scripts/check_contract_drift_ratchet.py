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

- ``pr``: strict non-worsening delta ratchet. Compares the five baseline-file
  counts at HEAD against the merge base (``--base-ref``); FAILS if any count
  increased or any integrity condition holds, PASSES on equal-or-lower counts
  even while the program schedule is red. The program status is still
  reported (informational) so PR authors see the burn-down state.
"""

from __future__ import annotations

import argparse
import json
import math
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
        append_only = _append_only_issues(base_items, head_items)
        integrity_issues.extend(append_only)
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
        result["pr_delta"] = {
            "base_ref": base_ref,
            "counts": deltas,
            "increased": increased,
        }
        result["passing"] = not increased and not integrity_issues
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
    parser.add_argument("--mode", choices=("program", "pr"), default="program")
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
