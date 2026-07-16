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
import json
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
        if not item.get("provenance"):
            issues.append(f"Inventory item missing provenance: {item_id}")
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
        except ValueError:
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
    inventory_path = args.inventory or repo_root / DEFAULT_INVENTORY
    as_of = date.fromisoformat(args.as_of)

    if args.provenance is not None and not PROVENANCE_REF.search(args.provenance):
        print("ERROR: --provenance must include a PR/issue reference (#123 or URL)")
        return 1

    try:
        current_ids = collect_ids(load_working_docs(repo_root))
        cohort_ids = collect_ids(load_git_docs(repo_root, args.cohort_commit))
    except (ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}")
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
