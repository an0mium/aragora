"""
Migrate legacy Gastown workspace/hook state into the canonical store.

This helper is intentionally conservative:
- Default is dry-run (prints actions without writing).
- When applied, merges workspace/rig state and hook metadata by ID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aragora.nomic.stores.paths import resolve_store_dir


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _merge_by_id(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {
        item.get("id"): item for item in existing if item.get("id")
    }
    for item in incoming:
        item_id = item.get("id")
        if not item_id:
            continue
        merged[item_id] = item
    return list(merged.values())


def _find_legacy_roots() -> list[Path]:
    candidates = []
    for raw in [
        ".gt",
        ".gastown",
        ".nomic/gastown",
        "~/.gt",
        "~/.gastown",
        "~/.aragora/gastown",
    ]:
        candidates.append(Path(raw).expanduser())
    seen: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate not in seen:
            seen.append(candidate)
    return seen


def _detect_layout(root: Path) -> str | None:
    if (root / "workspaces").exists() or (root / "hooks").exists():
        return "coordinator"
    if (root / "state.json").exists() or (root / "hooks.json").exists():
        return "workspace"
    return None


def _describe_root(root: Path, layout: str | None) -> None:
    label = layout or "unknown"
    print(f"- {root} ({label})")
    if layout == "coordinator":
        paths = [
            root / "workspaces" / "state.json",
            root / "hooks" / "hooks.json",
            root / "ledger.json",
        ]
    else:
        paths = [
            root / "state.json",
            root / "hooks.json",
            root / "ledger.json",
        ]
    for path in paths:
        if path.exists():
            print(f"  • found: {path}")


def _migrate_workspace_state(
    source_state: Path,
    target_state: Path,
    apply: bool,
) -> None:
    source_payload = _load_json(source_state) or {}
    target_payload = _load_json(target_state) or {}
    source_workspaces = source_payload.get("workspaces", [])
    source_rigs = source_payload.get("rigs", [])
    target_workspaces = target_payload.get("workspaces", [])
    target_rigs = target_payload.get("rigs", [])

    merged_workspaces = _merge_by_id(target_workspaces, source_workspaces)
    merged_rigs = _merge_by_id(target_rigs, source_rigs)

    if not apply:
        print(f"Would merge workspace state {source_state} -> {target_state}")
        return

    target_state.parent.mkdir(parents=True, exist_ok=True)
    target_state.write_text(
        json.dumps({"workspaces": merged_workspaces, "rigs": merged_rigs}, indent=2)
    )


def _migrate_hooks(
    source_hooks: Path,
    target_hooks: Path,
    apply: bool,
) -> None:
    source_payload = _load_json(source_hooks) or {}
    target_payload = _load_json(target_hooks) or {}
    source_hooks_list = source_payload.get("hooks", [])
    target_hooks_list = target_payload.get("hooks", [])
    merged_hooks = _merge_by_id(target_hooks_list, source_hooks_list)

    if not apply:
        print(f"Would merge hooks {source_hooks} -> {target_hooks}")
        return

    target_hooks.parent.mkdir(parents=True, exist_ok=True)
    target_hooks.write_text(json.dumps({"hooks": merged_hooks}, indent=2))


def _migrate_ledger(
    source_ledger: Path,
    target_ledger: Path,
    apply: bool,
) -> None:
    source_payload = _load_json(source_ledger) or {}
    target_payload = _load_json(target_ledger) or {}
    source_entries = source_payload.get("entries", [])
    target_entries = target_payload.get("entries", [])
    merged_entries = _merge_by_id(target_entries, source_entries)

    if not apply:
        print(f"Would merge ledger {source_ledger} -> {target_ledger}")
        return

    target_ledger.parent.mkdir(parents=True, exist_ok=True)
    target_ledger.write_text(json.dumps({"entries": merged_entries}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="source", help="Legacy gastown root dir")
    parser.add_argument("--to", dest="target", help="Target canonical store dir")
    parser.add_argument(
        "--mode",
        choices=["workspace", "coordinator"],
        default="workspace",
        help="Target layout for migration",
    )
    parser.add_argument("--apply", action="store_true", help="Apply changes")
    args = parser.parse_args()

    roots = [Path(args.source).expanduser()] if args.source else _find_legacy_roots()
    roots = [root for root in roots if root.exists()]

    if not roots:
        print("No legacy Gastown state found.")
        return

    target_base = Path(args.target).expanduser() if args.target else resolve_store_dir()

    print(f"Found {len(roots)} legacy root(s):")
    for root in roots:
        layout = _detect_layout(root)
        _describe_root(root, layout)
        if layout is None:
            print("  • skipped (unknown layout)")
            continue

        if args.mode == "coordinator":
            workspace_state = root / "workspaces" / "state.json"
            if not workspace_state.exists():
                workspace_state = root / "state.json"
            target_workspace = target_base / "workspaces" / "state.json"
            _migrate_workspace_state(workspace_state, target_workspace, args.apply)

            hooks_state = root / "hooks" / "hooks.json"
            if not hooks_state.exists():
                hooks_state = root / "hooks.json"
            target_hooks = target_base / "hooks" / "hooks.json"
            _migrate_hooks(hooks_state, target_hooks, args.apply)

            ledger_state = root / "ledger.json"
            if ledger_state.exists():
                target_ledger = target_base / "ledger.json"
                _migrate_ledger(ledger_state, target_ledger, args.apply)
            elif (root / "ledger.json").exists():
                print("  • ledger.json exists but was not migrated")
        else:
            workspace_state = root / "state.json"
            target_workspace = target_base / "state.json"
            _migrate_workspace_state(workspace_state, target_workspace, args.apply)

            hooks_state = root / "hooks.json"
            target_hooks = target_base / "hooks.json"
            _migrate_hooks(hooks_state, target_hooks, args.apply)

        if not args.apply:
            print("Run with --apply to perform migration.")


if __name__ == "__main__":
    main()
