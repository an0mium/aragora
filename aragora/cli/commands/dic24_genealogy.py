"""CLI command: ``aragora genealogy show``.

DIC-24 operator surface for the epistemic genealogy ledger (issue #6218).

Reads a JSONL store file where each line encodes one GenealogyEntry:
    {"code_unit_id": "...", "entry_kind": "...", "entry_id": "...",
     "checksum": "...", "timestamp": "...", "metadata": {...}}

Flag: ``ARAGORA_GENEALOGY_ENABLED`` (default OFF).
Live queue effect: none — read-only operator surface.
Advances: issue #6218 (DIC-24).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from aragora.epistemic.genealogy import GenealogyEntry, InMemoryGenealogyStore, get_genealogy

logger = logging.getLogger(__name__)

_FLAG = "ARAGORA_GENEALOGY_ENABLED"
_DEFAULT_STORE = ".aragora_genealogy.jsonl"


def _flag_enabled() -> bool:
    return os.environ.get(_FLAG, "").lower() in {"1", "true", "yes", "on"}


def _load_store(path: Path) -> InMemoryGenealogyStore:
    """Parse *path* line-by-line into an InMemoryGenealogyStore.

    Malformed lines are logged at WARNING and skipped.
    """
    store = InMemoryGenealogyStore()
    if not path.exists():
        return store
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj: dict[str, Any] = json.loads(raw)
            entry = GenealogyEntry(
                entry_kind=obj["entry_kind"],  # type: ignore[arg-type]
                entry_id=str(obj["entry_id"]),
                checksum=str(obj["checksum"]),
                timestamp=str(obj["timestamp"]),
                metadata=dict(obj.get("metadata") or {}),
            )
            store.add(str(obj["code_unit_id"]), entry)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("genealogy store line %d skipped: %s", lineno, exc)
    return store


def cmd_genealogy_show(args: argparse.Namespace) -> int:
    """Show lineage for one proof-carrying code unit."""
    if not _flag_enabled():
        print(
            f"error: {_FLAG} is not set; set it to '1' to enable genealogy commands",
            file=sys.stderr,
        )
        return 1

    code_unit_id: str = args.code_unit_id
    store_path = Path(getattr(args, "store_file", _DEFAULT_STORE)).expanduser()
    as_json: bool = getattr(args, "json", False)

    store = _load_store(store_path)
    try:
        genealogy = get_genealogy(code_unit_id, store, require_enabled=True)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(genealogy.to_dict(), indent=2))
        return 0

    entries = genealogy.entries
    print(f"Genealogy: {code_unit_id}")
    print(f"  entries       : {len(entries)}")
    print(f"  chain_checksum: {genealogy.chain_checksum[:16]}…")
    if entries:
        print(f"  oldest        : {entries[0].timestamp}")
        print(f"  newest        : {entries[-1].timestamp}")
        print()
        for e in entries:
            meta = f" | {e.metadata}" if e.metadata else ""
            print(f"  [{e.entry_kind}] {e.entry_id} @ {e.timestamp}{meta}")
    else:
        print("  (no entries)")
    return 0
