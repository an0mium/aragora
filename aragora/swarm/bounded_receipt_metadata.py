"""Bound worker-result receipt metadata before it reaches the signed-receipt path.

Background
----------
Boss-loop's ``_emit_lane_receipt`` historically spread ``worker_result["receipt_metadata"]``
directly into the receipt's metadata dict. That payload can carry the entire dispatch
gate context, prior worker results, harvest metadata, and ad-hoc debug fields — easily
hundreds of KB to multiple MB on real worker completions.

The downstream signing pipeline canonicalises receipts via
``json.dumps(..., sort_keys=True, default=str)``. ``sort_keys=True`` forces the
encoder to walk the entire structure, sorting nested dicts at every level. With a
multi-MB metadata payload nested 6+ levels deep, a single signing call takes 2+
seconds of CPU; chained through the post-worker asyncio path this stalls the
whole boss-loop process and looks like a hang.

Fix
---
``bound_receipt_metadata`` produces a small, audit-useful summary while writing
the full payload to a reference file under ``.aragora/worker-results/`` keyed by
sha256. Receipt consumers see the summary directly; anyone who needs the full
context loads it from the reference.

Design choices
--------------
- Keep all known scalar/short-list fields verbatim (cheap to preserve, often
  used by downstream consumers like outcome metrics emission).
- Truncate stdout/stderr-style fields to a bounded tail (last 4 KiB).
- Replace large dict/list fields (dispatch_gate, publish_result, harvest_result,
  raw_worker_output) with bounded summary dicts plus a reference if the original
  exceeded the per-field target.
- Always write the full payload to disk (best-effort). The reference path lives
  alongside the receipt, so receipt loaders can opt into the deep view.
- Hard size cap: total bounded summary <= 16 KiB after json-serialisation. If we
  somehow exceed it, fall back to a tiny ``{"reference": ..., "truncated": True}``.

The bound is applied **before** the receipt object is constructed, so the entire
downstream pipeline (signing, persistence, content-hash) sees only bounded
data and cannot spin.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# Total target size for the bounded summary after json-serialisation.
BOUNDED_METADATA_TARGET_BYTES: Final[int] = 16 * 1024

# Per-field tail size for stdout/stderr-like blobs (codex spec: 4 KiB each).
BOUNDED_TAIL_BYTES: Final[int] = 4 * 1024

# Per-field tail size for secondary text blobs (log/raw_output/blocker_evidence).
# Smaller than the primary tails because stdout/stderr already capture the gist.
BOUNDED_TAIL_BYTES_SECONDARY: Final[int] = 1024

# Default location for the full-payload reference store.
DEFAULT_RESULTS_DIR: Final[Path] = Path(".aragora/worker-results")

# Fields preserved verbatim when small. Order matters: these are the keys
# downstream consumers (boss_loop_outcome, follow-up dispatch logic, lane
# receipts test contract) read.
_PRESERVED_SCALAR_KEYS: Final[tuple[str, ...]] = (
    "issue_title",
    "issue_number",
    "worker_receipt_id",
    "actual_target_agent",
    "requested_target_agent",
    "runner_id",
    "runner_type",
    "cost_class",
    "blocker_kind",
    "terminal_outcome",
    "outcome",
    "status",
    "error_class",
    "branch",
    "pr_url",
    "pr_number",
    "head_sha",
    "base_sha",
    "lease_id",
    "postprocess_promoted_from_status",
    "postprocess_action",
    "postprocess_outcome",
)

# Per-field size threshold: dicts smaller than this stay verbatim; larger
# ones get summarised. 4 KiB is large enough for typical publish_result /
# issue_comment_result payloads (which only carry a handful of small keys),
# while keeping a hard ceiling on individual contributions to the total.
_DICT_VERBATIM_THRESHOLD_BYTES: Final[int] = 4 * 1024

# Field names that ALWAYS get summarised regardless of size, because they
# are known to carry potentially huge nested structures (e.g. dispatch_gate
# captures the full pre-dispatch context including prior worker outputs).
_DICT_KEYS_FORCE_SUMMARY: Final[frozenset[str]] = frozenset(
    {
        "dispatch_gate",
        "raw_worker_output",
        "raw_metadata",
    }
)

# Field names whose values are dicts/lists that MAY be either small enough
# to keep verbatim or large enough to need summarisation. These are the
# fields downstream consumers actually walk into.
_BOUNDED_DICT_KEYS: Final[tuple[str, ...]] = (
    "dispatch_gate",
    "publish_result",
    "harvest_result",
    "issue_comment_result",
    "postprocess_result",
)

# Primary text blobs — get the larger 4 KiB tail (codex spec).
_BOUNDED_TEXT_KEYS_PRIMARY: Final[tuple[str, ...]] = (
    "stdout",
    "stderr",
)

# Secondary text blobs — get the smaller 1 KiB tail. They tend to overlap with
# stdout/stderr semantically and inflating them past 1 KiB rarely adds debug value.
_BOUNDED_TEXT_KEYS_SECONDARY: Final[tuple[str, ...]] = (
    "log",
    "raw_output",
    "blocker_evidence",
)

# Combined for tests / introspection.
_BOUNDED_TEXT_KEYS: Final[tuple[str, ...]] = (
    *_BOUNDED_TEXT_KEYS_PRIMARY,
    *_BOUNDED_TEXT_KEYS_SECONDARY,
)

# Fields treated as bounded lists of strings.
_BOUNDED_LIST_KEYS: Final[tuple[str, ...]] = (
    "blocked_reasons",
    "needs_human_reasons",
    "changed_files",
    "validations_run",
)

_MAX_LIST_ITEMS: Final[int] = 32
_MAX_LIST_ITEM_BYTES: Final[int] = 256


def _scalar_or_short_str(value: Any, max_bytes: int = 512) -> Any:
    """Return value unchanged if scalar, else a tail-truncated string repr."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        encoded = value.encode("utf-8", errors="replace")
        if len(encoded) <= max_bytes:
            return value
        return _tail(encoded, max_bytes)
    # Fall through: convert non-scalar to string and truncate.
    return _tail(str(value).encode("utf-8", errors="replace"), max_bytes)


def _tail(blob: bytes, max_bytes: int) -> str:
    """Return the last ``max_bytes`` of ``blob`` as a UTF-8 string with a marker."""
    if len(blob) <= max_bytes:
        return blob.decode("utf-8", errors="replace")
    truncated = blob[-max_bytes:].decode("utf-8", errors="replace")
    return f"[truncated:{len(blob) - max_bytes}b]\n{truncated}"


def _bound_list(value: Any) -> list[Any]:
    """Cap list length and per-item size."""
    if not isinstance(value, list):
        return []
    bounded: list[Any] = []
    for item in value[:_MAX_LIST_ITEMS]:
        bounded.append(_scalar_or_short_str(item, max_bytes=_MAX_LIST_ITEM_BYTES))
    if len(value) > _MAX_LIST_ITEMS:
        bounded.append(f"[+{len(value) - _MAX_LIST_ITEMS} more items truncated]")
    return bounded


def _summarise_nested_dict(value: Any, *, max_bytes: int = 1024) -> dict[str, Any]:
    """Produce a small summary of a nested dict.

    Keeps top-level scalar fields verbatim if small. Replaces nested
    dict/list values with type+length descriptors. Result is bounded by
    ``max_bytes`` after json-serialisation.
    """
    if not isinstance(value, Mapping):
        return {"_kind": "non_mapping", "_repr": _scalar_or_short_str(value)}
    summary: dict[str, Any] = {}
    for key, child in value.items():
        skey = str(key)
        if isinstance(child, (str, int, float, bool)) or child is None:
            summary[skey] = _scalar_or_short_str(child, max_bytes=256)
        elif isinstance(child, Mapping):
            summary[skey] = {"_kind": "dict", "_keys": list(child.keys())[:16]}
        elif isinstance(child, list):
            summary[skey] = {"_kind": "list", "_len": len(child)}
        else:
            summary[skey] = {"_kind": type(child).__name__}
        # Cheap size guard: bail if we already exceed the budget.
        if len(json.dumps(summary, default=str)) > max_bytes:
            summary["_truncated"] = True
            break
    return summary


def _resolve_results_dir(repo_root: Path | str | None) -> Path:
    """Return the worker-results directory anchored at ``repo_root`` or cwd."""
    if repo_root is None:
        return DEFAULT_RESULTS_DIR
    return Path(repo_root) / DEFAULT_RESULTS_DIR


def _write_reference_payload(
    *,
    payload: Any,
    run_id: str,
    repo_root: Path | str | None,
) -> dict[str, Any] | None:
    """Persist the full payload to disk and return a reference descriptor.

    Best-effort: returns ``None`` if the write fails for any reason.
    """
    try:
        results_dir = _resolve_results_dir(repo_root)
        results_dir.mkdir(parents=True, exist_ok=True)
        # Include "_kind=reference" so consumers can detect a stub vs full payload.
        try:
            serialised = json.dumps(payload, sort_keys=True, default=str)
        except (TypeError, ValueError) as exc:
            logger.debug("Bounded receipt: payload not JSON-serialisable (%s)", exc)
            return None
        sha = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
        safe_run_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(run_id))
        target = results_dir / f"{safe_run_id or 'unknown'}.json"
        # Atomic-ish write: write to .tmp then rename.
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(serialised, encoding="utf-8")
        os.replace(tmp, target)
        return {
            "path": str(target),
            "sha256": sha,
            "size_bytes": len(serialised),
        }
    except OSError as exc:
        logger.debug("Bounded receipt: reference write failed (%s)", exc)
        return None


def bound_receipt_metadata(
    raw_metadata: Any,
    *,
    run_id: str,
    repo_root: Path | str | None = None,
    target_bytes: int = BOUNDED_METADATA_TARGET_BYTES,
) -> dict[str, Any]:
    """Return a bounded summary of ``raw_metadata`` suitable for the signed-receipt path.

    The full payload is persisted to ``.aragora/worker-results/<run_id>.json``
    and referenced from the bounded summary. Always returns a dict — never
    ``None`` — so downstream code can rely on the shape.

    Behaviour
    ---------
    - Empty / non-mapping input returns ``{"_bounded": True, "_empty": True}``.
    - Preserved scalar fields (issue_title, terminal_outcome, etc.) are kept
      verbatim with a soft per-field size cap.
    - Bounded dict fields (dispatch_gate, publish_result, harvest_result) are
      replaced with type+keys summaries.
    - Text blobs (stdout, stderr, log, raw_output, blocker_evidence) are
      tail-truncated to ``BOUNDED_TAIL_BYTES``.
    - List fields are length- and per-item-capped.
    - Reference to the full payload is included under ``_reference`` if the
      write succeeded.
    - If the bounded summary itself exceeds ``target_bytes`` after json
      serialisation, the function falls back to a minimal stub of just the
      reference plus a ``_overflow=True`` flag.
    """
    if not isinstance(raw_metadata, Mapping):
        return {"_bounded": True, "_empty": True}

    bounded: dict[str, Any] = {"_bounded": True}

    # 1. Preserved scalars (only include if present in raw metadata).
    for key in _PRESERVED_SCALAR_KEYS:
        if key in raw_metadata:
            bounded[key] = _scalar_or_short_str(raw_metadata[key], max_bytes=512)

    # 2. Nested dicts: keep small ones verbatim, summarise large or known-huge ones.
    for key in _BOUNDED_DICT_KEYS:
        if key not in raw_metadata:
            continue
        value = raw_metadata[key]
        if key in _DICT_KEYS_FORCE_SUMMARY:
            bounded[key] = _summarise_nested_dict(value)
            continue
        # Probe size: if the verbatim serialisation is small, keep it.
        try:
            size = len(json.dumps(value, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            size = _DICT_VERBATIM_THRESHOLD_BYTES + 1
        if size <= _DICT_VERBATIM_THRESHOLD_BYTES:
            # Small dict — pass through. Preserves runner_id/action/branch etc.
            bounded[key] = value
        else:
            bounded[key] = _summarise_nested_dict(value)

    # 2b. Pass-through for any other small dict-valued keys we don't explicitly know
    # about. This keeps test contracts that read ad-hoc keys (runner_id, cost_class,
    # postprocess_promoted_from_status) working when they're set as raw_metadata
    # top-level fields. Skip keys we've already handled.
    handled = (
        set(_PRESERVED_SCALAR_KEYS)
        | set(_BOUNDED_DICT_KEYS)
        | set(_BOUNDED_TEXT_KEYS)
        | set(_BOUNDED_LIST_KEYS)
    )
    for key, value in raw_metadata.items():
        if key in handled or key in bounded:
            continue
        # Only pass through small non-bytes scalar/dict values; everything else
        # gets dropped (the full payload is in the reference file anyway).
        if value is None or isinstance(value, (bool, int, float)):
            bounded[key] = value
        elif isinstance(value, str):
            if len(value.encode("utf-8")) <= 1024:
                bounded[key] = value
        elif isinstance(value, Mapping):
            try:
                size = len(json.dumps(value, default=str).encode("utf-8"))
            except (TypeError, ValueError):
                continue
            if size <= 1024:
                bounded[key] = dict(value)

    # 3a. Primary tail-truncated text blobs (stdout/stderr) — 4 KiB each.
    for key in _BOUNDED_TEXT_KEYS_PRIMARY:
        if key in raw_metadata:
            value = raw_metadata[key]
            if isinstance(value, str):
                encoded = value.encode("utf-8", errors="replace")
                if len(encoded) <= BOUNDED_TAIL_BYTES:
                    bounded[key] = value
                else:
                    bounded[key] = _tail(encoded, BOUNDED_TAIL_BYTES)
            else:
                bounded[key] = _scalar_or_short_str(value, max_bytes=BOUNDED_TAIL_BYTES)

    # 3b. Secondary tail-truncated text blobs (log/raw_output/blocker_evidence) — 1 KiB each.
    for key in _BOUNDED_TEXT_KEYS_SECONDARY:
        if key in raw_metadata:
            value = raw_metadata[key]
            if isinstance(value, str):
                encoded = value.encode("utf-8", errors="replace")
                if len(encoded) <= BOUNDED_TAIL_BYTES_SECONDARY:
                    bounded[key] = value
                else:
                    bounded[key] = _tail(encoded, BOUNDED_TAIL_BYTES_SECONDARY)
            else:
                bounded[key] = _scalar_or_short_str(value, max_bytes=BOUNDED_TAIL_BYTES_SECONDARY)

    # 4. Length-capped lists.
    for key in _BOUNDED_LIST_KEYS:
        if key in raw_metadata:
            bounded[key] = _bound_list(raw_metadata[key])

    # 5. Reference to the full payload (best-effort).
    reference = _write_reference_payload(
        payload=dict(raw_metadata),
        run_id=run_id,
        repo_root=repo_root,
    )
    if reference is not None:
        bounded["_reference"] = reference

    # 6. Progressive overflow handling. Rather than dropping everything when we
    # exceed the cap (which would lose audit fields downstream consumers read),
    # we drop the largest non-essential fields first in priority order:
    #   1. secondary text blobs (log, raw_output, blocker_evidence)
    #   2. primary text blobs (stderr, then stdout)
    #   3. nested-dict summaries (publish_result, harvest_result, etc.)
    # Preserved scalars and lists stay because consumers read them.
    def _serialised_size(d: dict[str, Any]) -> int:
        try:
            return len(json.dumps(d, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            return target_bytes + 1

    drop_order = (
        # Tier 1: secondary text blobs (low-value duplicates of stdout/stderr).
        list(_BOUNDED_TEXT_KEYS_SECONDARY),
        # Tier 2: stderr first (stdout is usually more useful for triage).
        ["stderr"],
        # Tier 3: stdout.
        ["stdout"],
        # Tier 4: large dict summaries.
        ["dispatch_gate", "harvest_result", "publish_result", "issue_comment_result"],
    )
    serialised_size = _serialised_size(bounded)
    for tier in drop_order:
        if serialised_size <= target_bytes:
            break
        for key in tier:
            bounded.pop(key, None)
        serialised_size = _serialised_size(bounded)

    if serialised_size > target_bytes:
        # Last-resort minimal stub: we couldn't fit even after stripping bulk fields.
        minimal: dict[str, Any] = {
            "_bounded": True,
            "_overflow": True,
            "_serialised_bytes": serialised_size,
        }
        if reference is not None:
            minimal["_reference"] = reference
        # Keep the most operationally-useful scalars even on extreme overflow.
        for key in (
            "issue_title",
            "terminal_outcome",
            "outcome",
            "status",
            "error_class",
            "branch",
            "pr_number",
        ):
            if key in bounded:
                minimal[key] = bounded[key]
        return minimal

    return bounded


def reference_keys() -> Iterable[str]:
    """Return the field keys that bounded summaries may use.

    Useful for tests asserting the bounded shape without hard-coding strings.
    """
    return (
        *_PRESERVED_SCALAR_KEYS,
        *_BOUNDED_DICT_KEYS,
        *_BOUNDED_TEXT_KEYS,
        *_BOUNDED_LIST_KEYS,
        "_bounded",
        "_empty",
        "_reference",
        "_overflow",
        "_serialised_bytes",
    )


__all__ = [
    "BOUNDED_METADATA_TARGET_BYTES",
    "BOUNDED_TAIL_BYTES",
    "DEFAULT_RESULTS_DIR",
    "bound_receipt_metadata",
    "reference_keys",
]
