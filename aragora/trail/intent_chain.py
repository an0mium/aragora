"""Append-only hash-chained intent ledger (TET phase T1).

Spec: ``docs/specs/TAMPER_EVIDENT_TRAIL.md`` Component 2 ("Intent anchoring").

Each repo-mutating agent action records an *intent* before acting. Records
are appended to a local JSONL working copy (default
``.aragora/trail/intent-chain.jsonl``) and chained: every record carries
``prev_hash`` (the previous record's ``record_hash``; 64 zeros for genesis)
and ``record_hash`` (SHA-256 over the RFC 8785 / JCS canonical JSON of all
other fields). Canonicalization is shared with the Open Decision Receipt
profile (``docs/specs/OPEN_DECISION_RECEIPT.md`` §5) via
:func:`aragora.gauntlet.odr_export.jcs_canonicalize`, so equal record content
hashes identically regardless of key order or unicode representation choices
at the call site.

Threat model honesty: an adversary with this machine's credentials can append
forged intents or truncate the local file. The chain alone only proves
*internal* consistency; tamper-evidence comes from anchoring the head hash on
infrastructure the laptop cannot rewrite (``scripts/anchor_intent_chain.py``,
TET phase T2). Rewriting any anchored prefix changes every subsequent hash
and breaks against the anchor.

Concurrency: appends take an exclusive ``flock`` on a sidecar lock file for
the read-tail-then-append critical section, then write the record as a single
``write`` on an ``O_APPEND`` descriptor. Prior lines are never rewritten.

Clock: timestamps flow through an injectable ``now`` callable (defaulting to
the module-level :func:`utc_now_iso`) so importable code paths never hardcode
wall-clock reads and tests stay deterministic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aragora.gauntlet.odr_export import jcs_canonicalize

__all__ = [
    "ACTOR_CLASSES",
    "GENESIS_PREV_HASH",
    "INTENT_TYPES",
    "ChainError",
    "append_intent",
    "chain_head_hash",
    "compute_record_hash",
    "default_chain_path",
    "read_records",
    "record_intent",
    "verify_chain",
    "utc_now_iso",
]

logger = logging.getLogger(__name__)

GENESIS_PREV_HASH = "0" * 64

ACTOR_CLASSES = frozenset(
    {
        "human",
        "agent-claude",
        "agent-codex",
        "agent-app",
        "daemon-publisher",
        "daemon-arbiter",
        "daemon-boss",
    }
)

INTENT_TYPES = frozenset(
    {
        "publish_pr",
        "merge_pr",
        "settle_pr",
        "close_pr",
        "branch_delete",
        "issue_create",
    }
)

_HASHED_FIELDS = (
    "seq",
    "ts",
    "actor_class",
    "intent_type",
    "target",
    "intent_id",
    "payload",
    "prev_hash",
)

_ENV_FLAG = "ARAGORA_TRAIL"
_ENV_CHAIN_PATH = "ARAGORA_TRAIL_CHAIN"


class ChainError(ValueError):
    """A record or chain violates the intent-chain contract."""


def utc_now_iso() -> str:
    """Module-level clock: current UTC time as an ISO-8601 string."""
    return datetime.now(tz=UTC).isoformat()


def default_chain_path() -> Path:
    """Default working-copy location, overridable via ``ARAGORA_TRAIL_CHAIN``."""
    override = os.environ.get(_ENV_CHAIN_PATH, "").strip()
    if override:
        return Path(override)
    return Path(".aragora") / "trail" / "intent-chain.jsonl"


def compute_record_hash(record: dict[str, Any]) -> str:
    """SHA-256 hex digest over the JCS canonical bytes of the hashed fields.

    Every field except ``record_hash`` participates (including ``prev_hash``),
    in a fixed field set — extra keys are rejected rather than silently
    dropped, so a record cannot carry unhashed content.
    """
    extras = set(record) - set(_HASHED_FIELDS) - {"record_hash"}
    if extras:
        raise ChainError(f"unhashable extra fields: {sorted(extras)}")
    payload = {field: record.get(field) for field in _HASHED_FIELDS}
    return hashlib.sha256(jcs_canonicalize(payload)).hexdigest()


def _validate_intent(actor_class: str, intent_type: str, target: dict[str, Any]) -> None:
    if actor_class not in ACTOR_CLASSES:
        raise ChainError(
            f"unknown actor_class {actor_class!r}; expected one of {sorted(ACTOR_CLASSES)}"
        )
    if intent_type not in INTENT_TYPES:
        raise ChainError(
            f"unknown intent_type {intent_type!r}; expected one of {sorted(INTENT_TYPES)}"
        )
    if not isinstance(target, dict) or not str(target.get("repo") or "").strip():
        raise ChainError("target must be a dict with a non-empty 'repo'")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """All records in file order; missing or empty file yields ``[]``.

    Raises:
        ChainError: when a line is not a JSON object (a corrupt ledger must
            never be silently truncated to its parseable prefix).
    """
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ChainError(f"line {line_number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ChainError(f"line {line_number} is not a JSON object")
            records.append(record)
    return records


def _tail_state(records: list[dict[str, Any]]) -> tuple[int, str]:
    """Next sequence number and the prev_hash the next record must carry."""
    if not records:
        return 0, GENESIS_PREV_HASH
    last = records[-1]
    try:
        seq = int(last["seq"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ChainError("tail record has no valid seq") from exc
    last_hash = str(last.get("record_hash") or "")
    if len(last_hash) != 64:
        raise ChainError("tail record has no valid record_hash")
    return seq + 1, last_hash


def append_intent(
    path: str | Path,
    *,
    actor_class: str,
    intent_type: str,
    target: dict[str, Any],
    payload: dict[str, Any] | None = None,
    intent_id: str | None = None,
    now: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Append one intent record, extending the hash chain atomically.

    The whole read-tail-then-append section runs under an exclusive ``flock``
    on ``<chain>.lock``, so concurrent appenders serialize and each record's
    ``seq``/``prev_hash`` reflect the true tail. The record line is written
    with a single ``write`` on an ``O_APPEND`` descriptor and fsync'd; prior
    lines are never modified.

    No stale-lock recovery is needed: ``flock`` is a kernel advisory lock
    released automatically when the holder's descriptor closes (including
    process crash), unlike ``O_EXCL`` lockfile schemes. A waiting appender
    blocks until the lock is free rather than timing out — appends are
    sub-millisecond, and a daemon that must not block can wrap the call
    (``record_intent`` already isolates callers from failures).

    Returns:
        The appended record, including ``record_hash``.
    """
    _validate_intent(actor_class, intent_type, target)
    clock = now if now is not None else utc_now_iso
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = file_path.with_name(file_path.name + ".lock")

    import fcntl  # POSIX-only by design; the trail runs on the operator fleet

    lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        seq, prev_hash = _tail_state(read_records(file_path))
        record: dict[str, Any] = {
            "seq": seq,
            "ts": str(clock()),
            "actor_class": actor_class,
            "intent_type": intent_type,
            "target": target,
            "intent_id": intent_id or str(uuid.uuid4()),
            "payload": payload or {},
            "prev_hash": prev_hash,
        }
        record["record_hash"] = compute_record_hash(record)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        data_fd = os.open(file_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
        try:
            os.write(data_fd, line.encode("utf-8"))
            os.fsync(data_fd)
        finally:
            os.close(data_fd)
        return record
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def verify_chain(path: str | Path) -> tuple[bool, int | None]:
    """Verify hash-chain integrity; return ``(ok, first_broken_seq)``.

    Checks, per record: ``seq`` is a contiguous 0-based sequence (duplicates
    and gaps both break), ``prev_hash`` equals the previous record's
    ``record_hash`` (genesis: 64 zeros), and ``record_hash`` recomputes from
    the record content. ``first_broken_seq`` is the claimed ``seq`` of the
    first failing record (falling back to its position when ``seq`` itself is
    unreadable). An empty or missing chain verifies trivially.
    """
    try:
        records = read_records(path)
    except ChainError:
        return False, 0
    expected_prev = GENESIS_PREV_HASH
    for position, record in enumerate(records):
        try:
            seq = int(record["seq"])
        except (KeyError, TypeError, ValueError):
            return False, position
        if seq != position:
            return False, seq
        try:
            recomputed = compute_record_hash(record)
        except ChainError:
            return False, seq
        if record.get("prev_hash") != expected_prev or record.get("record_hash") != recomputed:
            return False, seq
        expected_prev = recomputed
    return True, None


def chain_head_hash(path: str | Path) -> str | None:
    """``record_hash`` of the last record, or ``None`` for an empty chain."""
    records = read_records(path)
    if not records:
        return None
    head = str(records[-1].get("record_hash") or "")
    return head or None


def record_intent(
    *,
    actor_class: str,
    intent_type: str,
    target: dict[str, Any],
    payload: dict[str, Any] | None = None,
    path: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Best-effort intent recording for daemon/script call sites.

    Off unless ``ARAGORA_TRAIL=1`` (the TET contract rolls out call-site by
    call-site; default must not change daemon behavior). Never raises: a
    failed trail write logs a warning and returns ``None`` — the trail is a
    detection layer, not an availability dependency for the action itself.
    """
    environ = os.environ if env is None else env
    if str(environ.get(_ENV_FLAG) or "").strip() != "1":
        return None
    try:
        return append_intent(
            path if path is not None else default_chain_path(),
            actor_class=actor_class,
            intent_type=intent_type,
            target=target,
            payload=payload,
        )
    except (ChainError, OSError) as exc:
        logger.warning("trail intent record failed (non-fatal): %s", exc)
        return None
