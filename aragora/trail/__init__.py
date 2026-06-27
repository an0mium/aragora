"""Tamper-evident trail (TET) — intent anchoring primitives.

Implements Component 2 of ``docs/specs/TAMPER_EVIDENT_TRAIL.md``: an
append-only, hash-chained ledger of *intent records* for repo-mutating agent
actions, plus helpers to read, verify, and anchor the chain head outside the
machine that wrote it.

This package is the Plan v2 Pillar 5 seed (Open Receipt Standard): the chain
shares its canonicalization with the Open Decision Receipt profile (RFC 8785
JCS via :mod:`aragora.gauntlet.odr_export`), so an externally anchored chain
head commits to byte-stable record content.
"""

from aragora.trail.intent_chain import (
    ACTOR_CLASSES,
    GENESIS_PREV_HASH,
    INTENT_TYPES,
    ChainError,
    append_intent,
    chain_head_hash,
    compute_record_hash,
    default_chain_path,
    read_records,
    record_intent,
    verify_chain,
)

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
]
