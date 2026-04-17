"""AGT-05 on-chain anchoring — ReputationDelta → ERC-8004 ReputationRegistry.

This module closes the loop between the in-memory AGT-05 settlement
flow (see :mod:`aragora.reputation.settlement`) and the on-chain
ERC-8004 reputation registry at
:class:`aragora.blockchain.contracts.reputation.ReputationRegistryContract`.

Design principles:

- **Dry-run by default.** Every call returns an :class:`AnchorReceipt`
  describing what was or would be anchored. Live chain writes happen
  only when ``dry_run=False`` AND the ``ARAGORA_REPUTATION_ANCHORING_
  ENABLED`` flag is truthy AND a ``registry`` and ``signer`` are
  supplied.
- **Injected registry.** The registry client is passed in; no
  implicit provider resolution. Tests inject stubs; callers wire
  production providers.
- **Deterministic hash.** ``feedback_hash`` is SHA-256 over the
  canonical JSON of the ReputationDelta, so the on-chain record
  binds to the exact delta that was settled.
- **Value encoding.** The float delta is scaled to an int128 using
  ``value_decimals`` (default 6). Clamped to int128 range so
  extreme deltas cannot overflow.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.blockchain.contracts.reputation import ReputationRegistryContract
    from aragora.reputation.types import ReputationDelta

# Default scaling for float → int128 conversion
DEFAULT_VALUE_DECIMALS = 6

# int128 bounds (solidity int128 is 2's complement 128-bit)
INT128_MAX = (1 << 127) - 1
INT128_MIN = -(1 << 127)


class AnchorError(RuntimeError):
    """Raised when anchoring cannot proceed."""


def anchoring_enabled() -> bool:
    """Return True if callers should actually submit to the chain.

    Reads ``ARAGORA_REPUTATION_ANCHORING_ENABLED`` from the process
    environment. Default is False; when unset, ``anchor_delta`` forces
    dry-run mode regardless of the ``dry_run`` argument.
    """
    raw = str(os.environ.get("ARAGORA_REPUTATION_ANCHORING_ENABLED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def enable_anchoring() -> None:
    """Enable chain submission for the current process."""
    os.environ["ARAGORA_REPUTATION_ANCHORING_ENABLED"] = "1"


@dataclass(frozen=True)
class AnchorReceipt:
    """Record of an anchoring attempt — dry-run or live.

    - ``dry_run``: True iff no chain submission was attempted
    - ``tx_hash``: populated on successful live submission; ``None`` in
      dry-run mode or on failed submission
    - ``agent_id``: the ERC-8004 token id used
    - ``value``, ``value_decimals``: the int128 encoding submitted
    - ``feedback_hash_hex``: hex of the SHA-256 hash of the canonical
      ReputationDelta payload; matches the 32-byte ``feedback_hash``
      field the registry stores
    - ``tag1``: domain (e.g. ``"prediction_market"``)
    - ``tag2``: scoring rule (e.g. ``"brier_proper"``)
    - ``submitted_at``: timestamp of the anchor attempt
    - ``provenance``: delta_id, claim_id, resolution_id for audit
    - ``error``: populated when a live submission failed; ``None``
      otherwise
    """

    dry_run: bool
    agent_id: int
    value: int
    value_decimals: int
    feedback_hash_hex: str
    tag1: str
    tag2: str
    feedback_uri: str
    submitted_at: str
    provenance: dict[str, Any] = field(default_factory=dict)
    tx_hash: str | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "agent_id": self.agent_id,
            "value": self.value,
            "value_decimals": self.value_decimals,
            "feedback_hash_hex": self.feedback_hash_hex,
            "tag1": self.tag1,
            "tag2": self.tag2,
            "feedback_uri": self.feedback_uri,
            "submitted_at": self.submitted_at,
            "provenance": dict(self.provenance),
            "tx_hash": self.tx_hash,
            "error": self.error,
        }


def compute_feedback_value(delta: "ReputationDelta", value_decimals: int) -> int:
    """Encode ``delta.delta`` (float) as an int128, clamped to int128 range."""
    if value_decimals < 0 or value_decimals > 18:
        raise AnchorError(f"value_decimals must be in [0, 18]; got {value_decimals}")
    scaled = int(round(delta.delta * (10**value_decimals)))
    if scaled > INT128_MAX:
        return INT128_MAX
    if scaled < INT128_MIN:
        return INT128_MIN
    return scaled


def compute_feedback_hash(delta: "ReputationDelta") -> bytes:
    """Canonical SHA-256 over the delta's JSON. Returns 32 bytes."""
    payload = json.dumps(delta.to_json(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def delta_to_feedback_args(
    delta: "ReputationDelta",
    *,
    agent_id: int,
    value_decimals: int = DEFAULT_VALUE_DECIMALS,
    endpoint: str = "",
    feedback_uri: str = "",
) -> dict[str, Any]:
    """Build the keyword arguments to pass to
    :meth:`ReputationRegistryContract.give_feedback`.

    Does not submit. Returned mapping is guaranteed JSON-friendly
    except for the 32-byte ``feedback_hash`` which is raw bytes.
    """
    if agent_id < 0:
        raise AnchorError(f"agent_id must be non-negative; got {agent_id}")
    value = compute_feedback_value(delta, value_decimals)
    feedback_hash = compute_feedback_hash(delta)
    return {
        "agent_id": agent_id,
        "value": value,
        "value_decimals": value_decimals,
        "tag1": delta.domain,
        "tag2": delta.scoring_rule,
        "endpoint": endpoint,
        "feedback_uri": feedback_uri,
        "feedback_hash": feedback_hash,
    }


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def anchor_delta(
    delta: "ReputationDelta",
    *,
    agent_id: int,
    registry: "ReputationRegistryContract | None" = None,
    signer: Any = None,
    value_decimals: int = DEFAULT_VALUE_DECIMALS,
    endpoint: str = "",
    feedback_uri: str = "",
    dry_run: bool | None = None,
) -> AnchorReceipt:
    """Anchor a :class:`ReputationDelta` to the ERC-8004 Reputation Registry.

    Defaults to dry-run mode: computes the on-chain-shaped feedback
    args and returns an :class:`AnchorReceipt` describing what would
    be submitted, without touching the chain.

    Live submission requires all of:

    - ``dry_run`` is explicitly ``False`` (or left None with the env
      flag truthy)
    - ``ARAGORA_REPUTATION_ANCHORING_ENABLED`` is truthy
    - ``registry`` and ``signer`` are both supplied

    If any of those are missing, the call silently falls back to dry
    run — no exception. Callers read ``receipt.dry_run`` to detect
    the fallback and ``receipt.tx_hash`` to confirm live submission.

    Transport errors during live submission are captured in
    ``receipt.error`` and ``receipt.tx_hash`` is ``None``; the call
    does not re-raise, on the principle that AGT-05 settlement should
    not crash when chain writes fail.
    """
    feedback_args = delta_to_feedback_args(
        delta,
        agent_id=agent_id,
        value_decimals=value_decimals,
        endpoint=endpoint,
        feedback_uri=feedback_uri,
    )

    # Resolve dry-run mode
    flag_on = anchoring_enabled()
    if dry_run is None:
        dry_run = not flag_on
    elif dry_run is False and not flag_on:
        dry_run = True  # flag override: force dry-run
    if registry is None or signer is None:
        dry_run = True  # cannot submit without both

    base_receipt = AnchorReceipt(
        dry_run=dry_run,
        agent_id=int(feedback_args["agent_id"]),
        value=int(feedback_args["value"]),
        value_decimals=int(feedback_args["value_decimals"]),
        feedback_hash_hex=feedback_args["feedback_hash"].hex(),
        tag1=str(feedback_args["tag1"]),
        tag2=str(feedback_args["tag2"]),
        feedback_uri=str(feedback_args["feedback_uri"]),
        submitted_at=_utc_now_iso(),
        provenance={
            "delta_id": delta.delta_id,
            "agent_id_string": delta.agent_id,
            "claim_id": delta.claim_id,
            "resolution_id": delta.resolution_id,
            "domain": delta.domain,
            "scoring_rule": delta.scoring_rule,
        },
    )

    if dry_run:
        return base_receipt

    # Live submission path
    try:
        tx_hash = registry.give_feedback(
            agent_id=feedback_args["agent_id"],
            value=feedback_args["value"],
            signer=signer,
            value_decimals=feedback_args["value_decimals"],
            tag1=feedback_args["tag1"],
            tag2=feedback_args["tag2"],
            endpoint=feedback_args["endpoint"],
            feedback_uri=feedback_args["feedback_uri"],
            feedback_hash=feedback_args["feedback_hash"],
        )
    except Exception as exc:  # noqa: BLE001 - chain-write errors are wrapped, not re-raised
        return AnchorReceipt(
            dry_run=False,
            agent_id=base_receipt.agent_id,
            value=base_receipt.value,
            value_decimals=base_receipt.value_decimals,
            feedback_hash_hex=base_receipt.feedback_hash_hex,
            tag1=base_receipt.tag1,
            tag2=base_receipt.tag2,
            feedback_uri=base_receipt.feedback_uri,
            submitted_at=base_receipt.submitted_at,
            provenance=base_receipt.provenance,
            tx_hash=None,
            error=f"anchor submission failed: {type(exc).__name__}: {exc}",
        )

    return AnchorReceipt(
        dry_run=False,
        agent_id=base_receipt.agent_id,
        value=base_receipt.value,
        value_decimals=base_receipt.value_decimals,
        feedback_hash_hex=base_receipt.feedback_hash_hex,
        tag1=base_receipt.tag1,
        tag2=base_receipt.tag2,
        feedback_uri=base_receipt.feedback_uri,
        submitted_at=base_receipt.submitted_at,
        provenance=base_receipt.provenance,
        tx_hash=str(tx_hash),
        error=None,
    )


__all__ = [
    "DEFAULT_VALUE_DECIMALS",
    "INT128_MAX",
    "INT128_MIN",
    "AnchorError",
    "AnchorReceipt",
    "anchor_delta",
    "anchoring_enabled",
    "compute_feedback_hash",
    "compute_feedback_value",
    "delta_to_feedback_args",
    "enable_anchoring",
]
