"""Lane-completion receipt schema for swarm worker lanes.

Every completed lane emits a structured receipt that captures audit-relevant
provenance: which task, lease, agent, git refs, changed files, validations,
outcome, risks, and PR linkage.  Receipts are persisted via the operational
receipt pipeline and are queryable by integrator and dashboard flows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = frozenset(
    {
        "task_id",
        "lease_id",
        "agent_id",
        "outcome",
    }
)


@dataclass(slots=True)
class LaneCompletionReceipt:
    """Structured receipt emitted when a swarm worker lane finishes."""

    # Identity
    task_id: str
    lease_id: str
    agent_id: str
    receipt_id: str = field(default_factory=lambda: f"lane-{uuid.uuid4().hex[:12]}")

    # Git provenance
    base_sha: str | None = None
    head_sha: str | None = None
    changed_files: list[str] = field(default_factory=list)

    # Validation
    validations_run: list[dict[str, Any]] = field(default_factory=list)

    # Outcome
    outcome: str = "unknown"  # pass | fail | blocked | unknown
    risks: list[str] = field(default_factory=list)

    # PR linkage
    pr_url: str | None = None
    pr_number: int | None = None
    branch: str | None = None

    # Timing
    duration_seconds: float = 0.0
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Extensible metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """SHA-256 over identity + outcome fields for tamper detection."""
        payload = json.dumps(
            {
                "receipt_id": self.receipt_id,
                "task_id": self.task_id,
                "lease_id": self.lease_id,
                "agent_id": self.agent_id,
                "base_sha": self.base_sha,
                "head_sha": self.head_sha,
                "changed_files": sorted(self.changed_files),
                "outcome": self.outcome,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["content_hash"] = self.content_hash()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LaneCompletionReceipt:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def validate_receipt(receipt: LaneCompletionReceipt | dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty == valid)."""
    if isinstance(receipt, LaneCompletionReceipt):
        data = receipt.to_dict()
    else:
        data = dict(receipt)

    errors: list[str] = []
    for req in sorted(_REQUIRED_FIELDS):
        val = data.get(req)
        if not val or (isinstance(val, str) and not val.strip()):
            errors.append(f"missing or empty required field: {req}")

    outcome = data.get("outcome", "")
    if outcome not in ("pass", "fail", "blocked", "unknown"):
        errors.append(f"invalid outcome value: {outcome!r}")

    return errors


def emit_lane_receipt(receipt: LaneCompletionReceipt) -> str | None:
    """Persist a lane-completion receipt via the operational receipt pipeline.

    Returns the receipt_id on success, ``None`` on best-effort failure.
    Never raises to the caller.
    """
    try:
        from aragora.receipts.provenance import emit_operational_receipt

        return emit_operational_receipt(
            source="swarm_lane",
            action="lane_completed",
            actor=receipt.agent_id,
            inputs={
                "task_id": receipt.task_id,
                "lease_id": receipt.lease_id,
                "agent_id": receipt.agent_id,
                "base_sha": receipt.base_sha,
                "branch": receipt.branch,
            },
            outputs={
                "head_sha": receipt.head_sha,
                "changed_files": receipt.changed_files,
                "validations_run": receipt.validations_run,
                "outcome": receipt.outcome,
                "risks": receipt.risks,
                "pr_url": receipt.pr_url,
                "pr_number": receipt.pr_number,
            },
            verdict=receipt.outcome,
            duration_seconds=receipt.duration_seconds,
            receipt_id=receipt.receipt_id,
            metadata=receipt.metadata,
        )
    except Exception as exc:
        logger.debug("Lane receipt emission failed: %s", exc)
        return None


__all__ = [
    "LaneCompletionReceipt",
    "emit_lane_receipt",
    "validate_receipt",
]
