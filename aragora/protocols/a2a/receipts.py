"""Agent-readable decision receipts for the A2A consumer surface.

This module is the AGT-02 sub-deliverable that gives external software
agents a canonical, signed, machine-parseable shape they can fetch,
verify, parse, and act on without HTML scraping. See
``docs/plans/AGENT_CONSUMER_SURFACE.md`` §S4 and issue #6063.

The :class:`AgentReceipt` is a thin envelope that wraps any decision
artifact (decision text, CruxSet, claim, prediction outcome, debate
result) in a versioned, content-addressed, signature-verifiable
container. The wrapped artifact lives in ``subject``; the optional
:class:`aragora.reasoning.cruxset.CruxSet` (when AGT-01 emission is
enabled) lives in ``cruxset``; reputation deltas applied as a
consequence of this receipt (when AGT-05 settlement is wired) live in
``reputation_deltas_applied``.

Activation: this module is contract-only. Nothing publishes or signs
receipts automatically. Server endpoints that emit AgentReceipts will
land in a follow-up PR and remain gated behind the same AGT-* "no live
behavior change" rule from ``docs/status/NEXT_STEPS_CANONICAL.md``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

AGENT_RECEIPT_SCHEMA_VERSION = "1.0"
DEFAULT_FRESHNESS_SLA_SECONDS = 24 * 3600  # 1 day
DEFAULT_SETTLEMENT_WINDOW_SECONDS = 7 * 24 * 3600  # 7 days
DEFAULT_SIGNATURE_ALGORITHM = "hmac-sha256"
DEFAULT_SIGNATURE_KEY_ID = "default"

SignatureKey = str | bytes
SignatureKeyResolver = Callable[[str, str], SignatureKey | None]


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_hex(material: str) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _key_bytes(signing_key: SignatureKey) -> bytes:
    if isinstance(signing_key, bytes):
        key = signing_key
    else:
        key = str(signing_key).encode("utf-8")
    if not key:
        raise ValueError("signing_key must be non-empty")
    return key


def _hmac_sha256_hex(material: str, signing_key: SignatureKey) -> str:
    return hmac.new(_key_bytes(signing_key), material.encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class DissentEntry:
    """One dissenting view captured in a receipt."""

    agent_id: str
    statement: str
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")
        if not str(self.agent_id).strip():
            raise ValueError("agent_id must be non-empty")
        if not str(self.statement).strip():
            raise ValueError("statement must be non-empty")

    def to_json(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "statement": self.statement,
            "confidence": round(self.confidence, 6),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "DissentEntry":
        return cls(
            agent_id=str(data["agent_id"]),
            statement=str(data["statement"]),
            confidence=float(data.get("confidence") or 0.0),
        )


@dataclass(frozen=True)
class ReputationDelta:
    """A reputation change applied by AGT-05 settlement.

    This shape is forward-compatible with the
    :mod:`aragora.blockchain.contracts.reputation` registry; the
    actual on-chain anchoring lives in AGT-05 (issue #6066).
    """

    agent_id: str
    domain: str
    delta: float
    reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "domain": self.domain,
            "delta": round(self.delta, 6),
            "reason": self.reason,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ReputationDelta":
        return cls(
            agent_id=str(data["agent_id"]),
            domain=str(data["domain"]),
            delta=float(data["delta"]),
            reason=str(data.get("reason") or ""),
        )


@dataclass(frozen=True)
class AgentReceipt:
    """Agent-readable decision-receipt envelope.

    All fields are JSON-serializable. ``receipt_id`` is a content
    address computed from the canonical receipt payload so identical
    decisions deduplicate. ``signature`` is an HMAC-SHA256 over the
    canonical payload plus signature metadata, using the issuer's
    configured signing key. Verification therefore requires the issuer
    key (or a resolver for it); the public payload alone is not enough to
    mint a valid signature for changed content.

    Forward compatibility: the schema is versioned; readers should
    accept any schema_version with the same major number and tolerate
    unknown fields rather than rejecting them.
    """

    receipt_id: str
    schema_version: str
    issued_at: str
    issuer: str
    subject_kind: str
    subject: dict[str, Any]
    cruxset: dict[str, Any] | None
    dissent: tuple[DissentEntry, ...]
    reputation_deltas_applied: tuple[ReputationDelta, ...]
    freshness_sla_seconds: int
    settlement_window_seconds: int
    provenance: dict[str, Any]
    signature_key_id: str
    signature_algorithm: str
    signature: str

    @classmethod
    def build(
        cls,
        *,
        issuer: str,
        subject_kind: str,
        subject: dict[str, Any],
        cruxset: dict[str, Any] | None = None,
        dissent: tuple[DissentEntry, ...] = (),
        reputation_deltas_applied: tuple[ReputationDelta, ...] = (),
        freshness_sla_seconds: int = DEFAULT_FRESHNESS_SLA_SECONDS,
        settlement_window_seconds: int = DEFAULT_SETTLEMENT_WINDOW_SECONDS,
        provenance: dict[str, Any] | None = None,
        issued_at: str | None = None,
        signing_key: SignatureKey | None = None,
        signature_key_id: str = DEFAULT_SIGNATURE_KEY_ID,
    ) -> "AgentReceipt":
        if not str(issuer).strip():
            raise ValueError("issuer must be non-empty")
        if not str(subject_kind).strip():
            raise ValueError("subject_kind must be non-empty")
        if freshness_sla_seconds < 1:
            raise ValueError("freshness_sla_seconds must be >= 1")
        if settlement_window_seconds < 0:
            raise ValueError("settlement_window_seconds must be >= 0")
        if signing_key is None:
            raise ValueError("signing_key must be provided")
        if not str(signature_key_id).strip():
            raise ValueError("signature_key_id must be non-empty")

        timestamp = issued_at or _utc_now_iso()
        provenance_dict = dict(provenance or {})
        key_id = str(signature_key_id).strip()

        # Build the canonical payload in a deterministic order. receipt_id
        # is omitted because it is content-addressed from this payload.
        canonical_payload: dict[str, Any] = {
            "schema_version": AGENT_RECEIPT_SCHEMA_VERSION,
            "issued_at": timestamp,
            "issuer": issuer,
            "subject_kind": subject_kind,
            "subject": subject,
            "cruxset": cruxset,
            "dissent": [d.to_json() for d in dissent],
            "reputation_deltas_applied": [r.to_json() for r in reputation_deltas_applied],
            "freshness_sla_seconds": int(freshness_sla_seconds),
            "settlement_window_seconds": int(settlement_window_seconds),
            "provenance": provenance_dict,
        }
        canonical = _canonical(canonical_payload)
        receipt_id = "rcpt_a_" + _sha256_hex(canonical)[:16]
        signed_payload = {
            **canonical_payload,
            "signature_key_id": key_id,
            "signature_algorithm": DEFAULT_SIGNATURE_ALGORITHM,
        }
        signature = _hmac_sha256_hex(_canonical(signed_payload), signing_key)

        return cls(
            receipt_id=receipt_id,
            schema_version=AGENT_RECEIPT_SCHEMA_VERSION,
            issued_at=timestamp,
            issuer=issuer,
            subject_kind=subject_kind,
            subject=dict(subject),
            cruxset=(dict(cruxset) if cruxset is not None else None),
            dissent=tuple(dissent),
            reputation_deltas_applied=tuple(reputation_deltas_applied),
            freshness_sla_seconds=int(freshness_sla_seconds),
            settlement_window_seconds=int(settlement_window_seconds),
            provenance=provenance_dict,
            signature_key_id=key_id,
            signature_algorithm=DEFAULT_SIGNATURE_ALGORITHM,
            signature=signature,
        )

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "issued_at": self.issued_at,
            "issuer": self.issuer,
            "subject_kind": self.subject_kind,
            "subject": self.subject,
            "cruxset": self.cruxset,
            "dissent": [d.to_json() for d in self.dissent],
            "reputation_deltas_applied": [r.to_json() for r in self.reputation_deltas_applied],
            "freshness_sla_seconds": self.freshness_sla_seconds,
            "settlement_window_seconds": self.settlement_window_seconds,
            "provenance": self.provenance,
        }

    def _signed_payload(self) -> dict[str, Any]:
        return {
            **self._canonical_payload(),
            "signature_key_id": self.signature_key_id,
            "signature_algorithm": self.signature_algorithm,
        }

    def verify_signature(
        self,
        *,
        signing_key: SignatureKey | None = None,
        key_resolver: SignatureKeyResolver | None = None,
    ) -> bool:
        """Verify this receipt with the issuer key or a key resolver."""
        if self.signature_algorithm != DEFAULT_SIGNATURE_ALGORITHM:
            return False
        key = signing_key
        if key is None and key_resolver is not None:
            key = key_resolver(self.issuer, self.signature_key_id)
        if key is None:
            return False
        try:
            expected = _hmac_sha256_hex(_canonical(self._signed_payload()), key)
        except ValueError:
            return False
        return hmac.compare_digest(expected, self.signature)

    def is_fresh(self, *, now: datetime | None = None) -> bool:
        """Return True if the receipt is within its freshness SLA."""
        try:
            issued = datetime.fromisoformat(self.issued_at.replace("Z", "+00:00")).astimezone(UTC)
        except (ValueError, AttributeError):
            return False
        reference = (now or datetime.now(tz=UTC)).astimezone(UTC)
        age_seconds = (reference - issued).total_seconds()
        return age_seconds < self.freshness_sla_seconds

    def is_settled(self, *, now: datetime | None = None) -> bool:
        """Return True if the settlement window has elapsed.

        Once settled, a receipt's underlying outcome is considered
        final unless explicitly re-opened by a policy event.
        """
        try:
            issued = datetime.fromisoformat(self.issued_at.replace("Z", "+00:00")).astimezone(UTC)
        except (ValueError, AttributeError):
            return False
        reference = (now or datetime.now(tz=UTC)).astimezone(UTC)
        age_seconds = (reference - issued).total_seconds()
        return age_seconds >= self.settlement_window_seconds

    def to_json(self) -> dict[str, Any]:
        payload = self._signed_payload()
        payload["receipt_id"] = self.receipt_id
        payload["signature"] = self.signature
        return payload

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "AgentReceipt":
        return cls(
            receipt_id=str(data["receipt_id"]),
            schema_version=str(data.get("schema_version") or AGENT_RECEIPT_SCHEMA_VERSION),
            issued_at=str(data["issued_at"]),
            issuer=str(data["issuer"]),
            subject_kind=str(data["subject_kind"]),
            subject=dict(data.get("subject") or {}),
            cruxset=(dict(data["cruxset"]) if data.get("cruxset") is not None else None),
            dissent=tuple(DissentEntry.from_json(d) for d in (data.get("dissent") or [])),
            reputation_deltas_applied=tuple(
                ReputationDelta.from_json(r) for r in (data.get("reputation_deltas_applied") or [])
            ),
            freshness_sla_seconds=int(
                data.get("freshness_sla_seconds") or DEFAULT_FRESHNESS_SLA_SECONDS
            ),
            settlement_window_seconds=int(
                data.get("settlement_window_seconds") or DEFAULT_SETTLEMENT_WINDOW_SECONDS
            ),
            provenance=dict(data.get("provenance") or {}),
            signature_key_id=str(data.get("signature_key_id") or ""),
            signature_algorithm=str(data.get("signature_algorithm") or DEFAULT_SIGNATURE_ALGORITHM),
            signature=str(data.get("signature") or ""),
        )


__all__ = [
    "AGENT_RECEIPT_SCHEMA_VERSION",
    "DEFAULT_FRESHNESS_SLA_SECONDS",
    "DEFAULT_SETTLEMENT_WINDOW_SECONDS",
    "DEFAULT_SIGNATURE_ALGORITHM",
    "DEFAULT_SIGNATURE_KEY_ID",
    "AgentReceipt",
    "DissentEntry",
    "ReputationDelta",
    "SignatureKey",
    "SignatureKeyResolver",
]
