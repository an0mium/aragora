"""Canonical receipt enforcement helpers for action-taking paths.

Phase 1 adds a single enforcement surface that write-capable handlers can call
without changing default behavior. Enforcement stays disabled until the
per-domain feature flag is enabled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

from aragora.config.feature_flags import is_enabled
from aragora.gauntlet.receipt_models import DecisionReceipt
from aragora.gauntlet.receipt_store import (
    ReceiptState,
    ReceiptStateError,
    StoredReceipt,
    get_receipt_store,
)

logger = logging.getLogger(__name__)

_FEATURE_FLAG_BY_DOMAIN: Final[dict[str, str]] = {
    "openclaw": "receipt_enforcement_openclaw",
    "canvas": "receipt_enforcement_canvas",
    "computer_use": "receipt_enforcement_computer_use",
    "inbox": "receipt_enforcement_inbox",
    "shared_inbox": "receipt_enforcement_shared_inbox",
}


class ReceiptEnforcementError(RuntimeError):
    """Raised when an action-taking path lacks a valid execution receipt."""

    status_code = 428

    def __init__(
        self,
        message: str,
        *,
        action_domain: str,
        action_type: str,
        receipt_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.action_domain = action_domain
        self.action_type = action_type
        self.receipt_id = receipt_id


@dataclass(frozen=True, slots=True)
class ReceiptExemption:
    """Documented exemption for read-only or legacy-admin execution paths."""

    reason: str
    approved_by: str
    category: str = "read_only"


def _normalize_domain(action_domain: str) -> str:
    normalized = str(action_domain or "").strip().lower().replace("-", "_")
    if normalized not in _FEATURE_FLAG_BY_DOMAIN:
        raise ValueError(f"Unsupported receipt enforcement domain: {action_domain!r}")
    return normalized


def _log_enforcement_event(
    *,
    result: str,
    action_domain: str,
    action_type: str,
    actor_id: str,
    resource_id: str,
    receipt_id: str | None = None,
    detail: str | None = None,
) -> None:
    logger.info(
        "receipt_enforcement result=%s domain=%s action=%s actor=%s resource=%s receipt=%s detail=%s",
        result,
        action_domain,
        action_type,
        actor_id or "<unknown>",
        resource_id or "<unknown>",
        receipt_id or "<none>",
        detail or "",
    )


def is_receipt_enforcement_enabled(domain: str) -> bool:
    """Return whether receipt enforcement is enabled for one action domain."""

    normalized = _normalize_domain(domain)
    return is_enabled(_FEATURE_FLAG_BY_DOMAIN[normalized])


def require_receipt_gate(
    *,
    action_domain: str,
    action_type: str,
    actor_id: str,
    resource_id: str,
    receipt_id: str | None = None,
    exempt: ReceiptExemption | None = None,
) -> StoredReceipt | None:
    """Validate an approved receipt before an action-taking path proceeds."""

    normalized_domain = _normalize_domain(action_domain)
    normalized_receipt_id = str(receipt_id or "").strip() or None

    if exempt is not None:
        _log_enforcement_event(
            result="exempted",
            action_domain=normalized_domain,
            action_type=action_type,
            actor_id=actor_id,
            resource_id=resource_id,
            receipt_id=normalized_receipt_id,
            detail=f"{exempt.category}:{exempt.approved_by}:{exempt.reason}",
        )
        return None

    if not is_receipt_enforcement_enabled(normalized_domain):
        _log_enforcement_event(
            result="disabled",
            action_domain=normalized_domain,
            action_type=action_type,
            actor_id=actor_id,
            resource_id=resource_id,
            receipt_id=normalized_receipt_id,
        )
        return None

    if normalized_receipt_id is None:
        raise ReceiptEnforcementError(
            f"{normalized_domain} action {action_type!r} requires an approved execution receipt",
            action_domain=normalized_domain,
            action_type=action_type,
        )

    store = get_receipt_store()
    stored = store.get(normalized_receipt_id)
    if stored is None:
        raise ReceiptEnforcementError(
            f"Receipt {normalized_receipt_id!r} not found",
            action_domain=normalized_domain,
            action_type=action_type,
            receipt_id=normalized_receipt_id,
        )

    if stored.state != ReceiptState.APPROVED:
        raise ReceiptEnforcementError(
            f"Receipt {normalized_receipt_id!r} is in state {stored.state.value}, expected APPROVED",
            action_domain=normalized_domain,
            action_type=action_type,
            receipt_id=normalized_receipt_id,
        )

    if not stored.signature:
        raise ReceiptEnforcementError(
            f"Receipt {normalized_receipt_id!r} is missing a cryptographic signature",
            action_domain=normalized_domain,
            action_type=action_type,
            receipt_id=normalized_receipt_id,
        )

    if not store.verify_receipt(normalized_receipt_id):
        raise ReceiptEnforcementError(
            f"Receipt {normalized_receipt_id!r} failed signature verification",
            action_domain=normalized_domain,
            action_type=action_type,
            receipt_id=normalized_receipt_id,
        )

    try:
        receipt = DecisionReceipt.from_dict(stored.receipt_data)
    except (TypeError, ValueError, KeyError) as exc:
        raise ReceiptEnforcementError(
            f"Receipt {normalized_receipt_id!r} could not be reconstructed for integrity validation",
            action_domain=normalized_domain,
            action_type=action_type,
            receipt_id=normalized_receipt_id,
        ) from exc
    if not receipt.verify_integrity():
        raise ReceiptEnforcementError(
            f"Receipt {normalized_receipt_id!r} failed integrity verification",
            action_domain=normalized_domain,
            action_type=action_type,
            receipt_id=normalized_receipt_id,
        )

    _log_enforcement_event(
        result="allowed",
        action_domain=normalized_domain,
        action_type=action_type,
        actor_id=actor_id,
        resource_id=resource_id,
        receipt_id=normalized_receipt_id,
    )
    return stored


def transition_receipt_executed(receipt_id: str) -> StoredReceipt:
    """Transition an approved receipt to EXECUTED after successful mutation."""

    normalized_receipt_id = str(receipt_id or "").strip()
    if not normalized_receipt_id:
        raise ReceiptEnforcementError(
            "receipt_id is required to transition a receipt to EXECUTED",
            action_domain="unknown",
            action_type="transition",
        )

    store = get_receipt_store()
    stored = store.get(normalized_receipt_id)
    if stored is None:
        raise ReceiptEnforcementError(
            f"Receipt {normalized_receipt_id!r} not found",
            action_domain="unknown",
            action_type="transition",
            receipt_id=normalized_receipt_id,
        )

    if stored.state == ReceiptState.EXECUTED:
        _log_enforcement_event(
            result="already_executed",
            action_domain="unknown",
            action_type="transition",
            actor_id="system",
            resource_id=normalized_receipt_id,
            receipt_id=normalized_receipt_id,
        )
        return stored

    try:
        updated = store.transition(normalized_receipt_id, ReceiptState.EXECUTED)
    except ReceiptStateError as exc:
        raise ReceiptEnforcementError(
            f"Receipt {normalized_receipt_id!r} could not transition to EXECUTED from {stored.state.value}",
            action_domain="unknown",
            action_type="transition",
            receipt_id=normalized_receipt_id,
        ) from exc

    _log_enforcement_event(
        result="executed",
        action_domain="unknown",
        action_type="transition",
        actor_id="system",
        resource_id=normalized_receipt_id,
        receipt_id=normalized_receipt_id,
    )
    return updated


__all__ = [
    "ReceiptEnforcementError",
    "ReceiptExemption",
    "is_receipt_enforcement_enabled",
    "require_receipt_gate",
    "transition_receipt_executed",
]
