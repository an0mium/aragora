"""Best-effort provenance receipt emission for autonomous loops."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _get_facade() -> Any | None:
    try:
        from aragora.pipeline.receipt_store_facade import get_receipt_store_facade

        return get_receipt_store_facade()
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        logger.debug("Receipt facade unavailable: %s", exc)
        return None


def _get_signer() -> Any | None:
    try:
        from aragora.gauntlet.signing import get_default_signer

        return get_default_signer()
    except (ImportError, RuntimeError, OSError, ValueError, TypeError) as exc:
        logger.debug("Receipt signer unavailable: %s", exc)
        return None


def emit_operational_receipt(
    *,
    source: str,
    action: str,
    actor: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    verdict: str,
    confidence: float = 0.0,
    duration_seconds: float = 0.0,
    receipt_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """Persist a best-effort signed receipt and never raise to the caller."""

    facade = _get_facade()
    if facade is None:
        return None

    receipt_id = receipt_id or f"op-{uuid.uuid4().hex[:12]}"
    receipt_data = {
        "receipt_id": receipt_id,
        "receipt_type": "operational",
        "source": source,
        "action": action,
        "actor": actor,
        "inputs": inputs,
        "outputs": outputs,
        "verdict": verdict,
        "confidence": confidence,
        "duration_seconds": duration_seconds,
        "metadata": dict(metadata or {}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "content_hash": _content_hash(
            receipt_id=receipt_id,
            source=source,
            action=action,
            actor=actor,
            inputs=inputs,
            outputs=outputs,
            verdict=verdict,
            confidence=confidence,
        ),
    }

    signature = None
    signature_key_id = None
    signed_at = None
    signature_algorithm = None

    signer = _get_signer()
    if signer is not None:
        try:
            signed = signer.sign(receipt_data)
            metadata_obj = getattr(signed, "signature_metadata", None)
            signature = getattr(signed, "signature", None)
            signature_key_id = getattr(metadata_obj, "key_id", None)
            signed_at = getattr(metadata_obj, "timestamp", None)
            signature_algorithm = getattr(metadata_obj, "algorithm", None)
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            logger.debug("Receipt signing skipped for %s: %s", receipt_id, exc)

    try:
        facade.persist_and_save(
            receipt_id,
            receipt_data,
            signature=signature,
            signature_key_id=signature_key_id,
            signed_at=signed_at,
            signature_algorithm=signature_algorithm,
            state="CREATED",
        )
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        logger.debug("Receipt emission failed for %s: %s", receipt_id, exc)
        return None

    logger.info("Operational receipt emitted: %s (%s/%s)", receipt_id, source, action)
    return receipt_id


def _content_hash(
    *,
    receipt_id: str,
    source: str,
    action: str,
    actor: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    verdict: str,
    confidence: float,
) -> str:
    content = json.dumps(
        {
            "receipt_id": receipt_id,
            "source": source,
            "action": action,
            "actor": actor,
            "inputs": inputs,
            "outputs": outputs,
            "verdict": verdict,
            "confidence": confidence,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


__all__ = ["emit_operational_receipt"]
