"""Deterministic receipt helpers for heterogeneity probes."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

HETEROGENEITY_RECEIPT_SCHEMA_VERSION = "heterogeneity_probe_receipt.v1"


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Return canonical JSON used for receipt hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_receipt_id(receipt: Mapping[str, Any]) -> str:
    """Compute a stable receipt ID, excluding volatile receipt fields."""
    body = copy.deepcopy(dict(receipt))
    body.pop("receipt_id", None)
    body.pop("produced_at", None)
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def write_receipt(receipt: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Write a receipt under ``output_dir`` using its receipt ID."""
    receipt_id = receipt.get("receipt_id")
    if not isinstance(receipt_id, str) or not receipt_id:
        raise ValueError("receipt must include a non-empty receipt_id")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{receipt_id}.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
