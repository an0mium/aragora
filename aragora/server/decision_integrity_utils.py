"""Deprecated import location for the decision-integrity backbone helpers.

The decision-integrity helper surface moved down to
:mod:`aragora.pipeline.decision_integrity_utils` during the P4a layering work so
that lower-layer modules (e.g. ``aragora.core.decision_router``) can reach it
without importing ``aragora.server``. Importing from
``aragora.server.decision_integrity_utils`` still works but is deprecated; import
from ``aragora.pipeline.decision_integrity_utils`` instead.
"""

from __future__ import annotations

import warnings

from aragora.pipeline.decision_integrity_utils import (
    _normalize_execution_request_for_safety_mode,
    build_decision_integrity_payload,
    ensure_decision_plan_backbone_run,
    execute_decision_plan_with_backbone,
    extract_execution_overrides,
    maybe_emit_decision_integrity,
    sync_decision_plan_backbone_receipt,
)

warnings.warn(
    "aragora.server.decision_integrity_utils is deprecated; "
    "import from aragora.pipeline.decision_integrity_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "extract_execution_overrides",
    "ensure_decision_plan_backbone_run",
    "sync_decision_plan_backbone_receipt",
    "execute_decision_plan_with_backbone",
    "build_decision_integrity_payload",
    "maybe_emit_decision_integrity",
    "_normalize_execution_request_for_safety_mode",
]
