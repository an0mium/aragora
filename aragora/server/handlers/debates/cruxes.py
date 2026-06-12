"""
Crux operations handler mixin (ODR-4 / #8227).

Exposes the crux finder's recorded output for a debate:

    GET /api/v1/debates/{debate_id}/cruxes

The endpoint is read-only and strictly honest: it returns the crux map the
``crux_finder`` consensus mode actually recorded on the debate's consensus
proof. When the debate exists but carries no crux record (crux mode was not
enabled, or it fell back to another consensus), the response carries an
explicit ODR-style absent marker (``{"status": "absent", "reason": ...}``)
rather than a fabricated or empty-but-present crux set. See
``docs/specs/OPEN_DECISION_RECEIPT.md`` section 4.6 for the field semantics
this contract mirrors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from aragora.rbac.decorators import require_permission

from ..base import (
    HandlerResult,
    error_response,
    json_response,
)
from ..openapi_decorator import api_endpoint

logger = logging.getLogger(__name__)


class _DebatesHandlerProtocol(Protocol):
    """Interface expected by CruxOperationsMixin."""

    ctx: dict[str, Any]

    def get_storage(self) -> Any | None:
        """Get debate storage instance."""
        ...

    def get_nomic_dir(self) -> Path | None:
        """Get nomic directory path."""
        ...

    def _load_crux_source_from_trace(self, debate_id: str) -> Any | None:
        """Load a debate result from the nomic trace store."""
        ...


def _absent(reason: str) -> dict[str, str]:
    """ODR-profile absent marker: honest absence, never fabrication."""
    return {"status": "absent", "reason": reason}


class CruxOperationsMixin:
    """Mixin providing crux-finder read operations for DebatesHandler."""

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/{debate_id}/cruxes",
        summary="Get debate cruxes",
        description=(
            "Get the crux map recorded for a debate run with the crux_finder "
            "consensus mode: the load-bearing disagreements the verdict turns on. "
            "When crux mode was not enabled for the debate, the cruxes block "
            "carries an explicit absent marker instead of fabricated data."
        ),
        tags=["Debates", "Analysis"],
        parameters=[
            {"name": "debate_id", "in": "path", "schema": {"type": "string"}, "required": True},
        ],
        responses={
            "200": {
                "description": "Crux map or explicit absent marker returned",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "debate_id": {"type": "string"},
                                "cruxes": {
                                    "type": "object",
                                    "description": (
                                        "Either {status: 'present', items: [...]} or "
                                        "{status: 'absent', reason: '...'}"
                                    ),
                                },
                                "crux_count": {"type": "integer"},
                                "convergence_barrier": {"type": "number"},
                                "counterfactuals": {"type": "array", "items": {"type": "object"}},
                                "recommended_focus": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "consensus_mode": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "401": {"description": "Unauthorized"},
            "404": {"description": "Debate not found"},
        },
    )
    @require_permission("debates:read")
    def _get_cruxes(self: _DebatesHandlerProtocol, debate_id: str) -> HandlerResult:
        """Get the recorded crux map for a debate (honest-absence contract)."""
        from aragora.debate.crux_mode import extract_crux_payload, extract_crux_skip_reason

        debate_record: Any | None = None

        storage = self.get_storage()
        if storage is not None:
            try:
                debate_record = storage.get_debate(debate_id)
            except (KeyError, ValueError, TypeError, OSError, RuntimeError) as e:
                logger.warning("Crux lookup: storage error for %s: %s", debate_id, e)
                debate_record = None

        # Fallback: debate trace on disk (same source meta-critique uses).
        if debate_record is None:
            debate_record = self._load_crux_source_from_trace(debate_id)

        if debate_record is None:
            return error_response(f"Debate not found: {debate_id}", 404)

        try:
            payload = extract_crux_payload(debate_record)
            skip_reason = extract_crux_skip_reason(debate_record)
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Failed to extract cruxes for %s: %s", debate_id, e, exc_info=True)
            return error_response("Error extracting crux data", 500)

        if payload is None or not payload["cruxes"]:
            if skip_reason:
                reason = (
                    f"crux finder was requested but skipped ({skip_reason}); "
                    "no crux set was recorded for this debate"
                )
            elif payload is not None:
                reason = (
                    "crux finder ran but identified no cruxes above the configured score threshold"
                )
            else:
                reason = (
                    "crux mode was not enabled for this debate "
                    "(run with consensus='crux_finder' to record a crux set)"
                )
            return json_response(
                {
                    "debate_id": debate_id,
                    "cruxes": _absent(reason),
                    "crux_count": 0,
                }
            )

        return json_response(
            {
                "debate_id": debate_id,
                "cruxes": {"status": "present", "items": payload["cruxes"]},
                "crux_count": payload["crux_count"],
                "convergence_barrier": payload["convergence_barrier"],
                "counterfactuals": payload["counterfactuals"],
                "recommended_focus": payload["recommended_focus"],
                "consensus_mode": payload["consensus_mode"],
            }
        )

    def _load_crux_source_from_trace(self: _DebatesHandlerProtocol, debate_id: str) -> Any | None:
        """Load a debate result from the nomic trace store, if present."""
        try:
            from aragora.debate.traces import DebateTrace
        except ImportError:
            return None

        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return None

        trace_path = Path(nomic_dir) / "traces" / f"{debate_id}.json"
        if not trace_path.exists():
            return None

        try:
            trace = DebateTrace.load(trace_path)
            return trace.to_debate_result()
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
            logger.warning("Crux lookup: failed to load trace for %s: %s", debate_id, e)
            return None


__all__ = ["CruxOperationsMixin"]
