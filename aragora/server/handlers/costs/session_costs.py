"""
Debate Session Cost Calculation API Handler.

Provides endpoints for retrieving detailed cost breakdowns per debate session,
calculating cost-per-decision metrics, and generating receipt data:

- GET /api/v1/costs/sessions/{debate_id}/breakdown   — Detailed cost breakdown
- GET /api/v1/costs/sessions/{debate_id}/metrics      — Usage metrics summary
- GET /api/v1/costs/sessions/{debate_id}/receipt       — Cost receipt for audit
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
)
from aragora.rbac.decorators import require_permission
from aragora.server.handlers.utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

_session_cost_limiter = RateLimiter(requests_per_minute=60)

# Versioned and legacy path prefixes for dynamic matching
_V1_PREFIX = "/api/v1/costs/sessions/"
_LEGACY_PREFIX = "/api/costs/sessions/"


class SessionCostHandler:
    """Handler for debate session cost calculation endpoints.

    Exposes per-debate cost breakdowns, usage metrics, and receipt
    generation backed by the DebateCostTracker and global CostTracker.
    """

    ROUTES = [
        # Versioned (canonical)
        "/api/v1/costs/sessions/*/breakdown",
        "/api/v1/costs/sessions/*/metrics",
        "/api/v1/costs/sessions/*/receipt",
        # Legacy (unversioned)
        "/api/costs/sessions/*/breakdown",
        "/api/costs/sessions/*/metrics",
        "/api/costs/sessions/*/receipt",
    ]

    RESOURCE_TYPE = "session_costs"

    def __init__(self, ctx: dict | None = None):
        self.ctx = ctx or {}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def can_handle(self, path: str) -> bool:
        return self._parse_path(path) is not None

    def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
        method: str = "GET",
    ) -> HandlerResult | None:
        if hasattr(handler, "command"):
            method = handler.command

        if method != "GET":
            return error_response("Method not allowed", 405)

        client_ip = get_client_ip(handler)
        if not _session_cost_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        parsed = self._parse_path(path)
        if parsed is None:
            return error_response("Not found", 404)

        debate_id, action = parsed

        if action == "breakdown":
            return self._get_breakdown(debate_id, query_params)
        elif action == "metrics":
            return self._get_metrics(debate_id, query_params)
        elif action == "receipt":
            return self._get_receipt(debate_id, query_params)

        return error_response("Not found", 404)

    # ------------------------------------------------------------------
    # Path parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_path(path: str) -> tuple[str, str] | None:
        """Extract (debate_id, action) from the path.

        Returns None when the path does not match the expected pattern.
        """
        for prefix in (_V1_PREFIX, _LEGACY_PREFIX):
            if not path.startswith(prefix):
                continue
            remainder = path[len(prefix) :]
            parts = remainder.strip("/").split("/", 1)
            if len(parts) == 2 and parts[0] and parts[1] in ("breakdown", "metrics", "receipt"):
                return parts[0], parts[1]
        return None

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def _get_storage(self) -> Any | None:
        return self.ctx.get("storage")

    def _verify_debate_exists(self, debate_id: str) -> str | None:
        """Return an error message if the debate doesn't exist, else None."""
        storage = self._get_storage()
        if not storage:
            return None  # No storage — skip existence check
        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return f"Debate not found: {debate_id}"
        except (AttributeError, TypeError, KeyError):
            pass  # Storage doesn't support get_debate — skip
        return None

    def _get_debate_cost_summary(self, debate_id: str) -> dict[str, Any]:
        """Fetch the DebateCostSummary dict for *debate_id*.

        Tries the per-debate DebateCostTracker first, falls back to the
        global CostTracker, and ultimately returns a zero-value stub.
        """
        # Try per-debate tracker
        try:
            from aragora.billing.debate_costs import get_debate_cost_tracker

            tracker = get_debate_cost_tracker()
            summary = tracker.get_debate_cost(debate_id)
            if summary.total_calls > 0:
                return summary.to_dict()
        except (ImportError, RuntimeError, TypeError) as exc:
            logger.debug("DebateCostTracker unavailable: %s", exc)

        # Try global cost tracker
        try:
            from aragora.billing.cost_tracker import get_cost_tracker

            global_tracker = get_cost_tracker()
            debate_cost = global_tracker._debate_costs.get(debate_id)
            if debate_cost and debate_cost > 0:
                return {
                    "debate_id": debate_id,
                    "total_cost_usd": str(debate_cost),
                    "total_tokens_in": 0,
                    "total_tokens_out": 0,
                    "total_calls": 0,
                    "per_agent": {},
                    "per_round": {},
                    "model_usage": {},
                    "started_at": None,
                    "completed_at": None,
                }
        except (ImportError, RuntimeError, TypeError) as exc:
            logger.debug("Global CostTracker unavailable: %s", exc)

        # Empty stub
        return {
            "debate_id": debate_id,
            "total_cost_usd": "0",
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_calls": 0,
            "per_agent": {},
            "per_round": {},
            "model_usage": {},
            "started_at": None,
            "completed_at": None,
        }

    # ------------------------------------------------------------------
    # Endpoint implementations
    # ------------------------------------------------------------------

    @require_permission("billing:read")
    def _get_breakdown(
        self,
        debate_id: str,
        query_params: dict[str, Any],
    ) -> HandlerResult:
        """Return a detailed cost breakdown for a debate session.

        Includes per-agent costs, per-round costs, model usage, and
        cost-per-decision calculation.
        """
        err = self._verify_debate_exists(debate_id)
        if err:
            return error_response(err, 404)

        try:
            summary = self._get_debate_cost_summary(debate_id)

            total_cost = Decimal(summary.get("total_cost_usd", "0"))
            total_calls = summary.get("total_calls", 0)
            total_tokens = summary.get("total_tokens_in", 0) + summary.get("total_tokens_out", 0)
            num_rounds = len(summary.get("per_round", {}))

            # Cost-per-decision: total cost / number of rounds that produced a decision
            cost_per_round = str(total_cost / max(num_rounds, 1)) if total_cost > 0 else "0"

            return json_response(
                {
                    "debate_id": debate_id,
                    "total_cost_usd": str(total_cost),
                    "total_tokens": total_tokens,
                    "total_tokens_in": summary.get("total_tokens_in", 0),
                    "total_tokens_out": summary.get("total_tokens_out", 0),
                    "total_api_calls": total_calls,
                    "cost_per_round": cost_per_round,
                    "num_rounds": num_rounds,
                    "per_agent": summary.get("per_agent", {}),
                    "per_round": summary.get("per_round", {}),
                    "model_usage": summary.get("model_usage", {}),
                    "started_at": summary.get("started_at"),
                    "completed_at": summary.get("completed_at"),
                }
            )
        except (ImportError, RuntimeError, OSError, ValueError, KeyError, TypeError) as e:
            logger.error("Session cost breakdown error: %s", e, exc_info=True)
            return error_response("Failed to retrieve cost breakdown", 500)

    @require_permission("billing:read")
    def _get_metrics(
        self,
        debate_id: str,
        query_params: dict[str, Any],
    ) -> HandlerResult:
        """Return usage metrics for a debate session.

        Provides token efficiency, cost efficiency, and per-agent
        performance metrics useful for optimization.
        """
        err = self._verify_debate_exists(debate_id)
        if err:
            return error_response(err, 404)

        try:
            summary = self._get_debate_cost_summary(debate_id)

            total_cost = Decimal(summary.get("total_cost_usd", "0"))
            total_calls = summary.get("total_calls", 0)
            tokens_in = summary.get("total_tokens_in", 0)
            tokens_out = summary.get("total_tokens_out", 0)
            total_tokens = tokens_in + tokens_out

            # Efficiency metrics
            avg_cost_per_call = str(total_cost / max(total_calls, 1)) if total_cost > 0 else "0"
            avg_tokens_per_call = total_tokens / max(total_calls, 1) if total_tokens > 0 else 0
            input_output_ratio = round(tokens_in / max(tokens_out, 1), 2) if tokens_in > 0 else 0

            # Per-agent metrics
            agent_metrics = {}
            for name, agent_data in summary.get("per_agent", {}).items():
                a_cost = Decimal(agent_data.get("total_cost_usd", "0"))
                a_calls = agent_data.get("call_count", 0)
                a_tokens = agent_data.get("total_tokens_in", 0) + agent_data.get(
                    "total_tokens_out", 0
                )
                agent_metrics[name] = {
                    "total_cost_usd": str(a_cost),
                    "call_count": a_calls,
                    "total_tokens": a_tokens,
                    "avg_cost_per_call": str(a_cost / max(a_calls, 1)) if a_cost > 0 else "0",
                    "avg_tokens_per_call": a_tokens / max(a_calls, 1) if a_tokens > 0 else 0,
                    "cost_share_pct": (
                        round(float(a_cost / total_cost * 100), 1) if total_cost > 0 else 0
                    ),
                }

            # Model metrics
            model_metrics = {}
            for key, model_data in summary.get("model_usage", {}).items():
                m_cost = Decimal(model_data.get("total_cost_usd", "0"))
                m_calls = model_data.get("call_count", 0)
                model_metrics[key] = {
                    "provider": model_data.get("provider", ""),
                    "model": model_data.get("model", key),
                    "total_cost_usd": str(m_cost),
                    "call_count": m_calls,
                    "avg_cost_per_call": str(m_cost / max(m_calls, 1)) if m_cost > 0 else "0",
                }

            return json_response(
                {
                    "debate_id": debate_id,
                    "summary": {
                        "total_cost_usd": str(total_cost),
                        "total_api_calls": total_calls,
                        "total_tokens": total_tokens,
                        "total_tokens_in": tokens_in,
                        "total_tokens_out": tokens_out,
                    },
                    "efficiency": {
                        "avg_cost_per_call": avg_cost_per_call,
                        "avg_tokens_per_call": avg_tokens_per_call,
                        "input_output_ratio": input_output_ratio,
                    },
                    "per_agent": agent_metrics,
                    "per_model": model_metrics,
                }
            )
        except (ImportError, RuntimeError, OSError, ValueError, KeyError, TypeError) as e:
            logger.error("Session cost metrics error: %s", e, exc_info=True)
            return error_response("Failed to retrieve usage metrics", 500)

    @require_permission("billing:read")
    def _get_receipt(
        self,
        debate_id: str,
        query_params: dict[str, Any],
    ) -> HandlerResult:
        """Generate a cost receipt for a debate session.

        The receipt is a self-contained, auditable document with a
        SHA-256 checksum that can be used for billing verification and
        compliance.
        """
        err = self._verify_debate_exists(debate_id)
        if err:
            return error_response(err, 404)

        try:
            summary = self._get_debate_cost_summary(debate_id)

            total_cost = Decimal(summary.get("total_cost_usd", "0"))
            total_calls = summary.get("total_calls", 0)
            tokens_in = summary.get("total_tokens_in", 0)
            tokens_out = summary.get("total_tokens_out", 0)
            num_rounds = len(summary.get("per_round", {}))

            # Build line items from per-agent data
            line_items = []
            for name, agent_data in summary.get("per_agent", {}).items():
                line_items.append(
                    {
                        "description": f"Agent: {name}",
                        "tokens_in": agent_data.get("total_tokens_in", 0),
                        "tokens_out": agent_data.get("total_tokens_out", 0),
                        "api_calls": agent_data.get("call_count", 0),
                        "cost_usd": agent_data.get("total_cost_usd", "0"),
                    }
                )

            now = datetime.now(timezone.utc)

            receipt_data = {
                "receipt_id": f"rcpt-{debate_id}-{now.strftime('%Y%m%d%H%M%S')}",
                "debate_id": debate_id,
                "generated_at": now.isoformat(),
                "period": {
                    "started_at": summary.get("started_at"),
                    "completed_at": summary.get("completed_at"),
                },
                "totals": {
                    "cost_usd": str(total_cost),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "api_calls": total_calls,
                    "rounds": num_rounds,
                },
                "line_items": line_items,
                "model_usage": summary.get("model_usage", {}),
            }

            # Compute SHA-256 checksum for tamper detection
            canonical = json.dumps(receipt_data, sort_keys=True, default=str)
            checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            receipt_data["checksum"] = checksum

            return json_response(receipt_data)
        except (ImportError, RuntimeError, OSError, ValueError, KeyError, TypeError) as e:
            logger.error("Session cost receipt error: %s", e, exc_info=True)
            return error_response("Failed to generate cost receipt", 500)


__all__ = ["SessionCostHandler"]
