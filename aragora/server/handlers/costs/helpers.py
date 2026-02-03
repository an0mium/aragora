"""
Cost visibility helper functions.

Provides mock data generation, export formatting, and recommendation helpers.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from typing import Any

from aiohttp import web

from .models import CostSummary


def _generate_mock_summary(time_range: str) -> CostSummary:
    """Generate mock data for demo."""
    now = datetime.now(timezone.utc)

    # Generate daily data
    range_days = {"24h": 1, "7d": 7, "30d": 30, "90d": 90}.get(time_range, 7)
    daily_costs = []
    total_cost = 0
    total_tokens = 0

    for i in range(range_days):
        date = now - timedelta(days=range_days - 1 - i)
        cost = 15 + (i % 5) * 3 + (hash(date.strftime("%Y-%m-%d")) % 10)
        tokens = int(cost * 25000)
        daily_costs.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "cost": round(cost, 2),
                "tokens": tokens,
            }
        )
        total_cost += cost
        total_tokens += tokens

    return CostSummary(
        total_cost=round(total_cost, 2),
        budget=500.00,
        tokens_used=total_tokens,
        api_calls=int(total_cost * 100),
        last_updated=now,
        cost_by_provider=[
            {"name": "Anthropic", "cost": round(total_cost * 0.616, 2), "percentage": 61.6},
            {"name": "OpenAI", "cost": round(total_cost * 0.276, 2), "percentage": 27.6},
            {"name": "Mistral", "cost": round(total_cost * 0.069, 2), "percentage": 6.9},
            {"name": "OpenRouter", "cost": round(total_cost * 0.039, 2), "percentage": 3.9},
        ],
        cost_by_feature=[
            {"name": "Debates", "cost": round(total_cost * 0.432, 2), "percentage": 43.2},
            {"name": "Email Triage", "cost": round(total_cost * 0.255, 2), "percentage": 25.5},
            {"name": "Code Review", "cost": round(total_cost * 0.179, 2), "percentage": 17.9},
            {"name": "Knowledge Work", "cost": round(total_cost * 0.134, 2), "percentage": 13.4},
        ],
        daily_costs=daily_costs,
        alerts=[
            {
                "id": "1",
                "type": "budget_warning",
                "message": "Projected to reach 80% of monthly budget by Jan 25",
                "severity": "warning",
                "timestamp": (now - timedelta(hours=1)).isoformat(),
            },
            {
                "id": "2",
                "type": "spike_detected",
                "message": "Unusual spike in Debate costs detected (45% above average)",
                "severity": "info",
                "timestamp": (now - timedelta(hours=2)).isoformat(),
            },
        ],
    )


def _build_export_rows(summary: CostSummary, group_by: str) -> list[dict[str, Any]]:
    """Build export rows from cost summary based on grouping."""
    if group_by == "provider":
        return [
            {
                "name": entry["name"],
                "cost": entry["cost"],
                "percentage": entry["percentage"],
            }
            for entry in summary.cost_by_provider
        ]
    elif group_by == "feature":
        return [
            {
                "name": entry["name"],
                "cost": entry["cost"],
                "percentage": entry["percentage"],
            }
            for entry in summary.cost_by_feature
        ]
    else:
        # daily (default)
        return [
            {
                "date": entry.get("date", ""),
                "cost": entry.get("cost", 0),
                "tokens": entry.get("tokens", 0),
            }
            for entry in summary.daily_costs
        ]


def _export_csv_response(
    rows: list[dict[str, Any]], workspace_id: str, time_range: str
) -> web.Response:
    """Generate a CSV response from export rows."""
    if not rows:
        return web.Response(
            text="",
            content_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="costs_{workspace_id}_{time_range}.csv"'
            },
        )

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

    return web.Response(
        text=output.getvalue(),
        content_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="costs_{workspace_id}_{time_range}.csv"'
        },
    )


def _get_implementation_steps(rec_type: str) -> list[str]:
    """Get implementation steps for a recommendation type."""
    steps = {
        "model_downgrade": [
            "Identify tasks where lower-tier models would suffice",
            "Update model configuration for identified tasks",
            "Monitor quality metrics after switch",
            "Adjust if quality degrades below threshold",
        ],
        "caching": [
            "Enable response caching for repeated queries",
            "Configure cache TTL based on data freshness requirements",
            "Monitor cache hit rates",
            "Tune cache size based on usage patterns",
        ],
        "batching": [
            "Identify operations that can be batched together",
            "Implement request batching logic",
            "Set appropriate batch sizes and timeouts",
            "Monitor latency impact",
        ],
        "rate_limiting": [
            "Implement rate limiting for non-critical operations",
            "Set appropriate limits based on priority",
            "Add queue for overflow requests",
            "Monitor queue depths and wait times",
        ],
    }
    return steps.get(
        rec_type,
        [
            "Review recommendation details",
            "Plan implementation",
            "Execute changes",
            "Monitor results",
        ],
    )


def _get_implementation_difficulty(rec_type: str) -> str:
    """Get implementation difficulty for a recommendation type."""
    difficulties = {
        "model_downgrade": "easy",
        "caching": "medium",
        "batching": "medium",
        "rate_limiting": "easy",
    }
    return difficulties.get(rec_type, "medium")


def _get_implementation_time(rec_type: str) -> str:
    """Get estimated implementation time for a recommendation type."""
    times = {
        "model_downgrade": "< 1 hour",
        "caching": "2-4 hours",
        "batching": "4-8 hours",
        "rate_limiting": "1-2 hours",
    }
    return times.get(rec_type, "2-4 hours")
