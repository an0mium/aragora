---
title: Cost Visibility
description: Cost Visibility
---

# Cost Visibility

Aragora includes a cost visibility module for tracking AI usage, budgets, and
spend trends. It powers the `/costs` dashboard and provides an API for budget
alerts and breakdowns.

## Overview

The cost visibility layer provides:

- Total spend, tokens, and API call counts per workspace.
- Provider and feature breakdowns.
- Daily usage timelines for trend analysis.
- Budget alerts and projections.

The reference implementation uses in-memory storage. For production, connect a
durable store and wire cost ingestion from your usage tracker.

## Dashboard UI

Route: `/costs`

Key components:

- `aragora/live/src/components/costs/CostDashboard.tsx`
- `aragora/live/src/components/costs/CostBreakdownChart.tsx`
- `aragora/live/src/components/costs/UsageTimeline.tsx`
- `aragora/live/src/components/costs/BudgetAlerts.tsx`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/costs` | Cost dashboard summary |
| GET | `/api/costs/breakdown` | Breakdown by provider/feature |
| GET | `/api/costs/timeline` | Usage timeline |
| GET | `/api/costs/alerts` | Budget alerts |
| POST | `/api/costs/budget` | Set budget limits |
| POST | `/api/costs/alerts/\{alert_id\}/dismiss` | Dismiss alert |

### Fetch Cost Summary

```http
GET /api/costs?range=30d&workspace_id=default
```

### Set Budget

```http
POST /api/costs/budget
Content-Type: application/json

{
  "budget": 1500,
  "workspace_id": "default"
}
```

## Recording Usage

Use `record_cost` to ingest usage events:

```python
from aragora.server.handlers.costs import record_cost

record_cost(
    provider="Anthropic",
    feature="Debates",
    tokens_input=1200,
    tokens_output=800,
    cost=1.42,
    model="claude-sonnet-4-20250514",
    workspace_id="default",
    user_id="user_123",
)
```

## Notes

- Cost data is cached in memory by default.
- Budget alerts are triggered at 80% usage of the configured budget.
- Integrate a persistent store for multi-instance deployments.
