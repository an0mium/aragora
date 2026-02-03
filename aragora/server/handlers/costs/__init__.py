"""
Cost Visibility API Handler package.

Provides API endpoints for tracking and visualizing AI costs:
- Total cost and budget tracking
- Cost breakdown by provider and feature
- Usage timeline data
- Budget alerts and projections
- Optimization suggestions
- Usage data export (CSV/JSON)

All public symbols are re-exported here for backward compatibility.
"""

# Data models
from .models import (
    BudgetAlert,
    CostEntry,
    CostSummary,
    _get_active_alerts,
    _get_cost_tracker,
    get_cost_summary,
    record_cost,
)

# Handler class
from .handler import CostHandler

# Helper functions
from .helpers import (
    _build_export_rows,
    _export_csv_response,
    _generate_mock_summary,
    _get_implementation_difficulty,
    _get_implementation_steps,
    _get_implementation_time,
)

# Route registration
from .routes import register_routes

__all__ = [
    # Data models
    "BudgetAlert",
    "CostEntry",
    "CostSummary",
    # Tracker integration
    "_get_active_alerts",
    "_get_cost_tracker",
    "get_cost_summary",
    "record_cost",
    # Handler
    "CostHandler",
    # Helpers
    "_build_export_rows",
    "_export_csv_response",
    "_generate_mock_summary",
    "_get_implementation_difficulty",
    "_get_implementation_steps",
    "_get_implementation_time",
    # Routes
    "register_routes",
]
