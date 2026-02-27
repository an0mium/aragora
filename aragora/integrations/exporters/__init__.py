"""Decision Plan exporters for external project management tools.

Pluggable adapter system that formats DecisionPlan outputs as tickets
and exports them to Jira, Linear, or generic webhooks (n8n/Zapier).

Usage:
    from aragora.integrations.exporters import DecisionExporter, WebhookAdapter

    exporter = DecisionExporter()
    exporter.register_adapter(WebhookAdapter(url="https://hooks.example.com/abc"))
    receipt = await exporter.export(decision_plan)
"""

from aragora.integrations.exporters.base import (
    ExportAdapter,
    ExportReceipt,
    ExportStatus,
    TicketData,
)
from aragora.integrations.exporters.exporter import DecisionExporter
from aragora.integrations.exporters.jira_adapter import JiraAdapter
from aragora.integrations.exporters.linear_adapter import LinearAdapter
from aragora.integrations.exporters.webhook_adapter import WebhookAdapter

__all__ = [
    "DecisionExporter",
    "ExportAdapter",
    "ExportReceipt",
    "ExportStatus",
    "JiraAdapter",
    "LinearAdapter",
    "TicketData",
    "WebhookAdapter",
]
