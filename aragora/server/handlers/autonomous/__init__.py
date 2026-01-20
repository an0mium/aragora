"""Autonomous operations handlers (Phase 5).

Provides HTTP endpoints for:
- Approval flows (human-in-the-loop)
- Scheduled triggers
- Alert management
- Trend monitoring
- Anomaly detection
- Continuous learning
"""

from aragora.server.handlers.autonomous.approvals import ApprovalHandler
from aragora.server.handlers.autonomous.alerts import AlertHandler
from aragora.server.handlers.autonomous.triggers import TriggerHandler
from aragora.server.handlers.autonomous.monitoring import MonitoringHandler
from aragora.server.handlers.autonomous.learning import LearningHandler

__all__ = [
    "ApprovalHandler",
    "AlertHandler",
    "TriggerHandler",
    "MonitoringHandler",
    "LearningHandler",
]
