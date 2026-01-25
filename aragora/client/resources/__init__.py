"""
Resource API modules for the Aragora client.

These modules contain the API interface classes that are used by AragoraClient.
"""

from .agents import AgentsAPI
from .analytics import AnalyticsAPI
from .audit import AuditAPI
from .billing import BillingAPI
from .debates import DebatesAPI
from .consensus import ConsensusAPI
from .documents import DocumentsAPI
from .gauntlet import GauntletAPI
from .graph_debates import GraphDebatesAPI
from .leaderboard import LeaderboardAPI
from .matrix_debates import MatrixDebatesAPI
from .memory import MemoryAPI
from .pulse import PulseAPI
from .rbac import RBACAPI
from .replay import ReplayAPI
from .system import SystemAPI
from .tournaments import TournamentsAPI
from .verification import VerificationAPI

__all__ = [
    "AgentsAPI",
    "AnalyticsAPI",
    "AuditAPI",
    "BillingAPI",
    "ConsensusAPI",
    "DebatesAPI",
    "DocumentsAPI",
    "GauntletAPI",
    "GraphDebatesAPI",
    "LeaderboardAPI",
    "MatrixDebatesAPI",
    "MemoryAPI",
    "PulseAPI",
    "RBACAPI",
    "ReplayAPI",
    "SystemAPI",
    "TournamentsAPI",
    "VerificationAPI",
]
