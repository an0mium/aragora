"""
Resource API modules for the Aragora client.

These modules contain the API interface classes that are used by AragoraClient.
"""

from .agents import AgentsAPI
from .audit import AuditAPI
from .consensus import ConsensusAPI
from .documents import DocumentsAPI
from .leaderboard import LeaderboardAPI
from .memory import MemoryAPI
from .pulse import PulseAPI
from .system import SystemAPI
from .tournaments import TournamentsAPI
from .verification import VerificationAPI

__all__ = [
    "AgentsAPI",
    "AuditAPI",
    "ConsensusAPI",
    "DocumentsAPI",
    "LeaderboardAPI",
    "MemoryAPI",
    "PulseAPI",
    "SystemAPI",
    "TournamentsAPI",
    "VerificationAPI",
]
