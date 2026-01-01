"""
Tournament system for agent competitions.

Provides structured competitions with multiple formats and ELO tracking.
"""

from aagora.tournaments.tournament import (
    Tournament,
    TournamentFormat,
    TournamentTask,
    TournamentMatch,
    TournamentStanding,
    TournamentResult,
    create_default_tasks,
)

__all__ = [
    "Tournament",
    "TournamentFormat",
    "TournamentTask",
    "TournamentMatch",
    "TournamentStanding",
    "TournamentResult",
    "create_default_tasks",
]
