# aragora.ranking

Agent skill tracking and reputation system built on ELO ratings. Tracks agent
performance across debates, supports domain-specific ratings, calibration scoring,
tournament management, and red team integration.

## Modules

| Module | Purpose |
|--------|---------|
| `elo.py` | `EloSystem` facade -- main entry point delegating to sub-modules |
| `elo_core.py` | Pure ELO calculation functions |
| `elo_matchmaking.py` | Match recording orchestration |
| `elo_leaderboard.py` | Snapshot-based leaderboard access |
| `elo_calibration.py` | Calibration leaderboard queries |
| `elo_domain.py` | Knowledge Mound integration |
| `elo_analysis.py` | Learning efficiency and voting accuracy |
| `calibration_engine.py` | Calibration recording and scoring |
| `calibration_database.py` | Calibration persistence |
| `database.py` | Core ELO database layer |
| `postgres_database.py` | Postgres-backed ELO storage |
| `leaderboard_engine.py` | Core leaderboard queries |
| `match_recorder.py` | Match persistence helpers |
| `tournaments.py` | `TournamentManager` -- bracket/round-robin tournaments |
| `relationships.py` | Agent-to-agent relationship tracking |
| `redteam.py` | Red team vulnerability integration |
| `pattern_matcher.py` | Task pattern classification and agent affinity |
| `km_elo_bridge.py` | Sync ELO data to/from the Knowledge Mound |
| `verification.py` | Formal verification ELO adjustments |
| `snapshot.py` | JSON snapshot import/export |
| `performance_integrator.py` | Performance metric integration |

## Key Concepts

- **ELO Ratings** -- Each agent holds a numeric rating (default 1000) updated after
  every debate match using a configurable K-factor.
- **Domain Ratings** -- Agents are rated per-domain so a model strong at code review
  is distinguished from one strong at policy analysis.
- **Calibration** -- `CalibrationEngine` scores how well an agent's stated confidence
  aligns with actual outcomes, bucketed into probability ranges.
- **Tournaments** -- `TournamentManager` runs bracket or round-robin events, tracks
  standings, and emits `TournamentEvent` updates.
- **Knowledge Mound Bridge** -- `KMEloBridge` syncs ratings and match history into
  the broader Knowledge Mound for cross-system learning.

## Usage

```python
from aragora.ranking import EloSystem, MatchResult

elo = EloSystem()

# Record a debate outcome
result = MatchResult(
    winner="claude-opus",
    loser="gpt-4o",
    domain="security",
)
elo.record_match(result)

# Query the leaderboard
leaderboard = elo.get_leaderboard(limit=10)

# Check a specific agent's rating
rating = elo.get_rating("claude-opus")
print(f"Rating: {rating.rating}, Matches: {rating.matches}")
```
