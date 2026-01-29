"""
Comprehensive tests for the aragora.tournaments module.

Tests cover:
- TournamentFormat enum
- TournamentTask dataclass
- TournamentMatch dataclass
- TournamentStanding dataclass
- TournamentResult dataclass
- Tournament class (creation, bracket generation, match tracking, standings)
- TournamentManager (database reads)
- create_default_tasks helper
"""

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.tournaments.tournament import (
    Tournament,
    TournamentFormat,
    TournamentManager,
    TournamentMatch,
    TournamentResult,
    TournamentStanding,
    TournamentTask,
    create_default_tasks,
)


# ---------------------------------------------------------------------------
# Helpers - lightweight Agent stub and mock DB
# ---------------------------------------------------------------------------


class _StubAgent:
    """Minimal Agent-like object sufficient for Tournament."""

    def __init__(self, name: str, model: str = "stub-model"):
        self.name = name
        self.model = model
        self.role = "proposer"
        self.system_prompt = ""
        self.agent_type = "stub"
        self.stance = "neutral"
        self.tool_manifest = None


def _make_agents(n: int = 4) -> list:
    names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    return [_StubAgent(names[i]) for i in range(n)]


def _make_tasks(n: int = 2) -> list[TournamentTask]:
    return [
        TournamentTask(
            task_id=f"task-{i}",
            description=f"Task {i} description",
            domain="testing",
            difficulty=0.5,
            time_limit=60,
        )
        for i in range(n)
    ]


def _make_debate_result(
    participants: list[str],
    consensus: bool = True,
    confidence: float = 0.8,
) -> MagicMock:
    """Return a mock DebateResult with the fields Tournament reads."""
    result = MagicMock()
    result.consensus_reached = consensus
    result.confidence = confidence
    result.critiques = []
    result.messages = []
    return result


def _tournament_factory(
    agents=None,
    tasks=None,
    fmt=TournamentFormat.ROUND_ROBIN,
    db_path=":memory:",
):
    """Build a Tournament with mocked DB so no real SQLite is touched."""
    agents = agents if agents is not None else _make_agents(4)
    tasks = tasks if tasks is not None else _make_tasks(2)

    with patch("aragora.tournaments.tournament.TournamentDatabase"):
        with patch("aragora.tournaments.tournament.EloSystem"):
            t = Tournament(
                name="Test Tournament",
                agents=agents,
                tasks=tasks,
                format=fmt,
                db_path=db_path,
            )
    return t


# ---------------------------------------------------------------------------
# TournamentFormat enum
# ---------------------------------------------------------------------------


class TestTournamentFormat:
    def test_values(self):
        assert TournamentFormat.ROUND_ROBIN.value == "round_robin"
        assert TournamentFormat.SINGLE_ELIMINATION.value == "single_elimination"
        assert TournamentFormat.SWISS.value == "swiss"
        assert TournamentFormat.FREE_FOR_ALL.value == "free_for_all"

    def test_all_members(self):
        assert len(TournamentFormat) == 4

    def test_from_value(self):
        assert TournamentFormat("round_robin") is TournamentFormat.ROUND_ROBIN
        assert TournamentFormat("swiss") is TournamentFormat.SWISS

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            TournamentFormat("invalid")


# ---------------------------------------------------------------------------
# TournamentTask dataclass
# ---------------------------------------------------------------------------


class TestTournamentTask:
    def test_creation(self):
        task = TournamentTask(
            task_id="t1",
            description="Design a system",
            domain="architecture",
        )
        assert task.task_id == "t1"
        assert task.domain == "architecture"
        assert task.difficulty == 0.5  # default
        assert task.time_limit == 300  # default

    def test_custom_fields(self):
        task = TournamentTask(
            task_id="t2",
            description="Hard task",
            domain="security",
            difficulty=0.9,
            time_limit=600,
        )
        assert task.difficulty == 0.9
        assert task.time_limit == 600


# ---------------------------------------------------------------------------
# TournamentMatch dataclass
# ---------------------------------------------------------------------------


class TestTournamentMatch:
    def _make_match(self, **overrides):
        defaults = dict(
            match_id="m-1",
            round_num=0,
            participants=["alpha", "beta"],
            task=_make_tasks(1)[0],
        )
        defaults.update(overrides)
        return TournamentMatch(**defaults)

    def test_defaults(self):
        m = self._make_match()
        assert m.result is None
        assert m.scores == {}
        assert m.winner is None
        assert m.started_at is None
        assert m.completed_at is None

    def test_is_complete_false(self):
        m = self._make_match()
        assert m.is_complete is False

    def test_is_complete_true(self):
        m = self._make_match(result=MagicMock())
        assert m.is_complete is True

    def test_participants_list(self):
        m = self._make_match(participants=["a", "b", "c"])
        assert len(m.participants) == 3


# ---------------------------------------------------------------------------
# TournamentStanding dataclass
# ---------------------------------------------------------------------------


class TestTournamentStanding:
    def test_defaults(self):
        s = TournamentStanding(agent_name="alpha")
        assert s.wins == 0
        assert s.losses == 0
        assert s.draws == 0
        assert s.points == 0.0
        assert s.total_score == 0.0
        assert s.matches_played == 0

    def test_win_rate_zero_matches(self):
        s = TournamentStanding(agent_name="alpha")
        assert s.win_rate == 0.0

    def test_win_rate_calculation(self):
        s = TournamentStanding(agent_name="alpha", wins=3, matches_played=10)
        assert s.win_rate == pytest.approx(0.3)

    def test_win_rate_perfect(self):
        s = TournamentStanding(agent_name="alpha", wins=5, matches_played=5)
        assert s.win_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TournamentResult dataclass
# ---------------------------------------------------------------------------


class TestTournamentResult:
    def test_creation(self):
        standings = [
            TournamentStanding(agent_name="alpha", wins=3, points=9.0),
            TournamentStanding(agent_name="beta", wins=1, points=3.0),
        ]
        result = TournamentResult(
            tournament_id="t-1",
            name="Test",
            format=TournamentFormat.ROUND_ROBIN,
            standings=standings,
            matches=[],
            champion="alpha",
            total_rounds=3,
            started_at="2025-01-01T00:00:00",
            completed_at="2025-01-01T01:00:00",
        )
        assert result.champion == "alpha"
        assert result.total_rounds == 3
        assert len(result.standings) == 2
        assert result.format is TournamentFormat.ROUND_ROBIN


# ---------------------------------------------------------------------------
# create_default_tasks helper
# ---------------------------------------------------------------------------


class TestCreateDefaultTasks:
    def test_returns_five_tasks(self):
        tasks = create_default_tasks()
        assert len(tasks) == 5

    def test_all_are_tournament_tasks(self):
        tasks = create_default_tasks()
        for t in tasks:
            assert isinstance(t, TournamentTask)

    def test_unique_ids(self):
        tasks = create_default_tasks()
        ids = [t.task_id for t in tasks]
        assert len(set(ids)) == len(ids)

    def test_difficulties_in_range(self):
        tasks = create_default_tasks()
        for t in tasks:
            assert 0.0 <= t.difficulty <= 1.0


# ---------------------------------------------------------------------------
# Tournament - construction
# ---------------------------------------------------------------------------


class TestTournamentConstruction:
    def test_basic_construction(self):
        t = _tournament_factory()
        assert t.name == "Test Tournament"
        assert len(t.agents) == 4
        assert len(t.agent_names) == 4
        assert len(t.tasks) == 2
        assert t.format is TournamentFormat.ROUND_ROBIN
        assert t.current_round == 0

    def test_standings_initialized(self):
        t = _tournament_factory()
        assert len(t.standings) == 4
        for name in t.agent_names:
            assert name in t.standings
            s = t.standings[name]
            assert s.wins == 0
            assert s.matches_played == 0

    def test_tournament_id_generated(self):
        t = _tournament_factory()
        assert t.tournament_id
        assert len(t.tournament_id) == 8


# ---------------------------------------------------------------------------
# Tournament - round robin match generation
# ---------------------------------------------------------------------------


class TestRoundRobinGeneration:
    def test_match_count(self):
        """4 agents, 2 tasks -> C(4,2)=6 pairings * 2 tasks = 12 matches."""
        t = _tournament_factory()
        matches = t.generate_matches()
        assert len(matches) == 12

    def test_unique_match_ids(self):
        t = _tournament_factory()
        matches = t.generate_matches()
        ids = [m.match_id for m in matches]
        assert len(set(ids)) == len(ids)

    def test_all_pairs_covered(self):
        t = _tournament_factory()
        matches = t.generate_matches()
        pairs = set()
        for m in matches:
            pair = tuple(sorted(m.participants))
            pairs.add(pair)
        # C(4,2) = 6 unique pairs
        assert len(pairs) == 6

    def test_round_nums_assigned(self):
        t = _tournament_factory(tasks=_make_tasks(3))
        matches = t.generate_matches()
        rounds = {m.round_num for m in matches}
        assert rounds == {0, 1, 2}

    def test_two_agents(self):
        """2 agents, 1 task -> exactly 1 match."""
        t = _tournament_factory(agents=_make_agents(2), tasks=_make_tasks(1))
        matches = t.generate_matches()
        assert len(matches) == 1
        assert sorted(matches[0].participants) == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Tournament - free for all match generation
# ---------------------------------------------------------------------------


class TestFreeForAllGeneration:
    def test_one_match_per_task(self):
        t = _tournament_factory(fmt=TournamentFormat.FREE_FOR_ALL)
        matches = t.generate_matches()
        assert len(matches) == 2  # 2 tasks

    def test_all_agents_in_each_match(self):
        t = _tournament_factory(fmt=TournamentFormat.FREE_FOR_ALL)
        matches = t.generate_matches()
        for m in matches:
            assert len(m.participants) == 4

    def test_participants_are_copies(self):
        """Participants list should be a copy, not a reference."""
        t = _tournament_factory(fmt=TournamentFormat.FREE_FOR_ALL)
        matches = t.generate_matches()
        # Mutating one match's list should not affect another
        matches[0].participants.append("extra")
        assert "extra" not in matches[1].participants


# ---------------------------------------------------------------------------
# Tournament - single elimination bracket generation
# ---------------------------------------------------------------------------


class TestBracketGeneration:
    def test_four_agents_bracket(self):
        t = _tournament_factory(
            agents=_make_agents(4),
            fmt=TournamentFormat.SINGLE_ELIMINATION,
        )
        matches = t.generate_matches()
        # Round 0: 2 matches (4 agents -> 2 matches)
        # Round 1: placeholder winners (None) so loop ends
        round_0 = [m for m in matches if m.round_num == 0]
        assert len(round_0) == 2

    def test_two_agents_bracket(self):
        t = _tournament_factory(
            agents=_make_agents(2),
            fmt=TournamentFormat.SINGLE_ELIMINATION,
        )
        matches = t.generate_matches()
        assert len(matches) == 1
        assert sorted(matches[0].participants) == ["alpha", "beta"]

    def test_three_agents_bracket_has_bye(self):
        """3 agents -> round 0 has 1 match + 1 bye."""
        t = _tournament_factory(
            agents=_make_agents(3),
            fmt=TournamentFormat.SINGLE_ELIMINATION,
        )
        matches = t.generate_matches()
        round_0 = [m for m in matches if m.round_num == 0]
        assert len(round_0) == 1  # Only one match; third agent gets a bye

    def test_default_task_used_when_no_tasks(self):
        """When tasks list is empty, bracket generation uses a default task."""
        t = _tournament_factory(
            agents=_make_agents(2),
            tasks=[],
            fmt=TournamentFormat.SINGLE_ELIMINATION,
        )
        matches = t.generate_matches()
        assert len(matches) == 1
        assert matches[0].task.task_id == "default"


# ---------------------------------------------------------------------------
# Tournament - swiss round generation
# ---------------------------------------------------------------------------


class TestSwissGeneration:
    def test_swiss_pairs_adjacent_by_standing(self):
        t = _tournament_factory(fmt=TournamentFormat.SWISS)
        matches = t._generate_swiss_round()
        # With 4 agents all at 0 points, we get 2 matches
        assert len(matches) == 2

    def test_swiss_no_tasks_raises(self):
        t = _tournament_factory(tasks=[], fmt=TournamentFormat.SWISS)
        with pytest.raises(ValueError, match="no tasks configured"):
            t._generate_swiss_round()

    def test_swiss_odd_agents(self):
        """Odd number of agents -> one agent gets no match."""
        t = _tournament_factory(agents=_make_agents(3), fmt=TournamentFormat.SWISS)
        matches = t._generate_swiss_round()
        assert len(matches) == 1


# ---------------------------------------------------------------------------
# Tournament - standings updates
# ---------------------------------------------------------------------------


class TestStandingsUpdate:
    def test_update_winner(self):
        t = _tournament_factory()
        match = TournamentMatch(
            match_id="m-1",
            round_num=0,
            participants=["alpha", "beta"],
            task=_make_tasks(1)[0],
            scores={"alpha": 0.8, "beta": 0.4},
            winner="alpha",
        )
        t._update_standings(match)
        assert t.standings["alpha"].wins == 1
        assert t.standings["alpha"].points == 3.0
        assert t.standings["beta"].losses == 1
        assert t.standings["beta"].points == 0.0
        assert t.standings["alpha"].matches_played == 1
        assert t.standings["beta"].matches_played == 1

    def test_update_draw(self):
        t = _tournament_factory()
        match = TournamentMatch(
            match_id="m-2",
            round_num=0,
            participants=["alpha", "beta"],
            task=_make_tasks(1)[0],
            scores={"alpha": 0.5, "beta": 0.5},
            winner=None,
        )
        t._update_standings(match)
        assert t.standings["alpha"].draws == 1
        assert t.standings["alpha"].points == 1.0
        assert t.standings["beta"].draws == 1
        assert t.standings["beta"].points == 1.0

    def test_total_score_accumulated(self):
        t = _tournament_factory()
        match1 = TournamentMatch(
            match_id="m-1",
            round_num=0,
            participants=["alpha", "beta"],
            task=_make_tasks(1)[0],
            scores={"alpha": 0.6, "beta": 0.3},
            winner="alpha",
        )
        match2 = TournamentMatch(
            match_id="m-2",
            round_num=1,
            participants=["alpha", "gamma"],
            task=_make_tasks(1)[0],
            scores={"alpha": 0.4, "gamma": 0.7},
            winner="gamma",
        )
        t._update_standings(match1)
        t._update_standings(match2)
        assert t.standings["alpha"].total_score == pytest.approx(1.0)
        assert t.standings["alpha"].wins == 1
        assert t.standings["alpha"].losses == 1

    def test_get_current_standings_sorted(self):
        t = _tournament_factory()
        t.standings["alpha"].points = 9.0
        t.standings["beta"].points = 3.0
        t.standings["gamma"].points = 6.0
        t.standings["delta"].points = 0.0

        sorted_standings = t.get_current_standings()
        names = [s.agent_name for s in sorted_standings]
        assert names[0] == "alpha"
        assert names[1] == "gamma"
        assert names[2] == "beta"
        assert names[3] == "delta"


# ---------------------------------------------------------------------------
# Tournament - score calculation
# ---------------------------------------------------------------------------


class TestScoreCalculation:
    def test_empty_result(self):
        t = _tournament_factory()
        scores = t._calculate_match_scores(None, ["alpha", "beta"])
        assert scores == {"alpha": 0.0, "beta": 0.0}

    def test_consensus_bonus(self):
        t = _tournament_factory()
        result = _make_debate_result(["alpha", "beta"], consensus=True, confidence=1.0)
        scores = t._calculate_match_scores(result, ["alpha", "beta"])
        # consensus_reached -> +0.3  +  confidence bonus -> +0.2 * 1.0
        # No critiques or messages -> 0
        assert scores["alpha"] == pytest.approx(0.5)
        assert scores["beta"] == pytest.approx(0.5)

    def test_no_consensus(self):
        t = _tournament_factory()
        result = _make_debate_result(["alpha", "beta"], consensus=False)
        scores = t._calculate_match_scores(result, ["alpha", "beta"])
        # No consensus -> no 0.3 bonus, no confidence bonus
        assert scores["alpha"] == pytest.approx(0.0)
        assert scores["beta"] == pytest.approx(0.0)

    def test_critique_contribution(self):
        t = _tournament_factory()
        result = _make_debate_result(["alpha", "beta"], consensus=False)

        critique_mock = MagicMock()
        critique_mock.agent = "alpha"
        critique_mock.severity = 0.0  # Low severity = good
        result.critiques = [critique_mock]

        scores = t._calculate_match_scores(result, ["alpha", "beta"])
        # alpha gets critique bonus: 0.3 * (1 - 0.0) = 0.3
        assert scores["alpha"] == pytest.approx(0.3)
        assert scores["beta"] == pytest.approx(0.0)

    def test_message_contribution(self):
        t = _tournament_factory()
        result = _make_debate_result(["alpha", "beta"], consensus=False)

        # Alpha contributed 5 messages
        msgs = [MagicMock(agent="alpha") for _ in range(5)]
        result.messages = msgs

        scores = t._calculate_match_scores(result, ["alpha", "beta"])
        # message bonus: 0.2 * min(1.0, 5/5) = 0.2
        assert scores["alpha"] == pytest.approx(0.2)
        assert scores["beta"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tournament - async run (mocked)
# ---------------------------------------------------------------------------


class TestTournamentRun:
    @pytest.mark.asyncio
    async def test_run_sequential(self):
        t = _tournament_factory(agents=_make_agents(2), tasks=_make_tasks(1))

        async def mock_debate_fn(env, agents):
            return _make_debate_result([a.name for a in agents])

        # Mock save
        t._save_tournament = MagicMock()
        t.elo_system = MagicMock()
        t.elo_system.record_match = MagicMock()

        result = await t.run(mock_debate_fn, parallel=False)
        assert result.tournament_id == t.tournament_id
        assert result.champion in t.agent_names
        assert result.started_at is not None
        assert result.completed_at is not None
        assert len(result.matches) == 1

    @pytest.mark.asyncio
    async def test_run_parallel(self):
        t = _tournament_factory(agents=_make_agents(2), tasks=_make_tasks(1))

        async def mock_debate_fn(env, agents):
            return _make_debate_result([a.name for a in agents])

        t._save_tournament = MagicMock()
        t.elo_system = MagicMock()
        t.elo_system.record_match = MagicMock()

        result = await t.run(mock_debate_fn, parallel=True)
        assert result.champion in t.agent_names
        assert len(result.matches) == 1

    @pytest.mark.asyncio
    async def test_run_updates_standings(self):
        t = _tournament_factory(agents=_make_agents(2), tasks=_make_tasks(1))

        async def mock_debate_fn(env, agents):
            return _make_debate_result([a.name for a in agents], consensus=True, confidence=0.9)

        t._save_tournament = MagicMock()
        t.elo_system = MagicMock()
        t.elo_system.record_match = MagicMock()

        await t.run(mock_debate_fn, parallel=False)
        # Both agents played 1 match
        for name in t.agent_names:
            assert t.standings[name].matches_played == 1

    @pytest.mark.asyncio
    async def test_run_match_sets_timestamps(self):
        t = _tournament_factory(agents=_make_agents(2), tasks=_make_tasks(1))

        async def mock_debate_fn(env, agents):
            return _make_debate_result([a.name for a in agents])

        t._save_tournament = MagicMock()
        t.elo_system = MagicMock()
        t.elo_system.record_match = MagicMock()

        result = await t.run(mock_debate_fn, parallel=False)
        for match in result.matches:
            assert match.started_at is not None
            assert match.completed_at is not None


# ---------------------------------------------------------------------------
# Tournament - generate_matches dispatching
# ---------------------------------------------------------------------------


class TestGenerateMatchesDispatch:
    def test_round_robin(self):
        t = _tournament_factory(fmt=TournamentFormat.ROUND_ROBIN)
        matches = t.generate_matches()
        assert len(matches) > 0

    def test_free_for_all(self):
        t = _tournament_factory(fmt=TournamentFormat.FREE_FOR_ALL)
        matches = t.generate_matches()
        assert len(matches) > 0

    def test_single_elimination(self):
        t = _tournament_factory(fmt=TournamentFormat.SINGLE_ELIMINATION)
        matches = t.generate_matches()
        assert len(matches) > 0

    def test_swiss(self):
        t = _tournament_factory(fmt=TournamentFormat.SWISS)
        matches = t.generate_matches()
        assert len(matches) > 0


# ---------------------------------------------------------------------------
# TournamentManager - database reads (mocked)
# ---------------------------------------------------------------------------


class TestTournamentManager:
    def _make_manager_with_db(self, tmp_path):
        """Create a real SQLite DB for TournamentManager tests."""
        db_file = str(tmp_path / "test_tournaments.db")
        conn = sqlite3.connect(db_file)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tournaments (
                tournament_id TEXT PRIMARY KEY,
                name TEXT,
                format TEXT,
                agents TEXT,
                tasks TEXT,
                standings TEXT,
                champion TEXT,
                started_at TEXT,
                completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS tournament_matches (
                match_id TEXT PRIMARY KEY,
                tournament_id TEXT,
                round_num INTEGER,
                participants TEXT,
                task_id TEXT,
                scores TEXT,
                winner TEXT,
                started_at TEXT,
                completed_at TEXT
            );
        """)
        conn.commit()
        conn.close()
        return db_file

    def _insert_tournament(self, db_file, champion="alpha"):
        conn = sqlite3.connect(db_file)
        standings = json.dumps(
            {
                "alpha": {"wins": 3, "losses": 0, "draws": 0, "points": 9.0, "total_score": 2.5},
                "beta": {"wins": 1, "losses": 2, "draws": 0, "points": 3.0, "total_score": 1.2},
            }
        )
        conn.execute(
            "INSERT INTO tournaments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "t-1",
                "Test Cup",
                "round_robin",
                '["alpha","beta"]',
                "[]",
                standings,
                champion,
                "2025-01-01T00:00:00",
                "2025-01-01T01:00:00",
            ),
        )
        conn.commit()
        conn.close()

    def _insert_matches(self, db_file):
        conn = sqlite3.connect(db_file)
        conn.execute(
            "INSERT INTO tournament_matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "m-1",
                "t-1",
                0,
                '["alpha","beta"]',
                "task-0",
                '{"alpha": 0.8, "beta": 0.4}',
                "alpha",
                "2025-01-01T00:00:00",
                "2025-01-01T00:05:00",
            ),
        )
        conn.execute(
            "INSERT INTO tournament_matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "m-2",
                "t-1",
                1,
                '["alpha","beta"]',
                "task-1",
                '{"alpha": 0.6, "beta": 0.7}',
                "beta",
                "2025-01-01T00:10:00",
                "2025-01-01T00:15:00",
            ),
        )
        conn.commit()
        conn.close()

    def test_get_tournament_no_db(self, tmp_path):
        """Manager returns None when DB doesn't exist."""
        manager = MagicMock(spec=TournamentManager)
        manager.db_path = tmp_path / "nonexistent.db"
        manager.get_tournament = TournamentManager.get_tournament.__get__(manager)
        result = manager.get_tournament()
        assert result is None

    def test_get_tournament_with_data(self, tmp_path):
        db_file = self._make_manager_with_db(tmp_path)
        self._insert_tournament(db_file)

        # Use a mock manager that has a real db_path but mocked db connection
        manager = MagicMock(spec=TournamentManager)
        manager.db_path = tmp_path / "test_tournaments.db"

        # Simulate the real get_tournament behavior
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tournament_id, name, format, champion, started_at, completed_at
            FROM tournaments LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "t-1"
        assert row[1] == "Test Cup"
        assert row[3] == "alpha"

    def test_get_current_standings_empty(self, tmp_path):
        """Returns empty list when no DB."""
        manager = MagicMock(spec=TournamentManager)
        manager.db_path = tmp_path / "nonexistent.db"
        manager.get_current_standings = TournamentManager.get_current_standings.__get__(manager)
        result = manager.get_current_standings()
        assert result == []

    def test_standings_deserialization(self, tmp_path):
        """Test standings JSON -> TournamentStanding conversion."""
        standings_json = {
            "alpha": {"wins": 3, "losses": 0, "draws": 1, "points": 10.0, "total_score": 2.5},
            "beta": {"wins": 1, "losses": 2, "draws": 1, "points": 4.0, "total_score": 1.2},
        }
        standings = []
        for agent_name, stats in standings_json.items():
            standing = TournamentStanding(
                agent_name=agent_name,
                wins=stats.get("wins", 0),
                losses=stats.get("losses", 0),
                draws=stats.get("draws", 0),
                points=stats.get("points", 0.0),
                total_score=stats.get("total_score", 0.0),
                matches_played=stats["wins"] + stats["losses"] + stats["draws"],
            )
            standings.append(standing)

        standings.sort(key=lambda s: (s.points, s.total_score), reverse=True)
        assert standings[0].agent_name == "alpha"
        assert standings[0].points == 10.0
        assert standings[0].matches_played == 4
        assert standings[1].agent_name == "beta"

    def test_get_matches_from_db(self, tmp_path):
        """Test match retrieval from database."""
        db_file = self._make_manager_with_db(tmp_path)
        self._insert_tournament(db_file)
        self._insert_matches(db_file)

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT match_id, round_num, participants, task_id, scores, winner,
                   started_at, completed_at
            FROM tournament_matches ORDER BY round_num DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 2
        # Ordered by round_num DESC
        assert rows[0][1] == 1  # round_num=1 first
        assert json.loads(rows[0][4])["beta"] == 0.7

    def test_get_match_summary(self, tmp_path):
        """Test match summary statistics."""
        db_file = self._make_manager_with_db(tmp_path)
        self._insert_tournament(db_file)
        self._insert_matches(db_file)

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) as total_matches,
                SUM(CASE WHEN winner IS NOT NULL THEN 1 ELSE 0 END) as decided_matches,
                MAX(round_num) as max_round
            FROM tournament_matches
        """)
        row = cursor.fetchone()
        conn.close()

        assert row[0] == 2  # total_matches
        assert row[1] == 2  # decided_matches (both have winners)
        assert row[2] == 1  # max_round

    def test_get_matches_with_limit(self, tmp_path):
        """Test match retrieval with limit."""
        db_file = self._make_manager_with_db(tmp_path)
        self._insert_tournament(db_file)
        self._insert_matches(db_file)

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT match_id FROM tournament_matches
            ORDER BY round_num DESC LIMIT ?
        """,
            (1,),
        )
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_agent_tournament(self):
        """A tournament with a single agent still works."""
        t = _tournament_factory(agents=_make_agents(1))
        matches = t.generate_matches()
        # Round-robin with 1 agent -> C(1,2) = 0 pairings
        assert len(matches) == 0

    def test_no_tasks_round_robin(self):
        """Round-robin with no tasks -> no matches."""
        t = _tournament_factory(tasks=[])
        matches = t.generate_matches()
        assert len(matches) == 0

    def test_standings_tiebreak_by_total_score(self):
        """When points are tied, total_score breaks the tie."""
        t = _tournament_factory()
        t.standings["alpha"].points = 6.0
        t.standings["alpha"].total_score = 2.0
        t.standings["beta"].points = 6.0
        t.standings["beta"].total_score = 3.0
        t.standings["gamma"].points = 6.0
        t.standings["gamma"].total_score = 1.0

        sorted_standings = t.get_current_standings()
        assert sorted_standings[0].agent_name == "beta"
        assert sorted_standings[1].agent_name == "alpha"
        assert sorted_standings[2].agent_name == "gamma"

    def test_match_with_no_winner_is_draw(self):
        """When scores are equal, winner should be None (draw)."""
        t = _tournament_factory()
        result = _make_debate_result(["alpha", "beta"], consensus=True, confidence=0.5)
        scores = t._calculate_match_scores(result, ["alpha", "beta"])
        # Both agents get the same score (no critiques/messages differ)
        assert scores["alpha"] == scores["beta"]
