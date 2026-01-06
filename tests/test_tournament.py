"""
Tests for aragora/tournaments/tournament.py

Comprehensive tests for the tournament system including:
- Tournament formats (round-robin, bracket, swiss, free-for-all)
- Match generation and scoring
- Standings and rankings
- Database persistence
- TournamentManager read-only access
"""

import asyncio
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.tournaments.tournament import (
    TournamentFormat,
    TournamentTask,
    TournamentMatch,
    TournamentStanding,
    TournamentResult,
    Tournament,
    TournamentManager,
    create_default_tasks,
)
from aragora.core import Agent, DebateResult, Environment, Message, Critique


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        yield f.name
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass


@pytest.fixture
def sample_task():
    """Create a sample TournamentTask."""
    return TournamentTask(
        task_id="test-task-001",
        description="Design a test system",
        domain="testing",
        difficulty=0.5,
        time_limit=300,
    )


@pytest.fixture
def sample_tasks():
    """Create a list of sample tasks."""
    return [
        TournamentTask(
            task_id=f"task-{i}",
            description=f"Task {i} description",
            domain="testing",
            difficulty=0.5,
        )
        for i in range(3)
    ]


@pytest.fixture
def mock_agents():
    """Create mock agents for testing."""
    agents = []
    for name in ["agent-a", "agent-b", "agent-c", "agent-d"]:
        agent = MagicMock(spec=Agent)
        agent.name = name
        agents.append(agent)
    return agents


@pytest.fixture
def sample_match(sample_task):
    """Create a sample TournamentMatch."""
    return TournamentMatch(
        match_id="match-001",
        round_num=0,
        participants=["agent-a", "agent-b"],
        task=sample_task,
    )


@pytest.fixture
def sample_standing():
    """Create a sample TournamentStanding."""
    return TournamentStanding(
        agent_name="agent-a",
        wins=3,
        losses=1,
        draws=1,
        points=10.0,
        total_score=2.5,
        matches_played=5,
    )


@pytest.fixture
def mock_debate_result():
    """Create a mock DebateResult."""
    result = MagicMock(spec=DebateResult)
    result.consensus_reached = True
    result.confidence = 0.8
    result.messages = [
        MagicMock(agent="agent-a"),
        MagicMock(agent="agent-b"),
        MagicMock(agent="agent-a"),
    ]
    result.critiques = [
        MagicMock(agent="agent-a", severity=0.3),
        MagicMock(agent="agent-b", severity=0.5),
    ]
    return result


@pytest.fixture
def tournament_with_db(temp_db, mock_agents, sample_tasks):
    """Create a tournament with a temp database."""
    return Tournament(
        name="Test Tournament",
        agents=mock_agents,
        tasks=sample_tasks,
        format=TournamentFormat.ROUND_ROBIN,
        db_path=temp_db,
    )


# =============================================================================
# Test TournamentFormat Enum
# =============================================================================


class TestTournamentFormat:
    """Tests for TournamentFormat enum."""

    def test_round_robin_value(self):
        """Test ROUND_ROBIN enum value."""
        assert TournamentFormat.ROUND_ROBIN.value == "round_robin"

    def test_single_elimination_value(self):
        """Test SINGLE_ELIMINATION enum value."""
        assert TournamentFormat.SINGLE_ELIMINATION.value == "single_elimination"

    def test_swiss_value(self):
        """Test SWISS enum value."""
        assert TournamentFormat.SWISS.value == "swiss"

    def test_free_for_all_value(self):
        """Test FREE_FOR_ALL enum value."""
        assert TournamentFormat.FREE_FOR_ALL.value == "free_for_all"

    def test_all_values_unique(self):
        """Test all enum values are unique."""
        values = [f.value for f in TournamentFormat]
        assert len(values) == len(set(values))


# =============================================================================
# Test TournamentTask Dataclass
# =============================================================================


class TestTournamentTask:
    """Tests for TournamentTask dataclass."""

    def test_creation_with_required_fields(self):
        """Test creating TournamentTask with required fields."""
        task = TournamentTask(
            task_id="task-001",
            description="Test task",
            domain="testing",
        )
        assert task.task_id == "task-001"
        assert task.description == "Test task"
        assert task.domain == "testing"

    def test_default_difficulty(self):
        """Test default difficulty value."""
        task = TournamentTask(
            task_id="task-001",
            description="Test task",
            domain="testing",
        )
        assert task.difficulty == 0.5

    def test_default_time_limit(self):
        """Test default time_limit value."""
        task = TournamentTask(
            task_id="task-001",
            description="Test task",
            domain="testing",
        )
        assert task.time_limit == 300

    def test_custom_difficulty(self):
        """Test custom difficulty value."""
        task = TournamentTask(
            task_id="task-001",
            description="Test task",
            domain="testing",
            difficulty=0.8,
        )
        assert task.difficulty == 0.8

    def test_custom_time_limit(self):
        """Test custom time_limit value."""
        task = TournamentTask(
            task_id="task-001",
            description="Test task",
            domain="testing",
            time_limit=600,
        )
        assert task.time_limit == 600


# =============================================================================
# Test TournamentMatch Dataclass
# =============================================================================


class TestTournamentMatch:
    """Tests for TournamentMatch dataclass."""

    def test_creation_with_required_fields(self, sample_task):
        """Test creating TournamentMatch with required fields."""
        match = TournamentMatch(
            match_id="match-001",
            round_num=0,
            participants=["agent-a", "agent-b"],
            task=sample_task,
        )
        assert match.match_id == "match-001"
        assert match.round_num == 0
        assert match.participants == ["agent-a", "agent-b"]
        assert match.task == sample_task

    def test_default_result_none(self, sample_task):
        """Test default result is None."""
        match = TournamentMatch(
            match_id="match-001",
            round_num=0,
            participants=["agent-a", "agent-b"],
            task=sample_task,
        )
        assert match.result is None

    def test_default_scores_empty(self, sample_task):
        """Test default scores is empty dict."""
        match = TournamentMatch(
            match_id="match-001",
            round_num=0,
            participants=["agent-a", "agent-b"],
            task=sample_task,
        )
        assert match.scores == {}

    def test_default_winner_none(self, sample_task):
        """Test default winner is None."""
        match = TournamentMatch(
            match_id="match-001",
            round_num=0,
            participants=["agent-a", "agent-b"],
            task=sample_task,
        )
        assert match.winner is None

    def test_is_complete_false(self, sample_match):
        """Test is_complete is False when no result."""
        assert sample_match.is_complete is False

    def test_is_complete_true(self, sample_match, mock_debate_result):
        """Test is_complete is True when result exists."""
        sample_match.result = mock_debate_result
        assert sample_match.is_complete is True

    def test_timestamps_default_none(self, sample_task):
        """Test timestamps default to None."""
        match = TournamentMatch(
            match_id="match-001",
            round_num=0,
            participants=["agent-a", "agent-b"],
            task=sample_task,
        )
        assert match.started_at is None
        assert match.completed_at is None


# =============================================================================
# Test TournamentStanding Dataclass
# =============================================================================


class TestTournamentStanding:
    """Tests for TournamentStanding dataclass."""

    def test_creation_with_agent_name(self):
        """Test creating TournamentStanding with agent name."""
        standing = TournamentStanding(agent_name="agent-a")
        assert standing.agent_name == "agent-a"

    def test_default_values(self):
        """Test default values for standing."""
        standing = TournamentStanding(agent_name="agent-a")
        assert standing.wins == 0
        assert standing.losses == 0
        assert standing.draws == 0
        assert standing.points == 0.0
        assert standing.total_score == 0.0
        assert standing.matches_played == 0

    def test_win_rate_calculation(self, sample_standing):
        """Test win_rate property calculation."""
        # 3 wins out of 5 matches = 0.6
        assert sample_standing.win_rate == 0.6

    def test_win_rate_zero_matches(self):
        """Test win_rate returns 0 when no matches played."""
        standing = TournamentStanding(agent_name="agent-a")
        assert standing.win_rate == 0.0

    def test_win_rate_all_wins(self):
        """Test win_rate with all wins."""
        standing = TournamentStanding(
            agent_name="agent-a",
            wins=5,
            losses=0,
            draws=0,
            matches_played=5,
        )
        assert standing.win_rate == 1.0

    def test_win_rate_no_wins(self):
        """Test win_rate with no wins."""
        standing = TournamentStanding(
            agent_name="agent-a",
            wins=0,
            losses=5,
            draws=0,
            matches_played=5,
        )
        assert standing.win_rate == 0.0


# =============================================================================
# Test TournamentResult Dataclass
# =============================================================================


class TestTournamentResult:
    """Tests for TournamentResult dataclass."""

    def test_creation(self, sample_standing, sample_match):
        """Test creating TournamentResult."""
        result = TournamentResult(
            tournament_id="tourn-001",
            name="Test Tournament",
            format=TournamentFormat.ROUND_ROBIN,
            standings=[sample_standing],
            matches=[sample_match],
            champion="agent-a",
            total_rounds=3,
            started_at="2024-01-15T10:00:00",
            completed_at="2024-01-15T12:00:00",
        )
        assert result.tournament_id == "tourn-001"
        assert result.name == "Test Tournament"
        assert result.format == TournamentFormat.ROUND_ROBIN
        assert result.champion == "agent-a"
        assert result.total_rounds == 3

    def test_standings_list(self, sample_standing, sample_match):
        """Test standings is a list."""
        result = TournamentResult(
            tournament_id="tourn-001",
            name="Test Tournament",
            format=TournamentFormat.ROUND_ROBIN,
            standings=[sample_standing],
            matches=[sample_match],
            champion="agent-a",
            total_rounds=3,
            started_at="2024-01-15T10:00:00",
            completed_at="2024-01-15T12:00:00",
        )
        assert isinstance(result.standings, list)
        assert len(result.standings) == 1

    def test_matches_list(self, sample_standing, sample_match):
        """Test matches is a list."""
        result = TournamentResult(
            tournament_id="tourn-001",
            name="Test Tournament",
            format=TournamentFormat.ROUND_ROBIN,
            standings=[sample_standing],
            matches=[sample_match],
            champion="agent-a",
            total_rounds=3,
            started_at="2024-01-15T10:00:00",
            completed_at="2024-01-15T12:00:00",
        )
        assert isinstance(result.matches, list)
        assert len(result.matches) == 1


# =============================================================================
# Test Tournament Initialization
# =============================================================================


class TestTournamentInit:
    """Tests for Tournament initialization."""

    def test_init_creates_tournament(self, temp_db, mock_agents, sample_tasks):
        """Test initialization creates a tournament."""
        tournament = Tournament(
            name="Test Tournament",
            agents=mock_agents,
            tasks=sample_tasks,
            db_path=temp_db,
        )
        assert tournament.name == "Test Tournament"
        assert len(tournament.tournament_id) == 8

    def test_init_default_format(self, temp_db, mock_agents, sample_tasks):
        """Test default format is ROUND_ROBIN."""
        tournament = Tournament(
            name="Test Tournament",
            agents=mock_agents,
            tasks=sample_tasks,
            db_path=temp_db,
        )
        assert tournament.format == TournamentFormat.ROUND_ROBIN

    def test_init_custom_format(self, temp_db, mock_agents, sample_tasks):
        """Test custom format."""
        tournament = Tournament(
            name="Test Tournament",
            agents=mock_agents,
            tasks=sample_tasks,
            format=TournamentFormat.SINGLE_ELIMINATION,
            db_path=temp_db,
        )
        assert tournament.format == TournamentFormat.SINGLE_ELIMINATION

    def test_init_creates_standings(self, temp_db, mock_agents, sample_tasks):
        """Test initialization creates standings for all agents."""
        tournament = Tournament(
            name="Test Tournament",
            agents=mock_agents,
            tasks=sample_tasks,
            db_path=temp_db,
        )
        assert len(tournament.standings) == len(mock_agents)
        for agent in mock_agents:
            assert agent.name in tournament.standings

    def test_init_empty_matches(self, temp_db, mock_agents, sample_tasks):
        """Test initialization starts with empty matches."""
        tournament = Tournament(
            name="Test Tournament",
            agents=mock_agents,
            tasks=sample_tasks,
            db_path=temp_db,
        )
        assert tournament.matches == []

    def test_init_creates_db(self, temp_db, mock_agents, sample_tasks):
        """Test initialization creates database file."""
        tournament = Tournament(
            name="Test Tournament",
            agents=mock_agents,
            tasks=sample_tasks,
            db_path=temp_db,
        )
        assert Path(temp_db).exists()

    def test_init_creates_tables(self, temp_db, mock_agents, sample_tasks):
        """Test initialization creates required tables."""
        tournament = Tournament(
            name="Test Tournament",
            agents=mock_agents,
            tasks=sample_tasks,
            db_path=temp_db,
        )

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Check tournaments table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tournaments'"
        )
        assert cursor.fetchone() is not None

        # Check tournament_matches table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tournament_matches'"
        )
        assert cursor.fetchone() is not None

        conn.close()


# =============================================================================
# Test Match Generation
# =============================================================================


class TestMatchGeneration:
    """Tests for match generation."""

    def test_generate_round_robin(self, tournament_with_db):
        """Test round-robin match generation."""
        matches = tournament_with_db.generate_matches()
        # 4 agents, C(4,2) = 6 pairs per task, 3 tasks = 18 matches
        assert len(matches) > 0

    def test_round_robin_all_pairs(self, temp_db, mock_agents, sample_tasks):
        """Test round-robin includes all pairs."""
        tournament = Tournament(
            name="Test Tournament",
            agents=mock_agents,
            tasks=[sample_tasks[0]],  # Single task
            format=TournamentFormat.ROUND_ROBIN,
            db_path=temp_db,
        )
        matches = tournament.generate_matches()

        # 4 agents, C(4,2) = 6 pairs
        assert len(matches) == 6

        # Check all pairs are represented
        pairs = set()
        for match in matches:
            pair = tuple(sorted(match.participants))
            pairs.add(pair)
        assert len(pairs) == 6

    def test_generate_free_for_all(self, temp_db, mock_agents, sample_tasks):
        """Test free-for-all match generation."""
        tournament = Tournament(
            name="Test Tournament",
            agents=mock_agents,
            tasks=sample_tasks,
            format=TournamentFormat.FREE_FOR_ALL,
            db_path=temp_db,
        )
        matches = tournament.generate_matches()

        # One match per task
        assert len(matches) == len(sample_tasks)

        # All agents in each match
        for match in matches:
            assert len(match.participants) == len(mock_agents)

    def test_generate_bracket(self, temp_db, mock_agents, sample_tasks):
        """Test bracket match generation."""
        tournament = Tournament(
            name="Test Tournament",
            agents=mock_agents,
            tasks=sample_tasks,
            format=TournamentFormat.SINGLE_ELIMINATION,
            db_path=temp_db,
        )
        matches = tournament.generate_matches()

        # 4 agents = 2 matches (semifinals), plus 1 final = 3 total
        # But implementation generates initial round only
        assert len(matches) >= 2

    def test_generate_swiss_round(self, temp_db, mock_agents, sample_tasks):
        """Test Swiss-system match generation."""
        tournament = Tournament(
            name="Test Tournament",
            agents=mock_agents,
            tasks=sample_tasks,
            format=TournamentFormat.SWISS,
            db_path=temp_db,
        )
        matches = tournament.generate_matches()

        # Swiss pairs adjacent agents by standing
        # 4 agents = 2 matches per round
        assert len(matches) == 2

    def test_match_has_task(self, tournament_with_db):
        """Test generated matches have task assigned."""
        matches = tournament_with_db.generate_matches()
        for match in matches:
            assert match.task is not None
            assert match.task.task_id is not None

    def test_match_has_participants(self, tournament_with_db):
        """Test generated matches have participants."""
        matches = tournament_with_db.generate_matches()
        for match in matches:
            assert len(match.participants) >= 2


# =============================================================================
# Test Match Scoring
# =============================================================================


class TestMatchScoring:
    """Tests for match scoring."""

    def test_calculate_match_scores(self, tournament_with_db, mock_debate_result):
        """Test score calculation for a match."""
        scores = tournament_with_db._calculate_match_scores(
            mock_debate_result, ["agent-a", "agent-b"]
        )
        assert "agent-a" in scores
        assert "agent-b" in scores
        assert all(0 <= s <= 1 for s in scores.values())

    def test_calculate_scores_consensus_boost(self, tournament_with_db, mock_debate_result):
        """Test consensus reached adds to score."""
        mock_debate_result.consensus_reached = True
        scores = tournament_with_db._calculate_match_scores(
            mock_debate_result, ["agent-a", "agent-b"]
        )

        # With consensus, scores should be > 0
        assert all(s > 0 for s in scores.values())

    def test_calculate_scores_no_consensus(self, tournament_with_db, mock_debate_result):
        """Test score calculation without consensus."""
        mock_debate_result.consensus_reached = False
        mock_debate_result.confidence = 0.5
        scores = tournament_with_db._calculate_match_scores(
            mock_debate_result, ["agent-a", "agent-b"]
        )

        # Scores should still be calculated
        assert "agent-a" in scores
        assert "agent-b" in scores

    def test_calculate_scores_null_result(self, tournament_with_db):
        """Test score calculation with null result."""
        scores = tournament_with_db._calculate_match_scores(
            None, ["agent-a", "agent-b"]
        )
        assert scores == {"agent-a": 0.0, "agent-b": 0.0}


# =============================================================================
# Test Standings Update
# =============================================================================


class TestStandingsUpdate:
    """Tests for standings updates."""

    def test_update_standings_win(self, tournament_with_db, sample_task):
        """Test standings update for a win."""
        match = TournamentMatch(
            match_id="match-001",
            round_num=0,
            participants=["agent-a", "agent-b"],
            task=sample_task,
            scores={"agent-a": 0.8, "agent-b": 0.4},
            winner="agent-a",
        )

        tournament_with_db._update_standings(match)

        assert tournament_with_db.standings["agent-a"].wins == 1
        assert tournament_with_db.standings["agent-a"].points == 3
        assert tournament_with_db.standings["agent-b"].losses == 1
        assert tournament_with_db.standings["agent-b"].points == 0

    def test_update_standings_draw(self, tournament_with_db, sample_task):
        """Test standings update for a draw."""
        match = TournamentMatch(
            match_id="match-001",
            round_num=0,
            participants=["agent-a", "agent-b"],
            task=sample_task,
            scores={"agent-a": 0.5, "agent-b": 0.5},
            winner=None,  # Draw
        )

        tournament_with_db._update_standings(match)

        assert tournament_with_db.standings["agent-a"].draws == 1
        assert tournament_with_db.standings["agent-a"].points == 1
        assert tournament_with_db.standings["agent-b"].draws == 1
        assert tournament_with_db.standings["agent-b"].points == 1

    def test_update_standings_total_score(self, tournament_with_db, sample_task):
        """Test total_score accumulation."""
        match = TournamentMatch(
            match_id="match-001",
            round_num=0,
            participants=["agent-a", "agent-b"],
            task=sample_task,
            scores={"agent-a": 0.8, "agent-b": 0.4},
            winner="agent-a",
        )

        tournament_with_db._update_standings(match)

        assert tournament_with_db.standings["agent-a"].total_score == 0.8
        assert tournament_with_db.standings["agent-b"].total_score == 0.4

    def test_update_standings_matches_played(self, tournament_with_db, sample_task):
        """Test matches_played increment."""
        match = TournamentMatch(
            match_id="match-001",
            round_num=0,
            participants=["agent-a", "agent-b"],
            task=sample_task,
            scores={"agent-a": 0.8, "agent-b": 0.4},
            winner="agent-a",
        )

        tournament_with_db._update_standings(match)

        assert tournament_with_db.standings["agent-a"].matches_played == 1
        assert tournament_with_db.standings["agent-b"].matches_played == 1


# =============================================================================
# Test Tournament Run
# =============================================================================


class TestTournamentRun:
    """Tests for running tournaments."""

    @pytest.mark.asyncio
    async def test_run_tournament(self, tournament_with_db, mock_debate_result):
        """Test running a tournament."""
        async def mock_run_debate(env, agents):
            return mock_debate_result

        result = await tournament_with_db.run(mock_run_debate, parallel=False)

        assert result is not None
        assert result.tournament_id == tournament_with_db.tournament_id
        assert result.champion is not None

    @pytest.mark.asyncio
    async def test_run_sets_timestamps(self, tournament_with_db, mock_debate_result):
        """Test run sets started_at and completed_at."""
        async def mock_run_debate(env, agents):
            return mock_debate_result

        result = await tournament_with_db.run(mock_run_debate, parallel=False)

        assert result.started_at is not None
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_run_determines_champion(self, tournament_with_db, mock_debate_result):
        """Test champion determination."""
        async def mock_run_debate(env, agents):
            return mock_debate_result

        result = await tournament_with_db.run(mock_run_debate, parallel=False)

        # Champion should be one of the agents
        assert result.champion in [a.name for a in tournament_with_db.agents.values()]

    @pytest.mark.asyncio
    async def test_run_saves_to_db(self, tournament_with_db, mock_debate_result):
        """Test tournament is saved to database."""
        async def mock_run_debate(env, agents):
            return mock_debate_result

        await tournament_with_db.run(mock_run_debate, parallel=False)

        # Check database
        conn = sqlite3.connect(tournament_with_db.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT tournament_id FROM tournaments WHERE tournament_id = ?",
            (tournament_with_db.tournament_id,)
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None

    @pytest.mark.asyncio
    async def test_run_parallel(self, tournament_with_db, mock_debate_result):
        """Test parallel match execution."""
        async def mock_run_debate(env, agents):
            await asyncio.sleep(0.01)  # Small delay
            return mock_debate_result

        result = await tournament_with_db.run(mock_run_debate, parallel=True)

        assert result is not None
        assert len(result.matches) > 0


# =============================================================================
# Test Get Current Standings
# =============================================================================


class TestGetCurrentStandings:
    """Tests for getting current standings."""

    def test_get_current_standings_empty(self, tournament_with_db):
        """Test standings before any matches."""
        standings = tournament_with_db.get_current_standings()
        assert len(standings) == 4  # 4 agents
        for standing in standings:
            assert standing.matches_played == 0

    def test_get_current_standings_sorted(self, tournament_with_db, sample_task):
        """Test standings are sorted by points."""
        # Update standings manually
        tournament_with_db.standings["agent-a"].points = 10
        tournament_with_db.standings["agent-b"].points = 5
        tournament_with_db.standings["agent-c"].points = 15
        tournament_with_db.standings["agent-d"].points = 8

        standings = tournament_with_db.get_current_standings()

        assert standings[0].agent_name == "agent-c"
        assert standings[1].agent_name == "agent-a"
        assert standings[2].agent_name == "agent-d"
        assert standings[3].agent_name == "agent-b"


# =============================================================================
# Test create_default_tasks
# =============================================================================


class TestCreateDefaultTasks:
    """Tests for create_default_tasks function."""

    def test_returns_list(self):
        """Test returns a list."""
        tasks = create_default_tasks()
        assert isinstance(tasks, list)

    def test_returns_tournament_tasks(self):
        """Test returns TournamentTask objects."""
        tasks = create_default_tasks()
        assert all(isinstance(t, TournamentTask) for t in tasks)

    def test_has_multiple_tasks(self):
        """Test returns multiple tasks."""
        tasks = create_default_tasks()
        assert len(tasks) >= 3

    def test_tasks_have_domains(self):
        """Test all tasks have domains."""
        tasks = create_default_tasks()
        for task in tasks:
            assert task.domain is not None
            assert len(task.domain) > 0

    def test_tasks_have_descriptions(self):
        """Test all tasks have descriptions."""
        tasks = create_default_tasks()
        for task in tasks:
            assert task.description is not None
            assert len(task.description) > 0


# =============================================================================
# Test TournamentManager
# =============================================================================


class TestTournamentManager:
    """Tests for TournamentManager class."""

    def test_init(self, temp_db):
        """Test initialization."""
        manager = TournamentManager(temp_db)
        assert manager.db_path == Path(temp_db)

    def test_get_tournament_no_db(self):
        """Test get_tournament when database doesn't exist."""
        manager = TournamentManager("/nonexistent/path.db")
        result = manager.get_tournament()
        assert result is None

    def test_get_tournament_empty_db(self, temp_db):
        """Test get_tournament with empty database."""
        # Create empty database with schema
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE tournaments (
                tournament_id TEXT PRIMARY KEY,
                name TEXT,
                format TEXT,
                champion TEXT,
                started_at TEXT,
                completed_at TEXT
            )
        """)
        conn.commit()
        conn.close()

        manager = TournamentManager(temp_db)
        result = manager.get_tournament()
        assert result is None

    def test_get_tournament_with_data(self, temp_db):
        """Test get_tournament with tournament data."""
        # Create database with tournament
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE tournaments (
                tournament_id TEXT PRIMARY KEY,
                name TEXT,
                format TEXT,
                champion TEXT,
                started_at TEXT,
                completed_at TEXT
            )
        """)
        cursor.execute("""
            INSERT INTO tournaments VALUES (?, ?, ?, ?, ?, ?)
        """, ("tourn-001", "Test", "round_robin", "agent-a", "2024-01-15", "2024-01-16"))
        conn.commit()
        conn.close()

        manager = TournamentManager(temp_db)
        result = manager.get_tournament()

        assert result is not None
        assert result["tournament_id"] == "tourn-001"
        assert result["name"] == "Test"
        assert result["champion"] == "agent-a"

    def test_get_current_standings_no_db(self):
        """Test get_current_standings when database doesn't exist."""
        manager = TournamentManager("/nonexistent/path.db")
        result = manager.get_current_standings()
        assert result == []

    def test_get_current_standings_with_data(self, temp_db):
        """Test get_current_standings with standings data."""
        # Create database with standings
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE tournaments (
                tournament_id TEXT PRIMARY KEY,
                standings TEXT
            )
        """)
        standings_json = json.dumps({
            "agent-a": {"wins": 3, "losses": 1, "draws": 0, "points": 9, "total_score": 2.5},
            "agent-b": {"wins": 1, "losses": 3, "draws": 0, "points": 3, "total_score": 1.5},
        })
        cursor.execute(
            "INSERT INTO tournaments (tournament_id, standings) VALUES (?, ?)",
            ("tourn-001", standings_json)
        )
        conn.commit()
        conn.close()

        manager = TournamentManager(temp_db)
        standings = manager.get_current_standings()

        assert len(standings) == 2
        assert standings[0].agent_name == "agent-a"  # Higher points first
        assert standings[0].points == 9

    def test_get_matches_no_db(self):
        """Test get_matches when database doesn't exist."""
        manager = TournamentManager("/nonexistent/path.db")
        result = manager.get_matches()
        assert result == []

    def test_get_matches_with_data(self, temp_db):
        """Test get_matches with match data."""
        # Create database with matches
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE tournament_matches (
                match_id TEXT PRIMARY KEY,
                round_num INTEGER,
                participants TEXT,
                task_id TEXT,
                scores TEXT,
                winner TEXT,
                started_at TEXT,
                completed_at TEXT
            )
        """)
        cursor.execute("""
            INSERT INTO tournament_matches VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "match-001", 0,
            json.dumps(["agent-a", "agent-b"]),
            "task-1",
            json.dumps({"agent-a": 0.8, "agent-b": 0.4}),
            "agent-a",
            "2024-01-15", "2024-01-15"
        ))
        conn.commit()
        conn.close()

        manager = TournamentManager(temp_db)
        matches = manager.get_matches()

        assert len(matches) == 1
        assert matches[0]["match_id"] == "match-001"
        assert matches[0]["winner"] == "agent-a"

    def test_get_matches_with_limit(self, temp_db):
        """Test get_matches with limit."""
        # Create database with multiple matches
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE tournament_matches (
                match_id TEXT PRIMARY KEY,
                round_num INTEGER,
                participants TEXT,
                task_id TEXT,
                scores TEXT,
                winner TEXT,
                started_at TEXT,
                completed_at TEXT
            )
        """)
        for i in range(5):
            cursor.execute("""
                INSERT INTO tournament_matches VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"match-{i:03d}", i,
                json.dumps(["agent-a", "agent-b"]),
                "task-1", "{}", None, None, None
            ))
        conn.commit()
        conn.close()

        manager = TournamentManager(temp_db)
        matches = manager.get_matches(limit=3)

        assert len(matches) == 3

    def test_get_match_summary_no_db(self):
        """Test get_match_summary when database doesn't exist."""
        manager = TournamentManager("/nonexistent/path.db")
        result = manager.get_match_summary()
        assert result == {"total_matches": 0, "decided_matches": 0, "max_round": 0}

    def test_get_match_summary_with_data(self, temp_db):
        """Test get_match_summary with match data."""
        # Create database with matches
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE tournament_matches (
                match_id TEXT PRIMARY KEY,
                round_num INTEGER,
                winner TEXT
            )
        """)
        cursor.execute("INSERT INTO tournament_matches VALUES (?, ?, ?)", ("m1", 0, "agent-a"))
        cursor.execute("INSERT INTO tournament_matches VALUES (?, ?, ?)", ("m2", 0, None))
        cursor.execute("INSERT INTO tournament_matches VALUES (?, ?, ?)", ("m3", 1, "agent-b"))
        conn.commit()
        conn.close()

        manager = TournamentManager(temp_db)
        summary = manager.get_match_summary()

        assert summary["total_matches"] == 3
        assert summary["decided_matches"] == 2
        assert summary["max_round"] == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestTournamentIntegration:
    """Integration tests for the tournament system."""

    @pytest.mark.asyncio
    async def test_full_round_robin_tournament(self, temp_db, mock_agents, sample_tasks, mock_debate_result):
        """Test full round-robin tournament workflow."""
        tournament = Tournament(
            name="Integration Test",
            agents=mock_agents,
            tasks=sample_tasks[:1],  # Single task
            format=TournamentFormat.ROUND_ROBIN,
            db_path=temp_db,
        )

        async def mock_run_debate(env, agents):
            return mock_debate_result

        result = await tournament.run(mock_run_debate, parallel=False)

        assert result.tournament_id is not None
        assert result.champion is not None
        assert len(result.matches) == 6  # C(4,2) = 6 pairs
        assert len(result.standings) == 4

    @pytest.mark.asyncio
    async def test_full_free_for_all_tournament(self, temp_db, mock_agents, sample_tasks, mock_debate_result):
        """Test full free-for-all tournament workflow."""
        tournament = Tournament(
            name="FFA Test",
            agents=mock_agents,
            tasks=sample_tasks,
            format=TournamentFormat.FREE_FOR_ALL,
            db_path=temp_db,
        )

        async def mock_run_debate(env, agents):
            return mock_debate_result

        result = await tournament.run(mock_run_debate, parallel=False)

        assert result.format == TournamentFormat.FREE_FOR_ALL
        assert len(result.matches) == len(sample_tasks)

    def test_manager_reads_tournament_data(self, temp_db, mock_agents, sample_tasks, mock_debate_result):
        """Test TournamentManager can read tournament data after run."""
        # Create and save tournament data
        tournament = Tournament(
            name="Manager Test",
            agents=mock_agents,
            tasks=sample_tasks[:1],
            format=TournamentFormat.ROUND_ROBIN,
            db_path=temp_db,
        )

        # Generate matches
        matches = tournament.generate_matches()
        tournament.matches = matches

        # Update standings manually
        for match in matches:
            match.winner = match.participants[0]
            match.scores = {p: 0.5 for p in match.participants}
            tournament._update_standings(match)

        # Save to database
        tournament.started_at = datetime.now().isoformat()
        tournament.completed_at = datetime.now().isoformat()
        tournament._save_tournament("agent-a")

        # Use manager to read data
        manager = TournamentManager(temp_db)

        tournament_data = manager.get_tournament()
        assert tournament_data is not None
        assert tournament_data["name"] == "Manager Test"

        standings = manager.get_current_standings()
        assert len(standings) == 4

        matches = manager.get_matches()
        assert len(matches) == 6

        summary = manager.get_match_summary()
        assert summary["total_matches"] == 6
