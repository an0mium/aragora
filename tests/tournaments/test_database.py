"""Unit tests for tournament database schema and inherited helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.tournaments.database import TournamentDatabase


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "tournaments.sqlite"


@pytest.fixture
def tournament_db(db_path: Path) -> TournamentDatabase:
    db = TournamentDatabase(str(db_path))
    yield db
    db.close()


def _table_columns(db: TournamentDatabase, table_name: str) -> list[str]:
    return [row[1] for row in db.fetch_all(f"PRAGMA table_info({table_name})")]


def test_class_metadata_matches_tournament_schema() -> None:
    assert TournamentDatabase.DEFAULT_DB_PATH == "aragora_tournaments.db"
    assert TournamentDatabase.SCHEMA_NAME == "tournaments"
    assert TournamentDatabase.SCHEMA_VERSION == 1
    assert "CREATE TABLE IF NOT EXISTS tournaments" in TournamentDatabase.INITIAL_SCHEMA
    assert "CREATE TABLE IF NOT EXISTS tournament_matches" in TournamentDatabase.INITIAL_SCHEMA


def test_init_creates_database_file(db_path: Path) -> None:
    db = TournamentDatabase(str(db_path))
    try:
        assert db_path.exists()
        assert db.db_path == db_path
    finally:
        db.close()


def test_init_records_schema_version(tournament_db: TournamentDatabase) -> None:
    assert tournament_db.get_schema_version() == TournamentDatabase.SCHEMA_VERSION


def test_tournaments_table_has_expected_columns(
    tournament_db: TournamentDatabase,
) -> None:
    assert _table_columns(tournament_db, "tournaments") == [
        "tournament_id",
        "name",
        "format",
        "agents",
        "tasks",
        "standings",
        "champion",
        "started_at",
        "completed_at",
    ]


def test_tournament_matches_table_has_expected_columns(
    tournament_db: TournamentDatabase,
) -> None:
    assert _table_columns(tournament_db, "tournament_matches") == [
        "match_id",
        "tournament_id",
        "round_num",
        "participants",
        "task_id",
        "scores",
        "winner",
        "started_at",
        "completed_at",
    ]


def test_execute_write_and_fetch_one_round_trip(
    tournament_db: TournamentDatabase,
) -> None:
    tournament_db.execute_write(
        "INSERT INTO tournaments (tournament_id, name, format) VALUES (?, ?, ?)",
        ("tournament-1", "Qualifiers", "round_robin"),
    )

    row = tournament_db.fetch_one(
        "SELECT tournament_id, name, format FROM tournaments WHERE tournament_id = ?",
        ("tournament-1",),
    )

    assert tuple(row) == ("tournament-1", "Qualifiers", "round_robin")


def test_fetch_all_returns_match_rows_in_order(
    tournament_db: TournamentDatabase,
) -> None:
    tournament_db.executemany(
        """
        INSERT INTO tournament_matches (match_id, tournament_id, round_num, winner)
        VALUES (?, ?, ?, ?)
        """,
        [
            ("match-2", "tournament-1", 2, "beta"),
            ("match-1", "tournament-1", 1, "alpha"),
        ],
    )

    rows = tournament_db.fetch_all(
        "SELECT match_id, round_num, winner FROM tournament_matches ORDER BY round_num"
    )

    assert [tuple(row) for row in rows] == [
        ("match-1", 1, "alpha"),
        ("match-2", 2, "beta"),
    ]


def test_inherited_exists_count_and_delete_helpers(
    tournament_db: TournamentDatabase,
) -> None:
    tournament_db.execute_write(
        "INSERT INTO tournaments (tournament_id, name) VALUES (?, ?)",
        ("tournament-1", "Qualifiers"),
    )

    assert tournament_db.exists("tournaments", "tournament_id", "tournament-1")
    assert tournament_db.count("tournaments") == 1
    assert tournament_db.delete_by_id("tournaments", "tournament_id", "tournament-1")
    assert not tournament_db.exists("tournaments", "tournament_id", "tournament-1")
    assert tournament_db.count("tournaments") == 0


def test_connection_rolls_back_on_exception(
    tournament_db: TournamentDatabase,
) -> None:
    with pytest.raises(RuntimeError, match="abort tournament write"):
        with tournament_db.connection() as conn:
            conn.execute(
                "INSERT INTO tournaments (tournament_id, name) VALUES (?, ?)",
                ("tournament-1", "Qualifiers"),
            )
            raise RuntimeError("abort tournament write")

    assert tournament_db.count("tournaments") == 0


def test_invalid_table_name_is_rejected(tournament_db: TournamentDatabase) -> None:
    with pytest.raises(ValueError, match="Invalid table name"):
        tournament_db.count("tournaments; DROP TABLE tournaments")
