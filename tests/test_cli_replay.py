"""Tests for CLI replay command - debate replay functionality."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.replay import (
    find_replay_files,
    load_replay,
    format_duration,
    cmd_replay,
    _list_replays,
    _show_replay,
    _play_replay,
)


@pytest.fixture
def replay_dir(tmp_path):
    """Create directory with replay files."""
    replay_path = tmp_path / ".aragora" / "replays"
    replay_path.mkdir(parents=True)
    return replay_path


@pytest.fixture
def sample_replay():
    """Create sample replay data."""
    return {
        "task": "Design a rate limiter algorithm",
        "duration_seconds": 45.5,
        "rounds_used": 3,
        "confidence": 0.85,
        "consensus_reached": True,
        "agents": ["claude", "gpt4", "gemini"],
        "final_answer": "Token bucket algorithm with sliding window",
        "messages": [
            {"agent": "claude", "content": "I propose using token bucket", "round": 1},
            {"agent": "gpt4", "content": "Good idea, with modifications", "round": 1},
        ],
        "critiques": [
            {"critic": "gemini", "content": "Consider edge cases"},
        ],
    }


@pytest.fixture
def replay_file(replay_dir, sample_replay):
    """Create a replay file."""
    filepath = replay_dir / "debate_20260115_123456.json"
    filepath.write_text(json.dumps(sample_replay))
    return filepath


class TestFindReplayFiles:
    """Tests for find_replay_files function."""

    def test_finds_files_in_specified_directory(self, replay_dir, replay_file):
        """Find replay files in specified directory."""
        files = find_replay_files(str(replay_dir))

        assert len(files) == 1
        assert files[0] == replay_file

    def test_finds_files_in_default_locations(self, tmp_path, sample_replay, monkeypatch):
        """Find files in default locations."""
        monkeypatch.chdir(tmp_path)
        replay_dir = tmp_path / ".aragora" / "replays"
        replay_dir.mkdir(parents=True)
        filepath = replay_dir / "test.json"
        filepath.write_text(json.dumps(sample_replay))

        files = find_replay_files()

        assert len(files) == 1
        assert files[0].name == "test.json"

    def test_returns_empty_for_nonexistent_directory(self):
        """Return empty list for nonexistent directory."""
        files = find_replay_files("/nonexistent/path")

        assert files == []

    def test_returns_empty_when_no_default_locations(self, tmp_path, monkeypatch):
        """Return empty when no default locations exist."""
        monkeypatch.chdir(tmp_path)

        files = find_replay_files()

        assert files == []

    def test_sorts_by_modification_time(self, replay_dir, sample_replay):
        """Sort files by modification time (newest first)."""
        import time

        file1 = replay_dir / "old.json"
        file1.write_text(json.dumps(sample_replay))
        time.sleep(0.01)

        file2 = replay_dir / "new.json"
        file2.write_text(json.dumps(sample_replay))

        files = find_replay_files(str(replay_dir))

        assert len(files) == 2
        assert files[0].name == "new.json"
        assert files[1].name == "old.json"


class TestLoadReplay:
    """Tests for load_replay function."""

    def test_loads_valid_replay(self, replay_file, sample_replay):
        """Load valid replay file."""
        replay = load_replay(replay_file)

        assert replay is not None
        assert replay["task"] == sample_replay["task"]
        assert replay["confidence"] == sample_replay["confidence"]

    def test_returns_none_for_invalid_json(self, replay_dir):
        """Return None for invalid JSON."""
        filepath = replay_dir / "invalid.json"
        filepath.write_text("not valid json {{{")

        replay = load_replay(filepath)

        assert replay is None

    def test_returns_none_for_nonexistent_file(self, replay_dir):
        """Return None for nonexistent file."""
        filepath = replay_dir / "nonexistent.json"

        replay = load_replay(filepath)

        assert replay is None


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_formats_seconds(self):
        """Format durations under a minute."""
        assert format_duration(30.5) == "30.5s"
        assert format_duration(0.1) == "0.1s"
        assert format_duration(59.9) == "59.9s"

    def test_formats_minutes(self):
        """Format durations in minutes."""
        assert format_duration(60) == "1m 0s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(3599) == "59m 59s"

    def test_formats_hours(self):
        """Format durations in hours."""
        assert format_duration(3600) == "1h 0m"
        assert format_duration(3660) == "1h 1m"
        assert format_duration(7200) == "2h 0m"


class TestCmdReplay:
    """Tests for cmd_replay function."""

    def test_dispatches_to_list(self, replay_dir, replay_file, capsys):
        """Dispatch to list action by default."""
        args = MagicMock()
        args.action = "list"
        args.directory = str(replay_dir)
        args.limit = 10

        cmd_replay(args)

        captured = capsys.readouterr()
        assert "Recent Debates" in captured.out or "No replay files" in captured.out

    def test_dispatches_to_show(self, replay_dir, replay_file, capsys):
        """Dispatch to show action."""
        args = MagicMock()
        args.action = "show"
        args.id = replay_file.stem

        with patch("aragora.cli.replay.find_replay_files", return_value=[replay_file]):
            cmd_replay(args)

        captured = capsys.readouterr()
        assert "Replay:" in captured.out or "not found" in captured.out

    def test_defaults_to_list_for_unknown_action(self, capsys):
        """Default to list for unknown action."""
        args = MagicMock()
        args.action = "unknown"
        args.directory = None
        args.limit = 10

        cmd_replay(args)

        # Should not raise
        captured = capsys.readouterr()
        assert "No replay files" in captured.out or "Recent Debates" in captured.out


class TestListReplays:
    """Tests for _list_replays function."""

    def test_shows_no_files_message(self, tmp_path, monkeypatch, capsys):
        """Show message when no files found."""
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        args.directory = None
        args.limit = 10

        _list_replays(args)

        captured = capsys.readouterr()
        assert "No replay files found" in captured.out

    def test_shows_replay_list(self, replay_dir, replay_file, sample_replay, capsys):
        """Show list of replays."""
        args = MagicMock()
        args.directory = str(replay_dir)
        args.limit = 10

        _list_replays(args)

        captured = capsys.readouterr()
        assert "Recent Debates" in captured.out
        assert "Design a rate" in captured.out
        assert "Rounds: 3" in captured.out
        assert "Consensus: Yes" in captured.out

    def test_respects_limit(self, replay_dir, sample_replay, capsys):
        """Respect the limit parameter."""
        # Create multiple replay files
        for i in range(5):
            filepath = replay_dir / f"debate_{i}.json"
            filepath.write_text(json.dumps(sample_replay))

        args = MagicMock()
        args.directory = str(replay_dir)
        args.limit = 2

        _list_replays(args)

        captured = capsys.readouterr()
        assert "... and 3 more" in captured.out


class TestShowReplay:
    """Tests for _show_replay function."""

    def test_shows_replay_not_found(self, capsys):
        """Show not found message."""
        args = MagicMock()
        args.id = "nonexistent"

        with patch("aragora.cli.replay.find_replay_files", return_value=[]):
            _show_replay(args)

        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_shows_usage_without_id(self, capsys):
        """Show usage when no ID provided."""
        args = MagicMock()
        args.id = None

        _show_replay(args)

        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_shows_replay_details(self, replay_file, sample_replay, capsys):
        """Show replay details."""
        args = MagicMock()
        args.id = replay_file.stem

        with patch("aragora.cli.replay.find_replay_files", return_value=[replay_file]):
            _show_replay(args)

        captured = capsys.readouterr()
        assert f"Replay: {replay_file.stem}" in captured.out
        assert "Task: Design a rate limiter" in captured.out
        assert "Consensus: Yes" in captured.out
        assert "claude, gpt4, gemini" in captured.out

    def test_shows_final_answer(self, replay_file, capsys):
        """Show final answer section."""
        args = MagicMock()
        args.id = replay_file.stem

        with patch("aragora.cli.replay.find_replay_files", return_value=[replay_file]):
            _show_replay(args)

        captured = capsys.readouterr()
        assert "Final Answer:" in captured.out
        assert "Token bucket" in captured.out


class TestPlayReplay:
    """Tests for _play_replay function."""

    def test_shows_usage_without_id(self, capsys):
        """Show usage when no ID provided."""
        args = MagicMock()
        args.id = None

        _play_replay(args)

        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_shows_not_found(self, capsys):
        """Show not found for missing replay."""
        args = MagicMock()
        args.id = "missing"
        args.speed = 1.0

        with patch("aragora.cli.replay.find_replay_files", return_value=[]):
            _play_replay(args)

        captured = capsys.readouterr()
        assert "not found" in captured.out

    @patch("aragora.cli.replay.SpectatorStream")
    @patch("time.sleep")
    def test_plays_replay(self, mock_sleep, mock_stream, replay_file, capsys):
        """Play replay with spectator output."""
        args = MagicMock()
        args.id = replay_file.stem
        args.speed = 10.0  # Fast for tests

        mock_spectator = MagicMock()
        mock_stream.return_value = mock_spectator

        with patch("aragora.cli.replay.find_replay_files", return_value=[replay_file]):
            _play_replay(args)

        captured = capsys.readouterr()
        assert f"Playing: {replay_file.stem}" in captured.out
        assert "Speed: 10.0x" in captured.out
        assert "Replay complete" in captured.out

        # Spectator should emit events
        assert mock_spectator.emit.called


class TestIntegration:
    """Integration tests for replay module."""

    def test_full_replay_workflow(self, tmp_path, monkeypatch, sample_replay, capsys):
        """Test full workflow: create, list, show, play."""
        monkeypatch.chdir(tmp_path)

        # Create replay directory and file
        replay_dir = tmp_path / ".aragora" / "replays"
        replay_dir.mkdir(parents=True)
        replay_file = replay_dir / "test_debate.json"
        replay_file.write_text(json.dumps(sample_replay))

        # List replays
        args = MagicMock()
        args.action = "list"
        args.directory = None
        args.limit = 10
        cmd_replay(args)

        captured = capsys.readouterr()
        assert "test_debate" in captured.out

        # Show replay
        args.action = "show"
        args.id = "test_debate"
        cmd_replay(args)

        captured = capsys.readouterr()
        assert "Design a rate limiter" in captured.out
