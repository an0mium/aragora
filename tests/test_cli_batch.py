"""Tests for CLI batch command - batch debate processing."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.cli.batch import (
    _batch_local,
    _batch_via_server,
    _poll_batch_status,
    _read_input_file,
    cmd_batch,
    create_batch_parser,
)


@pytest.fixture
def mock_args():
    """Create mock args object."""
    args = MagicMock()
    args.input = "debates.jsonl"
    args.server = False
    args.url = "http://localhost:8080"
    args.token = None
    args.webhook = None
    args.wait = False
    args.agents = "anthropic-api,openai-api"
    args.rounds = 3
    args.output = None
    return args


@pytest.fixture
def sample_items():
    """Sample debate items."""
    return [
        {"question": "Design a rate limiter", "agents": "anthropic-api"},
        {"question": "Implement caching", "rounds": 4},
        {"question": "Security review"},
    ]


@pytest.fixture
def jsonl_file(tmp_path, sample_items):
    """Create a JSONL file with sample items."""
    file_path = tmp_path / "debates.jsonl"
    content = "\n".join(json.dumps(item) for item in sample_items)
    file_path.write_text(content)
    return file_path


@pytest.fixture
def json_file(tmp_path, sample_items):
    """Create a JSON file with sample items."""
    file_path = tmp_path / "debates.json"
    file_path.write_text(json.dumps(sample_items))
    return file_path


class TestCreateBatchParser:
    """Tests for create_batch_parser function."""

    def test_creates_parser(self):
        """Parser is created with correct arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        create_batch_parser(subparsers)

        args = parser.parse_args(["batch", "debates.jsonl"])
        assert args.input == "debates.jsonl"

    def test_server_mode_flag(self):
        """Server mode flag is parsed."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_batch_parser(subparsers)

        args = parser.parse_args(["batch", "debates.jsonl", "--server"])
        assert args.server is True

    def test_url_argument(self):
        """URL argument is parsed."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_batch_parser(subparsers)

        args = parser.parse_args(["batch", "debates.jsonl", "--url", "http://custom:9000"])
        assert args.url == "http://custom:9000"

    def test_token_argument(self):
        """Token argument is parsed."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_batch_parser(subparsers)

        args = parser.parse_args(["batch", "debates.jsonl", "-t", "secret-token"])
        assert args.token == "secret-token"

    def test_webhook_argument(self):
        """Webhook argument is parsed."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_batch_parser(subparsers)

        args = parser.parse_args(["batch", "debates.jsonl", "-w", "http://webhook.example.com"])
        assert args.webhook == "http://webhook.example.com"

    def test_wait_flag(self):
        """Wait flag is parsed."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_batch_parser(subparsers)

        args = parser.parse_args(["batch", "debates.jsonl", "--wait"])
        assert args.wait is True

    def test_agents_argument(self):
        """Agents argument is parsed."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_batch_parser(subparsers)

        args = parser.parse_args(["batch", "debates.jsonl", "-a", "claude,gpt4"])
        assert args.agents == "claude,gpt4"

    def test_rounds_argument(self):
        """Rounds argument is parsed."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_batch_parser(subparsers)

        args = parser.parse_args(["batch", "debates.jsonl", "-r", "5"])
        assert args.rounds == 5

    def test_output_argument(self):
        """Output argument is parsed."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_batch_parser(subparsers)

        args = parser.parse_args(["batch", "debates.jsonl", "-o", "results.json"])
        assert args.output == "results.json"


class TestReadInputFile:
    """Tests for _read_input_file function."""

    def test_reads_jsonl(self, jsonl_file, sample_items):
        """Read JSONL file."""
        items = _read_input_file(jsonl_file)

        assert len(items) == 3
        assert items[0]["question"] == "Design a rate limiter"

    def test_reads_json_array(self, json_file, sample_items):
        """Read JSON array file."""
        items = _read_input_file(json_file)

        assert len(items) == 3
        assert items[0]["question"] == "Design a rate limiter"

    def test_skips_comments(self, tmp_path):
        """Skip comment lines in JSONL."""
        file_path = tmp_path / "debates.jsonl"
        content = '# This is a comment\n{"question": "Q1"}\n# Another comment\n{"question": "Q2"}'
        file_path.write_text(content)

        items = _read_input_file(file_path)

        assert len(items) == 2

    def test_skips_empty_lines(self, tmp_path):
        """Skip empty lines in JSONL."""
        file_path = tmp_path / "debates.jsonl"
        content = '{"question": "Q1"}\n\n\n{"question": "Q2"}'
        file_path.write_text(content)

        items = _read_input_file(file_path)

        assert len(items) == 2

    def test_warns_on_invalid_json(self, tmp_path, capsys):
        """Warn on invalid JSON lines."""
        file_path = tmp_path / "debates.jsonl"
        content = '{"question": "Q1"}\ninvalid json\n{"question": "Q2"}'
        file_path.write_text(content)

        items = _read_input_file(file_path)

        assert len(items) == 2
        captured = capsys.readouterr()
        assert "Warning" in captured.out


class TestCmdBatch:
    """Tests for cmd_batch function."""

    def test_file_not_found(self, mock_args, tmp_path):
        """Exit on file not found."""
        mock_args.input = str(tmp_path / "nonexistent.jsonl")

        with pytest.raises(SystemExit) as exc_info:
            cmd_batch(mock_args)

        assert exc_info.value.code == 1

    def test_empty_file(self, mock_args, tmp_path):
        """Exit on empty file."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")
        mock_args.input = str(empty_file)

        with pytest.raises(SystemExit) as exc_info:
            cmd_batch(mock_args)

        assert exc_info.value.code == 1

    @patch("aragora.cli.batch._batch_via_server")
    def test_uses_server_mode(self, mock_server, mock_args, jsonl_file, capsys):
        """Use server mode when --server flag is set."""
        mock_args.input = str(jsonl_file)
        mock_args.server = True

        cmd_batch(mock_args)

        mock_server.assert_called_once()

    @patch("aragora.cli.batch._batch_local")
    def test_uses_local_mode(self, mock_local, mock_args, jsonl_file, capsys):
        """Use local mode by default."""
        mock_args.input = str(jsonl_file)
        mock_args.server = False

        cmd_batch(mock_args)

        mock_local.assert_called_once()


class TestBatchViaServer:
    """Tests for _batch_via_server function."""

    @patch("urllib.request.urlopen")
    def test_submits_batch(self, mock_urlopen, sample_items, mock_args, capsys):
        """Submit batch to server."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"success": True, "batch_id": "batch-123", "items_queued": 3}
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        _batch_via_server(sample_items, mock_args)

        captured = capsys.readouterr()
        assert "Batch submitted successfully" in captured.out
        assert "batch-123" in captured.out

    @patch("urllib.request.urlopen")
    def test_includes_webhook(self, mock_urlopen, sample_items, mock_args):
        """Include webhook URL in request."""
        mock_args.webhook = "http://webhook.example.com"
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"success": True, "batch_id": "batch-123"}
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        _batch_via_server(sample_items, mock_args)

        # Verify webhook was included in request
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        data = json.loads(request.data.decode())
        assert data["webhook_url"] == "http://webhook.example.com"

    @patch("urllib.request.urlopen")
    def test_server_error(self, mock_urlopen, sample_items, mock_args, capsys):
        """Handle server error."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "http://test", 500, "Server Error", {}, None
        )

        with pytest.raises(SystemExit) as exc_info:
            _batch_via_server(sample_items, mock_args)

        assert exc_info.value.code == 1

    @patch("urllib.request.urlopen")
    def test_connection_error(self, mock_urlopen, sample_items, mock_args, capsys):
        """Handle connection error."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(SystemExit) as exc_info:
            _batch_via_server(sample_items, mock_args)

        assert exc_info.value.code == 1


class TestBatchLocal:
    """Tests for _batch_local function."""

    @patch("aragora.cli.main.run_debate", new_callable=AsyncMock)
    def test_processes_items(self, mock_run_debate, sample_items, mock_args, capsys):
        """Process items locally."""
        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.final_answer = "Test answer"
        mock_run_debate.return_value = mock_result

        _batch_local(sample_items, mock_args)

        assert mock_run_debate.call_count == 3
        captured = capsys.readouterr()
        assert "BATCH COMPLETE" in captured.out
        assert "Succeeded: 3" in captured.out

    @patch("aragora.cli.main.run_debate", new_callable=AsyncMock)
    def test_handles_failures(self, mock_run_debate, sample_items, mock_args, capsys):
        """Handle failed debates."""
        mock_run_debate.side_effect = Exception("API error")

        _batch_local(sample_items, mock_args)

        captured = capsys.readouterr()
        assert "Failed: 3" in captured.out
        assert "ERROR" in captured.out

    @patch("aragora.cli.main.run_debate", new_callable=AsyncMock)
    def test_saves_results(self, mock_run_debate, sample_items, mock_args, tmp_path, capsys):
        """Save results to output file."""
        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.final_answer = "Test answer"
        mock_run_debate.return_value = mock_result

        output_file = tmp_path / "results.json"
        mock_args.output = str(output_file)

        _batch_local(sample_items, mock_args)

        assert output_file.exists()
        results = json.loads(output_file.read_text())
        assert len(results) == 3
        captured = capsys.readouterr()
        assert "Results saved" in captured.out

    @patch("aragora.cli.main.run_debate", new_callable=AsyncMock)
    def test_uses_item_agents(self, mock_run_debate, mock_args, capsys):
        """Use agents from item when specified."""
        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.final_answer = "Test"
        mock_run_debate.return_value = mock_result

        items = [{"question": "Q1", "agents": "custom-agent"}]
        _batch_local(items, mock_args)

        call_kwargs = mock_run_debate.call_args[1]
        assert call_kwargs["agents_str"] == "custom-agent"

    @patch("aragora.cli.main.run_debate", new_callable=AsyncMock)
    def test_uses_item_rounds(self, mock_run_debate, mock_args, capsys):
        """Use rounds from item when specified."""
        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.final_answer = "Test"
        mock_run_debate.return_value = mock_result

        items = [{"question": "Q1", "rounds": 5}]
        _batch_local(items, mock_args)

        call_kwargs = mock_run_debate.call_args[1]
        assert call_kwargs["rounds"] == 5


class TestPollBatchStatus:
    """Tests for _poll_batch_status function."""

    @patch("urllib.request.urlopen")
    @patch("time.sleep")
    def test_polls_until_complete(self, mock_sleep, mock_urlopen, capsys):
        """Poll until batch completes."""
        responses = [
            {"status": "processing", "progress_percent": 50, "completed": 1, "failed": 0, "total_items": 2},
            {"status": "completed", "progress_percent": 100, "completed": 2, "failed": 0, "total_items": 2},
        ]

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        call_count = [0]

        def side_effect(*args, **kwargs):
            response_data = responses[min(call_count[0], len(responses) - 1)]
            mock_response.read.return_value = json.dumps(response_data).encode()
            call_count[0] += 1
            return mock_response

        mock_urlopen.side_effect = side_effect

        _poll_batch_status("http://localhost:8080", "batch-123")

        captured = capsys.readouterr()
        assert "Batch completed successfully" in captured.out

    @patch("urllib.request.urlopen")
    @patch("time.sleep")
    def test_handles_partial_completion(self, mock_sleep, mock_urlopen, capsys):
        """Handle partial batch completion."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"status": "partial", "progress_percent": 100, "completed": 2, "failed": 1, "total_items": 3}
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        _poll_batch_status("http://localhost:8080", "batch-123")

        captured = capsys.readouterr()
        assert "partially completed" in captured.out

    @patch("urllib.request.urlopen")
    @patch("time.sleep")
    def test_handles_failure(self, mock_sleep, mock_urlopen, capsys):
        """Handle batch failure."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"status": "failed", "progress_percent": 50, "completed": 1, "failed": 2, "total_items": 3}
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        _poll_batch_status("http://localhost:8080", "batch-123")

        captured = capsys.readouterr()
        assert "Batch failed" in captured.out

    @patch("urllib.request.urlopen")
    @patch("time.sleep")
    def test_includes_auth_token(self, mock_sleep, mock_urlopen):
        """Include auth token in poll requests."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"status": "completed"}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        _poll_batch_status("http://localhost:8080", "batch-123", token="secret-token")

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.get_header("Authorization") == "Bearer secret-token"
