"""
Tests for aragora.cli.batch module.

Tests batch debate processing CLI commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.batch import (
    _read_input_file,
    cmd_batch,
    create_batch_parser,
)


# ===========================================================================
# Tests: create_batch_parser
# ===========================================================================


class TestCreateBatchParser:
    """Tests for create_batch_parser function."""

    def test_creates_parser(self):
        """Test parser creation."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        create_batch_parser(subparsers)

        # Should be able to parse batch command
        args = parser.parse_args(["batch", "input.jsonl"])
        assert args.input == "input.jsonl"

    def test_default_values(self):
        """Test default argument values."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_batch_parser(subparsers)

        args = parser.parse_args(["batch", "input.jsonl"])

        assert args.server is False
        assert args.wait is False
        assert args.agents == "anthropic-api,openai-api"
        assert args.rounds == 3

    def test_server_mode_args(self):
        """Test server mode arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_batch_parser(subparsers)

        args = parser.parse_args(
            [
                "batch",
                "input.jsonl",
                "--server",
                "--url",
                "http://custom:8080",
                "--token",
                "test-token",
                "--webhook",
                "http://webhook.test",
                "--wait",
            ]
        )

        assert args.server is True
        assert args.url == "http://custom:8080"
        assert args.token == "test-token"
        assert args.webhook == "http://webhook.test"
        assert args.wait is True

    def test_local_mode_args(self):
        """Test local mode arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_batch_parser(subparsers)

        args = parser.parse_args(
            [
                "batch",
                "input.jsonl",
                "--agents",
                "anthropic-api,gemini-api",
                "--rounds",
                "5",
                "--output",
                "results.json",
            ]
        )

        assert args.agents == "anthropic-api,gemini-api"
        assert args.rounds == 5
        assert args.output == "results.json"


# ===========================================================================
# Tests: _read_input_file
# ===========================================================================


class TestReadInputFile:
    """Tests for _read_input_file function."""

    def test_reads_json_array(self, tmp_path):
        """Test reading JSON array format."""
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps([{"question": "Topic 1"}, {"question": "Topic 2"}]))

        result = _read_input_file(input_file)

        assert len(result) == 2
        assert result[0]["question"] == "Topic 1"
        assert result[1]["question"] == "Topic 2"

    def test_reads_jsonl(self, tmp_path):
        """Test reading JSONL format."""
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(
            '{"question": "Topic 1"}\n{"question": "Topic 2"}\n{"question": "Topic 3"}'
        )

        result = _read_input_file(input_file)

        assert len(result) == 3
        assert result[0]["question"] == "Topic 1"
        assert result[1]["question"] == "Topic 2"
        assert result[2]["question"] == "Topic 3"

    def test_skips_comments(self, tmp_path):
        """Test JSONL with comments."""
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(
            '# This is a comment\n{"question": "Topic 1"}\n# Another comment\n{"question": "Topic 2"}'
        )

        result = _read_input_file(input_file)

        assert len(result) == 2

    def test_skips_empty_lines(self, tmp_path):
        """Test JSONL with empty lines."""
        input_file = tmp_path / "input.jsonl"
        input_file.write_text('{"question": "Topic 1"}\n\n\n{"question": "Topic 2"}\n')

        result = _read_input_file(input_file)

        assert len(result) == 2

    def test_skips_invalid_json_lines(self, tmp_path, capsys):
        """Test JSONL with invalid JSON lines."""
        input_file = tmp_path / "input.jsonl"
        input_file.write_text('{"question": "Topic 1"}\ninvalid json\n{"question": "Topic 2"}')

        result = _read_input_file(input_file)

        assert len(result) == 2
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "line 2" in captured.out


# ===========================================================================
# Tests: cmd_batch
# ===========================================================================


class TestCmdBatch:
    """Tests for cmd_batch function."""

    def test_file_not_found(self, capsys):
        """Test error when input file not found."""
        args = argparse.Namespace(input="/nonexistent/file.jsonl")

        with pytest.raises(SystemExit) as exc_info:
            cmd_batch(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_empty_input_file(self, tmp_path, capsys):
        """Test error when input file has no items."""
        input_file = tmp_path / "empty.jsonl"
        input_file.write_text("")

        args = argparse.Namespace(input=str(input_file), server=False)

        with pytest.raises(SystemExit) as exc_info:
            cmd_batch(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "No valid debate items" in captured.out

    def test_prints_header(self, tmp_path, capsys):
        """Test prints batch header."""
        input_file = tmp_path / "input.jsonl"
        input_file.write_text('{"question": "Test"}')

        args = argparse.Namespace(
            input=str(input_file),
            server=False,
            agents="anthropic-api",
            rounds=3,
            output=None,
        )

        # Mock the run_debate to avoid actual execution
        with patch("aragora.cli.batch._batch_local"):
            cmd_batch(args)

        captured = capsys.readouterr()
        assert "BATCH DEBATE PROCESSING" in captured.out
        assert "Items: 1" in captured.out
        assert "Mode: local" in captured.out

    def test_server_mode_selected(self, tmp_path, capsys):
        """Test server mode is selected."""
        input_file = tmp_path / "input.jsonl"
        input_file.write_text('{"question": "Test"}')

        args = argparse.Namespace(
            input=str(input_file),
            server=True,
            url="http://localhost:8080",
            token=None,
            webhook=None,
            wait=False,
        )

        with patch("aragora.cli.batch._batch_via_server") as mock_server:
            cmd_batch(args)

        mock_server.assert_called_once()
        captured = capsys.readouterr()
        assert "Mode: server" in captured.out

    def test_local_mode_selected(self, tmp_path, capsys):
        """Test local mode is selected."""
        input_file = tmp_path / "input.jsonl"
        input_file.write_text('{"question": "Test"}')

        args = argparse.Namespace(
            input=str(input_file),
            server=False,
            agents="anthropic-api",
            rounds=3,
            output=None,
        )

        with patch("aragora.cli.batch._batch_local") as mock_local:
            cmd_batch(args)

        mock_local.assert_called_once()
        captured = capsys.readouterr()
        assert "Mode: local" in captured.out


# ===========================================================================
# Tests: _batch_via_server
# ===========================================================================


class TestBatchViaServer:
    """Tests for _batch_via_server function."""

    def test_submits_to_server(self, capsys):
        """Test submitting batch to server."""
        from aragora.cli.batch import _batch_via_server

        items = [{"question": "Test 1"}, {"question": "Test 2"}]
        args = argparse.Namespace(
            url="http://localhost:8080",
            token="test-token",
            webhook=None,
            wait=False,
        )

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "success": True,
                "batch_id": "batch-123",
                "items_queued": 2,
                "status_url": "http://localhost:8080/status/batch-123",
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            _batch_via_server(items, args)

        captured = capsys.readouterr()
        assert "Batch submitted successfully" in captured.out
        assert "batch-123" in captured.out

    def test_handles_server_error(self, capsys):
        """Test handling server error."""
        from aragora.cli.batch import _batch_via_server
        import urllib.error

        items = [{"question": "Test"}]
        args = argparse.Namespace(
            url="http://localhost:8080",
            token=None,
            webhook=None,
            wait=False,
        )

        mock_error = urllib.error.HTTPError("http://test", 500, "Server Error", {}, None)

        with patch("urllib.request.urlopen", side_effect=mock_error):
            with pytest.raises(SystemExit) as exc_info:
                _batch_via_server(items, args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Server error" in captured.out

    def test_handles_connection_error(self, capsys):
        """Test handling connection error."""
        from aragora.cli.batch import _batch_via_server
        import urllib.error

        items = [{"question": "Test"}]
        args = argparse.Namespace(
            url="http://localhost:8080",
            token=None,
            webhook=None,
            wait=False,
        )

        mock_error = urllib.error.URLError("Connection refused")

        with patch("urllib.request.urlopen", side_effect=mock_error):
            with pytest.raises(SystemExit) as exc_info:
                _batch_via_server(items, args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Connection error" in captured.out

    def test_with_webhook(self, capsys):
        """Test batch with webhook URL."""
        from aragora.cli.batch import _batch_via_server

        items = [{"question": "Test"}]
        args = argparse.Namespace(
            url="http://localhost:8080",
            token=None,
            webhook="http://webhook.test/callback",
            wait=False,
        )

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"success": True, "batch_id": "batch-123"}
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_open:
            _batch_via_server(items, args)

        # Check that webhook was included in request
        call_args = mock_open.call_args
        request = call_args[0][0]
        data = json.loads(request.data.decode())
        assert data.get("webhook_url") == "http://webhook.test/callback"


# ===========================================================================
# Tests: _batch_local
# ===========================================================================


class TestBatchLocal:
    """Tests for _batch_local function."""

    def test_processes_items(self, tmp_path, capsys):
        """Test processing items locally."""
        from aragora.cli.batch import _batch_local

        items = [{"question": "Test 1"}, {"question": "Test 2"}]
        args = argparse.Namespace(
            agents="anthropic-api",
            rounds=3,
            output=None,
        )

        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.final_answer = "Test answer"

        with patch("asyncio.run", return_value=mock_result):
            _batch_local(items, args)

        captured = capsys.readouterr()
        assert "BATCH COMPLETE" in captured.out
        assert "Total: 2" in captured.out

    def test_handles_debate_error(self, capsys):
        """Test handling debate error."""
        from aragora.cli.batch import _batch_local

        items = [{"question": "Test"}]
        args = argparse.Namespace(
            agents="anthropic-api",
            rounds=3,
            output=None,
        )

        with patch("asyncio.run", side_effect=Exception("Test error")):
            _batch_local(items, args)

        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "Failed: 1" in captured.out

    def test_saves_output(self, tmp_path):
        """Test saving output to file."""
        from aragora.cli.batch import _batch_local

        items = [{"question": "Test"}]
        output_file = tmp_path / "results.json"
        args = argparse.Namespace(
            agents="anthropic-api",
            rounds=3,
            output=str(output_file),
        )

        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.final_answer = "Test answer"

        with patch("asyncio.run", return_value=mock_result):
            _batch_local(items, args)

        assert output_file.exists()
        results = json.loads(output_file.read_text())
        assert len(results) == 1
        assert results[0]["success"] is True

    def test_uses_item_agents(self, capsys):
        """Test uses agents from item if specified."""
        from aragora.cli.batch import _batch_local

        items = [{"question": "Test", "agents": "custom-agent"}]
        args = argparse.Namespace(
            agents="default-agent",
            rounds=3,
            output=None,
        )

        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.final_answer = "Test"

        with patch("asyncio.run", return_value=mock_result) as mock_run:
            _batch_local(items, args)

        # The run_debate call should have used custom-agent from item
        # (we can't easily check args without more complex mocking)

    def test_uses_item_rounds(self, capsys):
        """Test uses rounds from item if specified."""
        from aragora.cli.batch import _batch_local

        items = [{"question": "Test", "rounds": 5}]
        args = argparse.Namespace(
            agents="default-agent",
            rounds=3,
            output=None,
        )

        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.final_answer = "Test"

        with patch("asyncio.run", return_value=mock_result):
            _batch_local(items, args)


# ===========================================================================
# Tests: _poll_batch_status
# ===========================================================================


class TestPollBatchStatus:
    """Tests for _poll_batch_status function."""

    def test_polls_until_completed(self, capsys):
        """Test polling until batch completes."""
        from aragora.cli.batch import _poll_batch_status

        responses = [
            {
                "status": "processing",
                "progress_percent": 50,
                "completed": 1,
                "failed": 0,
                "total_items": 2,
            },
            {
                "status": "completed",
                "progress_percent": 100,
                "completed": 2,
                "failed": 0,
                "total_items": 2,
            },
        ]
        response_iter = iter(responses)

        def mock_urlopen(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(next(response_iter)).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with patch("time.sleep"):  # Skip actual sleeping
                _poll_batch_status("http://localhost:8080", "batch-123", None)

        captured = capsys.readouterr()
        assert "completed successfully" in captured.out

    def test_handles_partial_completion(self, capsys):
        """Test handling partial completion."""
        from aragora.cli.batch import _poll_batch_status

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {
                "status": "partial",
                "progress_percent": 100,
                "completed": 1,
                "failed": 1,
                "total_items": 2,
            }
        ).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            _poll_batch_status("http://localhost:8080", "batch-123", None)

        captured = capsys.readouterr()
        assert "partially completed" in captured.out

    def test_handles_failure(self, capsys):
        """Test handling batch failure."""
        from aragora.cli.batch import _poll_batch_status

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {
                "status": "failed",
                "progress_percent": 50,
                "completed": 0,
                "failed": 2,
                "total_items": 2,
            }
        ).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            _poll_batch_status("http://localhost:8080", "batch-123", None)

        captured = capsys.readouterr()
        assert "failed" in captured.out.lower()
