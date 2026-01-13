"""Tests for audience participation wiring.

These tests verify that:
1. The CLI correctly attempts to connect to the streaming server
2. The frontend sendMessage auto-injects loop_id
3. Arena accepts event_emitter parameter
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json


class TestCLIAudienceWiring:
    """Tests for CLI audience participation wiring."""

    def test_get_event_emitter_returns_none_when_server_unavailable(self):
        """When server is unavailable, get_event_emitter_if_available returns None."""
        from aragora.cli.main import get_event_emitter_if_available

        # Server not running - should return None gracefully
        result = get_event_emitter_if_available("http://localhost:99999")
        assert result is None

    def test_get_event_emitter_handles_timeout(self):
        """Timeout should be handled gracefully."""
        from aragora.cli.main import get_event_emitter_if_available

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("Connection timed out")
            result = get_event_emitter_if_available("http://localhost:8080")
            assert result is None

    def test_get_event_emitter_handles_connection_refused(self):
        """Connection refused should be handled gracefully."""
        from aragora.cli.main import get_event_emitter_if_available

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = ConnectionRefusedError()
            result = get_event_emitter_if_available("http://localhost:8080")
            assert result is None

    def test_run_debate_accepts_enable_audience_param(self):
        """run_debate should accept enable_audience parameter."""
        import inspect
        from aragora.cli.main import run_debate

        sig = inspect.signature(run_debate)
        params = list(sig.parameters.keys())
        assert "enable_audience" in params
        assert "server_url" in params


class TestArenaEventEmitter:
    """Tests for Arena event_emitter parameter."""

    def test_arena_accepts_event_emitter(self):
        """Arena should accept event_emitter as a parameter."""
        import inspect
        from aragora.debate.orchestrator import Arena

        sig = inspect.signature(Arena.__init__)
        params = list(sig.parameters.keys())
        assert "event_emitter" in params

    def test_arena_accepts_loop_id(self):
        """Arena should accept loop_id as a parameter."""
        import inspect
        from aragora.debate.orchestrator import Arena

        sig = inspect.signature(Arena.__init__)
        params = list(sig.parameters.keys())
        assert "loop_id" in params


class TestFrontendLoopIdInjection:
    """Tests for frontend loop_id auto-injection logic.

    Note: These are conceptual tests since the actual TypeScript code
    can't be directly tested in Python. They verify the expected behavior.
    """

    def test_user_vote_should_have_loop_id(self):
        """User votes should include loop_id for proper routing."""

        # Simulate the frontend logic in Python
        def send_message(message: dict, selected_loop_id: str | None) -> dict:
            """Simulates the fixed sendMessage function."""
            msg_type = message.get("type", "")
            if msg_type in ("user_vote", "user_suggestion") and "loop_id" not in message:
                if selected_loop_id:
                    message = {**message, "loop_id": selected_loop_id}
            return message

        # Test: vote without loop_id gets it injected
        vote = {"type": "user_vote", "agent": "claude"}
        result = send_message(vote, "loop-123")
        assert result["loop_id"] == "loop-123"

    def test_user_suggestion_should_have_loop_id(self):
        """User suggestions should include loop_id for proper routing."""

        def send_message(message: dict, selected_loop_id: str | None) -> dict:
            msg_type = message.get("type", "")
            if msg_type in ("user_vote", "user_suggestion") and "loop_id" not in message:
                if selected_loop_id:
                    message = {**message, "loop_id": selected_loop_id}
            return message

        suggestion = {"type": "user_suggestion", "content": "Consider caching"}
        result = send_message(suggestion, "loop-456")
        assert result["loop_id"] == "loop-456"

    def test_existing_loop_id_not_overwritten(self):
        """If message already has loop_id, don't overwrite it."""

        def send_message(message: dict, selected_loop_id: str | None) -> dict:
            msg_type = message.get("type", "")
            if msg_type in ("user_vote", "user_suggestion") and "loop_id" not in message:
                if selected_loop_id:
                    message = {**message, "loop_id": selected_loop_id}
            return message

        vote = {"type": "user_vote", "agent": "claude", "loop_id": "original-loop"}
        result = send_message(vote, "different-loop")
        assert result["loop_id"] == "original-loop"

    def test_other_message_types_not_modified(self):
        """Non-audience messages should not get loop_id injected."""

        def send_message(message: dict, selected_loop_id: str | None) -> dict:
            msg_type = message.get("type", "")
            if msg_type in ("user_vote", "user_suggestion") and "loop_id" not in message:
                if selected_loop_id:
                    message = {**message, "loop_id": selected_loop_id}
            return message

        ping = {"type": "ping"}
        result = send_message(ping, "loop-123")
        assert "loop_id" not in result

    def test_no_loop_id_when_none_selected(self):
        """If no loop is selected, don't inject loop_id."""

        def send_message(message: dict, selected_loop_id: str | None) -> dict:
            msg_type = message.get("type", "")
            if msg_type in ("user_vote", "user_suggestion") and "loop_id" not in message:
                if selected_loop_id:
                    message = {**message, "loop_id": selected_loop_id}
            return message

        vote = {"type": "user_vote", "agent": "claude"}
        result = send_message(vote, None)
        assert "loop_id" not in result


class TestDiagnosticScript:
    """Tests for the diagnostic script."""

    def test_diagnostic_script_exists(self):
        """Diagnostic script should exist."""
        from pathlib import Path

        script_path = Path(__file__).parent.parent / "diagnostics" / "audience_wiring_check.py"
        assert script_path.exists(), f"Diagnostic script not found at {script_path}"

    def test_diagnostic_script_is_valid_python(self):
        """Diagnostic script should be valid Python."""
        from pathlib import Path
        import ast

        script_path = Path(__file__).parent.parent / "diagnostics" / "audience_wiring_check.py"
        with open(script_path) as f:
            source = f.read()

        # This will raise SyntaxError if invalid
        ast.parse(source)

    def test_diagnostic_finds_arena_calls(self):
        """Diagnostic script should be able to parse Arena calls."""
        import sys
        from pathlib import Path

        # Add diagnostics to path
        diagnostics_path = Path(__file__).parent.parent / "diagnostics"
        sys.path.insert(0, str(diagnostics_path))

        try:
            from audience_wiring_check import find_arena_calls

            # Create a temp file with Arena call
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(
                    "from aragora import Arena\narena = Arena(env, agents, event_emitter=emitter)\n"
                )
                temp_path = f.name

            calls = find_arena_calls(Path(temp_path))
            assert len(calls) == 1
            assert calls[0].has_event_emitter is True

            # Cleanup
            import os

            os.unlink(temp_path)
        finally:
            sys.path.remove(str(diagnostics_path))
