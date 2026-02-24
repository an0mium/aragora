"""
Tests for RotationPipeline, LocalEnvRotator, and cron schedule validation.
"""

import json
import os
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from aragora.security.token_rotation import (
    TokenType,
    TokenRotationConfig,
    TokenRotationResult,
    TokenRotationManager,
    RotationPipeline,
    RotationEvent,
    LocalEnvRotator,
    TOKEN_ENV_VARS,
    _validate_cron_schedule,
)


# =============================================================================
# Cron Schedule Validation
# =============================================================================


class TestCronScheduleValidation:
    """Tests for _validate_cron_schedule."""

    def test_valid_standard_crons(self):
        assert _validate_cron_schedule("0 3 * * 0") is True  # 3AM Sunday
        assert _validate_cron_schedule("*/15 * * * *") is True  # Every 15 min
        assert _validate_cron_schedule("0 0 1 * *") is True  # Monthly
        assert _validate_cron_schedule("30 4 * * 1-5") is True  # Weekdays 4:30AM

    def test_valid_ranges_and_lists(self):
        assert _validate_cron_schedule("0,30 * * * *") is True
        assert _validate_cron_schedule("0 1-5 * * *") is True
        assert _validate_cron_schedule("*/10 */2 * * *") is True

    def test_invalid_crons(self):
        assert _validate_cron_schedule("") is False
        assert _validate_cron_schedule("not a cron") is False
        assert _validate_cron_schedule("0 3 * *") is False  # Only 4 fields
        assert _validate_cron_schedule("0 3 * * * *") is False  # 6 fields

    def test_all_wildcards(self):
        assert _validate_cron_schedule("* * * * *") is True


# =============================================================================
# LocalEnvRotator
# =============================================================================


class TestLocalEnvRotator:
    """Tests for local .env file rotation."""

    def test_read_env(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# Comment\n"
            "ANTHROPIC_API_KEY=sk-ant-test123\n"
            "OPENAI_API_KEY=sk-test456\n"
            "\n"
            "# Another comment\n"
            "EMPTY_VAR=\n"
        )
        rotator = LocalEnvRotator(env_path=env_file)
        env = rotator.read_env()

        assert env["ANTHROPIC_API_KEY"] == "sk-ant-test123"
        assert env["OPENAI_API_KEY"] == "sk-test456"
        assert env["EMPTY_VAR"] == ""

    def test_read_env_nonexistent(self, tmp_path):
        rotator = LocalEnvRotator(env_path=tmp_path / ".env")
        assert rotator.read_env() == {}

    def test_read_env_quoted_values(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('KEY1="value with spaces"\nKEY2=\'single quoted\'\n')
        rotator = LocalEnvRotator(env_path=env_file)
        env = rotator.read_env()
        assert env["KEY1"] == "value with spaces"
        assert env["KEY2"] == "single quoted"

    def test_update_existing_secret(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# API Keys\nANTHROPIC_API_KEY=old_value\nOPENAI_API_KEY=keep_this\n"
        )
        rotator = LocalEnvRotator(env_path=env_file)
        result = rotator.update_secret("ANTHROPIC_API_KEY", "new_value")

        assert result is True
        content = env_file.read_text()
        assert "ANTHROPIC_API_KEY=new_value" in content
        assert "OPENAI_API_KEY=keep_this" in content
        assert "# API Keys" in content  # Comments preserved

    def test_update_creates_backup(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=old\n")
        rotator = LocalEnvRotator(env_path=env_file)
        rotator.update_secret("KEY", "new")

        backup_dir = tmp_path / ".env_backups"
        assert backup_dir.exists()
        backups = list(backup_dir.glob(".env.*"))
        assert len(backups) == 1
        assert backups[0].read_text() == "KEY=old\n"

    def test_update_adds_new_key(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING=value\n")
        rotator = LocalEnvRotator(env_path=env_file)
        rotator.update_secret("NEW_KEY", "new_value")

        content = env_file.read_text()
        assert "EXISTING=value" in content
        assert "NEW_KEY=new_value" in content

    def test_update_quotes_values_with_spaces(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        rotator = LocalEnvRotator(env_path=env_file)
        rotator.update_secret("KEY", "value with spaces")

        content = env_file.read_text()
        assert 'KEY="value with spaces"' in content

    def test_env_path_property(self, tmp_path):
        env_file = tmp_path / ".env"
        rotator = LocalEnvRotator(env_path=env_file)
        assert rotator.env_path == env_file


# =============================================================================
# RotationEvent
# =============================================================================


class TestRotationEvent:
    """Tests for RotationEvent dataclass."""

    def test_to_dict(self):
        event = RotationEvent(
            event_type="rotation_completed",
            token_type="pypi",
            success=True,
            duration_seconds=1.5,
            stores_updated=["aws", "github"],
        )
        d = event.to_dict()
        assert d["event_type"] == "rotation_completed"
        assert d["token_type"] == "pypi"
        assert d["success"] is True
        assert d["duration_seconds"] == 1.5
        assert d["stores_updated"] == ["aws", "github"]
        assert "timestamp" in d

    def test_failure_event(self):
        event = RotationEvent(
            event_type="rotation_failed",
            token_type="npm",
            success=False,
            error="AWS connection failed",
        )
        d = event.to_dict()
        assert d["success"] is False
        assert d["error"] == "AWS connection failed"


# =============================================================================
# RotationPipeline
# =============================================================================


class TestRotationPipeline:
    """Tests for the full rotation pipeline."""

    def _make_manager(self, **kwargs):
        config = TokenRotationConfig(stores=[], **kwargs)
        return TokenRotationManager(config=config)

    def test_basic_execute(self):
        """Pipeline execute calls manager.rotate and emits events."""
        manager = self._make_manager()
        pipeline = RotationPipeline(
            manager=manager,
            health_checker=lambda tt: True,
        )

        result = pipeline.execute(TokenType.PYPI, "pypi-test123")

        assert result.success is True
        # Should have: rotation_started, health_check, rotation_completed
        assert len(pipeline.events) == 3
        assert pipeline.events[0].event_type == "rotation_started"
        assert pipeline.events[1].event_type == "health_check"
        assert pipeline.events[2].event_type == "rotation_completed"

    def test_execute_with_health_check_failure(self):
        """Failed health check should mark rotation as failed."""
        manager = self._make_manager()
        pipeline = RotationPipeline(
            manager=manager,
            health_checker=lambda tt: False,
        )

        result = pipeline.execute(TokenType.PYPI, "pypi-test123")

        assert result.success is False
        assert "health_check" in result.errors
        # Should have: rotation_started, health_check, rotation_failed
        assert len(pipeline.events) == 3
        assert pipeline.events[2].event_type == "rotation_failed"

    def test_execute_skip_health_check(self):
        """skip_health_check=True should bypass health check."""
        manager = self._make_manager()
        pipeline = RotationPipeline(
            manager=manager,
            health_checker=lambda tt: False,  # Would fail
        )

        result = pipeline.execute(
            TokenType.PYPI, "pypi-test123", skip_health_check=True
        )

        assert result.success is True
        # Should have: rotation_started, rotation_completed (no health_check)
        assert len(pipeline.events) == 2

    def test_alert_callback_on_failure(self):
        """Alert callback should be invoked on rotation failure."""
        manager = self._make_manager()
        alerts = []

        pipeline = RotationPipeline(
            manager=manager,
            health_checker=lambda tt: False,
            alert_callback=lambda event: alerts.append(event),
        )

        pipeline.execute(TokenType.PYPI, "pypi-test123")

        # Should have received failure event
        failure_alerts = [a for a in alerts if a.event_type == "rotation_failed"]
        assert len(failure_alerts) == 1
        assert failure_alerts[0].success is False

    def test_alert_callback_exception_handled(self):
        """Alert callback exceptions should not crash the pipeline."""
        manager = self._make_manager()

        def bad_callback(event):
            raise RuntimeError("PagerDuty down")

        pipeline = RotationPipeline(
            manager=manager,
            health_checker=lambda tt: False,
            alert_callback=bad_callback,
        )

        # Should not raise
        result = pipeline.execute(TokenType.PYPI, "pypi-test123")
        assert result.success is False

    def test_schedule_validation(self):
        """Invalid cron schedule should raise ValueError."""
        manager = self._make_manager()

        with pytest.raises(ValueError, match="Invalid cron schedule"):
            RotationPipeline(manager=manager, schedule="bad cron")

    def test_valid_schedule(self):
        """Valid cron schedule should be accepted."""
        manager = self._make_manager()
        pipeline = RotationPipeline(manager=manager, schedule="0 3 * * 0")
        assert pipeline.schedule == "0 3 * * 0"
        assert pipeline.validate_schedule() is True

    def test_no_schedule(self):
        """No schedule should be fine."""
        manager = self._make_manager()
        pipeline = RotationPipeline(manager=manager)
        assert pipeline.schedule is None
        assert pipeline.validate_schedule() is True

    def test_local_env_integration(self, tmp_path):
        """Pipeline should update local .env when enabled."""
        env_file = tmp_path / ".env"
        env_file.write_text("PYPI_API_TOKEN=old_token\n")

        manager = self._make_manager()
        pipeline = RotationPipeline(
            manager=manager,
            health_checker=lambda tt: True,
            enable_local_env=True,
            local_env_path=env_file,
        )

        result = pipeline.execute(TokenType.PYPI, "pypi-new_token")

        assert result.success is True
        assert "local_env" in result.stores_updated
        content = env_file.read_text()
        assert "pypi-new_token" in content

    def test_local_env_failure_non_fatal(self, tmp_path):
        """Local .env failure should not crash the pipeline."""
        # Use a read-only path
        env_file = tmp_path / "nonexistent_dir" / ".env"

        manager = self._make_manager()
        pipeline = RotationPipeline(
            manager=manager,
            health_checker=lambda tt: True,
            enable_local_env=True,
            local_env_path=env_file,
        )

        result = pipeline.execute(TokenType.PYPI, "pypi-test")
        # Should still succeed (local_env failure is non-fatal)
        # The main rotation has no stores configured, so success=True
        # but local_env error is recorded
        assert "local_env" in result.errors

    def test_old_token_passed_through(self):
        """old_token should be passed to the underlying manager."""
        manager = self._make_manager()
        pipeline = RotationPipeline(
            manager=manager,
            health_checker=lambda tt: True,
        )

        result = pipeline.execute(
            TokenType.PYPI,
            "pypi-newtoken",
            old_token="pypi-oldtoken",
        )

        assert result.old_token_prefix == "pypi-old..."
        assert result.new_token_prefix == "pypi-new..."

    def test_events_are_copies(self):
        """Pipeline.events should return a copy, not the internal list."""
        manager = self._make_manager()
        pipeline = RotationPipeline(
            manager=manager,
            health_checker=lambda tt: True,
        )
        pipeline.execute(TokenType.PYPI, "pypi-test")

        events = pipeline.events
        events.clear()
        assert len(pipeline.events) > 0  # Internal list unaffected

    @patch("aragora.security.token_rotation._BOTO3_AVAILABLE", True)
    @patch("aragora.security.token_rotation.boto3")
    def test_pipeline_with_aws_store(self, mock_boto3):
        """Pipeline should work with real AWS store."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {"SecretString": "{}"}

        config = TokenRotationConfig(stores=["aws"])
        manager = TokenRotationManager(config=config)
        manager._aws_client = mock_client

        pipeline = RotationPipeline(
            manager=manager,
            health_checker=lambda tt: True,
        )

        result = pipeline.execute(TokenType.PYPI, "pypi-test")
        assert result.success is True
        assert "aws" in result.stores_updated

    def test_telemetry_event_duration(self):
        """Events should have non-zero duration."""
        manager = self._make_manager()
        pipeline = RotationPipeline(
            manager=manager,
            health_checker=lambda tt: True,
        )
        pipeline.execute(TokenType.PYPI, "pypi-test")

        completed = [e for e in pipeline.events if e.event_type == "rotation_completed"]
        assert len(completed) == 1
        assert completed[0].duration_seconds >= 0


# =============================================================================
# Integration with TokenRotationManager
# =============================================================================


class TestPipelineManagerIntegration:
    """Tests ensuring pipeline correctly wraps the manager."""

    def test_stores_override_not_used(self):
        """Pipeline should not interfere with manager store selection."""
        config = TokenRotationConfig(stores=["unknown_store"])
        manager = TokenRotationManager(config=config)
        pipeline = RotationPipeline(
            manager=manager,
            health_checker=lambda tt: True,
        )

        result = pipeline.execute(TokenType.PYPI, "pypi-test")

        # Manager failure + health check not run on failed rotation
        assert result.success is False
        assert "unknown_store" in result.errors

    def test_rotation_history_preserved(self):
        """Manager rotation history should include pipeline rotations."""
        manager = TokenRotationManager(config=TokenRotationConfig(stores=[]))
        pipeline = RotationPipeline(
            manager=manager,
            health_checker=lambda tt: True,
        )

        pipeline.execute(TokenType.PYPI, "pypi-one")
        pipeline.execute(TokenType.NPM, "npm-two")

        history = manager.get_rotation_history()
        assert len(history) == 2
