"""
Tests for security audit schedule module.

Tests cover:
- add_daily_security_scan function
- run_security_scan_with_debate async function
- setup_default_security_schedules function
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.scheduler.security_audit_schedule import (
    add_daily_security_scan,
    run_security_scan_with_debate,
    setup_default_security_schedules,
)


class TestAddDailySecurityScan:
    """Tests for add_daily_security_scan."""

    @pytest.fixture
    def mock_scheduler(self):
        scheduler = MagicMock()
        scheduler.add_schedule.return_value = MagicMock(name="mock_job")
        return scheduler

    def test_creates_schedule_with_defaults(self, mock_scheduler):
        """Creates a schedule with default name and cron."""
        add_daily_security_scan(mock_scheduler)

        mock_scheduler.add_schedule.assert_called_once()
        config = mock_scheduler.add_schedule.call_args[0][0]
        assert config.name == "daily_security_scan"
        assert config.cron == "0 2 * * *"

    def test_custom_cron(self, mock_scheduler):
        """Respects a custom cron expression."""
        add_daily_security_scan(mock_scheduler, cron="0 4 * * 1")

        config = mock_scheduler.add_schedule.call_args[0][0]
        assert config.cron == "0 4 * * 1"

    def test_custom_workspace(self, mock_scheduler):
        """Passes workspace_id through to the config."""
        add_daily_security_scan(mock_scheduler, workspace_id="ws-123")

        config = mock_scheduler.add_schedule.call_args[0][0]
        assert config.workspace_id == "ws-123"

    def test_debate_on_critical_default(self, mock_scheduler):
        """debate_on_critical defaults to True in custom_config."""
        add_daily_security_scan(mock_scheduler)

        config = mock_scheduler.add_schedule.call_args[0][0]
        assert config.custom_config["debate_on_critical"] is True

    def test_returns_scheduled_job(self, mock_scheduler):
        """Returns the ScheduledJob from scheduler.add_schedule."""
        result = add_daily_security_scan(mock_scheduler)

        assert result is mock_scheduler.add_schedule.return_value


class TestRunSecurityScanWithDebate:
    """Tests for run_security_scan_with_debate."""

    @pytest.fixture
    def mock_report(self):
        report = MagicMock()
        report.scan_id = "scan-123"
        report.files_scanned = 10
        report.lines_scanned = 500
        report.risk_score = 25.0
        report.critical_count = 0
        report.high_count = 2
        report.medium_count = 1
        report.low_count = 0
        report.total_findings = 3
        report.findings = []
        return report

    @pytest.mark.asyncio
    async def test_returns_result_dict(self, mock_report):
        """Returns a dict with expected keys from the scan report."""
        mock_scanner_cls = MagicMock()
        mock_scanner_cls.return_value.scan_directory.return_value = mock_report

        with patch("aragora.audit.security_scanner.SecurityScanner", mock_scanner_cls):
            result = await run_security_scan_with_debate()

        assert result["scan_id"] == "scan-123"
        assert result["files_scanned"] == 10
        assert result["critical_count"] == 0
        assert result["debated"] is False

    @pytest.mark.asyncio
    async def test_no_debate_when_no_critical(self, mock_report):
        """Does not trigger debate when critical_count is 0."""
        mock_report.critical_count = 0
        mock_scanner_cls = MagicMock()
        mock_scanner_cls.return_value.scan_directory.return_value = mock_report

        with patch("aragora.audit.security_scanner.SecurityScanner", mock_scanner_cls):
            result = await run_security_scan_with_debate()

        assert result["debated"] is False

    @pytest.mark.asyncio
    async def test_no_debate_when_flag_false(self, mock_report):
        """Does not trigger debate when debate_on_critical is False even with critical findings."""
        mock_report.critical_count = 5
        mock_scanner_cls = MagicMock()
        mock_scanner_cls.return_value.scan_directory.return_value = mock_report

        with patch("aragora.audit.security_scanner.SecurityScanner", mock_scanner_cls):
            result = await run_security_scan_with_debate(debate_on_critical=False)

        assert result["debated"] is False

    @pytest.mark.asyncio
    async def test_debate_import_error_handled(self, mock_report):
        """Handles ImportError when debate modules are unavailable."""
        mock_report.critical_count = 1
        mock_scanner_cls = MagicMock()
        mock_scanner_cls.return_value.scan_directory.return_value = mock_report

        with (
            patch("aragora.audit.security_scanner.SecurityScanner", mock_scanner_cls),
            patch.dict(
                "sys.modules",
                {
                    "aragora.debate.security_debate": None,
                    "aragora.events.security_events": None,
                },
            ),
        ):
            result = await run_security_scan_with_debate()

        assert "debate_error" in result

    @pytest.mark.asyncio
    async def test_debate_runtime_error_handled(self, mock_report):
        """Handles RuntimeError during debate execution."""
        mock_report.critical_count = 1
        mock_scanner_cls = MagicMock()
        mock_scanner_cls.return_value.scan_directory.return_value = mock_report

        mock_debate_mod = MagicMock()
        mock_debate_mod.run_security_debate = AsyncMock(side_effect=RuntimeError("debate failed"))
        mock_events_mod = MagicMock()

        with (
            patch("aragora.audit.security_scanner.SecurityScanner", mock_scanner_cls),
            patch.dict(
                "sys.modules",
                {
                    "aragora.debate.security_debate": mock_debate_mod,
                    "aragora.events.security_events": mock_events_mod,
                },
            ),
        ):
            result = await run_security_scan_with_debate()

        assert "debate_error" in result


class TestSetupDefaultSecuritySchedules:
    """Tests for setup_default_security_schedules."""

    @pytest.fixture
    def mock_scheduler(self):
        scheduler = MagicMock()
        scheduler.add_schedule.return_value = MagicMock(name="mock_job")
        return scheduler

    def test_creates_two_schedules(self, mock_scheduler):
        """Calls scheduler.add_schedule twice (daily + weekly)."""
        setup_default_security_schedules(mock_scheduler)

        assert mock_scheduler.add_schedule.call_count == 2

    def test_returns_two_jobs(self, mock_scheduler):
        """Returns a list of two ScheduledJob instances."""
        result = setup_default_security_schedules(mock_scheduler)

        assert len(result) == 2

    def test_daily_and_weekly_crons(self, mock_scheduler):
        """Creates schedules with distinct daily and weekly cron expressions."""
        setup_default_security_schedules(mock_scheduler)

        calls = mock_scheduler.add_schedule.call_args_list
        config_1 = calls[0][0][0]
        config_2 = calls[1][0][0]

        crons = {config_1.cron, config_2.cron}
        assert "0 2 * * *" in crons
        assert "0 3 * * 0" in crons
