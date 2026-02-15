"""Tests for MCP gauntlet tools execution logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.mcp.tools_module.gauntlet import run_gauntlet_tool



class TestRunGauntletTool:
    """Tests for run_gauntlet_tool."""

    @pytest.mark.asyncio
    async def test_run_empty_content(self):
        """Test gauntlet with empty content."""
        result = await run_gauntlet_tool(content="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_run_quick_profile(self):
        """Test gauntlet with quick profile."""
        mock_vuln = MagicMock()
        mock_vuln.category = "security"
        mock_vuln.severity = "medium"
        mock_vuln.description = "Test vulnerability"

        mock_result = MagicMock()
        mock_result.verdict = MagicMock(value="pass")
        mock_result.risk_score = 0.3
        mock_result.vulnerabilities = [mock_vuln]

        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result

        with (
            patch(
                "aragora.gauntlet.GauntletRunner",
                return_value=mock_runner,
            ),
            patch(
                "aragora.gauntlet.GauntletConfig",
            ) as mock_config,
        ):
            result = await run_gauntlet_tool(
                content="Test spec content",
                content_type="spec",
                profile="quick",
            )

        assert result["verdict"] == "pass"
        assert result["risk_score"] == 0.3
        assert result["profile"] == "quick"
        assert result["vulnerabilities_count"] == 1

    @pytest.mark.asyncio
    async def test_run_security_profile(self):
        """Test gauntlet with security profile."""
        mock_result = MagicMock()
        mock_result.verdict = MagicMock(value="fail")
        mock_result.risk_score = 0.8
        mock_result.vulnerabilities = []

        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result

        with (
            patch(
                "aragora.gauntlet.GauntletRunner",
                return_value=mock_runner,
            ),
            patch(
                "aragora.gauntlet.GauntletConfig",
            ),
            patch(
                "aragora.gauntlet.config.AttackCategory",
            ),
        ):
            result = await run_gauntlet_tool(
                content="Insecure code",
                profile="security",
            )

        assert result["verdict"] == "fail"
        assert result["risk_score"] == 0.8
        assert result["profile"] == "security"

    @pytest.mark.asyncio
    async def test_run_code_profile(self):
        """Test gauntlet with code profile."""
        mock_result = MagicMock()
        mock_result.verdict = MagicMock(value="pass")
        mock_result.risk_score = 0.1
        mock_result.vulnerabilities = []

        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result

        with (
            patch(
                "aragora.gauntlet.GauntletRunner",
                return_value=mock_runner,
            ),
            patch(
                "aragora.gauntlet.GauntletConfig",
            ),
            patch(
                "aragora.gauntlet.config.AttackCategory",
            ),
        ):
            result = await run_gauntlet_tool(
                content="def hello(): pass",
                content_type="code",
                profile="code",
            )

        assert result["verdict"] == "pass"
        assert result["content_type"] == "code"
        assert result["profile"] == "code"

    @pytest.mark.asyncio
    async def test_run_thorough_profile(self):
        """Test gauntlet with thorough profile."""
        mock_result = MagicMock()
        mock_result.verdict = MagicMock(value="pass")
        mock_result.risk_score = 0.2
        mock_result.vulnerabilities = []

        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result

        with (
            patch(
                "aragora.gauntlet.GauntletRunner",
                return_value=mock_runner,
            ),
            patch(
                "aragora.gauntlet.GauntletConfig",
            ) as mock_config_class,
            patch(
                "aragora.gauntlet.config.AttackCategory",
            ),
        ):
            result = await run_gauntlet_tool(
                content="Architecture document",
                content_type="architecture",
                profile="thorough",
            )

        # Thorough profile should use 3 attack rounds
        mock_config_class.assert_called_once()
        call_kwargs = mock_config_class.call_args.kwargs
        assert call_kwargs["attack_rounds"] == 3

    @pytest.mark.asyncio
    async def test_run_limits_vulnerabilities(self):
        """Test that vulnerabilities are limited to 10."""
        mock_vulns = []
        for i in range(15):
            vuln = MagicMock()
            vuln.category = f"cat-{i}"
            vuln.severity = "low"
            vuln.description = f"Vuln {i}"
            mock_vulns.append(vuln)

        mock_result = MagicMock()
        mock_result.verdict = MagicMock(value="fail")
        mock_result.risk_score = 0.5
        mock_result.vulnerabilities = mock_vulns

        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result

        with (
            patch(
                "aragora.gauntlet.GauntletRunner",
                return_value=mock_runner,
            ),
            patch(
                "aragora.gauntlet.GauntletConfig",
            ),
            patch(
                "aragora.gauntlet.config.AttackCategory",
            ),
        ):
            result = await run_gauntlet_tool(content="Test content")

        assert result["vulnerabilities_count"] == 15  # Total count
        assert len(result["vulnerabilities"]) == 10  # Limited to 10

    @pytest.mark.asyncio
    async def test_run_result_without_verdict_attr(self):
        """Test handling result without verdict attribute."""
        mock_result = MagicMock()
        del mock_result.verdict  # Remove verdict attribute
        mock_result.risk_score = 0.5
        mock_result.vulnerabilities = []

        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result

        with (
            patch(
                "aragora.gauntlet.GauntletRunner",
                return_value=mock_runner,
            ),
            patch(
                "aragora.gauntlet.GauntletConfig",
            ),
            patch(
                "aragora.gauntlet.config.AttackCategory",
            ),
        ):
            result = await run_gauntlet_tool(content="Test content")

        assert result["verdict"] == "unknown"

    @pytest.mark.asyncio
    async def test_run_result_without_vulnerabilities(self):
        """Test handling result without vulnerabilities attribute."""
        mock_result = MagicMock()
        mock_result.verdict = MagicMock(value="pass")
        mock_result.risk_score = 0.1
        del mock_result.vulnerabilities  # Remove attribute

        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result

        with (
            patch(
                "aragora.gauntlet.GauntletRunner",
                return_value=mock_runner,
            ),
            patch(
                "aragora.gauntlet.GauntletConfig",
            ),
            patch(
                "aragora.gauntlet.config.AttackCategory",
            ),
        ):
            result = await run_gauntlet_tool(content="Test content")

        assert result["vulnerabilities_count"] == 0
        assert result["vulnerabilities"] == []
