"""Tests for MCP gauntlet tools.

Tests the document stress-testing functionality including:
- Input validation
- Profile selection
- Vulnerability extraction
- Risk scoring
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.mcp.tools_module.gauntlet import run_gauntlet_tool


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestGauntletInputValidation:
    """Tests for gauntlet tool input validation."""

    @pytest.mark.asyncio
    async def test_empty_content_returns_error(self):
        """Empty content returns error."""
        result = await run_gauntlet_tool(content="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_none_content_returns_error(self):
        """None content returns error."""
        result = await run_gauntlet_tool(content=None)  # type: ignore
        assert "error" in result

    @pytest.mark.asyncio
    async def test_whitespace_content_returns_error(self):
        """Whitespace-only content returns error."""
        result = await run_gauntlet_tool(content="   ")
        assert "error" in result


# =============================================================================
# Profile Selection Tests
# =============================================================================


class TestGauntletProfileSelection:
    """Tests for gauntlet profile selection."""

    @pytest.mark.asyncio
    async def test_quick_profile_uses_fewer_rounds(self):
        """Quick profile configures fewer attack rounds."""
        with patch("aragora.mcp.tools_module.gauntlet.GauntletRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.verdict.value = "pass"
            mock_result.risk_score = 0.2
            mock_result.vulnerabilities = []
            mock_instance.run = AsyncMock(return_value=mock_result)
            mock_runner.return_value = mock_instance

            with patch("aragora.mcp.tools_module.gauntlet.GauntletConfig") as mock_config:
                await run_gauntlet_tool(content="test content", profile="quick")
                # Check config was created with 2 rounds for quick
                call_kwargs = mock_config.call_args.kwargs
                assert call_kwargs.get("attack_rounds") == 2

    @pytest.mark.asyncio
    async def test_thorough_profile_uses_more_rounds(self):
        """Thorough profile configures more attack rounds."""
        with patch("aragora.mcp.tools_module.gauntlet.GauntletRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.verdict.value = "pass"
            mock_result.risk_score = 0.2
            mock_result.vulnerabilities = []
            mock_instance.run = AsyncMock(return_value=mock_result)
            mock_runner.return_value = mock_instance

            with patch("aragora.mcp.tools_module.gauntlet.GauntletConfig") as mock_config:
                await run_gauntlet_tool(content="test content", profile="thorough")
                call_kwargs = mock_config.call_args.kwargs
                assert call_kwargs.get("attack_rounds") == 3

    @pytest.mark.asyncio
    async def test_security_profile_uses_security_categories(self):
        """Security profile uses security-focused attack categories."""
        with patch("aragora.mcp.tools_module.gauntlet.GauntletRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.verdict.value = "pass"
            mock_result.risk_score = 0.1
            mock_result.vulnerabilities = []
            mock_instance.run = AsyncMock(return_value=mock_result)
            mock_runner.return_value = mock_instance

            with patch("aragora.mcp.tools_module.gauntlet.GauntletConfig") as mock_config:
                from aragora.gauntlet.config import AttackCategory

                await run_gauntlet_tool(content="test code", profile="security")
                call_kwargs = mock_config.call_args.kwargs
                categories = call_kwargs.get("attack_categories", [])
                assert AttackCategory.SECURITY in categories

    @pytest.mark.asyncio
    async def test_code_profile_uses_logic_categories(self):
        """Code profile uses logic-focused attack categories."""
        with patch("aragora.mcp.tools_module.gauntlet.GauntletRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.verdict.value = "pass"
            mock_result.risk_score = 0.1
            mock_result.vulnerabilities = []
            mock_instance.run = AsyncMock(return_value=mock_result)
            mock_runner.return_value = mock_instance

            with patch("aragora.mcp.tools_module.gauntlet.GauntletConfig") as mock_config:
                from aragora.gauntlet.config import AttackCategory

                await run_gauntlet_tool(content="def test(): pass", profile="code")
                call_kwargs = mock_config.call_args.kwargs
                categories = call_kwargs.get("attack_categories", [])
                assert AttackCategory.LOGIC in categories
                assert AttackCategory.EDGE_CASE in categories


# =============================================================================
# Result Processing Tests
# =============================================================================


class TestGauntletResultProcessing:
    """Tests for gauntlet result processing."""

    @pytest.mark.asyncio
    async def test_result_contains_verdict(self):
        """Result contains verdict field."""
        with patch("aragora.mcp.tools_module.gauntlet.GauntletRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.verdict.value = "pass"
            mock_result.risk_score = 0.1
            mock_result.vulnerabilities = []
            mock_instance.run = AsyncMock(return_value=mock_result)
            mock_runner.return_value = mock_instance

            with patch("aragora.mcp.tools_module.gauntlet.GauntletConfig"):
                result = await run_gauntlet_tool(content="test content")
                assert "verdict" in result
                assert result["verdict"] == "pass"

    @pytest.mark.asyncio
    async def test_result_contains_risk_score(self):
        """Result contains risk score."""
        with patch("aragora.mcp.tools_module.gauntlet.GauntletRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.verdict.value = "pass"
            mock_result.risk_score = 0.75
            mock_result.vulnerabilities = []
            mock_instance.run = AsyncMock(return_value=mock_result)
            mock_runner.return_value = mock_instance

            with patch("aragora.mcp.tools_module.gauntlet.GauntletConfig"):
                result = await run_gauntlet_tool(content="test content")
                assert "risk_score" in result
                assert result["risk_score"] == 0.75

    @pytest.mark.asyncio
    async def test_vulnerabilities_limited_to_ten(self):
        """Vulnerabilities are limited to 10 in output."""
        with patch("aragora.mcp.tools_module.gauntlet.GauntletRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.verdict.value = "fail"
            mock_result.risk_score = 0.9
            # Create 15 vulnerabilities
            mock_result.vulnerabilities = [
                MagicMock(category="test", severity="high", description=f"Vuln {i}")
                for i in range(15)
            ]
            mock_instance.run = AsyncMock(return_value=mock_result)
            mock_runner.return_value = mock_instance

            with patch("aragora.mcp.tools_module.gauntlet.GauntletConfig"):
                result = await run_gauntlet_tool(content="test content")
                assert len(result["vulnerabilities"]) == 10  # Limited to 10

    @pytest.mark.asyncio
    async def test_vulnerability_structure(self):
        """Vulnerability entries have correct structure."""
        with patch("aragora.mcp.tools_module.gauntlet.GauntletRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.verdict.value = "fail"
            mock_result.risk_score = 0.8
            mock_vuln = MagicMock()
            mock_vuln.category = "security"
            mock_vuln.severity = "critical"
            mock_vuln.description = "SQL injection vulnerability"
            mock_result.vulnerabilities = [mock_vuln]
            mock_instance.run = AsyncMock(return_value=mock_result)
            mock_runner.return_value = mock_instance

            with patch("aragora.mcp.tools_module.gauntlet.GauntletConfig"):
                result = await run_gauntlet_tool(content="test content")
                assert len(result["vulnerabilities"]) == 1
                vuln = result["vulnerabilities"][0]
                assert vuln["category"] == "security"
                assert vuln["severity"] == "critical"
                assert "SQL injection" in vuln["description"]

    @pytest.mark.asyncio
    async def test_result_includes_metadata(self):
        """Result includes content type and profile metadata."""
        with patch("aragora.mcp.tools_module.gauntlet.GauntletRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.verdict.value = "pass"
            mock_result.risk_score = 0.0
            mock_result.vulnerabilities = []
            mock_instance.run = AsyncMock(return_value=mock_result)
            mock_runner.return_value = mock_instance

            with patch("aragora.mcp.tools_module.gauntlet.GauntletConfig"):
                result = await run_gauntlet_tool(
                    content="test content",
                    content_type="code",
                    profile="security",
                )
                assert result["content_type"] == "code"
                assert result["profile"] == "security"


# =============================================================================
# Content Type Tests
# =============================================================================


class TestGauntletContentTypes:
    """Tests for different content types."""

    @pytest.mark.asyncio
    async def test_spec_content_type(self):
        """Spec content type is passed to config."""
        with patch("aragora.mcp.tools_module.gauntlet.GauntletRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.verdict.value = "pass"
            mock_result.risk_score = 0.1
            mock_result.vulnerabilities = []
            mock_instance.run = AsyncMock(return_value=mock_result)
            mock_runner.return_value = mock_instance

            with patch("aragora.mcp.tools_module.gauntlet.GauntletConfig") as mock_config:
                await run_gauntlet_tool(content="API spec", content_type="spec")
                call_kwargs = mock_config.call_args.kwargs
                assert call_kwargs.get("input_type") == "spec"

    @pytest.mark.asyncio
    async def test_code_content_type(self):
        """Code content type is passed to config."""
        with patch("aragora.mcp.tools_module.gauntlet.GauntletRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.verdict.value = "pass"
            mock_result.risk_score = 0.1
            mock_result.vulnerabilities = []
            mock_instance.run = AsyncMock(return_value=mock_result)
            mock_runner.return_value = mock_instance

            with patch("aragora.mcp.tools_module.gauntlet.GauntletConfig") as mock_config:
                await run_gauntlet_tool(content="def test(): pass", content_type="code")
                call_kwargs = mock_config.call_args.kwargs
                assert call_kwargs.get("input_type") == "code"

    @pytest.mark.asyncio
    async def test_policy_content_type(self):
        """Policy content type is passed to config."""
        with patch("aragora.mcp.tools_module.gauntlet.GauntletRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.verdict.value = "pass"
            mock_result.risk_score = 0.1
            mock_result.vulnerabilities = []
            mock_instance.run = AsyncMock(return_value=mock_result)
            mock_runner.return_value = mock_instance

            with patch("aragora.mcp.tools_module.gauntlet.GauntletConfig") as mock_config:
                await run_gauntlet_tool(content="Policy document", content_type="policy")
                call_kwargs = mock_config.call_args.kwargs
                assert call_kwargs.get("input_type") == "policy"
