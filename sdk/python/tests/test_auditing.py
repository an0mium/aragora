"""Tests for Auditing namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# Capability Probe Operations
# =========================================================================


class TestAuditingCapabilityProbe:
    """Tests for capability probe operations."""

    def test_probe_capability_basic(self) -> None:
        """Probe agent capability with basic parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "agent": "claude",
                "task": "Code review",
                "score": 0.92,
                "capabilities": {"analysis": True, "reasoning": True},
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.auditing.probe_capability(agent="claude", task="Code review")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates/capability-probe",
                json={"agent": "claude", "task": "Code review"},
            )
            assert result["score"] == 0.92
            client.close()

    def test_probe_capability_with_config(self) -> None:
        """Probe agent capability with configuration."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"agent": "gpt-4", "score": 0.88}

            client = AragoraClient(base_url="https://api.aragora.ai")
            config = {"temperature": 0.5, "max_tokens": 1000}
            client.auditing.probe_capability(agent="gpt-4", task="Summarization", config=config)

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["agent"] == "gpt-4"
            assert json_data["task"] == "Summarization"
            assert json_data["config"] == config
            client.close()


# =========================================================================
# Deep Audit Operations
# =========================================================================


class TestAuditingDeepAudit:
    """Tests for deep audit operations."""

    def test_deep_audit_basic(self) -> None:
        """Run deep audit with basic parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "audit_id": "audit_123",
                "task": "Security review",
                "findings": [],
                "summary": "No critical issues found",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.auditing.deep_audit(task="Security review")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates/deep-audit",
                json={"task": "Security review", "depth": "standard"},
            )
            assert result["audit_id"] == "audit_123"
            client.close()

    def test_deep_audit_with_all_options(self) -> None:
        """Run deep audit with all options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "audit_id": "audit_456",
                "findings": [
                    {
                        "category": "security",
                        "severity": "high",
                        "description": "SQL injection vulnerability",
                    }
                ],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.auditing.deep_audit(
                task="Code review",
                agents=["claude", "gpt-4"],
                depth="deep",
                config={"focus": "security"},
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["task"] == "Code review"
            assert json_data["agents"] == ["claude", "gpt-4"]
            assert json_data["depth"] == "deep"
            assert json_data["config"] == {"focus": "security"}
            client.close()


# =========================================================================
# Red Team Operations
# =========================================================================


class TestAuditingRedTeam:
    """Tests for red team operations."""

    def test_red_team_basic(self) -> None:
        """Run red team analysis with basic parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "debate_id": "deb_123",
                "attacks_run": 10,
                "vulnerabilities_found": 2,
                "findings": [],
                "summary": "Some vulnerabilities detected",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.auditing.red_team(debate_id="deb_123")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/debates/deb_123/red-team",
                json={"intensity": "medium"},
            )
            assert result["attacks_run"] == 10
            assert result["vulnerabilities_found"] == 2
            client.close()

    def test_red_team_with_all_options(self) -> None:
        """Run red team analysis with all options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debate_id": "deb_456", "attacks_run": 25}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.auditing.red_team(
                debate_id="deb_456",
                attack_types=["prompt_injection", "jailbreak"],
                intensity="high",
                config={"iterations": 5},
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["attack_types"] == ["prompt_injection", "jailbreak"]
            assert json_data["intensity"] == "high"
            assert json_data["config"] == {"iterations": 5}
            client.close()


# =========================================================================
# Attack Types Operations
# =========================================================================


class TestAuditingAttackTypes:
    """Tests for attack types operations."""

    def test_get_attack_types(self) -> None:
        """Get available attack types."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "attack_types": [
                    {
                        "id": "prompt_injection",
                        "name": "Prompt Injection",
                        "description": "Attempts to inject malicious prompts",
                        "category": "security",
                    },
                    {
                        "id": "jailbreak",
                        "name": "Jailbreak",
                        "description": "Attempts to bypass safety filters",
                        "category": "safety",
                    },
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.auditing.get_attack_types()

            mock_request.assert_called_once_with("GET", "/api/v1/redteam/attack-types")
            assert len(result["attack_types"]) == 2
            assert result["attack_types"][0]["id"] == "prompt_injection"
            client.close()


# =========================================================================
# Async Tests
# =========================================================================


class TestAsyncAuditing:
    """Tests for async Auditing API."""

    @pytest.mark.asyncio
    async def test_async_probe_capability(self) -> None:
        """Probe capability asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"agent": "claude", "score": 0.95}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.auditing.probe_capability(agent="claude", task="Analysis")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/debates/capability-probe",
                    json={"agent": "claude", "task": "Analysis"},
                )
                assert result["score"] == 0.95

    @pytest.mark.asyncio
    async def test_async_deep_audit(self) -> None:
        """Run deep audit asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"audit_id": "audit_789", "findings": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.auditing.deep_audit(task="Review code")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/debates/deep-audit",
                    json={"task": "Review code", "depth": "standard"},
                )
                assert result["audit_id"] == "audit_789"

    @pytest.mark.asyncio
    async def test_async_red_team(self) -> None:
        """Run red team asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"debate_id": "deb_999", "attacks_run": 15}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.auditing.red_team(debate_id="deb_999")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/debates/deb_999/red-team",
                    json={"intensity": "medium"},
                )
                assert result["attacks_run"] == 15

    @pytest.mark.asyncio
    async def test_async_get_attack_types(self) -> None:
        """Get attack types asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"attack_types": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.auditing.get_attack_types()

                mock_request.assert_called_once_with("GET", "/api/v1/redteam/attack-types")
                assert "attack_types" in result
