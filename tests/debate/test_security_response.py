"""
Tests for the security debate runner (aragora.debate.security_response).

Covers build_security_debate_question and trigger_security_debate, relocated
from aragora.events.security_events as part of the P4a events/queue layering
split (E7a): these functions are the domain-coupled half of the security
event module (they import Arena, DebateProtocol, and API agents), so they
live under aragora.debate rather than the domain-free events module.
"""

import json
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.security_response import (
    build_security_debate_question,
    trigger_security_debate,
)
from aragora.events.security_events import (
    SecurityEvent,
    SecurityFinding,
    SecuritySeverity,
)


# =============================================================================
# build_security_debate_question
# =============================================================================


class TestBuildSecurityDebateQuestion:
    """Tests for debate question construction from events."""

    def test_question_with_no_findings(self):
        """Should produce a fallback question when no findings exist."""
        event = SecurityEvent(repository="org/repo")
        q = build_security_debate_question(event)
        assert "org/repo" in q
        assert "remediation" in q.lower()

    def test_question_with_no_findings_no_repo(self):
        """Should use 'the codebase' when no repository is set."""
        event = SecurityEvent()
        q = build_security_debate_question(event)
        assert "the codebase" in q

    def test_question_with_vulnerability_findings(self):
        """Should include vulnerability details in the question."""
        finding = SecurityFinding(
            id="f-1",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="Remote Code Execution",
            description="RCE via unsafe deserialization in pickle module",
            cve_id="CVE-2024-12345",
            package_name="pickle-lib",
        )
        event = SecurityEvent(
            repository="org/app",
            findings=[finding],
        )
        q = build_security_debate_question(event)
        assert "CVE-2024-12345" in q
        assert "pickle-lib" in q
        assert "org/app" in q
        assert "remediation" in q.lower()

    def test_question_with_secret_findings(self):
        """Should include secret type information in the question."""
        finding = SecurityFinding(
            id="f-2",
            finding_type="secret",
            severity=SecuritySeverity.HIGH,
            title="Exposed API key",
            description="AWS access key found in source code",
            metadata={"secret_type": "aws_access_key"},
        )
        event = SecurityEvent(findings=[finding])
        q = build_security_debate_question(event)
        assert "aws_access_key" in q
        assert "secrets" in q.lower()

    def test_question_with_mixed_findings(self):
        """Should include both vulnerability and secret details."""
        vuln = SecurityFinding(
            id="f-v",
            finding_type="vulnerability",
            severity=SecuritySeverity.HIGH,
            title="SQL Injection",
            description="User input concatenated in SQL query",
            cve_id="CVE-2024-99999",
            package_name="sqlalchemy",
        )
        secret = SecurityFinding(
            id="f-s",
            finding_type="secret",
            severity=SecuritySeverity.HIGH,
            title="Exposed token",
            description="GitHub token in config file",
            metadata={"secret_type": "github_token"},
        )
        event = SecurityEvent(findings=[vuln, secret])
        q = build_security_debate_question(event)
        assert "vulnerabilities" in q.lower()
        assert "secrets" in q.lower()

    def test_question_limits_to_five_findings(self):
        """Should limit to at most 5 findings in the question."""
        findings = [
            SecurityFinding(
                id=f"f-{i}",
                finding_type="vulnerability",
                severity=SecuritySeverity.HIGH,
                title=f"Vuln {i}",
                description=f"Description {i}",
                cve_id=f"CVE-2024-{i:05d}",
                package_name=f"pkg-{i}",
            )
            for i in range(10)
        ]
        event = SecurityEvent(findings=findings)
        q = build_security_debate_question(event)
        # The details section should list at most 5 findings (limited at the top)
        detail_lines = [line for line in q.split("\n") if line.strip().startswith("- ")]
        assert len(detail_lines) <= 5

    def test_question_includes_remediation_structure(self):
        """Question should ask about mitigations, root cause, prevention."""
        finding = SecurityFinding(
            id="f-struct",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="Critical vuln",
            description="Description",
        )
        event = SecurityEvent(findings=[finding])
        q = build_security_debate_question(event)
        assert "Immediate mitigations" in q
        assert "Root cause" in q
        assert "Preventive measures" in q
        assert "Impact" in q


# =============================================================================
# trigger_security_debate integration
# =============================================================================


class TestTriggerSecurityDebate:
    """Tests for the trigger_security_debate function."""

    @pytest.mark.asyncio
    async def test_trigger_debate_returns_none_on_import_error(self):
        """Should return None gracefully when Arena is not importable."""
        event = SecurityEvent(
            severity=SecuritySeverity.CRITICAL,
            findings=[
                SecurityFinding(
                    id="f-1",
                    finding_type="vulnerability",
                    severity=SecuritySeverity.CRITICAL,
                    title="Test",
                    description="Test desc",
                )
            ],
        )

        with patch(
            "aragora.debate.security_response.build_security_debate_question",
            return_value="test question",
        ):
            # Simulate ImportError when trying to import Arena dependencies
            with patch.dict("sys.modules", {"aragora.core": None}):
                result = await trigger_security_debate(event)
                # Should gracefully return None (either ImportError or other exception)
                # The function catches ImportError and general Exception
                assert result is None

    @pytest.mark.asyncio
    async def test_trigger_debate_sets_debate_question(self):
        """Should set the debate_question on the event."""
        event = SecurityEvent(
            severity=SecuritySeverity.CRITICAL,
            repository="org/repo",
            findings=[
                SecurityFinding(
                    id="f-q",
                    finding_type="vulnerability",
                    severity=SecuritySeverity.CRITICAL,
                    title="RCE",
                    description="Remote code execution",
                    cve_id="CVE-2024-99999",
                    package_name="vuln-pkg",
                )
            ],
        )

        # Mock the entire chain: imports, Arena, result
        mock_arena_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.9
        mock_result.final_answer = "Fix it"
        mock_arena_instance.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "aragora.debate.security_response.build_security_debate_question",
                return_value="Generated question",
            ),
            patch(
                "aragora.debate.security_response._get_security_debate_agents",
                new_callable=AsyncMock,
                return_value=[MagicMock(), MagicMock()],
            ),
            patch(
                "aragora.debate.security_response._store_security_debate_result",
                new_callable=AsyncMock,
            ),
        ):
            # We need to mock the imports inside the function
            mock_env = MagicMock()
            mock_protocol = MagicMock()

            with patch.dict(
                "sys.modules",
                {
                    "aragora.core": MagicMock(Environment=mock_env, DebateResult=MagicMock()),
                    "aragora.debate.protocol": MagicMock(DebateProtocol=mock_protocol),
                    "aragora.debate.orchestrator": MagicMock(
                        Arena=MagicMock(return_value=mock_arena_instance)
                    ),
                },
            ):
                result = await trigger_security_debate(event)

                assert event.debate_question == "Generated question"
                assert result is not None
                assert result.startswith("security_debate_")

                # context must be a JSON string (Environment.context: str), not a
                # raw dict -- regression test for a pre-existing bug where this
                # was passed as cast(str, {...}) and only reached via the emitter;
                # the dispatcher default path now also depends on this being real
                # JSON (see aragora.events.security_dispatcher's default runner).
                _, call_kwargs = mock_env.call_args
                context = call_kwargs["context"]
                assert isinstance(context, str)
                decoded = json.loads(context)
                assert decoded["security_event_id"] == event.id
                assert decoded["repository"] == "org/repo"
                assert decoded["findings"][0]["cve_id"] == "CVE-2024-99999"


# =============================================================================
# Consumer registration side effect (real cold-import, not mocked)
# =============================================================================


class TestConsumerRegistrationSideEffect:
    """
    Regression tests proving that importing a real production consumer
    module -- without ever explicitly referencing aragora.debate -- leaves
    the security debate runner registered.

    Each check runs in a fresh subprocess so module-caching from other tests
    (or from importing aragora.debate.security_response directly earlier in
    this file) cannot mask a missing registration.
    """

    def _assert_runner_registered_after_import(self, import_line: str) -> None:
        script = (
            f"{import_line}\n"
            "from aragora.events.security_events import get_security_debate_runner\n"
            "runner = get_security_debate_runner()\n"
            "assert runner is not None, 'no security debate runner registered'\n"
            "assert runner.__name__ == 'trigger_security_debate'\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"subprocess failed (rc={result.returncode})\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "OK" in result.stdout

    def test_sast_scanner_import_registers_runner(self):
        """Importing the SAST scanner module alone must register the runner."""
        self._assert_runner_registered_after_import("import aragora.analysis.codebase.sast.scanner")

    def test_server_security_events_handler_import_registers_runner(self):
        """Importing the server security-events handler alone must register the runner."""
        self._assert_runner_registered_after_import(
            "import aragora.server.handlers.codebase.security.events"
        )
