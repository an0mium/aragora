"""
Tests for the security debate runner (aragora.debate.security_response).

Covers build_security_debate_question and trigger_security_debate, relocated
from aragora.events.security_events as part of the P4a events/queue layering
split (E7a): these functions are the domain-coupled half of the security
event module (they import Arena, DebateProtocol, and API agents), so they
live under aragora.debate rather than the domain-free events module.
"""

import subprocess
import sys
import types
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.security_response import (
    _get_security_debate_agents,
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
            description="AWS access key AKIA1234567890SECRET found in source code",
            metadata={"secret_type": "aws_access_key"},
        )
        event = SecurityEvent(findings=[finding])
        q = build_security_debate_question(event)
        assert "aws_access_key" in q
        assert "secrets" in q.lower()
        assert "AKIA1234567890SECRET" not in q
        assert "Exposed API key" not in q
        assert "[redacted secret finding description]" in q

    def test_question_redacts_secret_alias_findings(self):
        """Prompt redaction should match context redaction for aliases."""
        finding = SecurityFinding(
            id="f-token",
            finding_type="Credential",
            severity=SecuritySeverity.HIGH,
            title="Token abc123-secret",
            description="credential value abc123-secret",
            metadata={"secret_type": "api_token"},
        )
        event = SecurityEvent(findings=[finding])

        q = build_security_debate_question(event)

        assert "abc123-secret" not in q
        assert "Secret finding" in q

    def test_question_redacts_mislabeled_secret_from_summary(self):
        """Secret-like metadata should win over a misleading vulnerability type."""
        finding = SecurityFinding(
            id="f-mislabeled-token",
            finding_type="vulnerability",
            severity=SecuritySeverity.HIGH,
            title="Token abc123-secret",
            description="credential value abc123-secret",
            metadata={"secret_type": "api_token"},
        )
        event = SecurityEvent(findings=[finding])

        q = build_security_debate_question(event)

        assert "abc123-secret" not in q
        assert "api_token" in q
        assert "exposed secrets" in q
        assert "vulnerabilities" not in q

    def test_question_keeps_sast_token_vulnerability_context(self):
        """SAST vulnerabilities mentioning tokens should remain vulnerability context."""
        finding = SecurityFinding(
            id="f-csrf-token",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="Missing CSRF token validation",
            description="POST handler accepts requests without verifying the CSRF token.",
            metadata={
                "scanner": "semgrep",
                "rule_id": "python.django.security.csrf-token-missing",
                "message": "Missing CSRF token validation",
                "snippet": "csrf_token = request.headers.get('X-CSRF-Token')",
            },
        )
        event = SecurityEvent(repository="org/app", findings=[finding])

        q = build_security_debate_question(event)

        assert "vulnerabilities" in q
        assert "Missing CSRF token validation" in q
        assert "CSRF token" in q
        assert "Secret finding" not in q
        assert "exposed secrets" not in q

    def test_question_redacts_sast_scanner_bridge_secret_text(self):
        """Scanner-emitted generic vulnerabilities should not leak hardcoded secrets."""
        finding = SecurityFinding(
            id="f-hardcoded-password",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="hardcoded-password",
            description="Matched code: password = 'literal-secret'",
            metadata={
                "scanner": "semgrep",
                "snippet": "password = 'literal-secret'",
                "rule_source": "semgrep",
                "vulnerability_class": "hardcoded-password",
                "confidence": 0.95,
            },
        )
        event = SecurityEvent(repository="org/app", findings=[finding])

        q = build_security_debate_question(event)

        assert "exposed secrets" in q
        assert "Secret finding" in q
        assert "[redacted secret finding description]" in q
        assert "literal-secret" not in q
        assert "hardcoded-password" not in q
        assert "password =" not in q
        assert "vulnerabilities" not in q

    def test_question_uses_event_rule_metadata_for_sast_secret_redaction(self):
        """Event-level SAST secret rule metadata should redact prompt details."""
        finding = SecurityFinding(
            id="f-event-rule-secret",
            finding_type="sast",
            severity=SecuritySeverity.CRITICAL,
            title="Scanner finding",
            description="Matched code: literal-secret",
            metadata={
                "scanner": "semgrep",
                "snippet": "literal-secret",
            },
        )
        event = SecurityEvent(
            repository="org/app",
            findings=[finding],
            metadata={"rule_id": "python.lang.security.audit.hardcoded-credential"},
        )

        q = build_security_debate_question(event)

        assert "exposed secrets" in q
        assert "Secret finding" in q
        assert "literal-secret" not in q
        assert "Scanner finding" not in q

    def test_question_sanitizes_secret_type_metadata(self):
        """Prompt secret summaries should not echo scanner-controlled secret_type."""
        finding = SecurityFinding(
            id="f-malformed-secret-type",
            finding_type="secret",
            severity=SecuritySeverity.CRITICAL,
            title="Token literal-secret",
            description="literal-secret",
            metadata={"secret_type": "literal-secret"},
        )
        event = SecurityEvent(repository="org/app", findings=[finding])

        q = build_security_debate_question(event)

        assert "exposed secrets (unknown)" in q
        assert "literal-secret" not in q
        assert "Secret finding" in q

    def test_question_keeps_credential_vulnerability_context(self):
        """Credential-mentioning CVE text should remain vulnerability context."""
        finding = SecurityFinding(
            id="f-credential-redirect",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="Credential redirect vulnerability",
            description="Leaking Authorization credentials on cross-host redirect.",
            cve_id="CVE-2024-9999",
            package_name="requests",
            metadata={
                "scanner": "dependency-audit",
                "message": "Leaking Authorization credentials on cross-host redirect.",
            },
        )
        event = SecurityEvent(repository="org/app", findings=[finding])

        q = build_security_debate_question(event)

        assert "vulnerabilities" in q
        assert "CVE-2024-9999" in q
        assert "requests" in q
        assert "Credential redirect vulnerability" in q
        assert "Authorization credentials" in q
        assert "Secret finding" not in q
        assert "exposed secrets" not in q

    def test_question_keeps_mixed_event_vulnerability_context(self):
        """Event-level secret metadata should not erase unrelated CVEs."""
        secret_finding = SecurityFinding(
            id="metadata-secret-key",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="Config metadata finding",
            description="Scanner reported sensitive metadata.",
            metadata={"secret_key": "literal-secret"},
        )
        cve_finding = SecurityFinding(
            id="f-credential-redirect",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="Credential redirect vulnerability",
            description="Leaking Authorization credentials on cross-host redirect.",
            cve_id="CVE-2024-9999",
            package_name="requests",
        )
        event = SecurityEvent(
            repository="org/app",
            findings=[secret_finding, cve_finding],
            metadata={
                "rule_id": "python.lang.security.audit.hardcoded-credential",
                "raw_secret": "literal-secret",
            },
        )

        q = build_security_debate_question(event)

        assert "exposed secrets" in q
        assert "vulnerabilities" in q
        assert "CVE-2024-9999" in q
        assert "requests" in q
        assert "Credential redirect vulnerability" in q
        assert "Authorization credentials" in q
        assert "literal-secret" not in q

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
            # Simulate ImportError when trying to import the canonical runner
            with patch.dict("sys.modules", {"aragora.debate.security_debate": None}):
                result = await trigger_security_debate(event)
                # Should gracefully return None (either ImportError or other exception)
                # The function catches ImportError and general Exception
                assert result is None

    @pytest.mark.asyncio
    async def test_trigger_debate_uses_canonical_runner_and_stores_result(self):
        """Should use the canonical runner and store its arena debate ID."""
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

        mock_result = MagicMock()
        mock_result.debate_id = "arena-security-debate-1"
        mock_result.consensus_reached = True
        mock_result.confidence = 0.9
        mock_result.final_answer = "Fix it"
        mock_result.messages = [MagicMock()]
        mock_result.participants = ["security-auditor"]
        mock_result.rounds_used = 3
        mock_result.metadata = {"security_confidence_threshold_met": True}

        with (
            patch(
                "aragora.debate.security_debate.run_security_debate",
                new_callable=AsyncMock,
                return_value=mock_result,
            ) as mock_run,
            patch(
                "aragora.debate.security_response._store_security_debate_result",
                new_callable=AsyncMock,
            ) as mock_store,
        ):
            result = await trigger_security_debate(event)

            mock_run.assert_awaited_once_with(
                event=event,
                agents=None,
                confidence_threshold=0.7,
                timeout_seconds=300,
                org_id="default",
            )
            assert result == "arena-security-debate-1"
            assert event.debate_requested is True
            assert event.debate_id == "arena-security-debate-1"
            mock_store.assert_awaited_once()
            assert mock_store.await_args.args[0] == "arena-security-debate-1"

    @pytest.mark.asyncio
    async def test_trigger_debate_returns_none_when_canonical_runner_has_no_agents(self):
        """A canonical no-agent result should preserve trigger's None contract."""
        event = SecurityEvent(repository="org/repo")
        mock_result = MagicMock()
        mock_result.debate_id = "empty-debate"
        mock_result.consensus_reached = False
        mock_result.confidence = 0.0
        mock_result.final_answer = "No agents available for security debate"
        mock_result.messages = []
        mock_result.participants = []
        mock_result.rounds_used = 0
        mock_result.metadata = {"security_confidence_threshold_met": True}

        with (
            patch(
                "aragora.debate.security_debate.run_security_debate",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "aragora.debate.security_response._store_security_debate_result",
                new_callable=AsyncMock,
            ) as mock_store,
        ):
            result = await trigger_security_debate(event)

        assert result is None
        assert event.debate_requested is False
        assert event.debate_id is None
        mock_store.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_trigger_debate_stores_low_confidence_result_for_audit(self):
        """Low-confidence completed debates should stay retrievable for audit."""
        event = SecurityEvent(repository="org/repo")
        mock_result = MagicMock()
        mock_result.debate_id = "low-confidence-debate"
        mock_result.consensus_reached = False
        mock_result.confidence = 0.42
        mock_result.final_answer = "Weak consensus"
        mock_result.messages = [MagicMock()]
        mock_result.participants = ["security-auditor"]
        mock_result.rounds_used = 3
        mock_result.metadata = {"security_confidence_threshold_met": False}
        event.debate_requested = True
        event.debate_id = "low-confidence-debate"

        with (
            patch(
                "aragora.debate.security_debate.run_security_debate",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "aragora.debate.security_response._store_security_debate_result",
                new_callable=AsyncMock,
            ) as mock_store,
        ):
            result = await trigger_security_debate(event, confidence_threshold=0.7)

        assert result == "low-confidence-debate"
        assert event.debate_requested is True
        assert event.debate_id == "low-confidence-debate"
        mock_store.assert_awaited_once()
        assert mock_store.await_args.args == ("low-confidence-debate", event, mock_result)

    @pytest.mark.asyncio
    async def test_trigger_debate_fails_closed_without_threshold_metadata(self):
        """Canonical results must prove the threshold gate before counting."""
        event = SecurityEvent(repository="org/repo")
        mock_result = MagicMock()
        mock_result.debate_id = "missing-threshold-metadata"
        mock_result.consensus_reached = True
        mock_result.confidence = 1.0
        mock_result.final_answer = "Looks good"
        mock_result.messages = [MagicMock()]
        mock_result.participants = ["security-auditor"]
        mock_result.rounds_used = 3
        mock_result.metadata = {}

        with (
            patch(
                "aragora.debate.security_debate.run_security_debate",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch(
                "aragora.debate.security_response._store_security_debate_result",
                new_callable=AsyncMock,
            ) as mock_store,
        ):
            result = await trigger_security_debate(event)

        assert result is None
        assert event.debate_requested is False
        assert event.debate_id is None
        mock_store.assert_not_awaited()


# =============================================================================
# Agent selection
# =============================================================================


class TestSecurityDebateAgentSelection:
    """Tests for security debate agent discovery."""

    @pytest.mark.asyncio
    async def test_uses_deployment_agent_factory_before_direct_api_agents(self, monkeypatch):
        fake_agents = [MagicMock(name="factory-agent")]
        factory_module = types.ModuleType("aragora.agents.factory")
        factory_module.get_available_agents = AsyncMock(return_value=fake_agents)
        monkeypatch.setitem(sys.modules, "aragora.agents.factory", factory_module)

        agents = await _get_security_debate_agents()

        assert agents == fake_agents
        factory_module.get_available_agents.assert_awaited_once_with(
            capabilities=["security", "code_analysis"],
            min_count=2,
            max_count=4,
        )


# =============================================================================
# Consumer registration side effect (real cold-import, not mocked)
# =============================================================================


class TestConsumerRegistrationSideEffect:
    """
    Regression test proving that importing a real production consumer
    module and initializing it for event emission wires the debate runner
    without relying on unrelated transitive imports.

    Runs in a fresh subprocess so module-caching from other tests (or from
    importing aragora.debate.security_response directly earlier in this
    file) cannot mask a missing registration.
    """

    def test_sast_scanner_initialize_registers_runner(self):
        """Initializing the SAST scanner for event emission registers the runner."""
        script = textwrap.dedent(
            """
            import asyncio

            from aragora.analysis.codebase.sast.models import SASTConfig
            from aragora.analysis.codebase.sast.scanner import SASTScanner
            import aragora.events.security_events as security_events_mod
            from aragora.events.security_events import get_security_debate_runner

            # Reset to the never-set sentinel (NOT an explicit None-clear,
            # which now sticks and disables the lazy default import) so this
            # simulates a cold consumer even if transitive imports already
            # registered the default runner.
            security_events_mod._security_debate_runner = (
                security_events_mod._UNSET_RUNNER
            )
            assert get_security_debate_runner() is None

            async def main():
                scanner = SASTScanner(
                    config=SASTConfig(use_semgrep=False, emit_security_events=True)
                )
                await scanner.initialize()

            asyncio.run(main())
            runner = get_security_debate_runner()
            assert runner is not None, 'no security debate runner registered'
            assert runner.__name__ == 'trigger_security_debate'
            print('OK')
            """
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

    def test_default_runner_import_does_not_replace_registered_runner(self):
        """Default module import must not clobber an explicit runner hook."""
        script = textwrap.dedent(
            """
            from aragora.events.security_events import (
                get_security_debate_runner,
                register_security_debate_runner,
            )

            async def custom_runner(event, **kwargs):
                return 'custom-debate'

            register_security_debate_runner(custom_runner)
            import aragora.debate.security_response  # noqa: F401

            assert get_security_debate_runner() is custom_runner
            print('OK')
            """
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

    @pytest.mark.asyncio
    async def test_sast_initialize_keeps_emitter_when_debate_runner_import_fails(
        self,
        monkeypatch,
    ):
        """SAST event emission should not depend on optional debate imports."""
        import builtins

        import aragora.events.security_events as security_events
        from aragora.analysis.codebase.sast.models import SASTConfig
        from aragora.analysis.codebase.sast.scanner import SASTScanner

        sentinel_emitter = object()
        monkeypatch.setattr(security_events, "get_security_emitter", lambda: sentinel_emitter)

        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "aragora.debate" and "security_response" in (fromlist or ()):
                raise ImportError("debate stack unavailable")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        scanner = SASTScanner(config=SASTConfig(use_semgrep=False, emit_security_events=True))
        await scanner.initialize()

        assert scanner._security_emitter is sentinel_emitter
